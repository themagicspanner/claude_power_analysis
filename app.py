"""
app.py — Dash application for cycling power analysis.

Usage
─────
  python app.py          # starts on http://127.0.0.1:8050

Auto-watch
──────────
  Dropping a new .fit file into the raw_data/ folder is all that is needed.
  A background watchdog thread detects it, processes it into cycling.db,
  and the UI refreshes automatically within a few seconds.

Pages / sections
────────────────
  • Ride selector dropdown (auto-updates when new rides appear)
  • Power vs time for the selected ride
  • MMP for the selected ride vs 90-day best
  • All-rides MMP overview
  • PDC parameter history (AWC, Pmax, MAP over time)
"""

import datetime
import os
import sqlite3
import threading
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import dash
from dash import dcc, html, dash_table, Input, Output, State
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from build_database import (
    init_db, process_ride, backfill_pdc_params, recompute_all_pdc_params,
    _power_model, _fit_power_curve,
    PDC_K, PDC_INFLECTION, PDC_WINDOW,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "cycling.db")
FIT_DIR  = os.path.join(BASE_DIR, "raw_data")

# ── Shared data store (updated by watcher, read by callbacks) ─────────────────

_lock         = threading.Lock()
_data_version = 0   # incremented every time new data lands in the DB
_rides        = None
_mmp_all      = None
_pdc_params   = None


def _reload():
    """Re-read rides, mmp, and pdc_params from the DB and bump the version counter."""
    global _rides, _mmp_all, _pdc_params, _data_version
    r = _load_rides()
    m = _load_mmp_all(r)
    p = _load_pdc_params()
    with _lock:
        _rides        = r
        _mmp_all      = m
        _pdc_params   = p
        _data_version += 1


def get_data():
    """Return a consistent (version, rides, mmp_all, pdc_params) snapshot."""
    with _lock:
        return _data_version, _rides.copy(), _mmp_all.copy(), _pdc_params.copy()


# ── Duration formatting ────────────────────────────────────────────────────────

LOG_TICK_S   = [1, 5, 10, 30, 60, 120, 300, 600, 1200, 1800, 3600]
LOG_TICK_LBL = ["1s", "5s", "10s", "30s", "1min", "2min",
                 "5min", "10min", "20min", "30min", "1h"]

COLOURS = px.colors.qualitative.Light24


# ── Database helpers ───────────────────────────────────────────────────────────

def _load_rides() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        """SELECT id, name, ride_date,
                  round(duration_s / 60.0, 1) AS duration_min,
                  avg_power, max_power
           FROM rides ORDER BY ride_date, name""",
        conn,
    )
    conn.close()
    return df


def _load_mmp_all(rides: pd.DataFrame) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    mmp = pd.read_sql("SELECT ride_id, duration_s, power FROM mmp", conn)
    conn.close()
    mmp = mmp.merge(
        rides.rename(columns={"id": "ride_id"})[["ride_id", "name", "ride_date"]],
        on="ride_id",
    )
    return mmp


def _load_pdc_params() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT ride_id, AWC, Pmax, MAP, tau2, ftp, normalized_power, intensity_factor,"
        " tss, tss_map, tss_awc FROM pdc_params",
        conn,
    )
    conn.close()
    return df


def load_records(ride_id: int) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT elapsed_s, power FROM records WHERE ride_id = ? ORDER BY elapsed_s",
        conn,
        params=(ride_id,),
    )
    conn.close()
    df["elapsed_min"] = df["elapsed_s"] / 60.0
    return df


# ── One-time DB migration ─────────────────────────────────────────────────────

# Bump this when stored PDC params need to be fully recomputed.
_DB_SCHEMA_VERSION = 2


def _maybe_migrate(conn: sqlite3.Connection) -> None:
    """Recompute all PDC params once if the DB schema version is stale.

    Reads/writes the 'schema_version' key in the db_meta table.  After the
    first successful migration the version is stored so subsequent startups
    are fast.
    """
    row = conn.execute(
        "SELECT value FROM db_meta WHERE key='schema_version'"
    ).fetchone()
    current = int(row[0]) if row else 0
    if current < _DB_SCHEMA_VERSION:
        print(f"[migrate] DB schema v{current} → v{_DB_SCHEMA_VERSION}: recomputing PDC params …")
        recompute_all_pdc_params(conn)
        conn.execute(
            "INSERT OR REPLACE INTO db_meta (key, value) VALUES ('schema_version', ?)",
            (str(_DB_SCHEMA_VERSION),),
        )
        conn.commit()
        print("[migrate] Done.")


# ── Watchdog ──────────────────────────────────────────────────────────────────

class _FitHandler(FileSystemEventHandler):
    """Process any new .fit file that lands in FIT_DIR."""

    def _handle(self, path: str) -> None:
        if not path.lower().endswith(".fit"):
            return
        # Small delay to let the file finish copying before we read it
        time.sleep(1)
        print(f"[watcher] New file detected: {path}")
        try:
            conn = sqlite3.connect(DB_PATH)
            init_db(conn)
            process_ride(conn, path)
            conn.close()
            _reload()
            print(f"[watcher] Processed and reloaded data.")
        except Exception as exc:
            print(f"[watcher] Error processing {path}: {exc}")

    def on_created(self, event):
        if not event.is_directory:
            self._handle(event.src_path)

    def on_moved(self, event):
        if not event.is_directory:
            self._handle(event.dest_path)


def _start_watcher() -> None:
    os.makedirs(FIT_DIR, exist_ok=True)
    observer = Observer()
    observer.schedule(_FitHandler(), FIT_DIR, recursive=False)
    observer.daemon = True
    observer.start()
    print(f"[watcher] Watching {FIT_DIR} for new .fit files …")


# ── PDC on-the-fly helper ─────────────────────────────────────────────────────

def _fit_pdc_for_ride(ride: pd.Series, mmp_all: pd.DataFrame) -> dict | None:
    """Fit the PDC on-the-fly for a ride using the same method as fig_pdc_at_date.

    Returns a dict with keys AWC, Pmax, MAP, tau2, ftp (all floats), or None
    if there is insufficient data for a fit.  Using this for W'bal and TSS
    component charts guarantees they use the same parameters as the PDC chart.
    """
    ride_date     = ride["ride_date"]
    ride_date_obj = datetime.date.fromisoformat(ride_date)
    cutoff        = (ride_date_obj - datetime.timedelta(days=PDC_WINDOW)).isoformat()

    window = mmp_all[mmp_all["ride_date"].between(cutoff, ride_date)].copy()
    if window.empty:
        return None

    window["age_days"]   = window["ride_date"].apply(
        lambda d: (ride_date_obj - datetime.date.fromisoformat(d)).days
    )
    window["weight"]     = 1.0 / (1.0 + np.exp(PDC_K * (window["age_days"] - PDC_INFLECTION)))
    window["aged_power"] = window["power"] * window["weight"]

    aged = (
        window.groupby("duration_s")["aged_power"]
        .max().reset_index().sort_values("duration_s")
    )
    dur = aged["duration_s"].to_numpy(dtype=float)
    pwr = aged["aged_power"].to_numpy(dtype=float)

    if len(dur) < 4:
        return None

    popt, ok = _fit_power_curve(dur, pwr)
    if not ok:
        return None

    AWC, Pmax, MAP, tau2 = popt
    return {
        "AWC":  float(AWC),
        "Pmax": float(Pmax),
        "MAP":  float(MAP),
        "tau2": float(tau2),
        "ftp":  float(_power_model(3600.0, AWC, Pmax, MAP, tau2)),
    }


# ── Figure builders ───────────────────────────────────────────────────────────

def fig_power(records: pd.DataFrame, ride_name: str) -> go.Figure:
    fig = go.Figure()
    if records["power"].notna().any():
        fig.add_trace(go.Scatter(
            x=records["elapsed_min"], y=records["power"],
            mode="lines", name="Power",
            line=dict(color="darkorange", width=1),
        ))
    fig.update_xaxes(title_text="Elapsed Time (min)", showgrid=True, gridcolor="lightgrey")
    fig.update_yaxes(title_text="Power (W)", showgrid=True, gridcolor="lightgrey")
    fig.update_layout(
        title=dict(text=ride_name.replace("_", " "), font=dict(size=15)),
        height=260, margin=dict(t=55, b=40, l=60, r=20),
        showlegend=False, template="plotly_white",
    )
    return fig


def fig_mmp_pdc(ride: pd.Series, mmp_all: pd.DataFrame,
                live_pdc: dict | None = None) -> go.Figure:
    """Combined MMP and PDC chart for a single ride.

    Overlays this ride's raw MMP on the sigmoid-decayed MMP envelope and
    fitted two-component model (aerobic green fill / anaerobic orange fill).

    If live_pdc is provided its fitted parameters are used directly so the
    chart shares the same values as the W'bal and TSS component charts.
    """
    ride_date     = ride["ride_date"]
    ride_date_obj = datetime.date.fromisoformat(ride_date)
    cutoff        = (ride_date_obj - datetime.timedelta(days=PDC_WINDOW)).isoformat()

    window    = mmp_all[mmp_all["ride_date"].between(cutoff, ride_date)].copy()
    this_ride = mmp_all[mmp_all["ride_id"] == ride["id"]].sort_values("duration_s")

    fig = go.Figure()

    if not window.empty:
        window["age_days"]   = window["ride_date"].apply(
            lambda d: (ride_date_obj - datetime.date.fromisoformat(d)).days
        )
        window["weight"]     = 1.0 / (1.0 + np.exp(PDC_K * (window["age_days"] - PDC_INFLECTION)))
        window["aged_power"] = window["power"] * window["weight"]

        aged = (
            window.groupby("duration_s")["aged_power"]
            .max().reset_index().sort_values("duration_s")
        )
        # Best aged MMP from other rides in the window (excludes this ride)
        other = (
            window[window["ride_id"] != ride["id"]]
            .groupby("duration_s")["aged_power"]
            .max().reset_index().sort_values("duration_s")
        )
        dur = aged["duration_s"].to_numpy(dtype=float)
        pwr = aged["aged_power"].to_numpy(dtype=float)

        # Resolve model parameters
        ok = False
        if live_pdc is not None:
            AWC, Pmax, MAP, tau2 = (
                live_pdc["AWC"], live_pdc["Pmax"],
                live_pdc["MAP"], live_pdc["tau2"],
            )
            ok = True
        elif len(dur) >= 4:
            popt, ok = _fit_power_curve(dur, pwr)
            if ok:
                AWC, Pmax, MAP, tau2 = popt

        # Aerobic and total model fills (drawn first = behind data points)
        if ok:
            tau   = AWC / Pmax
            t_sm  = np.logspace(np.log10(dur.min()), np.log10(dur.max()), 400)
            p_aer = MAP * (1.0 - np.exp(-t_sm / tau2))
            p_tot = _power_model(t_sm, AWC, Pmax, MAP, tau2)
            fig.add_trace(go.Scatter(
                x=t_sm, y=p_aer,
                mode="lines", name="aerobic (MAP)",
                fill="tozeroy", fillcolor="rgba(46,139,87,0.20)",
                line=dict(color="rgba(46,139,87,0.7)", width=1.2),
            ))
            fig.add_trace(go.Scatter(
                x=t_sm, y=p_tot,
                mode="lines",
                name=f"model  AWC={AWC/1000:.1f} kJ  MAP={MAP:.0f} W",
                fill="tonexty", fillcolor="rgba(220,80,30,0.18)",
                line=dict(color="darkorange", width=2, dash="dash"),
            ))
            fig.add_annotation(
                xref="paper", yref="paper", x=0.02, y=0.08,
                text=(
                    f"AWC = {AWC/1000:.1f} kJ   Pmax = {Pmax:.0f} W<br>"
                    f"MAP = {MAP:.0f} W   τ₂ = {tau2:.0f} s   (τ = {tau:.0f} s)"
                ),
                showarrow=False, align="left",
                bgcolor="white", bordercolor="#bbb", borderwidth=1,
                font=dict(size=11),
            )

        # Best aged MMP from other rides in the PDC window
        if not other.empty:
            fig.add_trace(go.Scatter(
                x=other["duration_s"], y=other["aged_power"],
                mode="lines", name="other rides (best)",
                line=dict(color="steelblue", width=1.5, dash="dot"),
            ))

    # This ride's MMP on top
    if not this_ride.empty:
        fig.add_trace(go.Scatter(
            x=this_ride["duration_s"], y=this_ride["power"],
            mode="lines+markers", name=f"this ride ({ride_date})",
            marker=dict(size=5),
            line=dict(color="tomato", width=2.2),
        ))

    fig.update_xaxes(type="log", tickvals=LOG_TICK_S, ticktext=LOG_TICK_LBL,
                     title_text="Duration", showgrid=True, gridcolor="lightgrey")
    fig.update_yaxes(title_text="Power (W)", showgrid=True, gridcolor="lightgrey")
    fig.update_layout(
        title=dict(
            text=(
                f"MMP & Power Duration Curve — {ride['name'].replace('_', ' ')}<br>"
                f"<sup>Decayed MMP at {ride_date} "
                f"(inflection {PDC_INFLECTION} days · K={PDC_K})</sup>"
            ),
            font=dict(size=14),
        ),
        height=440, margin=dict(t=90, b=50, l=60, r=20),
        template="plotly_white",
        legend=dict(x=0.98, xanchor="right", y=0.98),
    )
    return fig





def _activities_table_data(rides: pd.DataFrame,
                           pdc_params: pd.DataFrame) -> list[dict]:
    """Join rides + pdc_params and return rows for the DataTable, newest first."""
    df = (
        rides
        .merge(pdc_params.rename(columns={"ride_id": "id"}), on="id", how="left")
        .sort_values("ride_date", ascending=False)
    )

    def _int(v):
        return int(round(v)) if pd.notna(v) else ""

    def _f1(v):
        return round(float(v), 1) if pd.notna(v) else ""

    def _f2(v):
        return round(float(v), 2) if pd.notna(v) else ""

    rows = []
    for _, r in df.iterrows():
        rows.append({
            "date":         r["ride_date"],
            "name":         r["name"].replace("_", " "),
            "duration_min": _f1(r.get("duration_min")),
            "avg_power":    _int(r.get("avg_power")),
            "max_power":    _int(r.get("max_power")),
            "ftp":          _int(r.get("ftp")),
            "np":           _int(r.get("normalized_power")),
            "if":           _f2(r.get("intensity_factor")),
            "tss":          _int(r.get("tss")),
            "tss_map":      _int(r.get("tss_map")),
            "tss_awc":      _int(r.get("tss_awc")),
            "map_w":        _int(r.get("MAP")),
            "awc_kj":       _f1(r["AWC"] / 1000 if pd.notna(r.get("AWC")) else None),
            "pmax":         _int(r.get("Pmax")),
        })
    return rows


def fig_90day_mmp(mmp_all: pd.DataFrame) -> go.Figure:
    today     = datetime.date.today()
    cutoff    = (today - datetime.timedelta(days=PDC_WINDOW)).isoformat()
    today_str = today.isoformat()

    window = mmp_all[mmp_all["ride_date"].between(cutoff, today_str)].copy()

    fig = go.Figure()

    if not window.empty:
        window["age_days"] = window["ride_date"].apply(
            lambda d: (today - datetime.date.fromisoformat(d)).days
        )
        window["weight"]      = 1.0 / (1.0 + np.exp(PDC_K * (window["age_days"] - PDC_INFLECTION)))
        window["aged_power"]  = window["power"] * window["weight"]

        aged = (
            window.groupby("duration_s")["aged_power"]
            .max().reset_index().sort_values("duration_s")
        )
        raw = (
            window.groupby("duration_s")["power"]
            .max().reset_index().sort_values("duration_s")
        )

        # Raw hard-max as a faint reference
        fig.add_trace(go.Scatter(
            x=raw["duration_s"], y=raw["power"],
            mode="lines", name="raw max",
            line=dict(color="lightsteelblue", width=1.5, dash="dot"),
        ))
        # Sigmoid-aged curve
        fig.add_trace(go.Scatter(
            x=aged["duration_s"], y=aged["aged_power"],
            mode="lines+markers", name="aged MMP",
            marker=dict(size=4),
            line=dict(color="steelblue", width=2.5),
        ))

        # ── Model fit with aerobic / anaerobic contributions ──────────────
        dur = aged["duration_s"].to_numpy(dtype=float)
        pwr = aged["aged_power"].to_numpy(dtype=float)
        popt, ok = _fit_power_curve(dur, pwr)
        if ok:
            AWC, Pmax, MAP, tau2 = popt
            tau      = AWC / Pmax
            t_smooth = np.logspace(np.log10(dur.min()), np.log10(dur.max()), 400)
            p_aerobic = MAP * (1.0 - np.exp(-t_smooth / tau2))
            p_total   = _power_model(t_smooth, *popt)
            # Aerobic component — fills from zero
            fig.add_trace(go.Scatter(
                x=t_smooth, y=p_aerobic,
                mode="lines", name="aerobic (MAP)",
                fill="tozeroy", fillcolor="rgba(46,139,87,0.20)",
                line=dict(color="rgba(46,139,87,0.7)", width=1.2),
            ))
            # Total model — fills from aerobic line, so the shaded band = anaerobic
            fig.add_trace(go.Scatter(
                x=t_smooth, y=p_total,
                mode="lines",
                name=f"model  AWC={AWC/1000:.1f} kJ  MAP={MAP:.0f} W",
                fill="tonexty", fillcolor="rgba(220,80,30,0.18)",
                line=dict(color="darkorange", width=2, dash="dash"),
            ))
            fig.add_annotation(
                xref="paper", yref="paper",
                x=0.02, y=0.08,
                text=(
                    f"AWC = {AWC/1000:.1f} kJ   Pmax = {Pmax:.0f} W<br>"
                    f"MAP = {MAP:.0f} W   τ₂ = {tau2:.0f} s   (τ = {tau:.0f} s)"
                ),
                showarrow=False, align="left",
                bgcolor="white", bordercolor="#bbb", borderwidth=1,
                font=dict(size=11),
            )

    fig.update_xaxes(type="log", tickvals=LOG_TICK_S, ticktext=LOG_TICK_LBL,
                     title_text="Duration", showgrid=True, gridcolor="lightgrey")
    fig.update_yaxes(title_text="Power (W)", showgrid=True, gridcolor="lightgrey")
    fig.update_layout(
        title=dict(
            text=(
                "90-Day Mean Maximal Power — S-curve aged<br>"
                f"<sup>Inflection {PDC_INFLECTION} days · K={PDC_K} · "
                f"reference date {today_str}</sup>"
            ),
            font=dict(size=14),
        ),
        height=440, margin=dict(t=90, b=50, l=60, r=20),
        template="plotly_white",
        legend=dict(x=0.98, xanchor="right", y=0.98),
    )
    return fig


def fig_pdc_params_history(pdc_params: pd.DataFrame,
                           rides: pd.DataFrame) -> go.Figure:
    """History of the fitted two-component PDC parameters over time.

    Left y-axis  : MAP and Pmax (W)
    Right y-axis : AWC (kJ)
    """
    if pdc_params.empty:
        return go.Figure()

    df = (
        pdc_params
        .merge(rides[["id", "ride_date"]], left_on="ride_id", right_on="id", how="left")
        .sort_values("ride_date")
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=df["ride_date"], y=df["MAP"],
        mode="lines+markers", name="MAP (W)",
        line=dict(color="seagreen", width=2),
        marker=dict(size=5),
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=df["ride_date"], y=df["Pmax"],
        mode="lines+markers", name="Pmax (W)",
        line=dict(color="crimson", width=2),
        marker=dict(size=5),
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=df["ride_date"], y=df["AWC"] / 1000,
        mode="lines+markers", name="AWC (kJ)",
        line=dict(color="steelblue", width=2, dash="dot"),
        marker=dict(size=5),
    ), secondary_y=True)

    fig.update_xaxes(title_text="Date", showgrid=True, gridcolor="lightgrey")
    fig.update_yaxes(title_text="Power (W)", showgrid=True, gridcolor="lightgrey",
                     secondary_y=False)
    fig.update_yaxes(title_text="AWC (kJ)", showgrid=False, secondary_y=True)
    fig.update_layout(
        title=dict(text="PDC Parameter History", font=dict(size=14)),
        height=380,
        margin=dict(t=60, b=50, l=60, r=60),
        template="plotly_white",
        legend=dict(x=0.02, y=0.98),
    )
    return fig




# ── W' balance ────────────────────────────────────────────────────────────────

def _wbal_series(elapsed_s: np.ndarray, power: np.ndarray,
                 AWC: float, CP: float) -> np.ndarray:
    """Skiba (2012) differential W' balance model.

    Parameters
    ----------
    elapsed_s : strictly-increasing array of elapsed time in seconds
    power     : power in watts at each sample (NaN → treated as 0 W)
    AWC       : W' — anaerobic work capacity in joules
    CP        : critical power in watts (use MAP from the two-component fit)

    Returns
    -------
    wbal : W'bal in joules, same shape as elapsed_s
    """
    p = np.where(np.isnan(power), 0.0, power.astype(float))
    n = len(elapsed_s)
    wbal = np.empty(n)
    wbal[0] = AWC
    for i in range(1, n):
        dt = elapsed_s[i] - elapsed_s[i - 1]
        if dt <= 0:
            wbal[i] = wbal[i - 1]
            continue
        pi = p[i - 1]
        if pi >= CP:
            wbal[i] = max(wbal[i - 1] + (CP - pi) * dt, 0.0)
        else:
            tau_w = 546.0 * np.exp(-0.01 * (CP - pi)) + 316.0
            wbal[i] = AWC - (AWC - wbal[i - 1]) * np.exp(-dt / tau_w)
    return wbal


def fig_wbal(records: pd.DataFrame, ride: pd.Series,
             pdc_params: pd.DataFrame,
             live_pdc: dict | None = None) -> go.Figure:
    """W' balance versus elapsed time for a single ride.

    AWC and CP for the W'bal calculation are taken from live_pdc (on-the-fly
    fit) when available, so the chart is guaranteed to match fig_pdc_at_date.
    TSS annotation metrics fall back to stored pdc_params.
    """
    if records["power"].isna().all():
        return go.Figure()

    params_row = pdc_params[pdc_params["ride_id"] == ride["id"]]

    # Use on-the-fly params for the calculation; stored params for annotations.
    if live_pdc is not None:
        AWC = live_pdc["AWC"]
        CP  = live_pdc["MAP"]
    elif not params_row.empty:
        r   = params_row.iloc[0]
        AWC = float(r["AWC"])
        CP  = float(r["MAP"])
    else:
        return go.Figure()

    awc_kj = AWC / 1000.0

    tss_val = np_val = if_val = ftp_val = tss_map_val = tss_awc_val = None
    if not params_row.empty:
        r           = params_row.iloc[0]
        tss_val     = r.get("tss")
        np_val      = r.get("normalized_power")
        if_val      = r.get("intensity_factor")
        ftp_val     = r.get("ftp")
        tss_map_val = r.get("tss_map")
        tss_awc_val = r.get("tss_awc")

    elapsed = records["elapsed_s"].to_numpy(dtype=float)
    power   = records["power"].to_numpy(dtype=float)
    wbal_kj = _wbal_series(elapsed, power, AWC, CP) / 1000.0
    t_min   = records["elapsed_min"].to_numpy(dtype=float)

    fig = go.Figure()

    fig.add_hline(y=awc_kj, line=dict(color="grey", dash="dot", width=1),
                  annotation_text=f"W' = {awc_kj:.1f} kJ",
                  annotation_position="top right",
                  annotation_font=dict(size=10, color="grey"))
    fig.add_hline(y=0, line=dict(color="crimson", dash="dot", width=1))

    fig.add_trace(go.Scatter(
        x=t_min, y=wbal_kj,
        mode="lines", name="W'bal",
        fill="tozeroy", fillcolor="rgba(70,130,180,0.15)",
        line=dict(color="steelblue", width=2),
    ))

    fig.update_xaxes(title_text="Elapsed Time (min)",
                     showgrid=True, gridcolor="lightgrey")
    fig.update_yaxes(title_text="W'bal (kJ)",
                     showgrid=True, gridcolor="lightgrey",
                     range=[-awc_kj * 0.05, awc_kj * 1.12])
    metrics = f"CP = {CP:.0f} W   W' = {awc_kj:.1f} kJ"
    if pd.notna(tss_val) and pd.notna(np_val) and pd.notna(if_val):
        metrics += (
            f"   |   FTP = {ftp_val:.0f} W   NP = {np_val:.0f} W"
            f"   IF = {if_val:.2f}   TSS = {tss_val:.0f}"
        )
    if pd.notna(tss_map_val) and pd.notna(tss_awc_val):
        metrics += (
            f"   (MAP: {tss_map_val:.0f}  AWC: {tss_awc_val:.0f})"
        )

    fig.update_layout(
        title=dict(
            text=f"W' Balance — {metrics}",
            font=dict(size=14),
        ),
        height=280,
        margin=dict(t=55, b=40, l=60, r=20),
        showlegend=False,
        template="plotly_white",
    )
    return fig


# ── TSS component series ──────────────────────────────────────────────────────

def _tss_rate_series(elapsed_s: np.ndarray, power: np.ndarray,
                     ftp: float, CP: float) -> tuple:
    """Return cumulative TSS_MAP and TSS_AWC series over the ride.

    Uses a 30-second rolling average (same-length via 'same' convolution),
    splits at CP proportionally, and integrates second-by-second.

    Returns (t_min, cum_tss_map, cum_tss_awc) — all same length as elapsed_s.
    """
    p = np.where(np.isnan(power), 0.0, power.astype(float))
    kernel = np.ones(30) / 30.0
    p_30s  = np.convolve(p, kernel, mode="same")

    dt = np.empty_like(elapsed_s)
    dt[0]  = 0.0
    dt[1:] = np.diff(elapsed_s)
    dt     = np.clip(dt, 0.0, None)

    p_map = np.minimum(p_30s, CP)
    p_awc = np.maximum(p_30s - CP, 0.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        f_map = np.where(p_30s > 0, p_map / p_30s, 0.0)
        f_awc = np.where(p_30s > 0, p_awc / p_30s, 0.0)

    tss_rate     = (p_30s / ftp) ** 2 * dt / 3600.0 * 100.0 if ftp > 0 else np.zeros_like(p_30s)
    cum_tss_map  = np.cumsum(tss_rate * f_map)
    cum_tss_awc  = np.cumsum(tss_rate * f_awc)
    t_min        = elapsed_s / 60.0
    return t_min, cum_tss_map, cum_tss_awc


def fig_tss_components(records: pd.DataFrame, ride: pd.Series,
                       pdc_params: pd.DataFrame,
                       live_pdc: dict | None = None) -> go.Figure:
    """Cumulative TSS_MAP and TSS_AWC stacked area chart over ride time."""
    if records["power"].isna().all():
        return go.Figure()

    params_row = pdc_params[pdc_params["ride_id"] == ride["id"]]

    if live_pdc is not None:
        CP  = live_pdc["MAP"]
        ftp = live_pdc["ftp"]
    elif not params_row.empty:
        r   = params_row.iloc[0]
        CP  = float(r["MAP"])
        ftp = float(r["ftp"]) if pd.notna(r.get("ftp")) else CP
    else:
        return go.Figure()

    elapsed = records["elapsed_s"].to_numpy(dtype=float)
    power   = records["power"].to_numpy(dtype=float)
    t_min, cum_map, cum_awc = _tss_rate_series(elapsed, power, ftp, CP)

    final_map = cum_map[-1]
    final_awc = cum_awc[-1]
    cum_total = cum_map + cum_awc

    fig = go.Figure()

    # Aerobic layer (fills from zero)
    fig.add_trace(go.Scatter(
        x=t_min, y=cum_map,
        mode="lines", name=f"TSS_MAP ({final_map:.0f})",
        fill="tozeroy", fillcolor="rgba(46,139,87,0.25)",
        line=dict(color="seagreen", width=1.5),
    ))
    # Anaerobic layer (fills from aerobic to total)
    fig.add_trace(go.Scatter(
        x=t_min, y=cum_total,
        mode="lines", name=f"TSS_AWC ({final_awc:.0f})",
        fill="tonexty", fillcolor="rgba(220,80,30,0.22)",
        line=dict(color="darkorange", width=1.5),
    ))

    fig.update_xaxes(title_text="Elapsed Time (min)",
                     showgrid=True, gridcolor="lightgrey")
    fig.update_yaxes(title_text="Cumulative TSS",
                     showgrid=True, gridcolor="lightgrey")
    fig.update_layout(
        title=dict(
            text=(
                f"TSS Components — MAP: {final_map:.0f}  "
                f"AWC: {final_awc:.0f}  "
                f"Total: {final_map + final_awc:.0f}"
            ),
            font=dict(size=14),
        ),
        height=260,
        margin=dict(t=55, b=40, l=60, r=20),
        template="plotly_white",
        legend=dict(x=0.02, y=0.98),
    )
    return fig


def fig_tss_history(pdc_params: pd.DataFrame, rides: pd.DataFrame) -> go.Figure:
    """TSS per ride as stacked bars — TSS_MAP (aerobic) + TSS_AWC (anaerobic)."""
    if pdc_params.empty or "tss_map" not in pdc_params.columns:
        return go.Figure()

    df = (
        pdc_params.dropna(subset=["tss_map", "tss_awc"])
        .merge(rides[["id", "ride_date", "name"]], left_on="ride_id", right_on="id", how="left")
        .sort_values("ride_date")
    )
    if df.empty:
        return go.Figure()

    cd = df[["ftp", "normalized_power", "intensity_factor", "tss_map", "tss_awc"]].values

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["ride_date"], y=df["tss_map"],
        name="TSS_MAP (aerobic)",
        marker_color="seagreen",
        customdata=cd,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "TSS_MAP = %{y:.0f}<br>"
            "TSS_AWC = %{customdata[4]:.0f}<br>"
            "Total = %{customdata[3]:.0f}<br>"
            "FTP = %{customdata[0]:.0f} W  NP = %{customdata[1]:.0f} W  "
            "IF = %{customdata[2]:.2f}<extra></extra>"
        ),
    ))
    fig.add_trace(go.Bar(
        x=df["ride_date"], y=df["tss_awc"],
        name="TSS_AWC (anaerobic)",
        marker_color="darkorange",
        customdata=cd,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "TSS_AWC = %{y:.0f}<br>"
            "TSS_MAP = %{customdata[3]:.0f}<br>"
            "Total = %{customdata[3]:.0f}<br>"
            "FTP = %{customdata[0]:.0f} W  NP = %{customdata[1]:.0f} W  "
            "IF = %{customdata[2]:.2f}<extra></extra>"
        ),
    ))

    fig.update_xaxes(title_text="Date", showgrid=False)
    fig.update_yaxes(title_text="TSS", showgrid=True, gridcolor="lightgrey")
    fig.update_layout(
        title=dict(text="Training Stress Score per Ride (MAP + AWC split)", font=dict(size=14)),
        barmode="stack",
        height=320,
        margin=dict(t=55, b=50, l=60, r=20),
        template="plotly_white",
    )
    return fig


# ── Performance Management Chart ──────────────────────────────────────────────

def _compute_pmc(daily_tss: pd.Series) -> pd.DataFrame:
    """Exponential-weighted ATL (τ=7 d) and CTL (τ=42 d) from a daily TSS series.

    daily_tss : Series with a DatetimeIndex.  Missing dates → 0 TSS (rest days).
    Returns DataFrame with columns: date, atl, ctl, tsb
    where TSB(d) = CTL(d-1) − ATL(d-1)  (form before today's ride).
    """
    if daily_tss.empty:
        return pd.DataFrame(columns=["date", "atl", "ctl", "tsb"])

    dates = pd.date_range(daily_tss.index.min(), daily_tss.index.max(), freq="D")
    tss   = daily_tss.reindex(dates, fill_value=0.0)

    k_atl = 1.0 - np.exp(-1.0 / 7.0)
    k_ctl = 1.0 - np.exp(-1.0 / 42.0)

    n   = len(dates)
    atl = np.zeros(n)
    ctl = np.zeros(n)
    tsb = np.zeros(n)

    for i in range(n):
        t = float(tss.iloc[i])
        if i == 0:
            atl[i] = t * k_atl
            ctl[i] = t * k_ctl
        else:
            tsb[i] = ctl[i - 1] - atl[i - 1]          # form before today's ride
            atl[i] = atl[i - 1] + k_atl * (t - atl[i - 1])
            ctl[i] = ctl[i - 1] + k_ctl * (t - ctl[i - 1])

    return pd.DataFrame({"date": dates, "atl": atl, "ctl": ctl, "tsb": tsb})


def fig_pmc(pdc_params: pd.DataFrame, rides: pd.DataFrame) -> go.Figure:
    """Three-panel Performance Management Chart.

    Row 1 — Total TSS (NP-based):   CTL (42d), ATL (7d), TSB
    Row 2 — MAP component TSS:      CTL (42d), ATL (7d), TSB
    Row 3 — AWC component TSS:      CTL (42d), ATL (7d), TSB

    TSB = CTL − ATL from the previous day.
    Green fill → positive TSB (fresh); red fill → negative TSB (tired).
    """
    required = {"tss", "tss_map", "tss_awc"}
    if pdc_params.empty or not required.issubset(pdc_params.columns):
        return go.Figure()

    df = (
        pdc_params.dropna(subset=["tss", "tss_map", "tss_awc"])
        .merge(rides[["id", "ride_date"]], left_on="ride_id", right_on="id", how="left")
    )
    if df.empty:
        return go.Figure()

    df["ride_date"] = pd.to_datetime(df["ride_date"])
    daily = df.groupby("ride_date")[["tss", "tss_map", "tss_awc"]].sum()

    pmc_tot = _compute_pmc(daily["tss"])
    pmc_map = _compute_pmc(daily["tss_map"])
    pmc_awc = _compute_pmc(daily["tss_awc"])

    # Align daily TSS to the continuous date grid used by pmc_tot
    _idx          = pd.DatetimeIndex(pmc_tot["date"])
    _tss_map_bars = daily["tss_map"].reindex(_idx, fill_value=0.0).round(1).values
    _tss_awc_bars = daily["tss_awc"].reindex(_idx, fill_value=0.0).round(1).values

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=[
            "Aerobic (MAP) — Daily bars + CTL / ATL / TSB",
            "Anaerobic (AWC) — Daily bars + CTL / ATL / TSB",
        ],
        vertical_spacing=0.09,
    )

    panels = [
        (1, pmc_map, "seagreen",     "darkorange", "MAP"),
        (2, pmc_awc, "mediumpurple", "tomato",     "AWC"),
    ]

    for row, pmc, ctl_col, atl_col, label in panels:
        show = row == 1   # show in legend only for the top panel
        dates    = pmc["date"].dt.strftime("%Y-%m-%d")
        tsb_pos  = np.where(pmc["tsb"] >= 0,  pmc["tsb"], 0.0)
        tsb_neg  = np.where(pmc["tsb"] <  0,  pmc["tsb"], 0.0)

        # TSS bars — drawn first so ATL/CTL/TSB lines sit on top
        if row == 1:
            fig.add_trace(go.Bar(
                x=dates, y=_tss_map_bars,
                name="TSS MAP", marker_color="rgba(46,139,87,0.45)",
                showlegend=show,
                hovertemplate="TSS MAP: %{y:.0f}<extra></extra>",
            ), row=row, col=1)
        if row == 2:
            fig.add_trace(go.Bar(
                x=dates, y=_tss_awc_bars,
                name="TSS AWC", marker_color="rgba(220,80,30,0.45)",
                showlegend=show,
                hovertemplate="TSS AWC: %{y:.0f}<extra></extra>",
            ), row=row, col=1)

        # TSB shading — drawn after bars so lines sit on top
        fig.add_trace(go.Scatter(
            x=dates, y=tsb_pos.round(1),
            mode="lines", fill="tozeroy",
            fillcolor="rgba(46,139,87,0.22)",
            line=dict(color="rgba(0,0,0,0)", width=0),
            name="TSB (fresh)", showlegend=show,
            hovertemplate=f"TSB ({label}): %{{y:.1f}}<extra></extra>",
        ), row=row, col=1)

        fig.add_trace(go.Scatter(
            x=dates, y=tsb_neg.round(1),
            mode="lines", fill="tozeroy",
            fillcolor="rgba(220,80,30,0.22)",
            line=dict(color="rgba(0,0,0,0)", width=0),
            name="TSB (tired)", showlegend=show,
            hovertemplate=f"TSB ({label}): %{{y:.1f}}<extra></extra>",
        ), row=row, col=1)

        # ATL — short-term fatigue (dashed)
        fig.add_trace(go.Scatter(
            x=dates, y=pmc["atl"].round(1),
            mode="lines", name=f"ATL 7d ({label})",
            line=dict(color=atl_col, width=1.8, dash="dash"),
            showlegend=show,
            hovertemplate=f"ATL ({label}): %{{y:.1f}}<extra></extra>",
        ), row=row, col=1)

        # CTL — long-term fitness (solid)
        fig.add_trace(go.Scatter(
            x=dates, y=pmc["ctl"].round(1),
            mode="lines", name=f"CTL 42d ({label})",
            line=dict(color=ctl_col, width=2.2),
            showlegend=show,
            hovertemplate=f"CTL ({label}): %{{y:.1f}}<extra></extra>",
        ), row=row, col=1)

        fig.add_hline(y=0, line=dict(color="grey", dash="dot", width=1), row=row, col=1)
        fig.update_yaxes(title_text="Load", showgrid=True, gridcolor="lightgrey",
                         zeroline=False, row=row, col=1)

    fig.update_xaxes(showgrid=True, gridcolor="lightgrey")
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_layout(
        title=dict(text="Performance Management Chart", font=dict(size=14)),
        barmode="stack",
        height=580,
        margin=dict(t=70, b=50, l=70, r=20),
        template="plotly_white",
        hovermode="x unified",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)"),
    )
    return fig


# ── App ───────────────────────────────────────────────────────────────────────

_boot_conn = sqlite3.connect(DB_PATH)
init_db(_boot_conn)
_maybe_migrate(_boot_conn)   # one-time recompute if DB schema is stale
backfill_pdc_params(_boot_conn)
_boot_conn.close()

_reload()          # initial load (picks up freshly computed pdc_params)
_start_watcher()   # background thread

app = dash.Dash(__name__, title="Cycling Power Analysis")

app.layout = html.Div(
    style={"fontFamily": "sans-serif", "padding": "20px"},
    children=[
        # Hidden stores / ticker
        dcc.Store(id="known-version", data=0),
        dcc.Interval(id="poll-interval", interval=3000, n_intervals=0),  # check every 3 s

        html.H1("Cycling Power Analysis", style={"marginBottom": "4px"}),

        # Status bar
        html.Div(id="status-bar", style={
            "fontSize": "12px", "color": "#888", "marginBottom": "12px",
        }),

        dcc.Tabs(
            id="tabs",
            value="tab-fitness",
            children=[

                # ── Fitness tab ────────────────────────────────────────────
                dcc.Tab(label="Fitness", value="tab-fitness", children=[
                    html.Div(style={"paddingTop": "20px"}, children=[

                        dcc.Graph(id="graph-90day-mmp"),

                        html.Hr(),

                        html.H2("Performance Management Chart", style={"marginBottom": "4px"}),
                        dcc.Graph(id="graph-pmc"),

                        html.Hr(),

                        html.H2("PDC Parameters over time", style={"marginBottom": "4px"}),
                        dcc.Graph(id="graph-pdc-params-history"),

                        html.Div(style={"height": "40px"}),
                    ]),
                ]),

                # ── Activities tab ─────────────────────────────────────────
                dcc.Tab(label="Activities", value="tab-activities", children=[
                    html.Div(style={"paddingTop": "20px"}, children=[

                        # Ride selector
                        html.Div([
                            html.Label("Select a ride:", style={"fontWeight": "bold", "marginRight": "10px"}),
                            dcc.Dropdown(
                                id="ride-dropdown",
                                clearable=False,
                                style={"width": "600px", "display": "inline-block", "verticalAlign": "middle"},
                            ),
                        ], style={"marginBottom": "20px"}),

                        # Per-ride charts
                        dcc.Graph(id="graph-power"),
                        dcc.Graph(id="graph-wbal"),
                        dcc.Graph(id="graph-tss-components"),
                        dcc.Graph(id="graph-mmp-pdc"),

                        html.Div(style={"height": "40px"}),
                    ]),
                ]),
            ],
        ),
    ],
)


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("known-version",   "data"),
    Output("ride-dropdown",   "options"),
    Output("ride-dropdown",   "value"),
    Output("graph-90day-mmp",          "figure"),
    Output("graph-pmc",                "figure"),
    Output("graph-pdc-params-history", "figure"),
    Output("status-bar",               "children"),
    Input("poll-interval",    "n_intervals"),
    State("known-version",    "data"),
    State("ride-dropdown",    "value"),
)
def poll_for_new_data(n_intervals, known_ver, current_ride_id):
    ver, rides, mmp_all, pdc_params = get_data()

    if ver == known_ver and n_intervals > 0:
        # Nothing changed — return no-update for everything
        raise dash.exceptions.PreventUpdate

    options = [
        {"label": f"{r['ride_date']}  {r['name'].replace('_', ' ')}", "value": r["id"]}
        for _, r in rides.iterrows()
    ]
    # Keep the currently selected ride if it still exists, else pick the latest
    if current_ride_id in rides["id"].values:
        selected = current_ride_id
    else:
        selected = int(rides.iloc[-1]["id"])

    ride_count = len(rides)
    status = (
        f"Watching {FIT_DIR} for new .fit files — "
        f"{ride_count} ride{'s' if ride_count != 1 else ''} loaded"
    )

    return (
        ver,
        options,
        selected,
        fig_90day_mmp(mmp_all),
        fig_pmc(pdc_params, rides),
        fig_pdc_params_history(pdc_params, rides),
        status,
    )


@app.callback(
    Output("graph-power",       "figure"),
    Output("graph-wbal",           "figure"),
    Output("graph-tss-components", "figure"),
    Output("graph-mmp-pdc",        "figure"),
    Input("ride-dropdown",    "value"),
    State("known-version",    "data"),
)
def update_ride_charts(ride_id, _ver):
    if ride_id is None:
        raise dash.exceptions.PreventUpdate
    _, rides, mmp_all, pdc_params = get_data()
    ride    = rides[rides["id"] == ride_id].iloc[0]
    records = load_records(ride_id)
    # Fit on-the-fly once; share params with W'bal and TSS components so all
    # activity charts use the same PDC parameters as the PDC curve chart.
    live_pdc = _fit_pdc_for_ride(ride, mmp_all)
    return (
        fig_power(records, ride["name"]),
        fig_wbal(records, ride, pdc_params, live_pdc),
        fig_tss_components(records, ride, pdc_params, live_pdc),
        fig_mmp_pdc(ride, mmp_all, live_pdc),
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=8050)
