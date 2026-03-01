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

import base64
import datetime
import os
import sqlite3
import threading
import time

import numpy as np
import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html, dash_table, Input, Output, State, ctx, Patch
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from build_database import (
    init_db, process_ride, backfill_pdc_params, backfill_mmh, backfill_gps_elevation,
    backfill_vi_aedec, backfill_zones,
    recompute_all_pdc_params,
    _power_model, _fit_power_curve,
    PDC_K, PDC_INFLECTION, PDC_WINDOW,
)

from graphs import (
    fig_power_hr, fig_mmh, fig_route_map, fig_elevation,
    fig_mmp_pdc, fig_90day_mmp, fig_90day_mmh,
    fig_pdc_params_history, fig_wbal, fig_tss_components,
    fig_tss_history, fig_pmc, fig_zone_distribution,
    _tss_rate_series, _compute_pmc,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "cycling.db")
FIT_DIR  = os.path.join(BASE_DIR, "raw_data")

# ── Shared data store (updated by watcher, read by callbacks) ─────────────────

_lock         = threading.Lock()
_data_version = 0   # incremented every time new data lands in the DB
_rides        = None
_mmp_all      = None
_mmh_all      = None
_pdc_params   = None
_gps_traces   = None  # dict: ride_id → [(lat, lon), ...] (downsampled)


def _reload():
    """Re-read rides, mmp, mmh, pdc_params and GPS traces from the DB and bump the version."""
    global _rides, _mmp_all, _mmh_all, _pdc_params, _gps_traces, _data_version
    r = _load_rides()
    m = _load_mmp_all(r)
    h = _load_mmh_all(r)
    p = _load_pdc_params()
    g = _load_gps_traces()
    with _lock:
        _rides        = r
        _mmp_all      = m
        _mmh_all      = h
        _pdc_params   = p
        _gps_traces   = g
        _data_version += 1


def get_data():
    """Return a consistent (version, rides, mmp_all, mmh_all, pdc_params, gps_traces) snapshot."""
    with _lock:
        return (_data_version, _rides.copy(), _mmp_all.copy(),
                _mmh_all.copy(), _pdc_params.copy(), _gps_traces)


# ── Duration formatting ────────────────────────────────────────────────────────

COLOURS = px.colors.qualitative.Light24


# ── Database helpers ───────────────────────────────────────────────────────────

def _load_rides() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        """SELECT id, name, ride_date,
                  round(duration_s / 60.0, 1) AS duration_min,
                  avg_power, max_power,
                  avg_heart_rate, max_heart_rate
           FROM rides ORDER BY ride_date, name""",
        conn,
    )
    conn.close()
    return df


def _load_gps_traces() -> dict[int, list[tuple[float, float]]]:
    """Load downsampled GPS traces for all rides. Returns {ride_id: [(lat, lon), ...]}."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT ride_id, latitude, longitude FROM records"
        " WHERE latitude IS NOT NULL ORDER BY ride_id, elapsed_s",
        conn,
    )
    conn.close()
    traces: dict[int, list[tuple[float, float]]] = {}
    for ride_id, group in df.groupby("ride_id"):
        pts = list(zip(group["latitude"], group["longitude"]))
        step = max(1, len(pts) // 300)   # at most ~300 points per thumbnail
        traces[int(ride_id)] = pts[::step]
    return traces


def _make_route_svg(latlons: list[tuple[float, float]]) -> str:
    """Return a minimal SVG polyline of the GPS route, sized to fit within 80×55 px."""
    if len(latlons) < 2:
        return ""
    lats = [p[0] for p in latlons]
    lons = [p[1] for p in latlons]
    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)
    lat_span = lat_max - lat_min or 1e-9
    lon_span = lon_max - lon_min or 1e-9
    max_w, max_h, pad = 80, 55, 3
    scale = min((max_w - 2 * pad) / lon_span, (max_h - 2 * pad) / lat_span)
    svg_w = lon_span * scale + 2 * pad
    svg_h = lat_span * scale + 2 * pad
    pts = " ".join(
        f"{(lon - lon_min) * scale + pad:.1f},{(lat_max - lat) * scale + pad:.1f}"
        for lat, lon in latlons
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w:.0f}" height="{svg_h:.0f}">'
        f'<polyline points="{pts}" fill="none" stroke="#4a90d9" stroke-width="1.5"'
        f' stroke-linejoin="round" stroke-linecap="round"/>'
        f'</svg>'
    )


def _build_table_data(rides: pd.DataFrame, pdc_params: pd.DataFrame,
                      gps_traces: dict | None = None) -> list[dict]:
    """Return list-of-dicts (most-recent first) for the activities DataTable."""
    merged = rides.merge(
        pdc_params[["ride_id", "normalized_power", "intensity_factor", "tss",
                    "variability_index", "aerobic_decoupling_pct"]],
        left_on="id", right_on="ride_id", how="left",
    )
    gps_traces = gps_traces or {}
    rows = []
    for _, r in merged.iterrows():
        dur = r["duration_min"]
        h, m = int(dur) // 60, int(dur) % 60
        latlons = gps_traces.get(int(r["id"]), [])
        if latlons:
            svg = _make_route_svg(latlons)
            b64 = base64.b64encode(svg.encode()).decode()
            thumbnail = f'<img src="data:image/svg+xml;base64,{b64}" style="display:block"/>'
        else:
            thumbnail = ""
        rows.append({
            "id":        int(r["id"]),
            "Route":     thumbnail,
            "Date":      r["ride_date"],
            "Name":      r["name"].replace("_", " "),
            "Duration":  f"{h}h {m:02d}m",
            "Avg Power": f"{r['avg_power']:.0f}" if pd.notna(r["avg_power"]) else "\u2014",
            "NP":        f"{r['normalized_power']:.0f}" if pd.notna(r.get("normalized_power")) else "\u2014",
            "TSS":       f"{r['tss']:.0f}" if pd.notna(r.get("tss")) else "\u2014",
            "IF":        f"{r['intensity_factor']:.2f}" if pd.notna(r.get("intensity_factor")) else "\u2014",
            "VI":        f"{r['variability_index']:.2f}" if pd.notna(r.get("variability_index")) else "\u2014",
            "AeDec%":    f"{r['aerobic_decoupling_pct']:.1f}" if pd.notna(r.get("aerobic_decoupling_pct")) else "\u2014",
        })
    return rows[::-1]   # most-recent first


def _load_mmp_all(rides: pd.DataFrame) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    mmp = pd.read_sql("SELECT ride_id, duration_s, power FROM mmp", conn)
    conn.close()
    mmp = mmp.merge(
        rides.rename(columns={"id": "ride_id"})[["ride_id", "name", "ride_date"]],
        on="ride_id",
    )
    return mmp


def _load_mmh_all(rides: pd.DataFrame) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    mmh = pd.read_sql("SELECT ride_id, duration_s, heart_rate FROM mmh", conn)
    conn.close()
    if mmh.empty:
        return mmh
    mmh = mmh.merge(
        rides.rename(columns={"id": "ride_id"})[["ride_id", "name", "ride_date"]],
        on="ride_id",
    )
    return mmh


def _load_pdc_params() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT ride_id, AWC, Pmax, MAP, tau2, ftp, normalized_power, intensity_factor,"
        " tss, tss_map, tss_awc, ltp, variability_index, aerobic_decoupling_pct"
        " FROM pdc_params",
        conn,
    )
    conn.close()
    return df


def _load_zones_for_ride(ride_id: int) -> dict[int, float]:
    """Return {zone: seconds} for the given ride from zone_distribution."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT zone, seconds FROM zone_distribution WHERE ride_id = ?",
        conn, params=(ride_id,),
    )
    conn.close()
    return {int(r["zone"]): float(r["seconds"]) for _, r in df.iterrows()}


def load_records(ride_id: int) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT elapsed_s, power, heart_rate, latitude, longitude, altitude_m"
        " FROM records WHERE ride_id = ? ORDER BY elapsed_s",
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
        conn = None
        try:
            conn = sqlite3.connect(DB_PATH)
            init_db(conn)
            process_ride(conn, path)
            backfill_vi_aedec(conn)
            backfill_zones(conn)
            _reload()
            print(f"[watcher] Processed and reloaded data.")
        except Exception as exc:
            print(f"[watcher] Error processing {path}: {exc}")
        finally:
            if conn is not None:
                conn.close()

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
        "ltp":  float(MAP * (1.0 - (5.0 / 2.0) * ((AWC / 1000.0) / MAP))),
    }


# ── Graph-level stat rows ─────────────────────────────────────────────────────

def _graph_stat_row(items: list[tuple]) -> html.Div:
    """Left-aligned row of compact stat cards to sit above a graph.

    items: list of (label, value, unit) tuples.
    """
    card_style = {
        "background": "#f8f9fa", "border": "1px solid #dee2e6",
        "borderRadius": "6px", "padding": "8px 16px", "minWidth": "80px",
        "textAlign": "center", "boxShadow": "0 1px 3px rgba(0,0,0,0.06)",
    }
    label_style = {"fontSize": "10px", "color": "#888", "marginBottom": "2px",
                   "textTransform": "uppercase", "letterSpacing": "0.05em"}
    value_style = {"fontSize": "18px", "fontWeight": "bold", "color": "#222"}
    unit_style  = {"fontSize": "11px", "color": "#666", "marginLeft": "2px"}

    return html.Div(
        style={"display": "flex", "gap": "10px", "marginBottom": "8px", "flexWrap": "wrap"},
        children=[
            _make_card(lbl, val, unit, card_style, label_style, value_style, unit_style)
            for lbl, val, unit in items
        ],
    )






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







# ── Freshness status ──────────────────────────────────────────────────────────

_FRESHNESS_CFG = {
    "green": {
        "color": "#16a34a", "label": "Ready",
        "desc": "All training",
        "bg": "#f0fdf4", "border": "#86efac",
    },
    "amber": {
        "color": "#d97706", "label": "Aerobic Only",
        "desc": "Low intensity",
        "bg": "#fffbeb", "border": "#fcd34d",
    },
    "red": {
        "color": "#dc2626", "label": "Fatigued",
        "desc": "Rest / recovery",
        "bg": "#fef2f2", "border": "#fca5a5",
    },
}


def _days_to_trainable(atl: float, ctl: float,
                        threshold_pct: float = 0.0,
                        max_days: int = 60) -> int | None:
    """Days of complete rest until TSB > -threshold_pct * CTL.

    threshold_pct = 0.30 for the MAP aerobic boundary (TSB > -30% CTL),
                  = 0.0  for the AWC high-intensity boundary (TSB > 0).
    Uses the same ATL τ=7 d / CTL τ=42 d decay as _compute_pmc.
    Returns None if the threshold is not crossed within max_days.
    """
    k_atl = 1.0 - np.exp(-1.0 / 7.0)
    k_ctl = 1.0 - np.exp(-1.0 / 42.0)
    for day in range(1, max_days + 1):
        tsb = ctl - atl                    # form before that day's TSS
        atl = atl + k_atl * (0.0 - atl)   # rest: TSS = 0
        ctl = ctl + k_ctl * (0.0 - ctl)
        if tsb > -threshold_pct * ctl:
            return day
    return None


def _compute_freshness_status(pdc_params: pd.DataFrame,
                               rides: pd.DataFrame) -> tuple:
    """Return (status, tsb_map, tsb_awc, map_threshold, atl_map, ctl_map, atl_awc, ctl_awc).

    The aerobic cutoff is -30 % of the current MAP CTL (training load), so a
    small fatigue deficit doesn't immediately block low-intensity work.

    status is 'green'  — TSB_MAP > threshold AND TSB_AWC > 0 (ready for anything)
              'amber'  — TSB_MAP > threshold but TSB_AWC ≤ 0  (aerobic / low-intensity only)
              'red'    — TSB_MAP ≤ threshold                   (rest / recovery)
    Returns a tuple of Nones when there is insufficient data.
    """
    _none = (None,) * 8
    if pdc_params.empty or not {"tss_map", "tss_awc"}.issubset(pdc_params.columns):
        return _none

    df = (
        pdc_params.dropna(subset=["tss_map", "tss_awc"])
        .merge(rides[["id", "ride_date"]], left_on="ride_id", right_on="id", how="left")
    )
    if df.empty:
        return _none

    df["ride_date"] = pd.to_datetime(df["ride_date"])
    daily = df.groupby("ride_date")[["tss_map", "tss_awc"]].sum()

    pmc_map = _compute_pmc(daily["tss_map"])
    pmc_awc = _compute_pmc(daily["tss_awc"])

    if pmc_map.empty or pmc_awc.empty:
        return _none

    tsb_map       = float(pmc_map["tsb"].iloc[-1])
    tsb_awc       = float(pmc_awc["tsb"].iloc[-1])
    ctl_map       = float(pmc_map["ctl"].iloc[-1])
    map_threshold = -0.30 * ctl_map          # −30 % of MAP training load

    if tsb_map > map_threshold and tsb_awc > 0:
        status = "green"
    elif tsb_map > map_threshold:
        status = "amber"
    else:
        status = "red"

    return (
        status, tsb_map, tsb_awc, map_threshold,
        float(pmc_map["atl"].iloc[-1]), ctl_map,
        float(pmc_awc["atl"].iloc[-1]), float(pmc_awc["ctl"].iloc[-1]),
    )


# ── Metric summary boxes ──────────────────────────────────────────────────────

def _make_card(label, value, unit, card_style, label_style, value_style, unit_style):
    return html.Div(style=card_style, children=[
        html.Div(label, style=label_style),
        html.Div([
            html.Span(value, style=value_style),
            html.Span(unit,  style=unit_style),
        ]),
    ])


def _activity_metric_boxes(ride: pd.Series, pdc_params: pd.DataFrame,
                           live_pdc: dict | None = None) -> list:
    """Metric cards showing the PDC state and ride metrics for a single activity."""
    card_style = {
        "background": "#f8f9fa", "border": "1px solid #dee2e6",
        "borderRadius": "8px", "padding": "12px 20px", "minWidth": "100px",
        "textAlign": "center", "boxShadow": "0 1px 3px rgba(0,0,0,0.08)",
    }
    label_style = {"fontSize": "11px", "color": "#888", "marginBottom": "4px",
                   "textTransform": "uppercase", "letterSpacing": "0.05em"}
    value_style = {"fontSize": "22px", "fontWeight": "bold", "color": "#222"}
    unit_style  = {"fontSize": "12px", "color": "#666", "marginLeft": "3px"}

    def card(label, value, unit=""):
        return _make_card(label, value, unit, card_style, label_style, value_style, unit_style)

    def _i(v):
        return f"{int(round(float(v)))}" if pd.notna(v) else "—"

    def _tss(v):
        """Show 1 decimal for values < 10 so e.g. 0.4 isn't displayed as 0."""
        if not pd.notna(v):
            return "—"
        f = float(v)
        return f"{f:.1f}" if f < 10 else f"{int(round(f))}"

    def _f2(v):
        return f"{float(v):.2f}" if pd.notna(v) else "—"

    params_row = pdc_params[pdc_params["ride_id"] == ride["id"]]
    stored = params_row.iloc[0] if not params_row.empty else None

    # PDC fit params — prefer on-the-fly, fall back to stored
    if live_pdc is not None:
        ftp_v  = _i(live_pdc.get("ftp"))
        map_v  = _i(live_pdc.get("MAP"))
        awc_v  = f"{live_pdc['AWC']/1000:.1f}" if live_pdc.get("AWC") else "—"
        pmax_v = _i(live_pdc.get("Pmax"))
        ltp_v  = _i(live_pdc.get("ltp"))
    elif stored is not None:
        ftp_v  = _i(stored.get("ftp"))
        map_v  = _i(stored.get("MAP"))
        awc_v  = f"{stored['AWC']/1000:.1f}" if pd.notna(stored.get("AWC")) else "—"
        pmax_v = _i(stored.get("Pmax"))
        ltp_v  = _i(stored.get("ltp"))
    else:
        ftp_v = map_v = awc_v = pmax_v = ltp_v = "—"

    # Ride performance metrics from stored pdc_params
    np_v      = _i(stored.get("normalized_power")) if stored is not None else "—"
    if_v      = _f2(stored.get("intensity_factor")) if stored is not None else "—"
    tss_v     = _i(stored.get("tss"))              if stored is not None else "—"
    tss_map_v = _tss(stored.get("tss_map"))         if stored is not None else "—"
    tss_awc_v = _tss(stored.get("tss_awc"))         if stored is not None else "—"

    ride_title = ride["name"].replace("_", " ")
    ride_date  = ride["ride_date"]

    return [
        # ── Title row ──────────────────────────────────────────────────────
        html.Div(style={"display": "flex", "alignItems": "baseline",
                        "gap": "12px", "marginBottom": "12px"}, children=[
            html.Span(ride_title, style={"fontSize": "20px", "fontWeight": "bold",
                                         "color": "#e8edf5"}),
            html.Span(ride_date,  style={"fontSize": "13px", "color": "#7a8fbb"}),
        ]),
        # ── Metrics row: ride stats left, PDC params right ─────────────────
        html.Div(style={
            "display": "flex", "justifyContent": "space-between",
            "alignItems": "flex-end", "flexWrap": "wrap", "gap": "12px",
            "marginBottom": "16px",
        }, children=[
            # Left — ride performance metrics
            html.Div(style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}, children=[
                card("NP",      np_v,      "W"),
                card("IF",      if_v,      ""),
                card("TSS",     tss_v,     ""),
                card("TSS MAP", tss_map_v, ""),
                card("TSS AWC", tss_awc_v, ""),
            ]),
            # Right — PDC fitness parameters
            html.Div(style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}, children=[
                card("FTP",  ftp_v,  "W"),
                card("MAP",  map_v,  "W"),
                card("LTP",  ltp_v,  "W"),
                card("AWC",  awc_v,  "kJ"),
                card("Pmax", pmax_v, "W"),
            ]),
        ]),
    ]


def _metric_boxes(pdc_params: pd.DataFrame, rides: pd.DataFrame) -> list:
    """Return a row of stat cards showing the most recent fitted PDC metrics."""
    card_style = {
        "background": "#f8f9fa", "border": "1px solid #dee2e6",
        "borderRadius": "8px", "padding": "12px 20px", "minWidth": "110px",
        "textAlign": "center", "boxShadow": "0 1px 3px rgba(0,0,0,0.08)",
    }
    label_style = {"fontSize": "11px", "color": "#888", "marginBottom": "4px",
                   "textTransform": "uppercase", "letterSpacing": "0.05em"}
    value_style = {"fontSize": "22px", "fontWeight": "bold", "color": "#222"}
    unit_style  = {"fontSize": "12px", "color": "#666", "marginLeft": "3px"}

    def card(label, value, unit=""):
        return _make_card(label, value, unit, card_style, label_style, value_style, unit_style)

    # Find the most recent ride that has PDC params
    if pdc_params.empty or rides.empty:
        return [card("FTP", "—", "W"), card("MAP", "—", "W"), card("LTP", "—", "W"),
                card("AWC", "—", "kJ"), card("Pmax", "—", "W")]

    merged = (
        pdc_params
        .merge(rides[["id", "ride_date"]], left_on="ride_id", right_on="id", how="left")
        .sort_values("ride_date", ascending=False)
        .dropna(subset=["MAP", "AWC", "Pmax"])
    )
    if merged.empty:
        return [card("FTP", "—", "W"), card("MAP", "—", "W"), card("LTP", "—", "W"),
                card("AWC", "—", "kJ"), card("Pmax", "—", "W")]

    latest = merged.iloc[0]
    ftp_v  = f"{int(round(latest['ftp']))}"  if pd.notna(latest.get("ftp"))  else "—"
    map_v  = f"{int(round(latest['MAP']))}"
    ltp_v  = f"{int(round(latest['ltp']))}"  if pd.notna(latest.get("ltp"))  else "—"
    awc_v  = f"{latest['AWC']/1000:.1f}"
    pmax_v = f"{int(round(latest['Pmax']))}"
    as_of  = latest["ride_date"]

    # Freshness status card
    status, tsb_map, tsb_awc, map_threshold, \
        atl_map, ctl_map, atl_awc, ctl_awc = _compute_freshness_status(pdc_params, rides)
    if status is not None:
        cfg = _FRESHNESS_CFG[status]

        # Days-until-trainable countdown
        if status == "red":
            days = _days_to_trainable(atl_map, ctl_map, threshold_pct=0.30)
            days_text  = (f"Aerobic in {days} day{'s' if days != 1 else ''}"
                          if days is not None else "Recovery > 60 days")
            days_color = _FRESHNESS_CFG["amber"]["color"]
        elif status == "amber":
            days = _days_to_trainable(atl_awc, ctl_awc, threshold_pct=0.0)
            days_text  = (f"High intensity in {days} day{'s' if days != 1 else ''}"
                          if days is not None else "High intensity > 60 days")
            days_color = _FRESHNESS_CFG["green"]["color"]
        else:
            days_text  = None
            days_color = None

        card_children = [
            html.Div("Freshness", style=label_style),
            html.Div(style={
                "display": "flex", "alignItems": "center",
                "justifyContent": "center", "gap": "7px",
            }, children=[
                html.Div(style={
                    "width": "11px", "height": "11px", "borderRadius": "50%",
                    "background": cfg["color"], "flexShrink": "0",
                }),
                html.Span(cfg["label"], style={
                    **value_style, "fontSize": "17px", "color": cfg["color"],
                }),
            ]),
            html.Div(
                f"{cfg['desc']}  ·  MAP {tsb_map:+.1f} (cut {map_threshold:.1f}) / AWC {tsb_awc:+.1f}",
                style={"fontSize": "10px", "color": "#888", "marginTop": "4px"},
            ),
        ]
        if days_text is not None:
            card_children.append(html.Div(days_text, style={
                "fontSize": "10px", "fontWeight": "600",
                "color": days_color, "marginTop": "3px",
            }))

        freshness_card = html.Div(style={
            **card_style,
            "background": cfg["bg"], "border": f"1px solid {cfg['border']}",
            "minWidth": "130px", "marginLeft": "auto",
        }, children=card_children)
    else:
        freshness_card = None

    children = [
        html.Div(style={**card_style, "textAlign": "left", "minWidth": "130px"}, children=[
            html.Div("Athlete", style=label_style),
            html.Div(style={"display": "flex", "alignItems": "center", "gap": "10px"}, children=[
                html.Img(
                    src="/assets/athlete_profile.jpg",
                    style={
                        "width": "40px", "height": "40px",
                        "borderRadius": "50%", "objectFit": "cover",
                        "border": "2px solid #dee2e6",
                    },
                ),
                html.Span("Mike Lauder", style={**value_style, "fontSize": "18px"}),
            ]),
        ]),
        card("FTP",  ftp_v,  "W"),
        card("MAP",  map_v,  "W"),
        card("LTP",  ltp_v,  "W"),
        card("AWC",  awc_v,  "kJ"),
        card("Pmax", pmax_v, "W"),
    ]
    if freshness_card is not None:
        children.append(freshness_card)

    return [
        html.Div(style={
            "display": "flex", "gap": "12px", "alignItems": "flex-end",
            "flexWrap": "wrap", "marginBottom": "16px",
        }, children=children),
    ]


# ── App ───────────────────────────────────────────────────────────────────────

_boot_conn = sqlite3.connect(DB_PATH)
init_db(_boot_conn)
_maybe_migrate(_boot_conn)   # one-time recompute if DB schema is stale
backfill_pdc_params(_boot_conn)
backfill_vi_aedec(_boot_conn)
backfill_zones(_boot_conn)
backfill_mmh(_boot_conn)
backfill_gps_elevation(_boot_conn)
_boot_conn.close()

_reload()          # initial load (picks up freshly computed pdc_params)
_start_watcher()   # background thread

app = dash.Dash(__name__, title="Cycling Power Analysis", update_title=None)

_NAV_BASE = {
    "display": "block", "width": "100%", "padding": "12px 20px",
    "border": "none", "textAlign": "left", "cursor": "pointer",
    "fontSize": "14px", "background": "transparent",
    "borderLeft": "3px solid transparent", "color": "#aab",
}
_NAV_ACTIVE = {**_NAV_BASE,
    "background": "rgba(255,255,255,0.08)", "color": "white",
    "fontWeight": "bold", "borderLeft": "3px solid #4a9eff",
}

app.layout = html.Div(
    style={"fontFamily": "sans-serif", "display": "flex", "height": "100vh", "margin": "0",
           "background": "#0d1117"},
    children=[
        # Hidden stores / ticker
        dcc.Store(id="known-version", data=0),
        dcc.Interval(id="poll-interval", interval=3000, n_intervals=0),  # check every 3 s

        # ── Sidebar ────────────────────────────────────────────────────────
        html.Div(style={
            "width": "220px", "minWidth": "220px", "background": "#1e2433",
            "display": "flex", "flexDirection": "column", "paddingTop": "24px",
        }, children=[
            html.Div("Cycling Power Analysis", style={
                "color": "white", "fontWeight": "bold", "fontSize": "13px",
                "padding": "0 20px", "marginBottom": "28px", "lineHeight": "1.4",
            }),
            html.Button("Fitness",    id="nav-fitness",     n_clicks=0, style=_NAV_ACTIVE),
            html.Button("Activities", id="nav-activities",  n_clicks=0, style=_NAV_BASE),
            html.Div(id="status-bar", style={
                "marginTop": "auto", "padding": "12px 20px",
                "fontSize": "11px", "color": "#556", "lineHeight": "1.5",
                "borderTop": "1px solid rgba(255,255,255,0.06)",
            }),
        ]),

        # ── Main content ───────────────────────────────────────────────────
        html.Div(style={"flex": "1", "padding": "24px", "overflowY": "auto",
                        "background": "#0d1117"}, children=[

            # ── Fitness page ───────────────────────────────────────────────
            html.Div(id="page-fitness", style={"display": "block"}, children=[

                # Current fitness metric boxes (moved here from global header)
                html.Div(id="metric-boxes", style={"marginBottom": "16px"}),

                html.Div(style={"display": "flex", "gap": "16px"}, children=[
                    html.Div(style={"flex": "1", "minWidth": "0"}, children=[
                        dcc.Graph(id="graph-90day-mmp"),
                    ]),
                    html.Div(style={"flex": "1", "minWidth": "0"}, children=[
                        dcc.Graph(id="graph-90day-mmh"),
                    ]),
                ]),

                html.Hr(),

                dcc.Graph(id="graph-pmc"),

                html.Hr(),

                dcc.Graph(id="graph-pdc-params-history"),

                html.Div(style={"height": "40px"}),
            ]),

            # ── Activities list page ────────────────────────────────────────
            html.Div(id="page-activities-list", style={"display": "none"}, children=[
                html.H2("Activities", style={"color": "#e8edf5", "marginBottom": "20px",
                                             "fontWeight": "600", "fontSize": "22px"}),
                dash_table.DataTable(
                    id="activities-table",
                    columns=[
                        {"name": "",           "id": "Route",     "presentation": "markdown"},
                        {"name": "Date",       "id": "Date"},
                        {"name": "Name",       "id": "Name"},
                        {"name": "Duration",   "id": "Duration"},
                        {"name": "Avg Power",  "id": "Avg Power"},
                        {"name": "NP",         "id": "NP"},
                        {"name": "TSS",        "id": "TSS"},
                        {"name": "IF",         "id": "IF"},
                        {"name": "VI",         "id": "VI"},
                        {"name": "AeDec%",     "id": "AeDec%"},
                    ],
                    markdown_options={"html": True},
                    sort_action="native",
                    style_table={"overflowX": "auto", "borderRadius": "8px", "overflow": "hidden"},
                    style_header={
                        "backgroundColor": "#ffffff",
                        "color": "#6b7280",
                        "fontWeight": "600",
                        "fontSize": "11px",
                        "letterSpacing": "0.06em",
                        "textTransform": "uppercase",
                        "borderTop": "none",
                        "borderLeft": "none",
                        "borderRight": "none",
                        "borderBottom": "2px solid #e5e7eb",
                    },
                    style_data={
                        "backgroundColor": "#ffffff",
                        "color": "#111827",
                        "borderTop": "none",
                        "borderLeft": "none",
                        "borderRight": "none",
                        "borderBottom": "1px solid #f3f4f6",
                        "cursor": "pointer",
                    },
                    style_data_conditional=[
                        {
                            "if": {"state": "active"},
                            "backgroundColor": "#f0f7ff",
                            "borderBottom": "1px solid #bfdbfe",
                            "color": "#111827",
                        },
                        {
                            "if": {"row_index": "odd"},
                            "backgroundColor": "#fafafa",
                        },
                    ],
                    style_cell={
                        "padding": "12px 16px",
                        "textAlign": "left",
                        "fontSize": "14px",
                        "fontFamily": "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                        "border": "none",
                    },
                    style_cell_conditional=[{
                        "if": {"column_id": "Route"},
                        "width": "90px", "minWidth": "90px", "maxWidth": "90px",
                        "padding": "6px 10px", "textAlign": "center",
                    }],
                ),
            ]),

            # ── Activities page ────────────────────────────────────────────
            html.Div(id="page-activities", style={"display": "none"}, children=[

                # Back navigation + hidden dropdown (dropdown kept in DOM for callback state)
                html.Button(
                    "\u2190 Back to Activities",
                    id="btn-back-to-list",
                    style={
                        "background": "transparent", "border": "none",
                        "color": "#6e9ac7", "cursor": "pointer",
                        "fontSize": "14px", "padding": "0 0 16px 0",
                    },
                ),
                dcc.Dropdown(id="ride-dropdown", clearable=False, style={"display": "none"}),

                # Ride title (left) + PDC fitness metrics (right)
                html.Div(style={
                    "display": "flex", "justifyContent": "space-between",
                    "alignItems": "flex-end", "marginBottom": "20px",
                }, children=[
                    html.Div(id="ride-header"),
                    html.Div(id="pdc-stats",
                             style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
                ]),

                # Per-ride charts
                html.Div(style={"display": "flex", "justifyContent": "space-between",
                                "alignItems": "flex-start"}, children=[
                    html.Div(id="power-stats"),
                    html.Div(id="hr-stats"),
                ]),
                dcc.Graph(id="graph-power-hr"),
                html.Hr(),
                dcc.Graph(id="graph-wbal"),
                html.Hr(),
                dcc.Graph(id="graph-tss-components"),
                html.Hr(),
                dcc.Graph(id="graph-zone-distribution"),
                html.Hr(),
                html.Div(id="graph-map-elevation-section", children=[
                    html.Div(style={"display": "flex", "gap": "16px"}, children=[
                        html.Div(style={"flex": "1", "minWidth": "0"}, children=[
                            dcc.Graph(id="graph-route-map"),
                        ]),
                        html.Div(style={"flex": "1", "minWidth": "0"}, children=[
                            dcc.Graph(id="graph-elevation"),
                        ]),
                    ]),
                    html.Hr(),
                ]),
                html.Div(style={"display": "flex", "gap": "16px"}, children=[
                    html.Div(style={"flex": "1", "minWidth": "0"}, children=[
                        dcc.Graph(id="graph-mmp-pdc"),
                    ]),
                    html.Div(id="mmh-section", style={"flex": "1", "minWidth": "0"}, children=[
                        dcc.Graph(id="graph-mmh"),
                    ]),
                ]),

                html.Div(style={"height": "40px"}),
            ]),
        ]),
    ],
)


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("page-fitness",          "style"),
    Output("page-activities-list",  "style"),
    Output("page-activities",       "style"),
    Output("nav-fitness",           "style"),
    Output("nav-activities",        "style"),
    Input("nav-fitness",            "n_clicks"),
    Input("nav-activities",         "n_clicks"),
)
def switch_page(_, __):
    show = {"display": "block"}
    hide = {"display": "none"}
    if ctx.triggered_id == "nav-activities":
        return hide, show, hide, _NAV_BASE, _NAV_ACTIVE
    return show, hide, hide, _NAV_ACTIVE, _NAV_BASE


@app.callback(
    Output("ride-dropdown",          "value", allow_duplicate=True),
    Output("page-activities-list",   "style", allow_duplicate=True),
    Output("page-activities",        "style", allow_duplicate=True),
    Input("activities-table",        "active_cell"),
    State("activities-table",        "data"),
    prevent_initial_call=True,
)
def open_activity(active_cell, table_data):
    if not active_cell or not table_data:
        raise dash.exceptions.PreventUpdate
    ride_id = table_data[active_cell["row"]]["id"]
    return ride_id, {"display": "none"}, {"display": "block"}


@app.callback(
    Output("page-activities-list",   "style", allow_duplicate=True),
    Output("page-activities",        "style", allow_duplicate=True),
    Input("btn-back-to-list",        "n_clicks"),
    prevent_initial_call=True,
)
def go_back_to_list(n_clicks):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    return {"display": "block"}, {"display": "none"}


@app.callback(
    Output("known-version",   "data"),
    Output("ride-dropdown",   "options"),
    Output("ride-dropdown",   "value"),
    Output("graph-90day-mmp",          "figure"),
    Output("graph-90day-mmh",          "figure"),
    Output("graph-pmc",                "figure"),
    Output("graph-pdc-params-history", "figure"),
    Output("status-bar",               "children"),
    Output("metric-boxes",             "children"),
    Output("activities-table",         "data"),
    Input("poll-interval",    "n_intervals"),
    State("known-version",    "data"),
    State("ride-dropdown",    "value"),
)
def poll_for_new_data(n_intervals, known_ver, current_ride_id):
    ver, rides, mmp_all, mmh_all, pdc_params, gps_traces = get_data()

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
    status = f"{ride_count} ride{'s' if ride_count != 1 else ''} loaded"

    return (
        ver,
        options,
        selected,
        fig_90day_mmp(mmp_all),
        fig_90day_mmh(mmh_all),
        fig_pmc(pdc_params, rides),
        fig_pdc_params_history(pdc_params, rides),
        status,
        _metric_boxes(pdc_params, rides),
        _build_table_data(rides, pdc_params, gps_traces),
    )


@app.callback(
    Output("graph-power-hr",              "figure"),
    Output("graph-wbal",                 "figure"),
    Output("graph-tss-components",       "figure"),
    Output("graph-zone-distribution",    "figure"),
    Output("graph-mmp-pdc",              "figure"),
    Output("graph-mmh",                  "figure"),
    Output("mmh-section",                "style"),
    Output("graph-route-map",            "figure"),
    Output("graph-elevation",            "figure"),
    Output("graph-map-elevation-section","style"),
    Output("ride-header",                "children"),
    Output("pdc-stats",                  "children"),
    Output("power-stats",                "children"),
    Output("hr-stats",                   "children"),
    Input("ride-dropdown",    "value"),
    State("known-version",    "data"),
)
def update_ride_charts(ride_id, _ver):
    if ride_id is None:
        raise dash.exceptions.PreventUpdate
    _, rides, mmp_all, mmh_all, pdc_params, _gps = get_data()
    ride    = rides[rides["id"] == ride_id].iloc[0]
    records = load_records(ride_id)
    # Fit on-the-fly once; share params with W'bal and TSS components so all
    # activity charts use the same PDC parameters as the PDC curve chart.
    live_pdc = _fit_pdc_for_ride(ride, mmp_all)

    has_hr    = "heart_rate" in records.columns and records["heart_rate"].notna().any()
    mmh_style = {"flex": "1", "minWidth": "0"} if has_hr else {"flex": "1", "minWidth": "0", "display": "none"}

    has_gps   = "latitude" in records.columns and records["latitude"].notna().any()
    map_style = {"display": "block"} if has_gps else {"display": "none"}

    def _i(v):
        return f"{int(round(float(v)))}" if pd.notna(v) else "—"

    def _tss(v):
        """Show 1 decimal for values < 10 so e.g. 0.4 isn't displayed as 0."""
        if not pd.notna(v):
            return "—"
        f = float(v)
        return f"{f:.1f}" if f < 10 else f"{int(round(f))}"

    def _f2(v):
        return f"{float(v):.2f}" if pd.notna(v) else "—"

    params_row = pdc_params[pdc_params["ride_id"] == ride["id"]]
    stored = params_row.iloc[0] if not params_row.empty else None

    # PDC fit params — prefer on-the-fly, fall back to stored
    if live_pdc is not None:
        ftp_v  = _i(live_pdc.get("ftp"))
        map_v  = _i(live_pdc.get("MAP"))
        awc_v  = f"{live_pdc['AWC']/1000:.1f}" if live_pdc.get("AWC") else "—"
        pmax_v = _i(live_pdc.get("Pmax"))
        ltp_v  = _i(live_pdc.get("ltp"))
    elif stored is not None:
        ftp_v  = _i(stored.get("ftp"))
        map_v  = _i(stored.get("MAP"))
        awc_v  = f"{stored['AWC']/1000:.1f}" if pd.notna(stored.get("AWC")) else "—"
        pmax_v = _i(stored.get("Pmax"))
        ltp_v  = _i(stored.get("ltp"))
    else:
        ftp_v = map_v = awc_v = pmax_v = ltp_v = "—"

    # Ride performance metrics from stored pdc_params
    np_v      = _i(stored.get("normalized_power"))    if stored is not None else "—"
    if_v      = _f2(stored.get("intensity_factor"))   if stored is not None else "—"
    tss_v     = _i(stored.get("tss"))                 if stored is not None else "—"
    tss_map_v = _tss(stored.get("tss_map"))            if stored is not None else "—"
    tss_awc_v = _tss(stored.get("tss_awc"))            if stored is not None else "—"
    vi_v      = (f"{float(stored['variability_index']):.2f}"
                 if stored is not None and pd.notna(stored.get("variability_index")) else "—")
    aedec_v   = (f"{float(stored['aerobic_decoupling_pct']):.1f}"
                 if stored is not None and pd.notna(stored.get("aerobic_decoupling_pct")) else "—")

    # Ride header: name + date
    ride_header = [
        html.Span(ride["name"].replace("_", " "),
                  style={"fontSize": "20px", "fontWeight": "bold", "color": "#e8edf5"}),
        html.Span(ride["ride_date"],
                  style={"fontSize": "13px", "color": "#7a8fbb", "marginLeft": "10px"}),
    ]

    # PDC fitness cards (top right, aligned with ride header)
    _cs = {"background": "#f8f9fa", "border": "1px solid #dee2e6",
           "borderRadius": "8px", "padding": "12px 20px", "minWidth": "100px",
           "textAlign": "center", "boxShadow": "0 1px 3px rgba(0,0,0,0.08)"}
    _ls = {"fontSize": "11px", "color": "#888", "marginBottom": "4px",
           "textTransform": "uppercase", "letterSpacing": "0.05em"}
    _vs = {"fontSize": "22px", "fontWeight": "bold", "color": "#222"}
    _us = {"fontSize": "12px", "color": "#666", "marginLeft": "3px"}

    def _card(lbl, val, unit=""):
        return _make_card(lbl, val, unit, _cs, _ls, _vs, _us)

    pdc_stats = [
        _card("FTP",  ftp_v,  "W"),
        _card("MAP",  map_v,  "W"),
        _card("LTP",  ltp_v,  "W"),
        _card("AWC",  awc_v,  "kJ"),
        _card("Pmax", pmax_v, "W"),
    ]

    # Difficulty: max 1-hour time-weighted rolling TSS rate over the ride
    difficulty_v = "—"
    if not records["power"].isna().all():
        if live_pdc is not None:
            _cp, _ftp = live_pdc["MAP"], live_pdc["ftp"]
        elif stored is not None:
            _cp  = float(stored["MAP"]) if pd.notna(stored.get("MAP")) else None
            _ftp = float(stored["ftp"]) if pd.notna(stored.get("ftp")) else _cp
        else:
            _cp = _ftp = None
        if _cp is not None and _ftp and _ftp > 0:
            _el  = records["elapsed_s"].to_numpy(dtype=float)
            _pw  = records["power"].to_numpy(dtype=float)
            *_, rate_1h_avg = _tss_rate_series(_el, _pw, float(_ftp), float(_cp))
            difficulty_v = _i(rate_1h_avg.max())

    # Power stats row (ride metrics + raw power, above the power graph)
    power_stats = _graph_stat_row([
        ("NP",         np_v,                         "W"),
        ("IF",         if_v,                         ""),
        ("TSS",        tss_v,                        ""),
        ("TSS MAP",    tss_map_v,                    ""),
        ("TSS AWC",    tss_awc_v,                    ""),
        ("Difficulty", difficulty_v,                 "TSS/h"),
        ("VI",         vi_v,                         ""),
        ("AeDec",      aedec_v,                      "%"),
        ("Avg Power",  _i(ride.get("avg_power")),    "W"),
        ("Max Power",  _i(ride.get("max_power")),    "W"),
    ])

    # HR stats row (above HR graph, hidden with hr-section when no HR data)
    hr_stats = _graph_stat_row([
        ("Avg HR", _i(ride.get("avg_heart_rate")), "bpm"),
        ("Max HR", _i(ride.get("max_heart_rate")), "bpm"),
    ])

    # Zone distribution chart
    zone_data = _load_zones_for_ride(int(ride_id))
    ltp_for_zones = (float(live_pdc["ltp"]) if live_pdc and live_pdc.get("ltp")
                     else (float(stored["ltp"]) if stored is not None and pd.notna(stored.get("ltp")) else 0.0))
    map_for_zones = (float(live_pdc["MAP"]) if live_pdc and live_pdc.get("MAP")
                     else (float(stored["MAP"]) if stored is not None and pd.notna(stored.get("MAP")) else 0.0))

    return (
        fig_power_hr(records, ride["name"]),
        fig_wbal(records, ride, pdc_params, live_pdc),
        fig_tss_components(records, ride, pdc_params, live_pdc),
        fig_zone_distribution(zone_data, ltp_for_zones, map_for_zones),
        fig_mmp_pdc(ride, mmp_all, live_pdc),
        fig_mmh(ride, mmh_all),
        mmh_style,
        fig_route_map(records),
        fig_elevation(records),
        map_style,
        ride_header,
        pdc_stats,
        power_stats,
        hr_stats,
    )


# ── Synced x-axis zoom/pan for the three ride-time charts ─────────────────────

@app.callback(
    Output("graph-power-hr",       "figure", allow_duplicate=True),
    Output("graph-wbal",           "figure", allow_duplicate=True),
    Output("graph-tss-components", "figure", allow_duplicate=True),
    Output("graph-elevation",      "figure", allow_duplicate=True),
    Input("graph-power-hr",        "relayoutData"),
    Input("graph-wbal",            "relayoutData"),
    Input("graph-tss-components",  "relayoutData"),
    Input("graph-elevation",       "relayoutData"),
    prevent_initial_call=True,
)
def _sync_ride_chart_xaxes(rld_phr, rld_wb, rld_tss, rld_elev):
    if not ctx.triggered_id:
        raise dash.exceptions.PreventUpdate

    rld = {
        "graph-power-hr":       rld_phr,
        "graph-wbal":           rld_wb,
        "graph-tss-components": rld_tss,
        "graph-elevation":      rld_elev,
    }[ctx.triggered_id]

    if not rld:
        raise dash.exceptions.PreventUpdate

    if "xaxis.range[0]" in rld and "xaxis.range[1]" in rld:
        new_range = [rld["xaxis.range[0]"], rld["xaxis.range[1]"]]
        autorange = False
    elif rld.get("xaxis.autorange") is True:
        autorange = True
        new_range = None
    else:
        raise dash.exceptions.PreventUpdate

    def make_patch():
        p = Patch()
        if autorange:
            p["layout"]["xaxis"]["autorange"] = True
        else:
            p["layout"]["xaxis"]["range"] = new_range
            p["layout"]["xaxis"]["autorange"] = False
        return p

    return make_patch(), make_patch(), make_patch(), make_patch()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=8050)
