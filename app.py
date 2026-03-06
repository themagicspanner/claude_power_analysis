"""
app.py — Dash application for cycling power analysis.

Usage
─────
  python app.py                 # starts on http://127.0.0.1:8050
  python app.py --lookback 365  # sync the last year of Strava rides

Strava sync
───────────
  On startup, the app imports any new Strava rides from the last 90 days
  (override with --lookback <days>).  A background thread re-checks every
  15 minutes, so new Strava uploads appear automatically.

  Requires a one-time OAuth setup:  python strava_import.py --setup

Pages / sections
────────────────
  • Ride selector dropdown (auto-updates when new rides appear)
  • Power vs time for the selected ride
  • MMP for the selected ride vs 90-day best
  • All-rides MMP overview
  • PDC parameter history (AWC, Pmax, MAP over time)
"""

import argparse
import base64
import datetime
import functools
import json
import os
import sqlite3
import sys
import threading
import time

# Force unbuffered stdout so prints appear immediately in IDE run windows
print = functools.partial(print, flush=True)  # type: ignore[assignment]

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
import dash_ag_grid as dag
from dash import dcc, html, Input, Output, State, ctx, Patch, ClientsideFunction
from build_database import (
    init_db, backfill_pdc_params, backfill_mmh, backfill_gps_elevation,
    backfill_vi_aedec, backfill_zones, backfill_missing_mmp,
    recompute_all_pdc_params, ensure_daily_pdc_current,
    _power_model, _power_model_extended, _fit_power_curve,
    _fit_with_endurance_tail, _normalized_power, _compute_tte_ltp,
    calculate_mmp, find_mmp_window, calculate_zones, MMP_DURATIONS,
    PDC_K, PDC_INFLECTION, PDC_WINDOW,
)
from strava_import import get_client, fetch_and_import, CONFIG_PATH

from graphs import (
    fig_power_hr, fig_hr, fig_mmh, fig_route_map, fig_elevation,
    fig_mmp_pdc, fig_90day_mmp, fig_90day_mmh,
    fig_pdc_params_history, fig_tss_components,
    fig_tss_history, fig_pmc, fig_pmc_combined, fig_zone_bars,
    fig_pdc_investigation, fig_sigmoid_decay,
    _tss_rate_series, _compute_pmc,
)

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DB_PATH        = os.path.join(BASE_DIR, "cycling.db")
WORKOUTS_PATH  = os.path.join(BASE_DIR, "saved_workouts.json")

STRAVA_SYNC_INTERVAL = 15 * 60  # seconds between background Strava syncs
STRAVA_SYNC_LOOKBACK = 90       # default days to look back when syncing

# ── Shared data store (updated by watcher, read by callbacks) ─────────────────

_lock         = threading.Lock()
_data_version = 0   # incremented every time new data lands in the DB
_rides        = None
_mmp_all      = None
_mmh_all      = None
_pdc_params   = None
_daily_pdc    = None
_gps_traces   = None  # dict: ride_id → [(lat, lon), ...] (downsampled)


def _fmt_tte_ltp(seconds) -> str:
    """Format TtE_LTP as 'H:MM' or 'MM' depending on length."""
    if seconds is None or (isinstance(seconds, float) and (seconds != seconds)):
        return "—"
    s = int(round(seconds))
    if s >= 3600:
        h, rem = divmod(s, 3600)
        m = rem // 60
        return f"{h}:{m:02d}"
    return f"{s // 60}"


def _reload():
    """Re-read rides, mmp, mmh, pdc_params, daily_pdc and GPS traces from the DB and bump the version."""
    global _rides, _mmp_all, _mmh_all, _pdc_params, _daily_pdc, _gps_traces, _data_version
    r  = _load_rides()
    m  = _load_mmp_all(r)
    h  = _load_mmh_all(r)
    p  = _load_pdc_params()
    dp = _load_daily_pdc()
    g  = _load_gps_traces()
    with _lock:
        _rides        = r
        _mmp_all      = m
        _mmh_all      = h
        _pdc_params   = p
        _daily_pdc    = dp
        _gps_traces   = g
        _data_version += 1
    print(f"[data] Loaded {len(r)} rides, {len(m)} MMP entries (v{_data_version}).")


def get_data():
    """Return a consistent (version, rides, mmp_all, mmh_all, pdc_params, daily_pdc, gps_traces) snapshot."""
    with _lock:
        return (_data_version, _rides.copy(), _mmp_all.copy(),
                _mmh_all.copy(), _pdc_params.copy(), _daily_pdc.copy(),
                _gps_traces)


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
    df = pd.read_sql("SELECT * FROM pdc_params", conn)
    conn.close()
    return df


def _load_daily_pdc() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT date, MAP, Pmax, AWC, ltp, tau2, tte, tte_b, tte_ltp FROM daily_pdc_params ORDER BY date",
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
_DB_SCHEMA_VERSION = 6


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
        # Force daily PDC recompute so new columns (e.g. tte_ltp) are populated
        conn.execute("DELETE FROM daily_pdc_params")
        conn.commit()
        conn.execute(
            "INSERT OR REPLACE INTO db_meta (key, value) VALUES ('schema_version', ?)",
            (str(_DB_SCHEMA_VERSION),),
        )
        conn.commit()
        print("[migrate] Done.")


# ── Strava background sync ────────────────────────────────────────────────────

def _strava_sync() -> None:
    """Fetch recent Strava rides and import any new ones into the database."""
    if not os.path.exists(CONFIG_PATH):
        print("[strava] No config found — run 'python strava_import.py --setup' first.")
        return

    after = (datetime.date.today() - datetime.timedelta(days=STRAVA_SYNC_LOOKBACK)).isoformat()
    conn = None
    try:
        print(f"[strava] Authenticating with Strava API …")
        client = get_client()
        conn = sqlite3.connect(DB_PATH)
        init_db(conn)
        print(f"[strava] Fetching activities since {after} …")
        fetch_and_import(client, conn, after)
        print("[strava] Backfilling VI / zones …")
        backfill_vi_aedec(conn)
        backfill_zones(conn)
        print("[strava] Reloading data …")
        _reload()
        print("[strava] Sync complete.")
    except Exception as exc:
        print(f"[strava] Sync error: {exc}")
    finally:
        if conn is not None:
            conn.close()


def _start_strava_timer(run_immediately: bool = False) -> None:
    """Run _strava_sync on a repeating background timer.

    If *run_immediately* is True the first sync fires right away (in the
    background thread) instead of waiting for the full interval.  This
    avoids blocking the main thread at startup.
    """
    def _loop():
        if not run_immediately:
            time.sleep(STRAVA_SYNC_INTERVAL)
        while True:
            print(f"[strava] Background sync starting …")
            _strava_sync()
            time.sleep(STRAVA_SYNC_INTERVAL)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    print(f"[strava] Background sync every {STRAVA_SYNC_INTERVAL // 60} minutes.")


# ── PDC on-the-fly helper ─────────────────────────────────────────────────────

def _fit_pdc_for_ride(ride: pd.Series, mmp_all: pd.DataFrame) -> dict | None:
    """Fit the PDC on-the-fly for a ride using the same method as fig_pdc_at_date.

    Returns a dict with keys AWC, Pmax, MAP, tau2, ftp (normalizing power),
    tte, tte_b, ltp (all floats), or None if there is insufficient data for
    a fit.  Using this for W'bal and TSS component charts guarantees they
    use the same parameters as the PDC chart.
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

    popt, ok, tte, tte_b = _fit_with_endurance_tail(dur, pwr)
    if not ok:
        return None

    AWC, Pmax, MAP, tau2 = popt
    ltp = float(MAP * (1.0 - (5.0 / 2.0) * ((AWC / 1000.0) / MAP)))
    return {
        "AWC":  float(AWC),
        "Pmax": float(Pmax),
        "MAP":  float(MAP),
        "tau2": float(tau2),
        "tte":  tte,
        "tte_b": tte_b,
        "tte_ltp": _compute_tte_ltp(AWC, Pmax, MAP, tau2, tte, tte_b, ltp),
        "ftp":  float(_power_model_extended(tte if tte is not None else 3600.0, AWC, Pmax, MAP, tau2, tte, tte_b)),
        "ltp":  ltp,
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
            "tss_ltp":      _int(r.get("tss_ltp")),
            "tss_map":      _int(r.get("tss_map")),
            "tss_awc":      _int(r.get("tss_awc")),
            "map_w":        _int(r.get("MAP")),
            "awc_kj":       _f1(r["AWC"] / 1000 if pd.notna(r.get("AWC")) else None),
            "pmax":         _int(r.get("Pmax")),
        })
    return rows







# ── Workout builder helpers ────────────────────────────────────────────────────

def _get_latest_pdc(pdc_params: pd.DataFrame,
                    rides: pd.DataFrame) -> dict | None:
    """Return the most recent PDC params as a dict, or None."""
    if pdc_params.empty or rides.empty:
        return None
    merged = (
        pdc_params
        .merge(rides[["id", "ride_date"]], left_on="ride_id", right_on="id", how="left")
        .sort_values("ride_date", ascending=False)
        .dropna(subset=["MAP", "AWC", "Pmax"])
    )
    if merged.empty:
        return None
    r = merged.iloc[0]
    tte   = float(r["tte"])   if pd.notna(r.get("tte"))   else None
    tte_b = float(r["tte_b"]) if pd.notna(r.get("tte_b")) else None
    return {
        "MAP":   float(r["MAP"]),
        "AWC":   float(r["AWC"]),
        "Pmax":  float(r["Pmax"]),
        "tau2":  float(r["tau2"]) if pd.notna(r.get("tau2")) else 300.0,
        "ftp":   float(r["ftp"]) if pd.notna(r.get("ftp")) else float(r["MAP"]),
        "ltp":   float(r["ltp"]) if pd.notna(r.get("ltp")) else 0.0,
        "tte":     tte,
        "tte_b":   tte_b,
        "tte_ltp": float(r["tte_ltp"]) if pd.notna(r.get("tte_ltp")) else None,
    }


# ── Saved workouts persistence ────────────────────────────────────────────────

def _load_workouts() -> dict[str, list[dict]]:
    """Load saved workouts from JSON file. Returns {name: rowData}."""
    if os.path.exists(WORKOUTS_PATH):
        with open(WORKOUTS_PATH, "r") as f:
            return json.load(f)
    return {}


def _save_workouts(workouts: dict[str, list[dict]]) -> None:
    """Persist workouts dict to JSON file."""
    with open(WORKOUTS_PATH, "w") as f:
        json.dump(workouts, f, indent=2)


def _resolve_ref_watts(ref: str, pdc: dict | None, map_watts: float) -> float:
    """Return the reference power in watts for a given zone label."""
    if pdc is None:
        return map_watts
    ref = (ref or "MAP").upper()
    if ref == "LTP":
        return float(pdc.get("ltp") or 0.0) or map_watts * 0.75
    if ref == "PMAX":
        return float(pdc.get("Pmax") or map_watts)
    return map_watts  # default: MAP


def _build_workout_records(row_data: list[dict],
                           map_watts: float,
                           pdc: dict | None = None) -> pd.DataFrame:
    """Generate a 1-Hz simulated power DataFrame from workout interval rows."""
    power_samples: list[float] = []
    for row in row_data:
        work_s = int(float(row.get("work_duration_min") or 0) * 60)
        rest_s = int(float(row.get("rest_duration_min") or 0) * 60)
        work_ref_w = _resolve_ref_watts(row.get("work_ref", "MAP"), pdc, map_watts)
        rest_ref_w = _resolve_ref_watts(row.get("rest_ref", "MAP"), pdc, map_watts)
        work_w = float(row.get("work_intensity_pct") or 0) / 100.0 * work_ref_w
        rest_w = float(row.get("rest_intensity_pct") or 0) / 100.0 * rest_ref_w
        reps   = int(row.get("repetitions") or 1)
        for _ in range(max(reps, 0)):
            power_samples.extend([work_w] * work_s)
            if rest_s > 0:
                power_samples.extend([rest_w] * rest_s)

    if not power_samples:
        return pd.DataFrame(columns=["elapsed_s", "elapsed_min", "power", "heart_rate"])

    n = len(power_samples)
    return pd.DataFrame({
        "elapsed_s":   np.arange(n, dtype=float),
        "elapsed_min": np.arange(n, dtype=float) / 60.0,
        "power":       np.array(power_samples, dtype=float),
        "heart_rate":  np.full(n, np.nan),
    })


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
        "desc": "Rest from intensity",
        "bg": "#fef2f2", "border": "#fca5a5",
    },
    "black": {
        "color": "#1e1e1e", "label": "Full Rest",
        "desc": "Deeply fatigued",
        "bg": "#f3f3f3", "border": "#a0a0a0",
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
                               rides: pd.DataFrame,
                               project_tomorrow: bool = False) -> tuple:
    """Return freshness tuple including base, threshold, and AWC components.

    Uses three zone-specific TSB values against CTL-relative cutoffs:
      • Base (≤ LTP):         cutoff = −50 % of base CTL
      • Threshold (LTP→MAP):  cutoff = −30 % of threshold CTL
      • AWC (> MAP):          cutoff = TSB > 0

    status is 'green'  — all three OK  (ready for anything)
              'amber'  — base & thresh OK but TSB_AWC ≤ 0  (aerobic only)
              'red'    — thresh below cutoff (rest from intensity; base OK)
              'black'  — base below cutoff  (full rest / recovery)

    If *project_tomorrow* is True, projects one extra day with 0 TSS so
    the returned values reflect tomorrow's expected readiness.

    Returns a tuple of Nones when there is insufficient data.
    """
    _none = (None,) * 12
    if pdc_params.empty or not {"tss_map", "tss_awc"}.issubset(pdc_params.columns):
        return _none

    df = (
        pdc_params.dropna(subset=["tss_map", "tss_awc"])
        .merge(rides[["id", "ride_date"]], left_on="ride_id", right_on="id", how="left")
    )
    if df.empty:
        return _none

    df["ride_date"] = pd.to_datetime(df["ride_date"])

    # Derive the threshold component (above LTP, below MAP)
    if "tss_ltp" in df.columns:
        df["tss_ltp"]    = df["tss_ltp"].fillna(df["tss_map"])
        df["tss_thresh"] = (df["tss_map"] - df["tss_ltp"]).clip(lower=0)
    else:
        df["tss_ltp"]    = df["tss_map"]   # fallback: no LTP data yet
        df["tss_thresh"] = df["tss_map"]

    daily = df.groupby("ride_date")[["tss_ltp", "tss_thresh", "tss_awc"]].sum()

    extra = 1 if project_tomorrow else 0
    pmc_base   = _compute_pmc(daily["tss_ltp"], future_days=extra)
    pmc_thresh = _compute_pmc(daily["tss_thresh"], future_days=extra)
    pmc_awc    = _compute_pmc(daily["tss_awc"], future_days=extra)

    if pmc_base.empty or pmc_thresh.empty or pmc_awc.empty:
        return _none

    tsb_base         = float(pmc_base["tsb"].iloc[-1])
    ctl_base         = float(pmc_base["ctl"].iloc[-1])
    base_threshold   = -0.50 * ctl_base    # −50 % of base training load

    tsb_thresh       = float(pmc_thresh["tsb"].iloc[-1])
    ctl_thresh       = float(pmc_thresh["ctl"].iloc[-1])
    thresh_threshold = -0.30 * ctl_thresh   # −30 % of threshold training load

    tsb_awc          = float(pmc_awc["tsb"].iloc[-1])

    if tsb_base <= base_threshold:
        status = "black"
    elif tsb_thresh <= thresh_threshold:
        status = "red"
    elif tsb_awc <= 0:
        status = "amber"
    else:
        status = "green"

    return (
        status, tsb_base, tsb_thresh, tsb_awc,
        base_threshold, thresh_threshold,
        float(pmc_base["atl"].iloc[-1]), ctl_base,
        float(pmc_thresh["atl"].iloc[-1]), ctl_thresh,
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

    def _f2(v):
        return f"{float(v):.2f}" if pd.notna(v) else "—"

    params_row = pdc_params[pdc_params["ride_id"] == ride["id"]]
    stored = params_row.iloc[0] if not params_row.empty else None

    # PDC fit params — prefer on-the-fly, fall back to stored
    if live_pdc is not None:
        map_v  = _i(live_pdc.get("MAP"))
        awc_v  = f"{live_pdc['AWC']/1000:.1f}" if live_pdc.get("AWC") else "—"
        pmax_v = _i(live_pdc.get("Pmax"))
        ltp_v  = _i(live_pdc.get("ltp"))
        tte_v  = f"{live_pdc['tte']/60:.0f}" if live_pdc.get("tte") else "—"
        tte_ltp_v = _fmt_tte_ltp(live_pdc.get("tte_ltp"))
    elif stored is not None:
        map_v  = _i(stored.get("MAP"))
        awc_v  = f"{stored['AWC']/1000:.1f}" if pd.notna(stored.get("AWC")) else "—"
        pmax_v = _i(stored.get("Pmax"))
        ltp_v  = _i(stored.get("ltp"))
        tte_v  = f"{stored['tte']/60:.0f}" if pd.notna(stored.get("tte")) else "—"
        tte_ltp_v = _fmt_tte_ltp(stored.get("tte_ltp") if pd.notna(stored.get("tte_ltp")) else None)
    else:
        map_v = awc_v = pmax_v = ltp_v = tte_v = tte_ltp_v = "—"

    # Ride performance metrics from stored pdc_params
    np_v      = _i(stored.get("normalized_power")) if stored is not None else "—"
    if_v      = _f2(stored.get("intensity_factor")) if stored is not None else "—"
    tss_v     = _i(stored.get("tss"))              if stored is not None else "—"

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
            ]),
            # Right — PDC fitness parameters
            html.Div(style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}, children=[
                card("Pmax", pmax_v, "W"),
                card("AWC",  awc_v,  "kJ"),
                card("MAP",  map_v,  "W"),
                card("TtE\u2098\u2090\u209a", tte_v, "min"),
                card("LTP",  ltp_v,  "W"),
                card("TtE\u2097\u209c\u209a", tte_ltp_v, "h:mm"),
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
        return [card("Pmax", "—", "W"), card("AWC", "—", "kJ"),
                card("MAP", "—", "W"), card("TtE\u2098\u2090\u209a", "—", "min"),
                card("LTP", "—", "W"), card("TtE\u2097\u209c\u209a", "—", "h:mm")]

    merged = (
        pdc_params
        .merge(rides[["id", "ride_date"]], left_on="ride_id", right_on="id", how="left")
        .sort_values("ride_date", ascending=False)
        .dropna(subset=["MAP", "AWC", "Pmax"])
    )
    if merged.empty:
        return [card("Pmax", "—", "W"), card("AWC", "—", "kJ"),
                card("MAP", "—", "W"), card("TtE\u2098\u2090\u209a", "—", "min"),
                card("LTP", "—", "W"), card("TtE\u2097\u209c\u209a", "—", "h:mm")]

    latest = merged.iloc[0]
    map_v  = f"{int(round(latest['MAP']))}"
    ltp_v  = f"{int(round(latest['ltp']))}"  if pd.notna(latest.get("ltp"))  else "—"
    awc_v  = f"{latest['AWC']/1000:.1f}"
    pmax_v = f"{int(round(latest['Pmax']))}"
    tte_v  = f"{latest['tte']/60:.0f}" if pd.notna(latest.get("tte")) else "—"
    tte_ltp_v = _fmt_tte_ltp(latest.get("tte_ltp") if pd.notna(latest.get("tte_ltp")) else None)
    as_of  = latest["ride_date"]

    # Training readiness card — show tomorrow's readiness if a ride was
    # recorded today, otherwise show today's.
    today_str = datetime.date.today().isoformat()
    rode_today = not rides.empty and (rides["ride_date"] == today_str).any()

    (status, tsb_base, tsb_thresh, tsb_awc,
     base_threshold, thresh_threshold,
     atl_base, ctl_base, atl_thresh, ctl_thresh,
     atl_awc, ctl_awc) = _compute_freshness_status(
        pdc_params, rides, project_tomorrow=rode_today)
    if status is not None:
        cfg = _FRESHNESS_CFG[status]

        # Per-zone status: (zone_label, ok?, tsb, cutoff_label)
        base_ok   = tsb_base  > base_threshold
        thresh_ok = tsb_thresh > thresh_threshold
        awc_ok    = tsb_awc   > 0

        _zone_dot_style = lambda ok: {
            "width": "9px", "height": "9px", "borderRadius": "50%",
            "background": "#16a34a" if ok else "#dc2626", "flexShrink": "0",
        }
        _zone_label_style = {"fontSize": "12px", "color": "#333", "fontWeight": "600"}
        _zone_detail_style = {"fontSize": "10px", "color": "#888"}

        def _zone_row(label, ok, tsb_val, cut_val, cut_str):
            return html.Div(style={
                "display": "flex", "alignItems": "center", "gap": "8px",
                "padding": "4px 0",
            }, children=[
                html.Div(style=_zone_dot_style(ok)),
                html.Span(label, style=_zone_label_style),
                html.Span(
                    f"TSB {tsb_val:+.1f}  /  cutoff {cut_val:.1f}",
                    style=_zone_detail_style,
                ),
            ])

        zone_rows = [
            _zone_row("Anaerobic", awc_ok,    tsb_awc,    0.0,              "0"),
            _zone_row("Threshold", thresh_ok, tsb_thresh, thresh_threshold, "−0.3 CTL"),
            _zone_row("Base",      base_ok,   tsb_base,   base_threshold,   "−0.5 CTL"),
        ]

        advice_day = "Tomorrow" if rode_today else "Today"

        # Summary text and next-zone countdown
        if status == "black":
            summary = f"{advice_day}: full rest recommended — base fatigue is too high for any riding."
            days = _days_to_trainable(atl_base, ctl_base, threshold_pct=0.50)
            countdown = (f"Base riding in ~{days} day{'s' if days != 1 else ''}"
                         if days is not None else "Recovery > 60 days")
            countdown_color = _FRESHNESS_CFG["red"]["color"]
        elif status == "red":
            summary = f"{advice_day}: recovery rides only — threshold fatigue needs to clear before intensity."
            days = _days_to_trainable(atl_thresh, ctl_thresh, threshold_pct=0.30)
            countdown = (f"Threshold sessions in ~{days} day{'s' if days != 1 else ''}"
                         if days is not None else "Recovery > 60 days")
            countdown_color = _FRESHNESS_CFG["amber"]["color"]
        elif status == "amber":
            summary = f"{advice_day}: aerobic training OK — endurance and tempo, but avoid VO2max / anaerobic work."
            days = _days_to_trainable(atl_awc, ctl_awc, threshold_pct=0.0)
            countdown = (f"High intensity in ~{days} day{'s' if days != 1 else ''}"
                         if days is not None else "High intensity > 60 days")
            countdown_color = _FRESHNESS_CFG["green"]["color"]
        else:
            summary = f"{advice_day}: all systems go — ready for any session including high-intensity intervals."
            countdown = None
            countdown_color = None

        readiness_children = [
            # Header: overall status + advice day
            html.Div(style={
                "display": "flex", "alignItems": "center", "gap": "8px",
                "marginBottom": "8px",
            }, children=[
                html.Div(style={
                    "width": "13px", "height": "13px", "borderRadius": "50%",
                    "background": cfg["color"], "flexShrink": "0",
                }),
                html.Span(cfg["label"], style={
                    "fontSize": "18px", "fontWeight": "bold", "color": cfg["color"],
                }),
                html.Span(f"({advice_day})", style={
                    "fontSize": "13px", "color": "#888", "marginLeft": "4px",
                }),
            ]),
            # Summary sentence
            html.Div(summary, style={
                "fontSize": "12px", "color": "#555", "marginBottom": "10px",
                "lineHeight": "1.4",
            }),
            # Per-zone breakdown
            html.Div(style={
                "borderTop": "1px solid #e5e7eb", "paddingTop": "6px",
            }, children=zone_rows),
        ]
        # Countdown to next zone clearing
        if countdown is not None:
            readiness_children.append(html.Div(countdown, style={
                "fontSize": "11px", "fontWeight": "600", "color": countdown_color,
                "marginTop": "8px", "paddingTop": "6px",
                "borderTop": "1px solid #e5e7eb",
            }))

        freshness_card = html.Div(style={
            "background": cfg["bg"], "border": f"1px solid {cfg['border']}",
            "borderRadius": "8px", "padding": "14px 18px",
            "minWidth": "300px", "marginLeft": "auto",
            "boxShadow": "0 1px 3px rgba(0,0,0,0.08)",
        }, children=readiness_children)
    else:
        freshness_card = None

    children = [
        card("Pmax", pmax_v, "W"),
        card("AWC",  awc_v,  "kJ"),
        card("MAP",  map_v,  "W"),
        card("TtE\u2098\u2090\u209a", tte_v, "min"),
        card("LTP",  ltp_v,  "W"),
        card("TtE\u2097\u209c\u209a", tte_ltp_v, "h:mm"),
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

# Parse CLI args early so --lookback is available before the first sync.
_parser = argparse.ArgumentParser(description="Cycling Power Analysis dashboard")
_parser.add_argument(
    "--lookback", type=int, default=None,
    help="Override Strava sync lookback (days). Default: 90.",
)
_args, _ = _parser.parse_known_args()  # parse_known_args so Dash flags pass through
if _args.lookback is not None:
    STRAVA_SYNC_LOOKBACK = _args.lookback
    print(f"[strava] Lookback overridden to {STRAVA_SYNC_LOOKBACK} days.")

print("[boot] Initialising database …")
_boot_conn = sqlite3.connect(DB_PATH)
init_db(_boot_conn)
_maybe_migrate(_boot_conn)   # one-time recompute if DB schema is stale
print("[boot] Backfilling PDC params …")
backfill_pdc_params(_boot_conn)
print("[boot] Backfilling missing MMP entries …")
backfill_missing_mmp(_boot_conn)
print("[boot] Backfilling VI / AE / DEC …")
backfill_vi_aedec(_boot_conn)
print("[boot] Backfilling power zones …")
backfill_zones(_boot_conn)
print("[boot] Backfilling MMH …")
backfill_mmh(_boot_conn)
print("[boot] Backfilling GPS / elevation …")
backfill_gps_elevation(_boot_conn)
print("[boot] Ensuring daily PDC is current …")
ensure_daily_pdc_current(_boot_conn)
_boot_conn.close()
print("[boot] Backfill complete.")

print("[boot] Loading data into memory …")
_reload()              # initial load (picks up freshly computed pdc_params)
print("[boot] Data loaded.")
_start_strava_timer(run_immediately=True)  # sync Strava in background, don't block startup

app = dash.Dash(__name__, title="Cycling Power Analysis", update_title=None,
                suppress_callback_exceptions=True)

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
        dcc.Store(id="known-ride-ids", data=[]),
        dcc.Store(id="toast-message", data=""),
        dcc.Interval(id="poll-interval", interval=3000, n_intervals=0),  # check every 3 s

        # Toast notification (top-right pop-over)
        html.Div(id="toast-container", children=[
            html.Div(id="toast-text"),
        ], style={
            "position": "fixed", "top": "20px", "right": "20px",
            "zIndex": "9999", "background": "#1e2a3a",
            "border": "1px solid #2d4a6f", "borderRadius": "8px",
            "padding": "14px 20px", "color": "#e8edf5",
            "fontSize": "14px", "boxShadow": "0 4px 20px rgba(0,0,0,0.4)",
            "maxWidth": "360px", "opacity": "0", "pointerEvents": "none",
            "transition": "opacity 0.3s ease",
        }),

        # ── Sidebar ────────────────────────────────────────────────────────
        html.Div(style={
            "width": "220px", "minWidth": "220px", "background": "#1e2433",
            "display": "flex", "flexDirection": "column", "paddingTop": "24px",
        }, children=[
            html.Div("Cycling Power Analysis", style={
                "color": "white", "fontWeight": "bold", "fontSize": "13px",
                "padding": "0 20px", "marginBottom": "16px", "lineHeight": "1.4",
            }),
            # Athlete card
            html.Div(style={
                "display": "flex", "alignItems": "center", "gap": "10px",
                "padding": "10px 20px", "marginBottom": "20px",
                "borderBottom": "1px solid rgba(255,255,255,0.06)",
                "borderTop": "1px solid rgba(255,255,255,0.06)",
            }, children=[
                html.Img(
                    src="/assets/athlete_profile.jpg",
                    style={
                        "width": "36px", "height": "36px",
                        "borderRadius": "50%", "objectFit": "cover",
                        "border": "2px solid #3a4a6a",
                    },
                ),
                html.Span("Mike Lauder", style={
                    "fontSize": "14px", "fontWeight": "600", "color": "#e8edf5",
                }),
            ]),
            html.Button("Fitness",    id="nav-fitness",     n_clicks=0, style=_NAV_ACTIVE),
            html.Button("Activities", id="nav-activities",  n_clicks=0, style=_NAV_BASE),
            html.Button("Workouts",   id="nav-workout",     n_clicks=0, style=_NAV_BASE),
            html.Button("PDC Model",  id="nav-pdc-model",   n_clicks=0, style=_NAV_BASE),
            html.Button("Testing",   id="nav-testing",     n_clicks=0, style=_NAV_BASE),
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

                dcc.Graph(id="graph-90day-mmp"),

                html.Hr(),

                dcc.Graph(id="graph-pmc-combined"),

                html.Details(
                    style={"marginTop": "8px", "marginBottom": "8px"},
                    children=[
                        html.Summary(
                            "Zone breakdown (Base / Threshold / Anaerobic)",
                            style={
                                "cursor": "pointer", "color": "#7a8fbb",
                                "fontSize": "13px", "padding": "8px 0",
                                "userSelect": "none",
                            },
                        ),
                        dcc.Graph(id="graph-pmc"),
                    ],
                ),

                html.Div(style={"height": "40px"}),
            ]),

            # ── PDC Model investigation page ────────────────────────────────
            html.Div(id="page-pdc-model", style={"display": "none"}, children=[
                html.H2("PDC Model",
                        style={"color": "#e8edf5", "marginBottom": "20px",
                               "fontWeight": "600", "fontSize": "22px"}),
                dcc.Graph(id="graph-pdc-history-power"),
                dcc.Graph(id="graph-pdc-history-energy"),
                dcc.Graph(id="graph-pdc-history-time"),
                html.Hr(),
                dcc.Store(id="pdc-slider-dates"),
                html.Div(id="pdc-historical-cards",
                         style={"display": "flex", "gap": "12px",
                                "flexWrap": "wrap", "marginBottom": "12px"}),
                dcc.Graph(id="graph-pdc-historical"),
                html.Div(style={"height": "40px"}),
            ]),

            # ── Testing page ─────────────────────────────────────────────────
            html.Div(id="page-testing", style={"display": "none"}, children=[
                html.H2("Testing",
                        style={"color": "#e8edf5", "marginBottom": "8px",
                               "fontWeight": "600", "fontSize": "22px"}),
                html.P(
                    "Each MMP data point is coloured by its sigmoid decay weight "
                    "(bright blue = recent / trustworthy, grey = old / may need retesting). "
                    "The dashed orange line is the fitted two-component PDC model. "
                    "Red residual bars show durations where the model overestimates "
                    "your data — focus testing efforts here first.",
                    style={"color": "#7a8fbb", "fontSize": "13px",
                           "marginBottom": "20px", "maxWidth": "860px"},
                ),
                dcc.Graph(id="graph-pdc-investigation"),
                html.Hr(),
                dcc.Graph(id="graph-sigmoid-decay"),
                html.Div(style={"height": "40px"}),
            ]),

            # ── Activities list page ────────────────────────────────────────
            html.Div(id="page-activities-list", style={"display": "none"}, children=[
                html.H2("Activities", style={"color": "#e8edf5", "marginBottom": "20px",
                                             "fontWeight": "600", "fontSize": "22px"}),
                dag.AgGrid(
                    id="activities-table",
                    columnDefs=[
                        {"headerName": "",          "field": "Route",
                         "cellRenderer": "markdown", "autoHeight": True,
                         "width": 90, "minWidth": 90, "maxWidth": 90,
                         "sortable": False, "cellStyle": {"textAlign": "center", "padding": "6px 10px"}},
                        {"headerName": "Date",      "field": "Date",      "sortable": True},
                        {"headerName": "Name",      "field": "Name",      "sortable": True},
                        {"headerName": "Duration",  "field": "Duration",  "sortable": True},
                        {"headerName": "Avg Power", "field": "Avg Power", "sortable": True},
                        {"headerName": "NP",        "field": "NP",        "sortable": True},
                        {"headerName": "TSS",       "field": "TSS",       "sortable": True},
                        {"headerName": "IF",        "field": "IF",        "sortable": True},
                        {"headerName": "VI",        "field": "VI",        "sortable": True},
                        {"headerName": "AeDec%",    "field": "AeDec%",    "sortable": True},
                    ],
                    defaultColDef={"resizable": True},
                    dangerously_allow_code=True,
                    getRowId="params.data.id",
                    dashGridOptions={
                        "domLayout": "autoHeight",
                        "rowHeight": 48,
                        "headerHeight": 36,
                    },
                    style={"width": "100%", "borderRadius": "8px", "overflow": "hidden"},
                    className="ag-theme-alpine",
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
                html.Div(id="power-stats"),
                # Hidden: kept in DOM for callback compatibility
                html.Div(id="hr-stats", style={"display": "none"}),
                dcc.Graph(id="graph-power-hr"),
                html.Hr(),
                dcc.Graph(id="graph-tss-components"),
                html.Hr(),
                dcc.Graph(id="graph-zone-bars"),
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
                html.Div(style={"display": "flex", "gap": "16px", "alignItems": "flex-start"}, children=[
                    html.Div(style={"flex": "1", "minWidth": "0"}, children=[
                        dcc.Graph(id="graph-mmp-pdc"),
                    ]),
                    html.Div(id="mmp-table-container",
                             style={"display": "none"}),
                ]),

                # Hidden: kept in DOM for callback compatibility
                dcc.Graph(id="graph-hr", style={"display": "none"}),
                html.Div(id="mmh-section", style={"display": "none"}, children=[
                    dcc.Graph(id="graph-mmh"),
                ]),

                # Store for MMP click → power highlight
                dcc.Store(id="ride-power-store", data=None),

                html.Div(style={"height": "40px"}),
            ]),

            # ── Workout list page ─────────────────────────────────────
            html.Div(id="page-workout-list", style={"display": "none"}, children=[
                html.Div(style={
                    "display": "flex", "justifyContent": "space-between",
                    "alignItems": "center", "marginBottom": "20px",
                }, children=[
                    html.H2("Workouts",
                            style={"color": "#e8edf5", "margin": "0",
                                   "fontWeight": "600", "fontSize": "22px"}),
                    html.Button("+ New Workout", id="workout-new-btn", n_clicks=0,
                                style={
                                    "padding": "8px 20px", "cursor": "pointer",
                                    "borderRadius": "4px",
                                    "border": "1px solid #4a9eff",
                                    "background": "#4a9eff", "color": "#fff",
                                    "fontSize": "13px", "fontWeight": "600",
                                }),
                ]),
                dag.AgGrid(
                    id="workout-list-table",
                    columnDefs=[
                        {"headerName": "",          "field": "Power",
                         "cellRenderer": "markdown", "autoHeight": True,
                         "width": 130, "minWidth": 130, "maxWidth": 130,
                         "sortable": False,
                         "cellStyle": {"textAlign": "center", "padding": "6px 10px"}},
                        {"headerName": "Name",      "field": "Name",       "sortable": True, "flex": 2},
                        {"headerName": "Type",      "field": "Type",       "sortable": True, "width": 120},
                        {"headerName": "Duration",  "field": "Duration",   "sortable": True, "width": 90},
                        {"headerName": "Avg Power", "field": "Avg Power",  "sortable": True, "width": 90},
                        {"headerName": "NP",        "field": "NP",         "sortable": True, "width": 70},
                        {"headerName": "IF",        "field": "IF",         "sortable": True, "width": 70},
                        {"headerName": "TSS",       "field": "TSS",        "sortable": True, "width": 70},
                        {"headerName": "Base",      "field": "Base TSS",   "sortable": True, "width": 70},
                        {"headerName": "Thresh",    "field": "Thresh TSS", "sortable": True, "width": 70},
                        {"headerName": "AWC",       "field": "AWC TSS",    "sortable": True, "width": 70},
                        {"headerName": "PDC Base",   "field": "PDC Base",   "sortable": False, "width": 90},
                        {"headerName": "PDC Thresh", "field": "PDC Thresh", "sortable": False, "width": 90},
                        {"headerName": "PDC AWC",    "field": "PDC AWC",    "sortable": False, "width": 90},
                    ],
                    rowData=[],
                    defaultColDef={"resizable": True},
                    dangerously_allow_code=True,
                    getRowId="params.data.Name",
                    dashGridOptions={
                        "domLayout": "autoHeight",
                        "rowHeight": 48,
                        "headerHeight": 36,
                    },
                    style={"width": "100%", "borderRadius": "8px", "overflow": "hidden"},
                    className="ag-theme-alpine",
                ),
            ]),

            # ── Workout editor page ──────────────────────────────────
            html.Div(id="page-workout", style={"display": "none"}, children=[
                html.Button(
                    "\u2190 Back to Workouts",
                    id="btn-back-to-workout-list",
                    style={
                        "background": "transparent", "border": "none",
                        "color": "#6e9ac7", "cursor": "pointer",
                        "fontSize": "14px", "padding": "0 0 16px 0",
                    },
                ),

                # Name + Save / Delete bar
                html.Div(style={
                    "display": "flex", "gap": "10px", "alignItems": "center",
                    "marginBottom": "16px", "flexWrap": "wrap",
                }, children=[
                    dcc.Input(
                        id="workout-name-input",
                        placeholder="Workout name",
                        type="text",
                        debounce=True,
                        style={
                            "width": "260px", "padding": "8px 12px",
                            "borderRadius": "4px",
                            "border": "1px solid #555",
                            "background": "#1e2a3a", "color": "#e8edf5",
                            "fontSize": "15px", "fontWeight": "600",
                        },
                    ),
                    html.Button(
                        "Save", id="workout-save-btn", n_clicks=0,
                        style={
                            "padding": "8px 20px", "cursor": "pointer",
                            "borderRadius": "4px",
                            "border": "1px solid #4a9eff",
                            "background": "#4a9eff", "color": "#fff",
                            "fontSize": "13px", "fontWeight": "600",
                        },
                    ),
                    html.Button(
                        "Delete", id="workout-delete-btn", n_clicks=0,
                        style={
                            "padding": "8px 20px", "cursor": "pointer",
                            "borderRadius": "4px",
                            "border": "1px solid #ff6b6b",
                            "background": "transparent", "color": "#ff6b6b",
                            "fontSize": "13px",
                        },
                    ),
                    html.Span(id="workout-lib-status", style={
                        "color": "#7a8fbb", "fontSize": "12px",
                        "marginLeft": "4px",
                    }),
                ]),

                html.P(
                    "Define intervals below. Charts update as you edit. "
                    "Intensities are expressed as % of the selected reference zone.",
                    style={"color": "#7a8fbb", "fontSize": "13px",
                           "marginBottom": "16px", "maxWidth": "660px"},
                ),

                html.Div(id="workout-pdc-cards",
                         style={"display": "flex", "gap": "12px",
                                "marginBottom": "16px", "flexWrap": "wrap"}),

                dag.AgGrid(
                    id="workout-table",
                    columnDefs=[
                        {"field": "work_ref", "rowDrag": True,
                         "headerName": "Work Ref", "editable": True,
                         "cellEditor": "agSelectCellEditor",
                         "cellEditorParams": {"values": ["MAP", "LTP", "Pmax"]},
                         "width": 90},
                        {"field": "work_intensity_pct",
                         "headerName": "Work %", "editable": True,
                         "type": "numericColumn", "cellDataType": "number"},
                        {"field": "work_duration_min",
                         "headerName": "Work (min)", "editable": True,
                         "type": "numericColumn", "cellDataType": "number"},
                        {"field": "rest_ref",
                         "headerName": "Rest Ref", "editable": True,
                         "cellEditor": "agSelectCellEditor",
                         "cellEditorParams": {"values": ["MAP", "LTP", "Pmax"]},
                         "width": 90},
                        {"field": "rest_intensity_pct",
                         "headerName": "Rest %", "editable": True,
                         "type": "numericColumn", "cellDataType": "number"},
                        {"field": "rest_duration_min",
                         "headerName": "Rest (min)", "editable": True,
                         "type": "numericColumn", "cellDataType": "number"},
                        {"field": "repetitions",
                         "headerName": "Reps", "editable": True,
                         "type": "numericColumn", "cellDataType": "number"},
                    ],
                    rowData=[
                        {"work_duration_min": 5, "work_ref": "MAP",
                         "work_intensity_pct": 100,
                         "rest_duration_min": 5, "rest_ref": "MAP",
                         "rest_intensity_pct": 80,
                         "repetitions": 5},
                    ],
                    defaultColDef={"flex": 1, "minWidth": 100, "sortable": False},
                    dashGridOptions={
                        "singleClickEdit": True,
                        "stopEditingWhenCellsLoseFocus": True,
                        "rowDragManaged": True,
                        "animateRows": True,
                        "domLayout": "autoHeight",
                    },
                    style={"width": "100%", "marginBottom": "12px"},
                ),

                html.Div(style={"display": "flex", "gap": "10px",
                                "marginBottom": "20px"}, children=[
                    html.Button("+ Add Row", id="workout-add-row", n_clicks=0,
                                style={"padding": "6px 16px", "cursor": "pointer",
                                       "borderRadius": "4px",
                                       "border": "1px solid #4a9eff",
                                       "background": "transparent",
                                       "color": "#4a9eff", "fontSize": "13px"}),
                    html.Button("Duplicate Last Row", id="workout-dup-row",
                                n_clicks=0,
                                style={"padding": "6px 16px", "cursor": "pointer",
                                       "borderRadius": "4px",
                                       "border": "1px solid #4a9eff",
                                       "background": "transparent",
                                       "color": "#4a9eff", "fontSize": "13px"}),
                    html.Button("- Remove Last Row", id="workout-remove-row",
                                n_clicks=0,
                                style={"padding": "6px 16px", "cursor": "pointer",
                                       "borderRadius": "4px",
                                       "border": "1px solid #888",
                                       "background": "transparent",
                                       "color": "#888", "fontSize": "13px"}),
                ]),

                html.Div(id="workout-stats", style={"marginBottom": "8px"}),

                dcc.Graph(id="graph-workout-power"),
                html.Hr(),
                dcc.Graph(id="graph-workout-tss-components"),
                html.Hr(),
                dcc.Graph(id="graph-workout-zone-bars"),
                html.Hr(),
                dcc.Graph(id="graph-workout-mmp-pdc"),

                html.Div(style={"height": "40px"}),
            ]),
        ]),
    ],
)


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("page-fitness",          "style"),
    Output("page-pdc-model",        "style"),
    Output("page-testing",          "style"),
    Output("page-activities-list",  "style"),
    Output("page-activities",       "style"),
    Output("page-workout-list",     "style"),
    Output("page-workout",          "style"),
    Output("nav-fitness",           "style"),
    Output("nav-activities",        "style"),
    Output("nav-workout",           "style"),
    Output("nav-pdc-model",         "style"),
    Output("nav-testing",           "style"),
    Input("nav-fitness",            "n_clicks"),
    Input("nav-activities",         "n_clicks"),
    Input("nav-workout",            "n_clicks"),
    Input("nav-pdc-model",          "n_clicks"),
    Input("nav-testing",            "n_clicks"),
)
def switch_page(_, __, ___, ____, _____):
    show = {"display": "block"}
    hide = {"display": "none"}
    if ctx.triggered_id == "nav-activities":
        return (hide, hide, hide, show, hide, hide, hide,
                _NAV_BASE, _NAV_ACTIVE, _NAV_BASE, _NAV_BASE, _NAV_BASE)
    if ctx.triggered_id == "nav-workout":
        return (hide, hide, hide, hide, hide, show, hide,
                _NAV_BASE, _NAV_BASE, _NAV_ACTIVE, _NAV_BASE, _NAV_BASE)
    if ctx.triggered_id == "nav-pdc-model":
        return (hide, show, hide, hide, hide, hide, hide,
                _NAV_BASE, _NAV_BASE, _NAV_BASE, _NAV_ACTIVE, _NAV_BASE)
    if ctx.triggered_id == "nav-testing":
        return (hide, hide, show, hide, hide, hide, hide,
                _NAV_BASE, _NAV_BASE, _NAV_BASE, _NAV_BASE, _NAV_ACTIVE)
    return (show, hide, hide, hide, hide, hide, hide,
            _NAV_ACTIVE, _NAV_BASE, _NAV_BASE, _NAV_BASE, _NAV_BASE)


@app.callback(
    Output("ride-dropdown",          "value", allow_duplicate=True),
    Output("page-activities-list",   "style", allow_duplicate=True),
    Output("page-activities",        "style", allow_duplicate=True),
    Input("activities-table",        "cellClicked"),
    prevent_initial_call=True,
)
def open_activity(cell_clicked):
    if not cell_clicked:
        raise dash.exceptions.PreventUpdate
    ride_id = cell_clicked["rowId"]
    return int(ride_id), {"display": "none"}, {"display": "block"}


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
    Output("known-ride-ids",  "data"),
    Output("toast-message",   "data"),
    Output("ride-dropdown",   "options"),
    Output("ride-dropdown",   "value"),
    Output("graph-90day-mmp",          "figure"),
    Output("graph-pmc-combined",       "figure"),
    Output("graph-pmc",                "figure"),
    Output("graph-pdc-history-power",  "figure"),
    Output("graph-pdc-history-energy", "figure"),
    Output("graph-pdc-history-time",   "figure"),
    Output("graph-pdc-investigation",  "figure"),
    Output("graph-sigmoid-decay",      "figure"),
    Output("status-bar",               "children"),
    Output("metric-boxes",             "children"),
    Output("activities-table",         "rowData"),
    Output("pdc-slider-dates",         "data"),
    Output("graph-pdc-historical",     "figure"),
    Output("pdc-historical-cards",     "children"),
    Input("poll-interval",    "n_intervals"),
    State("known-version",    "data"),
    State("known-ride-ids",   "data"),
    State("ride-dropdown",    "value"),
)
def poll_for_new_data(n_intervals, known_ver, known_ride_ids, current_ride_id):
    ver, rides, mmp_all, mmh_all, pdc_params, daily_pdc, gps_traces = get_data()

    if ver == known_ver and n_intervals > 0:
        # Nothing changed — return no-update for everything
        raise dash.exceptions.PreventUpdate

    print(f"[render] Fitness page: data version {ver}, {len(rides)} rides …")

    # Detect newly added rides
    current_ids = rides["id"].tolist()
    prev_set = set(known_ride_ids or [])
    new_ids = [rid for rid in current_ids if rid not in prev_set]
    toast_msg = ""
    if prev_set and new_ids:
        new_rides = rides[rides["id"].isin(new_ids)]
        names = [
            f"{r['ride_date']}  {r['name'].replace('strava_', '').replace('_', ' ')}"
            for _, r in new_rides.iterrows()
        ]
        if len(names) == 1:
            toast_msg = f"New activity loaded: {names[0]}"
        else:
            toast_msg = f"{len(names)} new activities loaded"

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

    # Build slider dates from daily_pdc
    slider_dates = sorted(daily_pdc["date"].dropna().unique().tolist())

    # Default PDC historical view: latest date
    if slider_dates:
        latest_str = slider_dates[-1]
        pdc_hist_fig = fig_90day_mmp(mmp_all, reference_date=datetime.date.fromisoformat(latest_str))
        pdc_hist_cards = _build_pdc_cards(daily_pdc, latest_str)
    else:
        pdc_hist_fig = go.Figure()
        pdc_hist_cards = []

    print("[render]   90-day MMP chart …")
    _fig_mmp = fig_90day_mmp(mmp_all)
    print("[render]   PMC combined chart …")
    _fig_pmc_c = fig_pmc_combined(pdc_params, rides)
    print("[render]   PMC chart …")
    _fig_pmc = fig_pmc(pdc_params, rides)
    print("[render]   PDC history charts …")
    _fig_pdc_h = fig_pdc_params_history(daily_pdc, rides)
    print("[render]   PDC investigation chart …")
    _fig_pdc_inv = fig_pdc_investigation(mmp_all)
    print("[render]   Sigmoid decay chart …")
    _fig_sig = fig_sigmoid_decay()
    print("[render]   Activities table …")
    _table = _build_table_data(rides, pdc_params, gps_traces)
    print("[render]   Metric boxes …")
    _metrics = _metric_boxes(pdc_params, rides)
    print("[render] Fitness page complete.")

    return (
        ver,
        current_ids,
        toast_msg,
        options,
        selected,
        _fig_mmp,
        _fig_pmc_c,
        _fig_pmc,
        *_fig_pdc_h,
        _fig_pdc_inv,
        _fig_sig,
        status,
        _metrics,
        _table,
        slider_dates,
        pdc_hist_fig,
        pdc_hist_cards,
    )


# Clientside callback: show toast pop-over and auto-dismiss after 5 s
app.clientside_callback(
    """
    function(msg) {
        var container = document.getElementById('toast-container');
        var text = document.getElementById('toast-text');
        if (!msg) {
            return '';
        }
        text.innerText = msg;
        container.style.opacity = '1';
        container.style.pointerEvents = 'auto';
        if (window._toastTimer) clearTimeout(window._toastTimer);
        window._toastTimer = setTimeout(function() {
            container.style.opacity = '0';
            container.style.pointerEvents = 'none';
        }, 5000);
        return '';
    }
    """,
    Output("toast-text", "children"),
    Input("toast-message", "data"),
)


def _build_pdc_cards(daily_pdc: pd.DataFrame, ref_str: str) -> list:
    """Build PDC parameter cards for a given date string."""
    _cs = {"background": "#f8f9fa", "border": "1px solid #dee2e6",
           "borderRadius": "8px", "padding": "12px 20px", "minWidth": "100px",
           "textAlign": "center", "boxShadow": "0 1px 3px rgba(0,0,0,0.08)"}
    _ls = {"fontSize": "11px", "color": "#888", "marginBottom": "4px",
           "textTransform": "uppercase", "letterSpacing": "0.05em"}
    _vs = {"fontSize": "22px", "fontWeight": "bold", "color": "#222"}
    _us = {"fontSize": "12px", "color": "#666", "marginLeft": "3px"}

    row = daily_pdc[daily_pdc["date"] == ref_str]
    if row.empty:
        return []
    r = row.iloc[0]
    map_v  = int(round(r["MAP"]))
    pmax_v = int(round(r["Pmax"]))
    awc_v  = f"{r['AWC']/1000:.1f}"
    ltp_v  = int(round(r["ltp"]))
    tte_v  = f"{r['tte']/60:.0f}" if pd.notna(r.get("tte")) else "—"
    tte_ltp_v = _fmt_tte_ltp(r.get("tte_ltp") if pd.notna(r.get("tte_ltp")) else None)
    return [
        _make_card(ref_str, "", "", {**_cs, "minWidth": "140px"},
                   {**_ls, "fontSize": "13px", "color": "#222"},
                   {**_vs, "fontSize": "16px", "color": "#7a8fbb"}, _us),
        _make_card("Pmax", str(pmax_v), "W", _cs, _ls, _vs, _us),
        _make_card("AWC",  awc_v,       "kJ", _cs, _ls, _vs, _us),
        _make_card("MAP",  str(map_v),  "W", _cs, _ls, _vs, _us),
        _make_card("TtE\u2098\u2090\u209a", tte_v, "min", _cs, _ls, _vs, _us),
        _make_card("LTP",  str(ltp_v),  "W", _cs, _ls, _vs, _us),
        _make_card("TtE\u2097\u209c\u209a", tte_ltp_v, "h:mm", _cs, _ls, _vs, _us),
    ]


# ── Historical PDC click callback ─────────────────────────────────────────────

@app.callback(
    Output("graph-pdc-historical",      "figure",   allow_duplicate=True),
    Output("pdc-historical-cards",      "children", allow_duplicate=True),
    Output("graph-pdc-history-power",   "figure",   allow_duplicate=True),
    Output("graph-pdc-history-energy",  "figure",   allow_duplicate=True),
    Output("graph-pdc-history-time",    "figure",   allow_duplicate=True),
    Input("graph-pdc-history-power",    "clickData"),
    Input("graph-pdc-history-energy",   "clickData"),
    Input("graph-pdc-history-time",     "clickData"),
    State("pdc-slider-dates", "data"),
    prevent_initial_call=True,
)
def update_pdc_historical(click_power, click_energy, click_time, dates):
    # Determine which chart was actually clicked (use triggered input)
    triggered = dash.ctx.triggered_id
    if triggered == "graph-pdc-history-power":
        click_data = click_power
    elif triggered == "graph-pdc-history-energy":
        click_data = click_energy
    elif triggered == "graph-pdc-history-time":
        click_data = click_time
    else:
        click_data = None
    if not dates or not click_data:
        raise dash.exceptions.PreventUpdate

    clicked_x = click_data["points"][0].get("x", "")
    try:
        clicked_date = datetime.date.fromisoformat(clicked_x[:10])
    except (ValueError, TypeError):
        raise dash.exceptions.PreventUpdate

    # Find nearest date in available dates
    best_idx = 0
    best_diff = abs((datetime.date.fromisoformat(dates[0]) - clicked_date).days)
    for i, d in enumerate(dates):
        diff = abs((datetime.date.fromisoformat(d) - clicked_date).days)
        if diff < best_diff:
            best_diff = diff
            best_idx = i

    ref_str = dates[best_idx]
    ref_date = datetime.date.fromisoformat(ref_str)
    print(f"[render] Historical PDC view for {ref_str} …")
    _, _rides, mmp_all, _mmh, _pdc, daily_pdc, *_ = get_data()
    fig = fig_90day_mmp(mmp_all, reference_date=ref_date)
    cards = _build_pdc_cards(daily_pdc, ref_str)
    fig_power, fig_energy, fig_time = fig_pdc_params_history(
        daily_pdc, _rides, reference_date=ref_date)
    print("[render] Historical PDC view complete.")
    return fig, cards, fig_power, fig_energy, fig_time


def _fmt_mmp_duration(seconds: int) -> str:
    """Format a duration in seconds to a human-readable label for the MMP table."""
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{m}min" if s == 0 else f"{m}:{s:02d}"
    h, rem = divmod(seconds, 3600)
    m = rem // 60
    return f"{h}h{m:02d}" if m else f"{h}h"


def _build_mmp_table(this_mmp: pd.DataFrame,
                     prior_best: dict[int, float]) -> html.Div:
    """Build a vertical MMP table with this ride's power, prior best, and delta.

    prior_best maps duration_s → best aged power from other rides in the
    PDC window, used to highlight improvements.
    """
    if this_mmp.empty:
        return html.Div()

    cell = {
        "padding": "7px 14px",
        "borderBottom": "1px solid #dee2e6",
        "fontSize": "13px",
        "color": "#222",
    }
    hdr_cell = {
        **cell,
        "background": "#f1f3f5", "fontWeight": "600",
        "color": "#555", "fontSize": "11px",
        "textTransform": "uppercase", "letterSpacing": "0.03em",
    }

    header = html.Tr([
        html.Th("Duration", style={**hdr_cell, "textAlign": "left"}),
        html.Th("Power", style={**hdr_cell, "textAlign": "right"}),
        html.Th("Prior Best", style={**hdr_cell, "textAlign": "right"}),
        html.Th("Delta", style={**hdr_cell, "textAlign": "right"}),
    ])

    rows = []
    for _, r in this_mmp.iterrows():
        d = int(r["duration_s"])
        p = float(r["power"])
        prior = prior_best.get(d)
        improved = prior is not None and p > prior
        delta = p - prior if prior is not None else None

        row_style = {"background": "rgba(0, 230, 118, 0.10)"} if improved else {}

        dur_style = {**cell, "textAlign": "left", "fontWeight": "500", **row_style}
        pwr_style = {**cell, "textAlign": "right", "fontWeight": "bold", **row_style}
        prior_style = {**cell, "textAlign": "right", "color": "#888", **row_style}
        delta_style = {**cell, "textAlign": "right", **row_style}

        if improved:
            delta_style["color"] = "#00c853"
            delta_style["fontWeight"] = "bold"
            delta_text = f"+{delta:.0f}"
        elif delta is not None:
            delta_style["color"] = "#999"
            delta_text = f"{delta:.0f}"
        else:
            delta_style["color"] = "#ccc"
            delta_text = "—"

        rows.append(html.Tr([
            html.Td(_fmt_mmp_duration(d), style=dur_style),
            html.Td(f"{p:.0f} W", style=pwr_style),
            html.Td(f"{prior:.0f} W" if prior is not None else "—", style=prior_style),
            html.Td(delta_text, style=delta_style),
        ]))

    table = html.Table(
        style={
            "borderCollapse": "collapse", "width": "100%",
            "background": "white", "borderRadius": "8px",
            "overflow": "hidden", "boxShadow": "0 1px 3px rgba(0,0,0,0.08)",
        },
        children=[
            html.Thead(header),
            html.Tbody(rows),
        ],
    )

    return html.Div(children=[
        html.H4("Mean Maximal Power", style={
            "color": "#e8edf5", "fontSize": "14px", "fontWeight": "600",
            "marginBottom": "8px",
        }),
        table,
    ])


@app.callback(
    Output("graph-power-hr",              "figure"),
    Output("graph-hr",                   "figure"),
    Output("graph-tss-components",       "figure"),
    Output("graph-zone-bars",            "figure"),
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
    Output("mmp-table-container",        "children"),
    Output("ride-power-store",           "data"),
    Input("ride-dropdown",    "value"),
    State("known-version",    "data"),
)
def update_ride_charts(ride_id, _ver):
    if ride_id is None:
        raise dash.exceptions.PreventUpdate
    _, rides, mmp_all, mmh_all, pdc_params, _daily_pdc, _gps = get_data()
    ride    = rides[rides["id"] == ride_id].iloc[0]
    print(f"[render] Activity page: {ride['name']} ({ride['ride_date']}) …")
    print("[render]   Loading records …")
    records = load_records(ride_id)
    # Fit on-the-fly once; share params with TSS components so all
    # activity charts use the same PDC parameters as the PDC curve chart.
    print("[render]   Fitting PDC …")
    live_pdc = _fit_pdc_for_ride(ride, mmp_all)

    has_hr    = "heart_rate" in records.columns and records["heart_rate"].notna().any()
    mmh_style = {"display": "none"}

    has_gps   = "latitude" in records.columns and records["latitude"].notna().any()
    map_style = {"display": "block"} if has_gps else {"display": "none"}

    def _i(v):
        return f"{int(round(float(v)))}" if pd.notna(v) else "—"

    def _f2(v):
        return f"{float(v):.2f}" if pd.notna(v) else "—"

    params_row = pdc_params[pdc_params["ride_id"] == ride["id"]]
    stored = params_row.iloc[0] if not params_row.empty else None

    # PDC fit params — prefer on-the-fly, fall back to stored
    if live_pdc is not None:
        map_v  = _i(live_pdc.get("MAP"))
        awc_v  = f"{live_pdc['AWC']/1000:.1f}" if live_pdc.get("AWC") else "—"
        pmax_v = _i(live_pdc.get("Pmax"))
        ltp_v  = _i(live_pdc.get("ltp"))
        tte_v  = f"{live_pdc['tte']/60:.0f}" if live_pdc.get("tte") else "—"
        tte_ltp_v = _fmt_tte_ltp(live_pdc.get("tte_ltp"))
    elif stored is not None:
        map_v  = _i(stored.get("MAP"))
        awc_v  = f"{stored['AWC']/1000:.1f}" if pd.notna(stored.get("AWC")) else "—"
        pmax_v = _i(stored.get("Pmax"))
        ltp_v  = _i(stored.get("ltp"))
        tte_v  = f"{stored['tte']/60:.0f}" if pd.notna(stored.get("tte")) else "—"
        tte_ltp_v = _fmt_tte_ltp(stored.get("tte_ltp") if pd.notna(stored.get("tte_ltp")) else None)
    else:
        map_v = awc_v = pmax_v = ltp_v = tte_v = tte_ltp_v = "—"

    # Ride performance metrics from stored pdc_params
    np_v      = _i(stored.get("normalized_power"))    if stored is not None else "—"
    if_v      = _f2(stored.get("intensity_factor"))   if stored is not None else "—"
    tss_v     = _i(stored.get("tss"))                 if stored is not None else "—"
    vi_v      = (f"{float(stored['variability_index']):.2f}"
                 if stored is not None and pd.notna(stored.get("variability_index")) else "—")
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
        _card("Pmax", pmax_v, "W"),
        _card("AWC",  awc_v,  "kJ"),
        _card("MAP",  map_v,  "W"),
        _card("TtE\u2098\u2090\u209a", tte_v, "min"),
        _card("LTP",  ltp_v,  "W"),
        _card("TtE\u2097\u209c\u209a", tte_ltp_v, "h:mm"),
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
        ("Difficulty", difficulty_v,                 "TSS/h"),
        ("VI",         vi_v,                         ""),
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

    # Compute TSS per zone from the power stream using stored (ingest-time) PDC
    # Compute TSS per zone from the power stream using stored (ingest-time) PDC
    # params so the bars and cumulative TSS chart are consistent with the PMC.
    zone_tss = (0.0, 0.0, 0.0)
    if not records["power"].isna().all() and stored is not None:
        _d_cp  = float(stored["MAP"]) if pd.notna(stored.get("MAP")) else None
        _d_ftp = float(stored["ftp"]) if pd.notna(stored.get("ftp")) else _d_cp
        _d_ltp = float(stored["ltp"]) if pd.notna(stored.get("ltp")) else None
        _d_awc  = float(stored["AWC"])  if pd.notna(stored.get("AWC"))  else None
        _d_pmax = float(stored["Pmax"]) if pd.notna(stored.get("Pmax")) else None
        _d_tau2 = float(stored["tau2"]) if pd.notna(stored.get("tau2")) else None
        if _d_cp is not None and _d_ftp and _d_ftp > 0:
            _el = records["elapsed_s"].to_numpy(dtype=float)
            _pw = records["power"].to_numpy(dtype=float)
            (_t, cum_ltp_s, cum_thresh_s, cum_awc_s,
             *_) = _tss_rate_series(_el, _pw, float(_d_ftp), float(_d_cp), ltp=_d_ltp,
                                    AWC=_d_awc, Pmax=_d_pmax, tau2=_d_tau2)
            zone_tss = (cum_ltp_s[-1], cum_thresh_s[-1], cum_awc_s[-1])

    mmp_table = []

    # Store power array for MMP-click highlighting (elapsed_s + power, NaN→0)
    power_store = None
    if not records["power"].isna().all():
        power_store = {
            "elapsed_s": records["elapsed_s"].tolist(),
            "power": records["power"].fillna(0).tolist(),
        }

    print("[render]   Power chart …")
    _fig_phr = fig_power_hr(records, ride["name"], ltp=ltp_for_zones, map_power=map_for_zones)
    print("[render]   HR chart …")
    _fig_hr = fig_hr(records)
    print("[render]   TSS components chart …")
    _fig_tss = fig_tss_components(records, ride, pdc_params)
    print("[render]   Zone bars chart …")
    _fig_zones = fig_zone_bars(
        zone_data,
        zone_tss[0], zone_tss[1], zone_tss[2],
        ltp_for_zones, map_for_zones,
    )
    print("[render]   MMP / PDC chart …")
    _fig_mmp = fig_mmp_pdc(ride, mmp_all, live_pdc)
    print("[render]   MMH chart …")
    _fig_mmh = fig_mmh(ride, mmh_all)
    print("[render]   Route map …")
    _fig_map = fig_route_map(records)
    print("[render]   Elevation chart …")
    _fig_elev = fig_elevation(records)
    print(f"[render] Activity page complete.")

    return (
        _fig_phr,
        _fig_hr,
        _fig_tss,
        _fig_zones,
        _fig_mmp,
        _fig_mmh,
        mmh_style,
        _fig_map,
        _fig_elev,
        map_style,
        ride_header,
        pdc_stats,
        power_stats,
        hr_stats,
        mmp_table,
        power_store,
    )


# ── Synced x-axis zoom/pan for the three ride-time charts ─────────────────────

@app.callback(
    Output("graph-power-hr",         "figure", allow_duplicate=True),
    Output("graph-hr",              "figure", allow_duplicate=True),
    Output("graph-tss-components",   "figure", allow_duplicate=True),
    Output("graph-elevation",        "figure", allow_duplicate=True),
    Input("graph-power-hr",          "relayoutData"),
    Input("graph-hr",               "relayoutData"),
    Input("graph-tss-components",    "relayoutData"),
    Input("graph-elevation",         "relayoutData"),
    prevent_initial_call=True,
)
def _sync_ride_chart_xaxes(rld_phr, rld_hr, rld_tss_z, rld_elev):
    if not ctx.triggered_id:
        raise dash.exceptions.PreventUpdate

    rld = {
        "graph-power-hr":       rld_phr,
        "graph-hr":             rld_hr,
        "graph-tss-components": rld_tss_z,
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


# ── MMP click → highlight on power trace ──────────────────────────────────────

@app.callback(
    Output("graph-power-hr", "figure", allow_duplicate=True),
    Input("graph-mmp-pdc",   "clickData"),
    State("ride-power-store", "data"),
    prevent_initial_call=True,
)
def _highlight_mmp_on_power(click_data, power_store):
    """When user clicks an MMP point, highlight the matching window on power."""
    if not click_data or not power_store:
        raise dash.exceptions.PreventUpdate

    # The x value of the MMP chart is duration in seconds (log scale)
    point = click_data["points"][0]
    duration = int(round(point["x"]))
    if duration < 1:
        raise dash.exceptions.PreventUpdate

    power = np.array(power_store["power"], dtype=float)
    elapsed = np.array(power_store["elapsed_s"], dtype=float)

    start_idx = find_mmp_window(power, duration)
    if start_idx is None:
        raise dash.exceptions.PreventUpdate

    end_idx = start_idx + duration - 1
    x0 = float(elapsed[start_idx])
    x1 = float(elapsed[min(end_idx, len(elapsed) - 1)])

    patch = Patch()
    # Clear any previous highlight, then add the new one
    patch["layout"]["shapes"] = [
        {
            "type": "rect",
            "xref": "x", "yref": "paper",
            "x0": x0, "x1": x1,
            "y0": 0, "y1": 1,
            "fillcolor": "rgba(255, 165, 0, 0.18)",
            "line": {"width": 1, "color": "rgba(255, 140, 0, 0.5)"},
            "layer": "below",
        },
    ]
    return patch


# ── Workout builder callbacks ─────────────────────────────────────────────────

def _make_power_trace_svg(power: np.ndarray, width: int = 120, height: int = 40) -> str:
    """Return a mini SVG sparkline of the power trace for the workout list."""
    if len(power) < 2:
        return ""
    # Downsample to ~width points for a compact SVG
    step = max(1, len(power) // width)
    p = power[::step]
    n = len(p)
    p_min, p_max = float(np.nanmin(p)), float(np.nanmax(p))
    p_range = p_max - p_min or 1.0
    pad = 2
    x_scale = (width - 2 * pad) / max(n - 1, 1)
    y_scale = (height - 2 * pad) / p_range
    pts = " ".join(
        f"{i * x_scale + pad:.1f},{height - ((p[i] - p_min) * y_scale + pad):.1f}"
        for i in range(n)
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        f'<polyline points="{pts}" fill="none" stroke="#4a9eff" stroke-width="1.2"'
        f' stroke-linejoin="round" stroke-linecap="round"/>'
        f'</svg>'
    )


def _pdc_duration_for_zone_pcts(base_pct: float, thresh_pct: float, awc_pct: float,
                                 pdc: dict | None) -> tuple[str, str, str]:
    """Find the PDC duration where each zone's fraction matches the given percentage.

    Returns formatted duration strings (e.g. "5:00", "1:30:00") for base,
    threshold, and AWC zones, or "—" if no match.
    """
    if (pdc is None or not pdc.get("AWC") or not pdc.get("Pmax")
            or not pdc.get("tau2") or not pdc.get("MAP") or not pdc.get("ltp")):
        return ("—", "—", "—")

    AWC  = pdc["AWC"]
    Pmax = pdc["Pmax"]
    MAP  = pdc["MAP"]
    tau2 = pdc["tau2"]
    ltp  = pdc["ltp"]
    if MAP <= 0 or ltp <= 0:
        return ("—", "—", "—")

    ltp_r = ltp / MAP

    # Compute zone fractions across the PDC at a fine grid of durations
    tte   = pdc.get("tte")
    tte_b = pdc.get("tte_b")
    t_grid = np.logspace(np.log10(0.5), np.log10(7200), 2000)
    p_total = _power_model_extended(t_grid, AWC, Pmax, MAP, tau2, tte, tte_b)
    p_aer_base = MAP * (1.0 - np.exp(-t_grid / tau2))
    # Beyond TtE the total power drops below the raw aerobic ramp; cap aerobic
    p_aer   = np.minimum(p_aer_base, p_total) if tte is not None else p_aer_base

    f_base_grid   = (p_aer * ltp_r) / p_total * 100.0
    f_thresh_grid = (p_aer * (1.0 - ltp_r)) / p_total * 100.0
    f_awc_grid    = (p_total - p_aer) / p_total * 100.0

    def _fmt_duration(seconds: float) -> str:
        s = int(round(seconds))
        if s >= 3600:
            h, rem = divmod(s, 3600)
            m, sec = divmod(rem, 60)
            return f"{h}:{m:02d}:{sec:02d}"
        m, sec = divmod(s, 60)
        return f"{m}:{sec:02d}"

    def _lookup(frac_grid: np.ndarray, target_pct: float) -> str:
        """Interpolate to find the duration where frac_grid == target_pct."""
        if target_pct <= 0:
            return "—"
        # base and threshold fractions increase with duration; AWC decreases.
        if frac_grid[-1] > frac_grid[0]:
            # Increasing
            if target_pct < frac_grid[0] or target_pct > frac_grid[-1]:
                return "—"
            t = float(np.interp(target_pct, frac_grid, t_grid))
        else:
            # Decreasing — reverse for np.interp
            if target_pct < frac_grid[-1] or target_pct > frac_grid[0]:
                return "—"
            t = float(np.interp(target_pct, frac_grid[::-1], t_grid[::-1]))
        return _fmt_duration(t)

    return (
        _lookup(f_base_grid,   base_pct),
        _lookup(f_thresh_grid, thresh_pct),
        _lookup(f_awc_grid,    awc_pct),
    )


def _workout_summary_row(name: str, rows: list[dict],
                         pdc: dict | None) -> dict:
    """Build a summary dict for the workout list table with full metrics."""
    map_w = pdc["MAP"] if pdc else 300.0
    ftp_w = pdc["ftp"] if pdc else map_w
    ltp_w = pdc.get("ltp", 0.0) if pdc else 0.0
    awc_w  = pdc.get("AWC")  if pdc else None
    pmax_w = pdc.get("Pmax") if pdc else None
    tau2_w = pdc.get("tau2") if pdc else None

    records = _build_workout_records(rows, map_w, pdc)
    total_s = len(records)

    if records.empty or total_s < 2:
        return {"Name": name, "Power": "", "Type": "—", "Duration": "0m",
                "Avg Power": "—", "NP": "—", "IF": "—",
                "TSS": "—", "Base TSS": "—", "Thresh TSS": "—",
                "AWC TSS": "—",
                "PDC Base": "—", "PDC Thresh": "—", "PDC AWC": "—"}

    power = records["power"].to_numpy(dtype=float)
    elapsed = records["elapsed_s"].to_numpy(dtype=float)

    avg_w  = float(np.nanmean(power))
    np_val = _normalized_power(power)
    if_val = np_val / ftp_w if ftp_w > 0 else 0.0
    tss    = (total_s / 3600.0) * (np_val / ftp_w) ** 2 * 100.0 if ftp_w > 0 else 0.0

    # Zone TSS breakdown
    (_, cum_ltp, cum_thresh, cum_awc, *_rest) = _tss_rate_series(
        elapsed, power, ftp_w, map_w, ltp=ltp_w,
        AWC=awc_w, Pmax=pmax_w, tau2=tau2_w,
    )

    mins = total_s // 60
    dur_str = f"{mins // 60}h{mins % 60:02d}m" if mins >= 60 else f"{mins}m"

    # Power trace SVG
    svg = _make_power_trace_svg(power)
    thumb = ""
    if svg:
        b64 = base64.b64encode(svg.encode()).decode()
        thumb = f'<img src="data:image/svg+xml;base64,{b64}" style="display:block"/>'

    base_tss   = cum_ltp[-1]
    thresh_tss = cum_thresh[-1]
    awc_tss    = cum_awc[-1]
    total_zone = base_tss + thresh_tss + awc_tss

    if awc_tss >= 1:
        zone_type = "Anaerobic"
    elif thresh_tss >= 1:
        zone_type = "Threshold"
    else:
        zone_type = "Base"

    # PDC equivalent durations for each zone's TSS percentage
    if total_zone > 0:
        base_pct   = base_tss / total_zone * 100.0
        thresh_pct = thresh_tss / total_zone * 100.0
        awc_pct    = awc_tss / total_zone * 100.0
        pdc_base, pdc_thresh, pdc_awc = _pdc_duration_for_zone_pcts(
            base_pct, thresh_pct, awc_pct, pdc)
    else:
        pdc_base = pdc_thresh = pdc_awc = "—"

    return {
        "Name":       name,
        "Power":      thumb,
        "Type":       zone_type,
        "Duration":   dur_str,
        "Avg Power":  f"{avg_w:.0f}",
        "NP":         f"{np_val:.0f}",
        "IF":         f"{if_val:.2f}",
        "TSS":        f"{tss:.0f}",
        "Base TSS":   f"{base_tss:.0f}",
        "Thresh TSS": f"{thresh_tss:.0f}",
        "AWC TSS":    f"{awc_tss:.0f}",
        "PDC Base":   pdc_base,
        "PDC Thresh": pdc_thresh,
        "PDC AWC":    pdc_awc,
    }


@app.callback(
    Output("workout-list-table", "rowData"),
    Input("page-workout-list",   "style"),
    prevent_initial_call=True,
)
def populate_workout_list(style):
    """Refresh the workout list table when the page becomes visible."""
    if style and style.get("display") == "none":
        raise dash.exceptions.PreventUpdate
    workouts = _load_workouts()
    _, rides, _, _, pdc_params, _, _ = get_data()
    pdc = _get_latest_pdc(pdc_params, rides)
    return [_workout_summary_row(k, v, pdc) for k, v in sorted(workouts.items())]


@app.callback(
    Output("workout-table",          "rowData"),
    Output("workout-name-input",     "value"),
    Output("page-workout-list",      "style", allow_duplicate=True),
    Output("page-workout",           "style", allow_duplicate=True),
    Input("workout-list-table",      "cellClicked"),
    Input("workout-new-btn",         "n_clicks"),
    prevent_initial_call=True,
)
def open_workout(cell_clicked, new_clicks):
    show = {"display": "block"}
    hide = {"display": "none"}
    if ctx.triggered_id == "workout-new-btn":
        default_rows = [{
            "work_duration_min": 5, "work_ref": "MAP",
            "work_intensity_pct": 100,
            "rest_duration_min": 5, "rest_ref": "MAP",
            "rest_intensity_pct": 80,
            "repetitions": 5,
        }]
        return default_rows, "", hide, show
    if ctx.triggered_id == "workout-list-table" and cell_clicked:
        name = cell_clicked["rowId"]
        workouts = _load_workouts()
        if name in workouts:
            return workouts[name], name, hide, show
    raise dash.exceptions.PreventUpdate


@app.callback(
    Output("page-workout-list",  "style", allow_duplicate=True),
    Output("page-workout",       "style", allow_duplicate=True),
    Input("btn-back-to-workout-list", "n_clicks"),
    prevent_initial_call=True,
)
def go_back_to_workout_list(n_clicks):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    return {"display": "block"}, {"display": "none"}


@app.callback(
    Output("workout-table", "rowData", allow_duplicate=True),
    Input("workout-add-row",    "n_clicks"),
    Input("workout-dup-row",    "n_clicks"),
    Input("workout-remove-row", "n_clicks"),
    State("workout-table",      "virtualRowData"),
    prevent_initial_call=True,
)
def manage_workout_rows(add_clicks, dup_clicks, remove_clicks, current_rows):
    if ctx.triggered_id == "workout-add-row":
        current_rows.append({
            "work_duration_min": 5, "work_ref": "MAP",
            "work_intensity_pct": 100,
            "rest_duration_min": 5, "rest_ref": "MAP",
            "rest_intensity_pct": 80,
            "repetitions": 3,
        })
        return current_rows
    elif ctx.triggered_id == "workout-dup-row" and current_rows:
        current_rows.append(dict(current_rows[-1]))
        return current_rows
    elif ctx.triggered_id == "workout-remove-row" and len(current_rows) > 1:
        current_rows.pop()
        return current_rows
    raise dash.exceptions.PreventUpdate


@app.callback(
    Output("workout-lib-status", "children"),
    Output("workout-name-input", "value", allow_duplicate=True),
    Output("page-workout-list",  "style", allow_duplicate=True),
    Output("page-workout",       "style", allow_duplicate=True),
    Input("workout-save-btn",    "n_clicks"),
    Input("workout-delete-btn",  "n_clicks"),
    State("workout-name-input",  "value"),
    State("workout-table",       "virtualRowData"),
    prevent_initial_call=True,
)
def save_or_delete_workout(save_clicks, del_clicks, name, row_data):
    trigger = ctx.triggered_id
    workouts = _load_workouts()

    if trigger == "workout-save-btn":
        if not name or not name.strip():
            return "Enter a name first", dash.no_update, dash.no_update, dash.no_update
        name = name.strip()
        workouts[name] = row_data
        _save_workouts(workouts)
        return f"Saved '{name}'", name, dash.no_update, dash.no_update

    elif trigger == "workout-delete-btn":
        target = name.strip() if name else None
        if not target or target not in workouts:
            return "Enter the name of a saved workout to delete", dash.no_update, dash.no_update, dash.no_update
        del workouts[target]
        _save_workouts(workouts)
        # Go back to list after deleting
        return dash.no_update, "", {"display": "block"}, {"display": "none"}

    raise dash.exceptions.PreventUpdate


@app.callback(
    Output("graph-workout-power",          "figure"),
    Output("graph-workout-tss-components", "figure"),
    Output("graph-workout-zone-bars",      "figure"),
    Output("graph-workout-mmp-pdc",        "figure"),
    Output("workout-stats",                "children"),
    Output("workout-pdc-cards",            "children"),
    Input("workout-table",                 "cellValueChanged"),
    Input("workout-table",                 "virtualRowData"),
    State("known-version",                 "data"),
)
def update_workout_charts(cell_changed, row_data, _ver):
    if not row_data:
        raise dash.exceptions.PreventUpdate

    _, rides, mmp_all, _, pdc_params, _, _ = get_data()
    latest_pdc = _get_latest_pdc(pdc_params, rides)
    if latest_pdc is None:
        empty = go.Figure()
        empty.add_annotation(
            text="Import rides to establish your power profile first",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color="grey"),
        )
        empty.update_layout(height=200, template="plotly_white")
        return empty, empty, empty, empty, [], []

    map_w = latest_pdc["MAP"]
    ltp_w = latest_pdc["ltp"]
    ftp_w = latest_pdc["ftp"]

    records = _build_workout_records(row_data, map_w, latest_pdc)
    if records.empty or len(records) < 2:
        raise dash.exceptions.PreventUpdate

    # Power chart (zone-coloured line)
    wk_fig_power = fig_power_hr(records, "Workout", ltp=ltp_w, map_power=map_w)

    # TSS components (stacked areas + difficulty)
    dummy_ride = pd.Series({
        "id": -1, "name": "Workout",
        "ride_date": datetime.date.today().isoformat(),
    })
    wk_fig_tss = fig_tss_components(
        records, dummy_ride, pd.DataFrame(columns=["ride_id"]),
        live_pdc=latest_pdc,
    )

    # Zone bars
    zone_data = calculate_zones(records, ltp_w, map_w)
    elapsed = records["elapsed_s"].to_numpy(dtype=float)
    power   = records["power"].to_numpy(dtype=float)
    (_, cum_ltp, cum_thresh, cum_awc, *_rest) = _tss_rate_series(
        elapsed, power, ftp_w, map_w, ltp=ltp_w,
        AWC=latest_pdc.get("AWC"), Pmax=latest_pdc.get("Pmax"),
        tau2=latest_pdc.get("tau2"),
    )
    wk_fig_zones = fig_zone_bars(
        zone_data,
        cum_ltp[-1], cum_thresh[-1], cum_awc[-1],
        ltp_w, map_w,
    )

    # MMP / PDC overlay
    workout_mmp = calculate_mmp(records, MMP_DURATIONS)
    today_str = datetime.date.today().isoformat()
    workout_mmp_rows = pd.DataFrame([
        {"ride_id": -1, "duration_s": d, "power": p, "ride_date": today_str}
        for d, p in workout_mmp.items()
    ])
    combined_mmp = pd.concat([mmp_all, workout_mmp_rows], ignore_index=True)
    wk_fig_mmp = fig_mmp_pdc(dummy_ride, combined_mmp, live_pdc=latest_pdc)

    # Summary stats
    total_s = len(records)
    np_val  = _normalized_power(power)
    tss     = (total_s / 3600.0) * (np_val / ftp_w) ** 2 * 100.0 if ftp_w > 0 else 0.0
    if_val  = np_val / ftp_w if ftp_w > 0 else 0.0
    avg_w   = float(np.nanmean(power))
    stats = _graph_stat_row([
        ("Duration", f"{total_s // 60}", "min"),
        ("Avg Power", f"{int(avg_w)}", "W"),
        ("NP", f"{int(np_val)}", "W"),
        ("IF", f"{if_val:.2f}", ""),
        ("TSS", f"{int(tss)}", ""),
    ])

    # PDC reference cards
    _cs = {"background": "#f8f9fa", "border": "1px solid #dee2e6",
           "borderRadius": "8px", "padding": "12px 20px", "minWidth": "100px",
           "textAlign": "center", "boxShadow": "0 1px 3px rgba(0,0,0,0.08)"}
    _ls = {"fontSize": "11px", "color": "#888", "marginBottom": "4px",
           "textTransform": "uppercase", "letterSpacing": "0.05em"}
    _vs = {"fontSize": "22px", "fontWeight": "bold", "color": "#222"}
    _us = {"fontSize": "12px", "color": "#666", "marginLeft": "3px"}
    pdc_cards = [
        _make_card("Pmax", f"{int(latest_pdc['Pmax'])}", "W", _cs, _ls, _vs, _us),
        _make_card("AWC",  f"{latest_pdc['AWC']/1000:.1f}", "kJ", _cs, _ls, _vs, _us),
        _make_card("MAP",  f"{int(map_w)}", "W", _cs, _ls, _vs, _us),
        _make_card("TtE\u2098\u2090\u209a",
                   f"{latest_pdc['tte']/60:.0f}" if latest_pdc.get("tte") else "—",
                   "min", _cs, _ls, _vs, _us),
        _make_card("LTP",  f"{int(ltp_w)}", "W", _cs, _ls, _vs, _us),
        _make_card("TtE\u2097\u209c\u209a",
                   _fmt_tte_ltp(latest_pdc.get("tte_ltp")),
                   "h:mm", _cs, _ls, _vs, _us),
    ]

    return wk_fig_power, wk_fig_tss, wk_fig_zones, wk_fig_mmp, stats, pdc_cards


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[boot] Starting Dash server on http://127.0.0.1:8050 …")
    app.run(debug=True, port=8050)
