"""workouts.py — Workout builder helpers and persistence.

Extracted from app.py to keep the dashboard module focused on layout and
callbacks.  Contains workout simulation, persistence (JSON), reference-power
resolution, and summary row generation.
"""

import json
import os

import numpy as np
import pandas as pd

from pdc_fitting import power_model, normalized_power


BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
WORKOUTS_PATH = os.path.join(BASE_DIR, "saved_workouts.json")


# ── PDC param helpers ─────────────────────────────────────────────────────────

def get_latest_pdc(pdc_params: pd.DataFrame,
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
    return {
        "MAP":  float(r["MAP"]),
        "AWC":  float(r["AWC"]),
        "Pmax": float(r["Pmax"]),
        "tau2": float(r["tau2"]) if pd.notna(r.get("tau2")) else 300.0,
        "ftp":  float(r["ftp"]) if pd.notna(r.get("ftp")) else float(r["MAP"]),
        "ltp":  float(r["ltp"]) if pd.notna(r.get("ltp")) else 0.0,
    }


# ── Saved workouts persistence ───────────────────────────────────────────────

def load_workouts() -> dict[str, list[dict]]:
    """Load saved workouts from JSON file. Returns {name: rowData}."""
    if os.path.exists(WORKOUTS_PATH):
        with open(WORKOUTS_PATH, "r") as f:
            return json.load(f)
    return {}


def save_workouts(workouts: dict[str, list[dict]]) -> None:
    """Persist workouts dict to JSON file."""
    with open(WORKOUTS_PATH, "w") as f:
        json.dump(workouts, f, indent=2)


# ── Reference power resolution ───────────────────────────────────────────────

def resolve_ref_watts(ref: str, pdc: dict | None, map_watts: float) -> float:
    """Return the reference power in watts for a given zone label."""
    if pdc is None:
        return map_watts
    ref = (ref or "MAP").upper()
    if ref == "FTP":
        return float(pdc.get("ftp") or map_watts)
    if ref == "LTP":
        return float(pdc.get("ltp") or 0.0) or map_watts * 0.75
    if ref == "PMAX":
        return float(pdc.get("Pmax") or map_watts)
    return map_watts  # default: MAP


# ── Workout simulation ───────────────────────────────────────────────────────

def build_workout_records(row_data: list[dict],
                          map_watts: float,
                          pdc: dict | None = None) -> pd.DataFrame:
    """Generate a 1-Hz simulated power DataFrame from workout interval rows."""
    power_samples: list[float] = []
    for row in row_data:
        work_s = int(float(row.get("work_duration_min") or 0) * 60)
        rest_s = int(float(row.get("rest_duration_min") or 0) * 60)
        work_ref_w = resolve_ref_watts(row.get("work_ref", "MAP"), pdc, map_watts)
        rest_ref_w = resolve_ref_watts(row.get("rest_ref", "MAP"), pdc, map_watts)
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


# ── Power trace SVG sparkline ────────────────────────────────────────────────

def make_power_trace_svg(power: np.ndarray, width: int = 120, height: int = 40) -> str:
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


# ── PDC duration for zone percentages ────────────────────────────────────────

def pdc_duration_for_zone_pcts(base_pct: float, thresh_pct: float, awc_pct: float,
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
    t_grid = np.logspace(np.log10(0.5), np.log10(7200), 2000)
    p_total = power_model(t_grid, AWC, Pmax, MAP, tau2)
    p_aer   = MAP * (1.0 - np.exp(-t_grid / tau2))

    f_base_grid   = (p_aer * ltp_r) / p_total * 100.0
    f_thresh_grid = (p_aer * (1.0 - ltp_r)) / p_total * 100.0
    f_awc_grid    = (p_total - p_aer) / p_total * 100.0

    def _fmt(seconds: float) -> str:
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
        return _fmt(t)

    return (
        _lookup(f_base_grid,   base_pct),
        _lookup(f_thresh_grid, thresh_pct),
        _lookup(f_awc_grid,    awc_pct),
    )


# ── Workout summary row ─────────────────────────────────────────────────────

def workout_summary_row(name: str, rows: list[dict],
                        pdc: dict | None,
                        tss_rate_fn=None) -> dict:
    """Build a summary dict for the workout list table with full metrics.

    *tss_rate_fn* should be the _tss_rate_series function from graphs.py,
    passed by the caller to avoid a circular import.
    """
    import base64

    map_w = pdc["MAP"] if pdc else 300.0
    ftp_w = pdc["ftp"] if pdc else map_w
    ltp_w = pdc.get("ltp", 0.0) if pdc else 0.0
    awc_w  = pdc.get("AWC")  if pdc else None
    pmax_w = pdc.get("Pmax") if pdc else None
    tau2_w = pdc.get("tau2") if pdc else None

    records = build_workout_records(rows, map_w, pdc)
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
    np_val = normalized_power(power)
    if_val = np_val / ftp_w if ftp_w > 0 else 0.0
    tss    = (total_s / 3600.0) * (np_val / ftp_w) ** 2 * 100.0 if ftp_w > 0 else 0.0

    # Zone TSS breakdown
    if tss_rate_fn is not None:
        (_, cum_ltp, cum_thresh, cum_awc, *_rest) = tss_rate_fn(
            elapsed, power, ftp_w, map_w, ltp=ltp_w,
            AWC=awc_w, Pmax=pmax_w, tau2=tau2_w,
        )
        base_tss   = cum_ltp[-1]
        thresh_tss = cum_thresh[-1]
        awc_tss    = cum_awc[-1]
    else:
        base_tss = thresh_tss = awc_tss = 0.0

    mins = total_s // 60
    dur_str = f"{mins // 60}h{mins % 60:02d}m" if mins >= 60 else f"{mins}m"

    # Power trace SVG
    svg = make_power_trace_svg(power)
    thumb = ""
    if svg:
        b64 = base64.b64encode(svg.encode()).decode()
        thumb = f'<img src="data:image/svg+xml;base64,{b64}" style="display:block"/>'

    total_zone = base_tss + thresh_tss + awc_tss

    if awc_tss >= 1:
        zone_type = "Anaerobic"
    elif thresh_tss >= 1:
        zone_type = "Threshold"
    else:
        zone_type = "Base"

    # PDC equivalent durations for each zone's TSS percentage
    if total_zone > 0:
        b_pct = base_tss / total_zone * 100.0
        t_pct = thresh_tss / total_zone * 100.0
        a_pct = awc_tss / total_zone * 100.0
        pdc_base, pdc_thresh, pdc_awc = pdc_duration_for_zone_pcts(
            b_pct, t_pct, a_pct, pdc)
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
