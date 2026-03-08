"""helpers.py — Shared utility functions for the cycling power analysis project.

Centralises commonly repeated logic (sigmoid aging, LTP calculation,
rolling-window max averages, DB queries, PDC parameter extraction) to
eliminate duplication across build_database.py, app.py, and graphs.py.
"""

import datetime
import sqlite3

import numpy as np
import pandas as pd


# ── Sigmoid aging constants for decayed MMP / PDC fitting ─────────────────────

PDC_K          = 0.15   # steepness of the S-curve
PDC_INFLECTION = 97     # days to midpoint (weight = 0.5)
PDC_WINDOW     = 150    # days of history to include


# ── LTP calculation ───────────────────────────────────────────────────────────

def calculate_ltp(AWC: float, MAP: float) -> float:
    """Lower Threshold Power from the two-component PDC model.

    LTP = MAP * (1 - (5/2) * (AWC_kJ / MAP))
    """
    return float(MAP * (1.0 - (5.0 / 2.0) * ((AWC / 1000.0) / MAP)))


# ── Sigmoid aging ─────────────────────────────────────────────────────────────

def apply_sigmoid_aging(df: pd.DataFrame,
                        reference_date: datetime.date,
                        value_col: str = "power") -> pd.DataFrame:
    """Apply sigmoid decay weighting to an MMP/MMH DataFrame.

    Expects *df* to have columns ``ride_date`` (ISO-8601 str) and *value_col*.
    Returns a copy with added columns ``age_days``, ``weight``, and
    ``aged_<value_col>``, then groups by ``duration_s`` taking the max of
    the aged value.

    The returned DataFrame has columns ``duration_s`` and
    ``aged_<value_col>``, sorted by ``duration_s``.
    """
    out = df.copy()
    aged_col = f"aged_{value_col}"
    out["age_days"] = out["ride_date"].apply(
        lambda d: (reference_date - datetime.date.fromisoformat(d)).days
    )
    out["weight"]   = 1.0 / (1.0 + np.exp(PDC_K * (out["age_days"] - PDC_INFLECTION)))
    out[aged_col]   = out[value_col] * out["weight"]
    return out


def aged_envelope(df: pd.DataFrame,
                  reference_date: datetime.date,
                  value_col: str = "power") -> pd.DataFrame:
    """Apply sigmoid aging and return the best aged value per duration.

    Convenience wrapper around apply_sigmoid_aging that groups by duration_s
    and takes the max.
    """
    aged_col = f"aged_{value_col}"
    out = apply_sigmoid_aging(df, reference_date, value_col)
    return (
        out.groupby("duration_s")[aged_col]
        .max().reset_index().sort_values("duration_s")
    )


# ── Rolling-window max average ───────────────────────────────────────────────

def calculate_rolling_max_avg(df: pd.DataFrame,
                              column: str,
                              durations: list[int]) -> dict[int, float]:
    """Return {duration_s: best_avg} for each requested duration.

    Uses a cumulative-sum sliding window (O(n) per duration).
    NaN values are filled with 0.
    """
    if df.empty or column not in df.columns or df[column].isna().all():
        return {}

    data = df[column].fillna(0).to_numpy(dtype=float)
    n = len(data)
    cumsum = data.cumsum()

    result: dict[int, float] = {}
    for d in durations:
        if n < d:
            continue
        window_sums = cumsum[d - 1:].copy()
        window_sums[1:] -= cumsum[:n - d]
        result[d] = float(window_sums.max() / d)

    return result


# ── DB query helper ───────────────────────────────────────────────────────────

def query_db(db_path: str, sql: str,
             params: tuple | None = None) -> pd.DataFrame:
    """Run a SQL query against the database and return a DataFrame."""
    conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql(sql, conn, params=params)
    finally:
        conn.close()


# ── PDC parameter extraction ─────────────────────────────────────────────────

def extract_pdc_params(live_pdc: dict | None,
                       params_row: pd.DataFrame) -> dict | None:
    """Extract PDC model parameters from live_pdc dict or pdc_params row.

    Returns a dict with keys CP, ftp, ltp, AWC, Pmax, tau2 or None if
    no valid source is available.
    """
    if live_pdc is not None:
        return {
            "CP":   live_pdc["MAP"],
            "ftp":  live_pdc["ftp"],
            "ltp":  live_pdc.get("ltp"),
            "AWC":  live_pdc.get("AWC"),
            "Pmax": live_pdc.get("Pmax"),
            "tau2": live_pdc.get("tau2"),
        }
    if not params_row.empty:
        r = params_row.iloc[0]
        return {
            "CP":   float(r["MAP"]),
            "ftp":  float(r["ftp"]) if pd.notna(r.get("ftp")) else float(r["MAP"]),
            "ltp":  float(r["ltp"]) if pd.notna(r.get("ltp")) else None,
            "AWC":  float(r["AWC"])  if pd.notna(r.get("AWC"))  else None,
            "Pmax": float(r["Pmax"]) if pd.notna(r.get("Pmax")) else None,
            "tau2": float(r["tau2"]) if pd.notna(r.get("tau2")) else None,
        }
    return None


# ── Duration formatting ──────────────────────────────────────────────────────

def fmt_duration(s: int) -> str:
    """Format seconds into a compact human-readable string (e.g. '5min', '1h30min')."""
    if s < 60:
        return f"{s}s"
    if s < 3600:
        m, rem = divmod(s, 60)
        return f"{m}min" if rem == 0 else f"{m}:{rem:02d}"
    h, rem = divmod(s, 3600)
    return f"{h}h" if rem == 0 else f"{h}h{rem // 60}min"
