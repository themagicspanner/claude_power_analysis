"""freshness.py — Training freshness / readiness status computation.

Extracted from app.py.  Uses zone-specific TSB (Training Stress Balance) to
determine whether the athlete is ready for all training, aerobic only, needs
rest from intensity, or needs full rest.
"""

import numpy as np
import pandas as pd

from helpers import compute_pmc


# ── Freshness status config ──────────────────────────────────────────────────

FRESHNESS_CFG = {
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


# ── Days to trainable ────────────────────────────────────────────────────────

def days_to_trainable(atl: float, ctl: float,
                      threshold_pct: float = 0.0,
                      max_days: int = 60) -> int | None:
    """Days of complete rest until TSB > -threshold_pct * CTL.

    threshold_pct = 0.30 for the MAP aerobic boundary (TSB > -30% CTL),
                  = 0.0  for the AWC high-intensity boundary (TSB > 0).
    Uses the same ATL τ=7 d / CTL τ=42 d decay as compute_pmc.
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


# ── Freshness status computation ─────────────────────────────────────────────

def compute_freshness_status(pdc_params: pd.DataFrame,
                              rides: pd.DataFrame) -> tuple:
    """Return freshness tuple including base, threshold, and AWC components.

    Uses three zone-specific TSB values against CTL-relative cutoffs:
      - Base (≤ LTP):         cutoff = −50 % of base CTL
      - Threshold (LTP→MAP):  cutoff = −30 % of threshold CTL
      - AWC (> MAP):          cutoff = TSB > 0

    status is 'green'  — all three OK  (ready for anything)
              'amber'  — base & thresh OK but TSB_AWC ≤ 0  (aerobic only)
              'red'    — thresh below cutoff (rest from intensity; base OK)
              'black'  — base below cutoff  (full rest / recovery)
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

    pmc_base   = compute_pmc(daily["tss_ltp"])
    pmc_thresh = compute_pmc(daily["tss_thresh"])
    pmc_awc    = compute_pmc(daily["tss_awc"])

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
