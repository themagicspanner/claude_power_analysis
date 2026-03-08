"""pdc_fitting.py — Power-duration curve model, fitting, and derived metrics.

Centralises the two-component PDC model, IRLS fitting, normalized power,
TSS component splitting, and aerobic decoupling — previously scattered
across build_database.py and imported by app.py and graphs.py.
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


# ── Two-component power-duration model ────────────────────────────────────────

def power_model(t, AWC, Pmax, MAP, tau2):
    """Two-component power-duration model.

    P(t) = AWC/t * (1 - exp(-t/tau))  +  MAP * (1 - exp(-t/tau2))

    where tau = AWC/Pmax  (Pmax is the instantaneous power limit as t → 0).
    """
    tau = AWC / Pmax
    return AWC / t * (1.0 - np.exp(-t / tau)) + MAP * (1.0 - np.exp(-t / tau2))


def fit_power_curve(dur: np.ndarray, pwr: np.ndarray,
                    n_iter: int = 8, asymmetry: float = 10.0,
                    p0_init: list | None = None):
    """Skimming fit via iteratively reweighted least squares (IRLS).

    Points above the model (hard efforts) receive weight `asymmetry`; points
    below receive weight 1, so the curve rides the upper envelope.

    Returns (popt, True) on success or (None, False) on failure.
    popt = [AWC, Pmax, MAP, tau2]

    Pass p0_init (a previous popt) to warm-start the solver; it is treated as
    the result of iteration 0, skipping the cold-start phase.  Only used when
    it lies within bounds.
    """
    p0_default = [20_000, float(pwr.max()) * 1.1, float(np.percentile(pwr, 90)) * 0.9, 300.0]
    bounds  = ([0, 0, 0, 1], [500_000, 5_000, 3_000, 3_600])
    weights = np.ones(len(dur))
    # Warm-start: validate p0_init against bounds before using it
    popt: list | None = None
    if p0_init is not None:
        lo, hi = bounds
        if all(lo[j] <= p0_init[j] <= hi[j] for j in range(4)):
            popt = list(p0_init)

    for i in range(n_iter):
        try:
            popt, _ = curve_fit(
                power_model, dur, pwr,
                p0=p0_default if popt is None else popt,
                bounds=bounds,
                sigma=1.0 / weights,
                absolute_sigma=False,
                maxfev=10_000,
            )
        except Exception as exc:
            print(f"[fit] IRLS iter {i} failed: {exc}")
            return None, False
        residuals = pwr - power_model(dur, *popt)
        weights   = np.where(residuals > 0, asymmetry, 1.0)

    return popt, True


# ── Normalized Power ─────────────────────────────────────────────────────────

def normalized_power(power: np.ndarray, sample_hz: float = 1.0) -> float:
    """Coggan normalized power — 4th-root of the mean 4th-power of the
    30-second rolling average. NaN samples are treated as 0 W."""
    p = np.where(np.isnan(power), 0.0, power.astype(float))
    window = max(1, int(30 * sample_hz))
    if len(p) < window:
        return float(np.mean(p))
    kernel  = np.ones(window) / window
    rolling = np.convolve(p, kernel, mode="valid")
    return float(np.mean(rolling ** 4) ** 0.25)


# ── TSS component splitting ──────────────────────────────────────────────────

def tss_components(elapsed_s: np.ndarray, power: np.ndarray,
                   ftp: float, CP: float, tss_total: float,
                   sample_hz: float = 1.0,
                   ltp: float | None = None,
                   AWC_val: float | None = None,
                   Pmax_val: float | None = None,
                   tau2_val: float | None = None) -> tuple[float, float, float]:
    """Split the NP-based TSS into base (LTP), threshold, and anaerobic parts.

    Uses p_30s² as a time-weighting kernel (same as NP methodology) to decide
    how much of each second's training stress should be credited to each zone.

    When AWC_val/Pmax_val/tau2_val are provided, zone fractions use the PDC
    model's time-dependent aerobic ramp-up so that sprint-level powers
    attribute only a small fraction to base/threshold (matching the sigmoidal
    shape of the PDC chart).

    The final values are scaled so that tss_ltp + tss_map + tss_awc = tss_total
    exactly (tss_map is the *total* aerobic component, unchanged from before;
    tss_ltp is the sub-component at or below LTP).

    Returns (tss_ltp, tss_map, tss_awc).
    """
    p = np.where(np.isnan(power), 0.0, power.astype(float))
    window = max(1, int(30 * sample_hz))
    kernel = np.ones(window) / window
    p_30s  = np.convolve(p, kernel, mode="same")   # same length as input

    dt = np.empty_like(elapsed_s)
    dt[0]  = 0.0
    dt[1:] = np.diff(elapsed_s)
    dt     = np.clip(dt, 0.0, None)

    # Split fractions from instantaneous power using PDC model when available
    use_pdc = (AWC_val is not None and Pmax_val is not None and tau2_val is not None
               and AWC_val > 0 and Pmax_val > 0 and tau2_val > 0
               and CP > 0 and ltp is not None and ltp > 0)

    if use_pdc:
        t_grid = np.logspace(-1, np.log10(7200), 2000)
        p_total_grid = power_model(t_grid, AWC_val, Pmax_val, CP, tau2_val)
        p_aer_grid   = CP * (1.0 - np.exp(-t_grid / tau2_val))
        p_total_rev  = p_total_grid[::-1]
        p_aer_rev    = p_aer_grid[::-1]
        ltp_frac_r   = ltp / CP

        with np.errstate(invalid="ignore", divide="ignore"):
            above_map = p > CP
            p_aer_at_p = np.where(
                above_map,
                np.interp(p, p_total_rev, p_aer_rev, left=CP, right=0.0),
                p,
            )
            p_base   = np.where(above_map, p_aer_at_p * ltp_frac_r, np.minimum(p, ltp))
            p_thresh = np.where(above_map, p_aer_at_p - p_base,
                                np.maximum(np.minimum(p, CP) - ltp, 0.0))
            f_awc = np.where(p > 0, np.maximum(p - p_base - p_thresh, 0.0) / p, 0.0)
            f_ltp = np.where(p > 0, p_base / p, 1.0)
    elif ltp is not None and ltp > 0 and CP > 0:
        with np.errstate(invalid="ignore", divide="ignore"):
            f_awc = np.where(p > 0, np.maximum(p - CP, 0.0) / p, 0.0)
            f_ltp = np.where(p > 0, np.minimum(p, ltp) / p, 1.0)
    else:
        with np.errstate(invalid="ignore", divide="ignore"):
            f_awc = np.where(p > 0, np.maximum(p - CP, 0.0) / p, 0.0)
        f_ltp = 1.0 - f_awc

    # p_30s² weights — same basis as NP; used as split ratio only
    weights = (p_30s ** 2) * dt if ftp > 0 else np.zeros_like(p_30s)
    w_total = float(np.sum(weights))
    if w_total > 0 and tss_total > 0:
        awc_frac = float(np.sum(weights * f_awc)) / w_total
        ltp_frac = float(np.sum(weights * f_ltp)) / w_total
        tss_awc  = tss_total * awc_frac
        tss_map  = tss_total - tss_awc
        tss_ltp  = min(tss_total * ltp_frac, tss_map)  # clamp for float safety
    else:
        tss_awc = 0.0
        tss_map = float(tss_total)
        tss_ltp = float(tss_total)
    return tss_ltp, tss_map, tss_awc


# ── Aerobic decoupling ────────────────────────────────────────────────────────

def aerobic_decoupling(df: pd.DataFrame) -> float | None:
    """Aerobic decoupling: drift in power:HR efficiency ratio between ride halves.

    Splits the ride in two equal halves and computes the standard TrainingPeaks
    aerobic decoupling metric:

        AeDec% = (PHR_first − PHR_second) / PHR_first × 100

    where PHR = average(power) / average(heart_rate) for each half.

    Positive value = HR drifted up relative to power (cardiac drift / fatigue).
    Negative value = efficiency improved in second half (rare).
    < 5 % = well-paced with good aerobic base; > 10 % = notable cardiac drift.

    Returns None when there are fewer than 60 samples with both power and HR data.
    """
    valid = df[["power", "heart_rate"]].dropna()
    if len(valid) < 60:
        return None
    n = len(valid) // 2
    first, second = valid.iloc[:n], valid.iloc[n:]
    phr1 = first["power"].mean() / first["heart_rate"].mean()
    phr2 = second["power"].mean() / second["heart_rate"].mean()
    return round((phr1 - phr2) / phr1 * 100, 2) if phr1 > 0 else None
