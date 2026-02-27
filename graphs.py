"""graphs.py — Plotly figure builders for the cycling power analysis dashboard."""

import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from build_database import (
    _power_model, _fit_power_curve, _normalized_power,
    PDC_K, PDC_INFLECTION, PDC_WINDOW,
)

LOG_TICK_S   = [1, 5, 10, 30, 60, 120, 300, 600, 1200, 1800, 3600]
LOG_TICK_LBL = ["1s", "5s", "10s", "30s", "1min", "2min",
                 "5min", "10min", "20min", "30min", "1h"]


# ── Figure builders ───────────────────────────────────────────────────────────

def fig_power_hr(records: pd.DataFrame, ride_name: str) -> go.Figure:
    """Power and heart rate on a shared time axis with dual y-axes."""
    has_power = records["power"].notna().any()
    has_hr    = "heart_rate" in records.columns and records["heart_rate"].notna().any()

    fig = go.Figure()

    if has_power:
        fig.add_trace(go.Scatter(
            x=records["elapsed_min"], y=records["power"],
            mode="lines", name="Power",
            line=dict(color="darkorange", width=1),
            yaxis="y1",
        ))
    if has_hr:
        fig.add_trace(go.Scatter(
            x=records["elapsed_min"], y=records["heart_rate"],
            mode="lines", name="Heart Rate",
            line=dict(color="crimson", width=1),
            yaxis="y2",
        ))

    fig.update_layout(
        title=dict(text="Power & Heart Rate", font=dict(size=15)),
        height=300, margin=dict(t=55, b=40, l=60, r=60),
        template="plotly_white",
        showlegend=False,
        hovermode="x unified",
        xaxis=dict(title_text="Elapsed Time (min)", showgrid=True, gridcolor="lightgrey"),
        yaxis=dict(title=dict(text="Power (W)", font=dict(color="darkorange")),
                   showgrid=True, gridcolor="lightgrey",
                   tickfont=dict(color="darkorange"),
                   fixedrange=True),
        yaxis2=dict(title=dict(text="Heart Rate (bpm)", font=dict(color="crimson")),
                    overlaying="y", side="right", showgrid=False,
                    tickfont=dict(color="crimson"),
                    fixedrange=True),
    )
    return fig


def fig_mmh(ride: pd.Series, mmh_all: pd.DataFrame) -> go.Figure:
    """Mean-maximal heart rate for this ride, with the best of other recent rides."""
    ride_date     = ride["ride_date"]
    ride_date_obj = datetime.date.fromisoformat(ride_date)
    cutoff        = (ride_date_obj - datetime.timedelta(days=PDC_WINDOW)).isoformat()

    this_ride = mmh_all[mmh_all["ride_id"] == ride["id"]].sort_values("duration_s")
    others    = mmh_all[
        mmh_all["ride_date"].between(cutoff, ride_date) &
        (mmh_all["ride_id"] != ride["id"])
    ]

    fig = go.Figure()

    if not others.empty:
        best_other = (
            others.groupby("duration_s")["heart_rate"]
            .max().reset_index().sort_values("duration_s")
        )
        fig.add_trace(go.Scatter(
            x=best_other["duration_s"], y=best_other["heart_rate"],
            mode="lines", name="other rides (best)",
            line=dict(color="steelblue", width=1.5, dash="dot"),
        ))

    if not this_ride.empty:
        fig.add_trace(go.Scatter(
            x=this_ride["duration_s"], y=this_ride["heart_rate"],
            mode="lines+markers", name=f"this ride ({ride_date})",
            marker=dict(size=5),
            line=dict(color="crimson", width=2.2),
        ))

    if this_ride.empty and others.empty:
        fig.add_annotation(
            text="No heart rate data for this ride",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=13, color="grey"),
        )

    fig.update_xaxes(type="log", tickvals=LOG_TICK_S, ticktext=LOG_TICK_LBL,
                     title_text="Duration", showgrid=True, gridcolor="lightgrey")
    fig.update_yaxes(title_text="Heart Rate (bpm)", showgrid=True, gridcolor="lightgrey")
    fig.update_layout(
        title=dict(text="Mean Maximal Heart Rate", font=dict(size=14)),
        height=440, margin=dict(t=70, b=50, l=60, r=20),
        template="plotly_white",
        showlegend=False,
        hovermode="x unified",
    )
    return fig


def fig_route_map(records: pd.DataFrame) -> go.Figure:
    """Scattermap route map using OpenStreetMap tiles (no token required)."""
    gps = records.dropna(subset=["latitude", "longitude"]) if "latitude" in records.columns else pd.DataFrame()
    if gps.empty:
        fig = go.Figure()
        fig.update_layout(
            height=350, margin=dict(t=55, b=5, l=5, r=5),
            title=dict(text="Route Map", font=dict(size=14)),
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            annotations=[dict(text="No GPS data", showarrow=False,
                              font=dict(color="#888", size=14),
                              xref="paper", yref="paper", x=0.5, y=0.5)],
        )
        return fig

    fig = go.Figure(go.Scattermap(
        lat=gps["latitude"], lon=gps["longitude"],
        mode="lines",
        line=dict(width=3, color="#4a90d9"),
        hoverinfo="skip",
    ))
    fig.update_layout(
        title=dict(text="Route Map", font=dict(size=14)),
        height=350,
        margin=dict(t=55, b=5, l=5, r=5),
        map=dict(
            style="open-street-map",
            center=dict(lat=float(gps["latitude"].mean()),
                        lon=float(gps["longitude"].mean())),
            zoom=11,
        ),
    )
    return fig


def fig_elevation(records: pd.DataFrame) -> go.Figure:
    """Elevation profile plotted against elapsed time."""
    alt = records.dropna(subset=["altitude_m"]) if "altitude_m" in records.columns else pd.DataFrame()
    if alt.empty:
        fig = go.Figure()
        fig.update_layout(
            height=350, template="plotly_white",
            margin=dict(t=55, b=40, l=60, r=20),
            title=dict(text="Elevation", font=dict(size=14)),
            annotations=[dict(text="No elevation data", showarrow=False,
                              font=dict(color="#888", size=14),
                              xref="paper", yref="paper", x=0.5, y=0.5)],
        )
        return fig

    fig = go.Figure(go.Scatter(
        x=alt["elapsed_min"], y=alt["altitude_m"],
        mode="lines", name="Elevation",
        fill="tozeroy",
        line=dict(color="#6aaa64", width=2),
        fillcolor="rgba(106,170,100,0.25)",
    ))
    fig.update_layout(
        title=dict(text="Elevation", font=dict(size=14)),
        height=350,
        margin=dict(t=55, b=40, l=60, r=20),
        template="plotly_white",
        hovermode="x unified",
        showlegend=False,
    )
    fig.update_xaxes(title_text="Elapsed Time (min)", showgrid=True, gridcolor="lightgrey")
    fig.update_yaxes(title_text="Elevation (m)", showgrid=True, gridcolor="lightgrey",
                     fixedrange=True)
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
                name="model",
                fill="tonexty", fillcolor="rgba(220,80,30,0.18)",
                line=dict(color="darkorange", width=2, dash="dash"),
            ))

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
        showlegend=False,
        hovermode="x unified",
    )
    return fig


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
                name="model",
                fill="tonexty", fillcolor="rgba(220,80,30,0.18)",
                line=dict(color="darkorange", width=2, dash="dash"),
            ))

    fig.update_xaxes(type="log", tickvals=LOG_TICK_S, ticktext=LOG_TICK_LBL,
                     title_text="Duration", showgrid=True, gridcolor="lightgrey")
    fig.update_yaxes(title_text="Power (W)", showgrid=True, gridcolor="lightgrey")
    fig.update_layout(
        title=dict(
            text=(
                "Power Duration Model<br>"
                f"<sup>Inflection {PDC_INFLECTION} days · K={PDC_K} · "
                f"reference date {today_str}</sup>"
            ),
            font=dict(size=14),
        ),
        height=440, margin=dict(t=90, b=50, l=60, r=20),
        template="plotly_white",
        showlegend=False,
        hovermode="x unified",
    )
    return fig


def fig_90day_mmh(mmh_all: pd.DataFrame) -> go.Figure:
    today     = datetime.date.today()
    cutoff    = (today - datetime.timedelta(days=PDC_WINDOW)).isoformat()
    today_str = today.isoformat()

    window = mmh_all[mmh_all["ride_date"].between(cutoff, today_str)].copy()

    fig = go.Figure()

    if not window.empty:
        window["age_days"]  = window["ride_date"].apply(
            lambda d: (today - datetime.date.fromisoformat(d)).days
        )
        window["weight"]    = 1.0 / (1.0 + np.exp(PDC_K * (window["age_days"] - PDC_INFLECTION)))
        window["aged_hr"]   = window["heart_rate"] * window["weight"]

        aged = (
            window.groupby("duration_s")["aged_hr"]
            .max().reset_index().sort_values("duration_s")
        )
        fig.add_trace(go.Scatter(
            x=aged["duration_s"], y=aged["aged_hr"],
            mode="lines+markers", name="aged MMH",
            marker=dict(size=4),
            line=dict(color="crimson", width=2.5),
        ))

    fig.update_xaxes(type="log", tickvals=LOG_TICK_S, ticktext=LOG_TICK_LBL,
                     title_text="Duration", showgrid=True, gridcolor="lightgrey")
    fig.update_yaxes(title_text="Heart Rate (bpm)", showgrid=True, gridcolor="lightgrey")
    fig.update_layout(
        title=dict(
            text=(
                "Mean Max Heartrate<br>"
                f"<sup>Inflection {PDC_INFLECTION} days · K={PDC_K} · "
                f"reference date {today_str}</sup>"
            ),
            font=dict(size=14),
        ),
        height=440, margin=dict(t=90, b=50, l=60, r=20),
        template="plotly_white",
        showlegend=False,
        hovermode="x unified",
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
        x=df["ride_date"], y=df["ltp"],
        mode="lines+markers", name="LTP (W)",
        line=dict(color="darkorange", width=2),
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
                     rangemode="tozero", secondary_y=False)
    fig.update_yaxes(title_text="AWC (kJ)", showgrid=False,
                     rangemode="tozero", secondary_y=True)
    fig.update_layout(
        title=dict(text="PDC Parameter History", font=dict(size=14)),
        height=380,
        margin=dict(t=60, b=50, l=60, r=60),
        template="plotly_white",
        showlegend=False,
        hovermode="x unified",
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
                     range=[-awc_kj * 0.05, awc_kj * 1.12],
                     fixedrange=True)
    fig.update_layout(
        title=dict(text="W' Balance", font=dict(size=14)),
        height=280,
        margin=dict(t=55, b=40, l=60, r=20),
        showlegend=False,
        hovermode="x unified",
        template="plotly_white",
    )
    return fig


# ── TSS component series ──────────────────────────────────────────────────────

def _tss_rate_series(elapsed_s: np.ndarray, power: np.ndarray,
                     ftp: float, CP: float) -> tuple:
    """Return TSS rate and cumulative TSS_MAP / TSS_AWC series over the ride.

    Uses a 30-second rolling average (same-length via 'same' convolution),
    splits at CP proportionally, and integrates second-by-second.

    Returns (t_min, cum_tss_map, cum_tss_awc, rate_map_ph, rate_awc_ph, rate_1h_avg)
    where rate_*_ph are the instantaneous TSS rates in TSS/hour and
    rate_1h_avg is the 1-hour time-weighted rolling average of total TSS rate.
    """
    p = np.where(np.isnan(power), 0.0, power.astype(float))

    dt = np.empty_like(elapsed_s)
    dt[0]  = 0.0
    dt[1:] = np.diff(elapsed_s)
    dt     = np.clip(dt, 0.0, None)

    # Detect sample rate from the data so the 30-second window is correct for
    # any recording frequency (1 Hz, 2 Hz, …).
    hz     = 1.0 / float(np.median(dt[dt > 0])) if (dt > 0).any() else 1.0
    window = max(1, int(30 * hz))
    kernel = np.ones(window) / window
    p_30s  = np.convolve(p, kernel, mode="same")

    # Split fractions from instantaneous power so that brief above-CP efforts register
    with np.errstate(invalid="ignore", divide="ignore"):
        f_awc = np.where(p > 0, np.maximum(p - CP, 0.0) / p, 0.0)
    f_map = 1.0 - f_awc

    if ftp > 0:
        tss_rate_ph = (p_30s / ftp) ** 2 * 100.0        # TSS per hour at each point
        tss_rate    = tss_rate_ph * dt / 3600.0          # TSS increment per sample
    else:
        tss_rate_ph = np.zeros_like(p_30s)
        tss_rate    = np.zeros_like(p_30s)

    # Scale so cumulative values match the NP-based TSS (same basis as stat cards).
    # Uses _normalized_power — the exact same function that produced the stored TSS —
    # so tss_np == stored TSS and the chart endpoints match the stat cards precisely.
    dur_s   = float(elapsed_s[-1] - elapsed_s[0]) if len(elapsed_s) > 1 else 0.0
    np_val  = _normalized_power(power, hz) if ftp > 0 else 0.0
    tss_np  = (dur_s / 3600.0) * (np_val / ftp) ** 2 * 100.0 if ftp > 0 else 0.0
    tss_p30 = float(np.sum(tss_rate))
    scale   = (tss_np / tss_p30) if tss_p30 > 0 else 1.0

    cum_tss_map  = np.cumsum(tss_rate * f_map) * scale
    cum_tss_awc  = np.cumsum(tss_rate * f_awc) * scale
    rate_map_ph  = tss_rate_ph * f_map * scale
    rate_awc_ph  = tss_rate_ph * f_awc * scale
    t_min        = elapsed_s / 60.0

    # 1-hour time-weighted rolling average of total TSS rate.
    # Uses prefix sums for O(n log n) efficiency; dt-weighted so that
    # pauses and irregular sampling are handled correctly.
    cum_dt_ext      = np.empty(len(dt) + 1)
    cum_dt_ext[0]   = 0.0
    cum_dt_ext[1:]  = np.cumsum(dt)
    cum_rdt_ext     = np.empty(len(dt) + 1)
    cum_rdt_ext[0]  = 0.0
    cum_rdt_ext[1:] = np.cumsum(tss_rate_ph * dt * scale)
    left            = np.searchsorted(elapsed_s, elapsed_s - 3600.0, side="left")
    idx             = np.arange(len(dt))
    window_dt       = cum_dt_ext[idx + 1] - cum_dt_ext[left]
    window_rdt      = cum_rdt_ext[idx + 1] - cum_rdt_ext[left]
    with np.errstate(invalid="ignore", divide="ignore"):
        rate_1h_avg = np.where(window_dt > 0, window_rdt / window_dt, 0.0)

    return t_min, cum_tss_map, cum_tss_awc, rate_map_ph, rate_awc_ph, rate_1h_avg


def fig_tss_components(records: pd.DataFrame, ride: pd.Series,
                       pdc_params: pd.DataFrame,
                       live_pdc: dict | None = None) -> go.Figure:
    """Instantaneous TSS rate (TSS/h) split into MAP and AWC components."""
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
    t_min, cum_map, cum_awc, rate_map, rate_awc, rate_1h_avg = _tss_rate_series(elapsed, power, ftp, CP)

    final_map = cum_map[-1]
    final_awc = cum_awc[-1]
    rate_total = rate_map + rate_awc

    fig = go.Figure()

    # Aerobic layer (fills from zero)
    fig.add_trace(go.Scatter(
        x=t_min, y=rate_map,
        mode="lines", name=f"TSS_MAP (total {final_map:.0f})",
        fill="tozeroy", fillcolor="rgba(46,139,87,0.25)",
        line=dict(color="seagreen", width=1.5),
    ))
    # Anaerobic layer (fills from aerobic to total)
    fig.add_trace(go.Scatter(
        x=t_min, y=rate_total,
        mode="lines", name=f"TSS_AWC (total {final_awc:.0f})",
        fill="tonexty", fillcolor="rgba(220,80,30,0.22)",
        line=dict(color="darkorange", width=1.5),
    ))
    # 1-hour time-weighted rolling average of total TSS rate
    fig.add_trace(go.Scatter(
        x=t_min, y=rate_1h_avg,
        mode="lines", name="Difficulty",
        line=dict(color="midnightblue", width=1.5),
    ))

    fig.update_xaxes(title_text="Elapsed Time (min)",
                     showgrid=True, gridcolor="lightgrey")
    fig.update_yaxes(title_text="TSS Rate (TSS/h)",
                     showgrid=True, gridcolor="lightgrey",
                     fixedrange=True)
    fig.update_layout(
        title=dict(text="TSS Rate", font=dict(size=14)),
        height=260,
        margin=dict(t=55, b=40, l=60, r=20),
        template="plotly_white",
        showlegend=False,
        hovermode="x unified",
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
        showlegend=False,
        hovermode="x unified",
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

    dates = pd.date_range(daily_tss.index.min(), pd.Timestamp.today().normalize(), freq="D")
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
        showlegend=False,
    )
    return fig
