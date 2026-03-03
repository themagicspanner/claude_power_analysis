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


# ── Zone distribution ─────────────────────────────────────────────────────────

def fig_zone_distribution(zone_data: dict[int, float],
                          ltp: float, map_: float) -> go.Figure:
    """Horizontal stacked bar showing time in each of the three physiological zones.

    Zone 1 (≤ LTP)       — base / recovery
    Zone 2 (LTP – MAP)   — threshold / sweet-spot
    Zone 3 (> MAP)       — high intensity / VO₂ max+
    """
    if not zone_data:
        fig = go.Figure()
        fig.update_layout(
            height=150, template="plotly_white",
            margin=dict(t=45, b=10, l=10, r=10),
            title=dict(text="Zone Distribution", font=dict(size=14)),
            annotations=[dict(text="No zone data available", showarrow=False,
                              font=dict(color="#888", size=13),
                              xref="paper", yref="paper", x=0.5, y=0.5)],
        )
        return fig

    total = sum(zone_data.values())

    def _fmt(seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        return f"{m}:{s:02d}"

    zones = [
        (1, f"Z1  ≤{ltp:.0f} W",            zone_data.get(1, 0.0), "#4a90d9"),
        (2, f"Z2  {ltp:.0f}–{map_:.0f} W",  zone_data.get(2, 0.0), "#f5a623"),
        (3, f"Z3  >{map_:.0f} W",            zone_data.get(3, 0.0), "#e74c3c"),
    ]

    fig = go.Figure()
    for _, name, secs, color in zones:
        pct = secs / total * 100 if total > 0 else 0.0
        fig.add_trace(go.Bar(
            y=[""],
            x=[pct],
            name=f"{name}  {pct:.0f}%  ({_fmt(secs)})",
            orientation="h",
            marker_color=color,
            hovertemplate=f"{name}<br>{pct:.1f}%  ·  {_fmt(secs)}<extra></extra>",
        ))

    fig.update_xaxes(range=[0, 100], showgrid=False, showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(
        title=dict(text="Zone Distribution  (LTP / MAP 3-zone model)", font=dict(size=14)),
        barmode="stack",
        height=150,
        margin=dict(t=55, b=10, l=10, r=10),
        template="plotly_white",
        showlegend=True,
        legend=dict(
            orientation="h", x=0, y=-0.05,
            font=dict(size=11), traceorder="normal",
        ),
        hovermode="closest",
    )
    return fig


# ── TSS component series ──────────────────────────────────────────────────────

def _tss_rate_series(elapsed_s: np.ndarray, power: np.ndarray,
                     ftp: float, CP: float,
                     ltp: float | None = None) -> tuple:
    """Return TSS rate and cumulative LTP / Threshold / AWC series over the ride.

    Uses a 30-second rolling average (same-length via 'same' convolution),
    splits at CP and LTP proportionally, and integrates second-by-second.

    Returns (t_min, cum_ltp, cum_thresh, cum_awc,
             rate_ltp_ph, rate_thresh_ph, rate_awc_ph, rate_1h_avg)
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

    # Sub-split aerobic fraction at LTP
    if ltp is not None and ltp > 0:
        with np.errstate(invalid="ignore", divide="ignore"):
            f_ltp = np.where(p > 0, np.minimum(p, ltp) / p, 0.0)
        f_ltp = np.minimum(f_ltp, f_map)
    else:
        f_ltp = f_map
    f_thresh = f_map - f_ltp

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

    cum_ltp      = np.cumsum(tss_rate * f_ltp)    * scale
    cum_thresh   = np.cumsum(tss_rate * f_thresh)  * scale
    cum_awc      = np.cumsum(tss_rate * f_awc)     * scale
    rate_ltp_ph    = tss_rate_ph * f_ltp    * scale
    rate_thresh_ph = tss_rate_ph * f_thresh * scale
    rate_awc_ph    = tss_rate_ph * f_awc    * scale
    t_min          = elapsed_s / 60.0

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

    return (t_min, cum_ltp, cum_thresh, cum_awc,
            rate_ltp_ph, rate_thresh_ph, rate_awc_ph, rate_1h_avg)


def fig_tss_components(records: pd.DataFrame, ride: pd.Series,
                       pdc_params: pd.DataFrame,
                       live_pdc: dict | None = None) -> go.Figure:
    """TSS Rate (TSS/h) and cumulative TSS split into Base / Threshold / AWC."""
    if records["power"].isna().all():
        return go.Figure()

    params_row = pdc_params[pdc_params["ride_id"] == ride["id"]]

    if live_pdc is not None:
        CP  = live_pdc["MAP"]
        ftp = live_pdc["ftp"]
        ltp = live_pdc.get("ltp")
    elif not params_row.empty:
        r   = params_row.iloc[0]
        CP  = float(r["MAP"])
        ftp = float(r["ftp"]) if pd.notna(r.get("ftp")) else CP
        ltp = float(r["ltp"]) if pd.notna(r.get("ltp")) else None
    else:
        return go.Figure()

    elapsed = records["elapsed_s"].to_numpy(dtype=float)
    power   = records["power"].to_numpy(dtype=float)
    (t_min, cum_ltp, cum_thresh, cum_awc,
     rate_ltp, rate_thresh, rate_awc, rate_1h_avg) = _tss_rate_series(
        elapsed, power, ftp, CP, ltp=ltp)

    final_ltp    = cum_ltp[-1]
    final_thresh = cum_thresh[-1]
    final_awc    = cum_awc[-1]
    rate_ltp_top    = rate_ltp
    rate_thresh_top = rate_ltp + rate_thresh
    rate_total      = rate_thresh_top + rate_awc
    cum_ltp_top     = cum_ltp
    cum_thresh_top  = cum_ltp + cum_thresh
    cum_total       = cum_thresh_top + cum_awc

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        row_heights=[0.5, 0.5],
    )

    # ── Row 1: instantaneous TSS rate ────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=t_min, y=rate_ltp_top,
        mode="lines", name=f"Base ≤LTP ({final_ltp:.0f})",
        fill="tozeroy", fillcolor="rgba(46,139,87,0.25)",
        line=dict(color="seagreen", width=1.5),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=t_min, y=rate_thresh_top,
        mode="lines", name=f"Threshold ({final_thresh:.0f})",
        fill="tonexty", fillcolor="rgba(70,130,180,0.25)",
        line=dict(color="steelblue", width=1.5),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=t_min, y=rate_total,
        mode="lines", name=f"AWC ({final_awc:.0f})",
        fill="tonexty", fillcolor="rgba(220,80,30,0.22)",
        line=dict(color="darkorange", width=1.5),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=t_min, y=rate_1h_avg,
        mode="lines", name="Difficulty",
        line=dict(color="midnightblue", width=1.5),
    ), row=1, col=1)

    # ── Row 2: cumulative TSS ────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=t_min, y=cum_ltp_top,
        mode="lines", name="Cumulative Base",
        fill="tozeroy", fillcolor="rgba(46,139,87,0.25)",
        line=dict(color="seagreen", width=1.5),
        showlegend=False,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=t_min, y=cum_thresh_top,
        mode="lines", name="Cumulative Threshold",
        fill="tonexty", fillcolor="rgba(70,130,180,0.25)",
        line=dict(color="steelblue", width=1.5),
        showlegend=False,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=t_min, y=cum_total,
        mode="lines", name="Cumulative AWC",
        fill="tonexty", fillcolor="rgba(220,80,30,0.22)",
        line=dict(color="darkorange", width=1.5),
        showlegend=False,
    ), row=2, col=1)

    fig.update_xaxes(title_text="Elapsed Time (min)",
                     showgrid=True, gridcolor="lightgrey", row=2, col=1)
    fig.update_xaxes(showgrid=True, gridcolor="lightgrey", row=1, col=1)
    fig.update_yaxes(title_text="TSS Rate (TSS/h)",
                     showgrid=True, gridcolor="lightgrey",
                     fixedrange=True, row=1, col=1)
    fig.update_yaxes(title_text="Cumulative TSS",
                     showgrid=True, gridcolor="lightgrey",
                     fixedrange=True, row=2, col=1)
    fig.update_layout(
        title=dict(text="TSS Rate", font=dict(size=14)),
        height=420,
        margin=dict(t=55, b=40, l=60, r=20),
        template="plotly_white",
        showlegend=False,
        hovermode="x unified",
    )
    return fig


def fig_tss_history(pdc_params: pd.DataFrame, rides: pd.DataFrame) -> go.Figure:
    """TSS per ride as stacked bars — Base (≤LTP) + Threshold (LTP→MAP) + AWC."""
    if pdc_params.empty or "tss_map" not in pdc_params.columns:
        return go.Figure()

    df = (
        pdc_params.dropna(subset=["tss_map", "tss_awc"])
        .merge(rides[["id", "ride_date", "name"]], left_on="ride_id", right_on="id", how="left")
        .sort_values("ride_date")
    )
    if df.empty:
        return go.Figure()

    # Derive the threshold component; tss_ltp may be missing on old DBs
    if "tss_ltp" in df.columns:
        df["tss_ltp"]    = df["tss_ltp"].fillna(df["tss_map"])
        df["tss_thresh"] = (df["tss_map"] - df["tss_ltp"]).clip(lower=0)
    else:
        df["tss_ltp"]    = df["tss_map"]
        df["tss_thresh"] = 0.0

    cd = df[["ftp", "normalized_power", "intensity_factor",
             "tss_ltp", "tss_thresh", "tss_awc"]].values

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["ride_date"], y=df["tss_ltp"],
        name="Base (≤LTP)",
        marker_color="seagreen",
        customdata=cd,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Base ≤LTP = %{y:.0f}<br>"
            "Threshold = %{customdata[4]:.0f}<br>"
            "AWC = %{customdata[5]:.0f}<br>"
            "FTP = %{customdata[0]:.0f} W  NP = %{customdata[1]:.0f} W  "
            "IF = %{customdata[2]:.2f}<extra></extra>"
        ),
    ))
    fig.add_trace(go.Bar(
        x=df["ride_date"], y=df["tss_thresh"],
        name="Threshold (LTP→MAP)",
        marker_color="steelblue",
        customdata=cd,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Threshold = %{y:.0f}<br>"
            "Base ≤LTP = %{customdata[3]:.0f}<br>"
            "AWC = %{customdata[5]:.0f}<br>"
            "FTP = %{customdata[0]:.0f} W  NP = %{customdata[1]:.0f} W  "
            "IF = %{customdata[2]:.2f}<extra></extra>"
        ),
    ))
    fig.add_trace(go.Bar(
        x=df["ride_date"], y=df["tss_awc"],
        name="AWC (anaerobic)",
        marker_color="darkorange",
        customdata=cd,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "AWC = %{y:.0f}<br>"
            "Base ≤LTP = %{customdata[3]:.0f}<br>"
            "Threshold = %{customdata[4]:.0f}<br>"
            "FTP = %{customdata[0]:.0f} W  NP = %{customdata[1]:.0f} W  "
            "IF = %{customdata[2]:.2f}<extra></extra>"
        ),
    ))

    fig.update_xaxes(title_text="Date", showgrid=False)
    fig.update_yaxes(title_text="TSS", showgrid=True, gridcolor="lightgrey")
    fig.update_layout(
        title=dict(text="Training Stress Score per Ride (Base + Threshold + AWC)", font=dict(size=14)),
        barmode="stack",
        height=320,
        margin=dict(t=55, b=50, l=60, r=20),
        template="plotly_white",
        showlegend=False,
        hovermode="x unified",
    )
    return fig


# ── Performance Management Chart ──────────────────────────────────────────────

def _compute_pmc(daily_tss: pd.Series, future_days: int = 0) -> pd.DataFrame:
    """Exponential-weighted ATL (τ=7 d) and CTL (τ=42 d) from a daily TSS series.

    daily_tss : Series with a DatetimeIndex.  Missing dates → 0 TSS (rest days).
    future_days : number of extra days (0 TSS) to project beyond today.
    Returns DataFrame with columns: date, atl, ctl, tsb
    where TSB(d) = CTL(d-1) − ATL(d-1)  (form before today's ride).
    """
    if daily_tss.empty:
        return pd.DataFrame(columns=["date", "atl", "ctl", "tsb"])

    end = pd.Timestamp.today().normalize() + pd.Timedelta(days=future_days)
    dates = pd.date_range(daily_tss.index.min(), end, freq="D")
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


def fig_pdc_investigation(mmp_all: pd.DataFrame) -> go.Figure:
    """PDC model investigation: MMP decay weights + residuals vs the fitted curve.

    Helps the athlete identify which durations need targeted testing efforts by
    showing how fresh each MMP data point is and how well it supports the model.

    Panel 1 — Decayed MMP scatter + PDC curve:
        Each (ride, duration) point is coloured by its sigmoid decay weight.
        Bright blue = recent (trustworthy data); grey = old (may need retesting).
        The envelope (best aged power per duration) and the fitted PDC curve are
        overlaid so the athlete can see where the model sits relative to the data.

    Panel 2 — Residuals (decayed envelope − PDC model):
        Green bar: athlete is above model here (data strongly supports the curve).
        Red bar:   athlete is below model (testing opportunity / model overestimates).
    """
    today     = datetime.date.today()
    today_str = today.isoformat()
    cutoff    = (today - datetime.timedelta(days=PDC_WINDOW)).isoformat()

    window = mmp_all[mmp_all["ride_date"].between(cutoff, today_str)].copy()

    if window.empty:
        fig = go.Figure()
        fig.update_layout(
            title=dict(text="PDC Investigation — no data in window", font=dict(size=14)),
            height=500, template="plotly_white",
            annotations=[dict(text="No MMP data in the current PDC window",
                              showarrow=False, font=dict(color="#888", size=14),
                              xref="paper", yref="paper", x=0.5, y=0.5)],
        )
        return fig

    window["age_days"]   = window["ride_date"].apply(
        lambda d: (today - datetime.date.fromisoformat(d)).days
    )
    window["weight"]     = 1.0 / (1.0 + np.exp(PDC_K * (window["age_days"] - PDC_INFLECTION)))
    window["aged_power"] = window["power"] * window["weight"]

    # Envelope: best aged power per duration, plus metadata of the contributing ride
    env_rows = []
    for dur, grp in window.groupby("duration_s"):
        best = grp.loc[grp["aged_power"].idxmax()]
        env_rows.append({
            "duration_s": dur,
            "aged_power": best["aged_power"],
            "raw_power":  best["power"],
            "weight":     best["weight"],
            "age_days":   best["age_days"],
            "ride_date":  best["ride_date"],
        })
    env_df = pd.DataFrame(env_rows).sort_values("duration_s")

    dur_arr = env_df["duration_s"].to_numpy(dtype=float)
    pwr_arr = env_df["aged_power"].to_numpy(dtype=float)

    popt, ok = (_fit_power_curve(dur_arr, pwr_arr) if len(dur_arr) >= 4
                else (None, False))

    if ok:
        AWC, Pmax, MAP, tau2 = popt
        model_vals             = _power_model(dur_arr, *popt)
        env_df["model_power"]  = model_vals
        env_df["residual"]     = env_df["aged_power"] - env_df["model_power"]
        env_df["residual_pct"] = (env_df["residual"] / env_df["model_power"] * 100).round(1)
        ftp = float(_power_model(3600.0, *popt))
        ltp = float(MAP * (1.0 - (5.0 / 2.0) * ((AWC / 1000.0) / MAP)))

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.62, 0.38],
        vertical_spacing=0.10,
        subplot_titles=[
            "Decayed MMP vs PDC model  (point colour = decay weight)",
            "Residual: aged MMP − PDC model  (green = above model; red = below / test needed)",
        ],
    )

    # ── Panel 1: all individual MMP points ────────────────────────────────────
    _weight_colorscale = [
        [0.0, "rgba(160,160,160,0.20)"],
        [0.4, "rgba(100,150,200,0.55)"],
        [1.0, "rgba(41,128,185,0.90)"],
    ]

    fig.add_trace(go.Scatter(
        x=window["duration_s"],
        y=window["aged_power"],
        mode="markers",
        name="All MMP points (aged)",
        marker=dict(
            size=5,
            color=window["weight"].to_numpy(dtype=float),
            colorscale=_weight_colorscale,
            cmin=0, cmax=1,
            showscale=False,
        ),
        customdata=np.column_stack([
            window["weight"].round(3),
            window["age_days"].round(0),
            window["ride_date"].to_numpy(),
            window["power"].round(0),
        ]),
        hovertemplate=(
            "<b>%{x:.0f} s</b><br>"
            "Aged power: %{y:.0f} W<br>"
            "Raw power:  %{customdata[3]:.0f} W<br>"
            "Weight: %{customdata[0]}  ·  Age: %{customdata[1]:.0f} d<br>"
            "Ride: %{customdata[2]}<extra></extra>"
        ),
    ), row=1, col=1)

    # Envelope: best aged power per duration, marker colour = weight
    fig.add_trace(go.Scatter(
        x=env_df["duration_s"],
        y=env_df["aged_power"],
        mode="lines+markers",
        name="Decayed MMP envelope",
        line=dict(color="steelblue", width=1.5, dash="dot"),
        marker=dict(
            size=10,
            color=env_df["weight"].to_numpy(dtype=float),
            colorscale=_weight_colorscale,
            cmin=0, cmax=1,
            line=dict(color="white", width=1.2),
            showscale=True,
            colorbar=dict(
                title=dict(text="Decay weight", side="right"),
                thickness=13, len=0.52, y=0.78,
                tickvals=[0, 0.5, 1],
                ticktext=["0 (old)", "0.5", "1 (fresh)"],
            ),
        ),
        customdata=np.column_stack([
            env_df["weight"].round(3),
            env_df["age_days"].round(0),
            env_df["ride_date"].to_numpy(),
            env_df["raw_power"].round(0),
        ]),
        hovertemplate=(
            "<b>%{x:.0f} s  (envelope best)</b><br>"
            "Aged power: %{y:.0f} W<br>"
            "Raw power:  %{customdata[3]:.0f} W<br>"
            "Weight: %{customdata[0]}  ·  Age: %{customdata[1]:.0f} d<br>"
            "Ride: %{customdata[2]}<extra></extra>"
        ),
    ), row=1, col=1)

    # PDC model curve
    if ok:
        t_sm  = np.logspace(np.log10(dur_arr.min()), np.log10(dur_arr.max()), 400)
        p_aer = MAP * (1.0 - np.exp(-t_sm / tau2))
        p_tot = _power_model(t_sm, *popt)
        fig.add_trace(go.Scatter(
            x=t_sm, y=p_aer,
            mode="lines", name="aerobic (MAP)",
            fill="tozeroy", fillcolor="rgba(46,139,87,0.15)",
            line=dict(color="rgba(46,139,87,0.6)", width=1),
            hoverinfo="skip",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=t_sm, y=p_tot,
            mode="lines", name="PDC model",
            fill="tonexty", fillcolor="rgba(220,80,30,0.10)",
            line=dict(color="darkorange", width=2.5, dash="dash"),
            hovertemplate="Model: %{y:.0f} W<extra></extra>",
        ), row=1, col=1)

    # ── Panel 2: residuals ────────────────────────────────────────────────────
    if ok:
        residuals  = env_df["residual"].to_numpy(dtype=float)
        bar_colors = ["seagreen" if r >= 0 else "crimson" for r in residuals]
        fig.add_trace(go.Bar(
            x=dur_arr,
            y=residuals,
            name="Residual",
            marker_color=bar_colors,
            customdata=np.column_stack([
                env_df["residual_pct"].to_numpy(),
                env_df["aged_power"].round(0),
                env_df["model_power"].round(0),
                env_df["weight"].round(3),
            ]),
            hovertemplate=(
                "<b>%{x:.0f} s</b><br>"
                "Residual: %{y:+.0f} W (%{customdata[0]:+.1f}%)<br>"
                "Envelope: %{customdata[1]:.0f} W  ·  Model: %{customdata[2]:.0f} W<br>"
                "Weight: %{customdata[3]}<extra></extra>"
            ),
        ), row=2, col=1)
        fig.add_hline(y=0, line=dict(color="grey", dash="dot", width=1), row=2, col=1)

    # ── Axes ──────────────────────────────────────────────────────────────────
    for row in [1, 2]:
        fig.update_xaxes(
            type="log", tickvals=LOG_TICK_S, ticktext=LOG_TICK_LBL,
            showgrid=True, gridcolor="lightgrey",
            row=row, col=1,
        )
    fig.update_xaxes(title_text="Duration", row=2, col=1)
    fig.update_yaxes(title_text="Power (W)",    showgrid=True, gridcolor="lightgrey",
                     row=1, col=1)
    fig.update_yaxes(title_text="Residual (W)", showgrid=True, gridcolor="lightgrey",
                     row=2, col=1)

    if ok:
        title_sub = (
            f"MAP {MAP:.0f} W  ·  FTP {ftp:.0f} W  ·  LTP {ltp:.0f} W  ·  "
            f"AWC {AWC / 1000:.1f} kJ  ·  Pmax {Pmax:.0f} W  ·  "
            f"window {PDC_WINDOW} d  ·  inflection {PDC_INFLECTION} d  ·  "
            f"K={PDC_K}  ·  ref {today_str}"
        )
    else:
        title_sub = (
            f"window {PDC_WINDOW} d  ·  inflection {PDC_INFLECTION} d  ·  "
            f"K={PDC_K}  ·  ref {today_str}"
        )

    fig.update_layout(
        title=dict(
            text=f"PDC Model Investigation<br><sup>{title_sub}</sup>",
            font=dict(size=14),
        ),
        height=640,
        margin=dict(t=110, b=50, l=70, r=90),
        template="plotly_white",
        showlegend=False,
        hovermode="x unified",
    )
    return fig


def fig_sigmoid_decay() -> go.Figure:
    """Plot the sigmoid decay curve used by the PDC model.

    X-axis is age in days (0 = today, rightward = older), Y-axis is the
    decay weight (0–1).  Key landmarks (today, inflection, window cutoff)
    are annotated.
    """
    days = np.linspace(0, PDC_WINDOW + 20, 500)
    weights = 1.0 / (1.0 + np.exp(PDC_K * (days - PDC_INFLECTION)))

    fig = go.Figure()

    # Sigmoid curve
    fig.add_trace(go.Scatter(
        x=days, y=weights,
        mode="lines",
        line=dict(color="steelblue", width=2.5),
        hovertemplate="Age: %{x:.0f} d<br>Weight: %{y:.3f}<extra></extra>",
        name="Decay weight",
    ))

    # Shaded area under the curve
    fig.add_trace(go.Scatter(
        x=days, y=weights,
        mode="lines", line=dict(width=0),
        fill="tozeroy", fillcolor="rgba(70,130,180,0.12)",
        showlegend=False, hoverinfo="skip",
    ))

    # Today marker (day 0)
    fig.add_vline(x=0, line=dict(color="limegreen", width=2))
    fig.add_annotation(
        x=0, y=1.02, yref="paper",
        text="<b>Today</b>", showarrow=False,
        font=dict(color="limegreen", size=11),
        xanchor="left", xshift=4,
    )

    # Inflection point
    fig.add_vline(x=PDC_INFLECTION, line=dict(color="orange", width=1.5, dash="dash"))
    fig.add_annotation(
        x=PDC_INFLECTION, y=0.5,
        text=f"  inflection {PDC_INFLECTION} d  (w = 0.50)",
        showarrow=False,
        font=dict(color="orange", size=10),
        xanchor="left",
    )

    # Window cutoff
    fig.add_vline(x=PDC_WINDOW, line=dict(color="crimson", width=1.5, dash="dot"))
    w_at_cutoff = 1.0 / (1.0 + np.exp(PDC_K * (PDC_WINDOW - PDC_INFLECTION)))
    fig.add_annotation(
        x=PDC_WINDOW, y=1.02, yref="paper",
        text=f"<b>window {PDC_WINDOW} d</b>",
        showarrow=False,
        font=dict(color="crimson", size=10),
        xanchor="right", xshift=-4,
    )

    fig.update_layout(
        title=dict(
            text=(f"Sigmoid Decay Curve<br>"
                  f"<sup>K = {PDC_K}  ·  inflection = {PDC_INFLECTION} d  ·  "
                  f"window = {PDC_WINDOW} d</sup>"),
            font=dict(size=14),
        ),
        xaxis=dict(title="Age (days)", showgrid=True, gridcolor="lightgrey",
                   range=[-5, PDC_WINDOW + 20]),
        yaxis=dict(title="Decay weight", showgrid=True, gridcolor="lightgrey",
                   range=[-0.02, 1.05]),
        height=340,
        margin=dict(t=70, b=50, l=60, r=20),
        template="plotly_white",
        showlegend=False,
    )
    return fig


def fig_pdc_testing_summary(mmp_all: pd.DataFrame) -> go.Figure:
    """Priority testing table for the PDC investigation page.

    Each MMP duration is ranked by testing urgency using a combined score:

        score = max(0, −residual_pct) × 0.7  +  (1 − weight) × 30

    Residual term: being below the model adds up to ~70 pts per 100 % gap.
    Staleness term: fully decayed data (weight = 0) adds 30 pts regardless
    of residual, because old efforts need re-confirming even when they beat
    the model.

    Action labels
    ─────────────
    Test now  (score > 20) — stale data AND/OR well below the model
    Monitor   (score >  8) — one factor mildly out of range
    On track  (score ≤  8) — fresh, recent data supporting the model
    """
    today  = datetime.date.today()
    cutoff = (today - datetime.timedelta(days=PDC_WINDOW)).isoformat()

    window = mmp_all[mmp_all["ride_date"].between(cutoff, today.isoformat())].copy()

    if window.empty:
        fig = go.Figure()
        fig.update_layout(height=80, template="plotly_white",
                          margin=dict(t=10, b=10, l=10, r=10))
        return fig

    window["age_days"]   = window["ride_date"].apply(
        lambda d: (today - datetime.date.fromisoformat(d)).days
    )
    window["weight"]     = 1.0 / (1.0 + np.exp(PDC_K * (window["age_days"] - PDC_INFLECTION)))
    window["aged_power"] = window["power"] * window["weight"]

    env_rows = []
    for dur, grp in window.groupby("duration_s"):
        best = grp.loc[grp["aged_power"].idxmax()]
        env_rows.append({
            "duration_s": dur,
            "aged_power": best["aged_power"],
            "raw_power":  best["power"],
            "weight":     best["weight"],
            "age_days":   int(best["age_days"]),
            "ride_date":  best["ride_date"],
        })
    env_df = pd.DataFrame(env_rows).sort_values("duration_s")

    dur_arr = env_df["duration_s"].to_numpy(dtype=float)
    pwr_arr = env_df["aged_power"].to_numpy(dtype=float)

    popt, ok = (_fit_power_curve(dur_arr, pwr_arr) if len(dur_arr) >= 4
                else (None, False))

    if ok:
        AWC, Pmax, MAP, tau2 = popt
        env_df["model_power"]  = _power_model(dur_arr, *popt)
        env_df["residual"]     = env_df["aged_power"] - env_df["model_power"]
        env_df["residual_pct"] = env_df["residual"] / env_df["model_power"] * 100.0
    else:
        env_df["model_power"]  = float("nan")
        env_df["residual"]     = float("nan")
        env_df["residual_pct"] = float("nan")

    def _score(row: pd.Series) -> float:
        below = max(0.0, -row["residual_pct"]) if pd.notna(row["residual_pct"]) else 0.0
        stale = (1.0 - row["weight"]) * 30.0
        return below * 0.7 + stale

    env_df["score"] = env_df.apply(_score, axis=1)
    env_df = env_df.sort_values("score", ascending=False)

    def _action(score: float) -> str:
        if score > 20: return "Test now"
        if score > 8:  return "Monitor"
        return "On track"

    env_df["action"] = env_df["score"].apply(_action)

    def _dur_label(s: float) -> str:
        s = int(s)
        if s < 60:   return f"{s}s"
        if s < 3600:
            m, sec = divmod(s, 60)
            return f"{m}:{sec:02d}" if sec else f"{m} min"
        return f"{s // 3600}h"

    dur_labels     = [_dur_label(d) for d in env_df["duration_s"]]
    raw_power_vals = [f"{int(r)} W" for r in env_df["raw_power"]]
    weight_vals    = [f"{w:.2f}" for w in env_df["weight"]]
    age_vals       = [f"{a}d  ({rd})" for a, rd in
                      zip(env_df["age_days"], env_df["ride_date"])]
    residual_vals  = [
        f"{r:+.0f} W  ({p:+.1f}%)" if pd.notna(r) else "—"
        for r, p in zip(env_df["residual"], env_df["residual_pct"])
    ]
    actions = env_df["action"].tolist()
    n       = len(env_df)

    # Row colours for the Action column
    action_fill  = {"Test now": "rgba(220,53,69,0.12)",
                    "Monitor":  "rgba(255,193,7,0.15)",
                    "On track": "rgba(40,167,69,0.10)"}
    action_font  = {"Test now": "#dc3545",
                    "Monitor":  "#856404",
                    "On track": "#155724"}

    row_action_fill = [action_fill[a] for a in actions]
    row_action_font = [action_font[a] for a in actions]

    # Weight column: grey (low) → blue (high)
    def _weight_color(w: float) -> str:
        r = int(160 - (160 - 41)  * w)
        g = int(160 - (160 - 128) * w)
        b = int(160 + (185 - 160) * w)
        return f"rgba({r},{g},{b},0.25)"

    weight_fill = [_weight_color(w) for w in env_df["weight"]]

    fig = go.Figure(go.Table(
        columnwidth=[60, 80, 60, 160, 130, 80],
        header=dict(
            values=["Duration", "Best effort", "Weight", "Age of best effort",
                    "vs Model", "Action"],
            fill_color="#eef2f7",
            align=["left", "right", "center", "left", "right", "center"],
            font=dict(size=12, color="#374151"),
            height=30,
        ),
        cells=dict(
            values=[dur_labels, raw_power_vals, weight_vals,
                    age_vals, residual_vals, actions],
            fill_color=[
                ["white"] * n,
                ["white"] * n,
                weight_fill,
                ["white"] * n,
                ["white"] * n,
                row_action_fill,
            ],
            font=dict(
                size=12,
                color=[
                    ["#111827"] * n,
                    ["#111827"] * n,
                    ["#374151"] * n,
                    ["#6b7280"] * n,
                    ["#374151"] * n,
                    row_action_font,
                ],
            ),
            align=["left", "right", "center", "left", "right", "center"],
            height=26,
        ),
    ))

    fig.update_layout(
        title=dict(
            text=(
                "Testing Priority  "
                "<sup>(ranked by urgency — highest first)</sup>"
            ),
            font=dict(size=14),
        ),
        height=max(180, 75 + 26 * n),
        margin=dict(t=55, b=10, l=10, r=10),
        template="plotly_white",
    )
    return fig


def fig_pmc(pdc_params: pd.DataFrame, rides: pd.DataFrame) -> go.Figure:
    """Three-panel Performance Management Chart.

    Row 1 — Base (≤ LTP) TSS:        CTL (42d), ATL (7d), TSB
    Row 2 — Threshold (LTP→MAP) TSS: CTL (42d), ATL (7d), TSB
    Row 3 — Anaerobic (> MAP) TSS:   CTL (42d), ATL (7d), TSB

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

    # Derive tss_ltp and tss_thresh; handle missing column gracefully
    if "tss_ltp" in df.columns:
        df["tss_ltp"]    = df["tss_ltp"].fillna(df["tss_map"])
        df["tss_thresh"] = (df["tss_map"] - df["tss_ltp"]).clip(lower=0)
    else:
        df["tss_ltp"]    = df["tss_map"]
        df["tss_thresh"] = 0.0

    daily = df.groupby("ride_date")[["tss", "tss_ltp", "tss_thresh", "tss_awc"]].sum()

    FUTURE_DAYS = 7
    pmc_ltp    = _compute_pmc(daily["tss_ltp"],    future_days=FUTURE_DAYS)
    pmc_thresh = _compute_pmc(daily["tss_thresh"], future_days=FUTURE_DAYS)
    pmc_awc    = _compute_pmc(daily["tss_awc"],    future_days=FUTURE_DAYS)

    # Align daily TSS to the continuous date grid used by pmc_ltp
    _idx             = pd.DatetimeIndex(pmc_ltp["date"])
    _tss_ltp_bars    = daily["tss_ltp"].reindex(_idx, fill_value=0.0).round(1).values
    _tss_thresh_bars = daily["tss_thresh"].reindex(_idx, fill_value=0.0).round(1).values
    _tss_awc_bars    = daily["tss_awc"].reindex(_idx, fill_value=0.0).round(1).values

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=[
            "Base (≤ LTP) — CTL / ATL / TSB",
            "Threshold (LTP → MAP) — CTL / ATL / TSB",
            "Anaerobic (> MAP) — CTL / ATL / TSB",
        ],
        vertical_spacing=0.07,
        specs=[[{"secondary_y": True}],
               [{"secondary_y": True}],
               [{"secondary_y": True}]],
    )

    panels = [
        # (row, pmc_df,     bar_data,         bar_color,                 ctl_col,        atl_col,      label)
        (1, pmc_ltp,    _tss_ltp_bars,    "rgba(46,139,87,0.45)",    "seagreen",     "darkorange", "Base"),
        (2, pmc_thresh, _tss_thresh_bars, "rgba(70,130,180,0.45)",   "steelblue",    "goldenrod",  "Threshold"),
        (3, pmc_awc,    _tss_awc_bars,    "rgba(220,80,30,0.45)",    "mediumpurple", "tomato",     "AWC"),
    ]

    for row, pmc, bar_vals, bar_col, ctl_col, atl_col, label in panels:
        show  = row == 1   # show in legend only for the top panel
        dates = pmc["date"].dt.strftime("%Y-%m-%d")
        tsb_pos = np.where(pmc["tsb"] >= 0, pmc["tsb"], 0.0)
        tsb_neg = np.where(pmc["tsb"] <  0, pmc["tsb"], 0.0)

        # TSS bars on LEFT axis (background context)
        fig.add_trace(go.Bar(
            x=dates, y=bar_vals,
            name=f"TSS {label}", marker_color=bar_col,
            showlegend=show,
            hovertemplate=f"TSS {label}: %{{y:.0f}}<extra></extra>",
        ), row=row, col=1, secondary_y=False)

        # TSB shading on RIGHT axis
        fig.add_trace(go.Scatter(
            x=dates, y=tsb_pos.round(1),
            mode="lines", fill="tozeroy",
            fillcolor="rgba(46,139,87,0.22)",
            line=dict(color="rgba(0,0,0,0)", width=0),
            name="TSB (fresh)", showlegend=show,
            hovertemplate=f"TSB ({label}): %{{y:.1f}}<extra></extra>",
        ), row=row, col=1, secondary_y=True)

        fig.add_trace(go.Scatter(
            x=dates, y=tsb_neg.round(1),
            mode="lines", fill="tozeroy",
            fillcolor="rgba(220,80,30,0.22)",
            line=dict(color="rgba(0,0,0,0)", width=0),
            name="TSB (tired)", showlegend=show,
            hovertemplate=f"TSB ({label}): %{{y:.1f}}<extra></extra>",
        ), row=row, col=1, secondary_y=True)

        # ATL on RIGHT axis — short-term fatigue (dashed)
        fig.add_trace(go.Scatter(
            x=dates, y=pmc["atl"].round(1),
            mode="lines", name=f"ATL 7d ({label})",
            line=dict(color=atl_col, width=1.8, dash="dash"),
            showlegend=show,
            hovertemplate=f"ATL ({label}): %{{y:.1f}}<extra></extra>",
        ), row=row, col=1, secondary_y=True)

        # CTL on RIGHT axis — long-term fitness (solid)
        fig.add_trace(go.Scatter(
            x=dates, y=pmc["ctl"].round(1),
            mode="lines", name=f"CTL 42d ({label})",
            line=dict(color=ctl_col, width=2.2),
            showlegend=show,
            hovertemplate=f"CTL ({label}): %{{y:.1f}}<extra></extra>",
        ), row=row, col=1, secondary_y=True)

        fig.add_hline(y=0, line=dict(color="grey", dash="dot", width=1),
                      row=row, col=1, secondary_y=True)

        # ── Align zero across both axes ──────────────────────────────────
        # Left axis is non-negative (TSS bars); right axis spans negative
        # TSB to positive CTL/ATL.  Extend the left range below 0 so that
        # zero sits at the same vertical fraction on both axes.
        r_min = float(min(pmc["tsb"].min(), 0))
        r_max = float(max(pmc["ctl"].max(), pmc["atl"].max(),
                          pmc["tsb"].max(), 1))
        l_max = float(max(np.max(bar_vals), 1))
        pad   = 0.05                              # 5 % visual padding
        r_min, r_max = r_min * (1 + pad), r_max * (1 + pad)
        l_max *= (1 + pad)
        r_span = r_max - r_min
        if r_span > 0 and r_min < 0:
            zero_frac = -r_min / r_span            # fraction of right axis below 0
            l_min = -l_max * zero_frac / (1 - zero_frac)
        else:
            l_min = 0.0

        fig.update_yaxes(title_text="Daily TSS", showgrid=False,
                         zeroline=False, range=[l_min, l_max],
                         row=row, col=1, secondary_y=False)
        fig.update_yaxes(title_text="CTL / ATL / TSB", showgrid=True,
                         gridcolor="lightgrey", zeroline=False,
                         range=[r_min, r_max],
                         row=row, col=1, secondary_y=True)

    # Default visible window: last 90 days + 7-day projection
    today = pd.Timestamp.today().normalize()
    x_start = (today - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
    x_end   = (today + pd.Timedelta(days=FUTURE_DAYS)).strftime("%Y-%m-%d")

    fig.update_xaxes(showgrid=True, gridcolor="lightgrey",
                     range=[x_start, x_end])
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_layout(
        title=dict(text="Performance Management Chart", font=dict(size=14)),
        barmode="stack",
        height=780,
        margin=dict(t=70, b=50, l=70, r=20),
        template="plotly_white",
        hovermode="x unified",
        showlegend=False,
    )
    return fig


def fig_pmc_combined(pdc_params: pd.DataFrame, rides: pd.DataFrame) -> go.Figure:
    """Single-panel PMC chart with stacked CTL components and zone TSB lines.

    TSS bars (left axis) are stacked by zone: Base (≤ LTP), Threshold
    (LTP→MAP), Anaerobic (> MAP).  CTL (right axis) is shown as stacked
    areas for the same three zones.  TSB_AWC (red) and aTSB_MAP (green)
    are plotted as lines on the right axis.
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

    # Derive tss_ltp and tss_thresh; handle missing column gracefully
    if "tss_ltp" in df.columns:
        df["tss_ltp"]    = df["tss_ltp"].fillna(df["tss_map"])
        df["tss_thresh"] = (df["tss_map"] - df["tss_ltp"]).clip(lower=0)
    else:
        df["tss_ltp"]    = df["tss_map"]
        df["tss_thresh"] = 0.0

    daily = df.groupby("ride_date")[["tss_ltp", "tss_thresh", "tss_awc", "tss_map"]].sum()

    FUTURE_DAYS = 7
    pmc_ltp    = _compute_pmc(daily["tss_ltp"],    future_days=FUTURE_DAYS)
    pmc_thresh = _compute_pmc(daily["tss_thresh"], future_days=FUTURE_DAYS)
    pmc_awc    = _compute_pmc(daily["tss_awc"],    future_days=FUTURE_DAYS)
    pmc_map    = _compute_pmc(daily["tss_map"],    future_days=FUTURE_DAYS)

    # Align daily TSS to the continuous date grid
    _idx             = pd.DatetimeIndex(pmc_ltp["date"])
    _tss_ltp_bars    = daily["tss_ltp"].reindex(_idx, fill_value=0.0).round(1).values
    _tss_thresh_bars = daily["tss_thresh"].reindex(_idx, fill_value=0.0).round(1).values
    _tss_awc_bars    = daily["tss_awc"].reindex(_idx, fill_value=0.0).round(1).values

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    dates = pmc_ltp["date"].dt.strftime("%Y-%m-%d")

    # ── Stacked TSS bars on LEFT axis ──────────────────────────────────
    fig.add_trace(go.Bar(
        x=dates, y=_tss_ltp_bars,
        name="TSS Base", marker_color="rgba(46,139,87,0.45)",
        hovertemplate="TSS Base: %{y:.0f}<extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Bar(
        x=dates, y=_tss_thresh_bars,
        name="TSS Threshold", marker_color="rgba(70,130,180,0.45)",
        hovertemplate="TSS Thresh: %{y:.0f}<extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Bar(
        x=dates, y=_tss_awc_bars,
        name="TSS Anaerobic", marker_color="rgba(220,80,30,0.45)",
        hovertemplate="TSS AWC: %{y:.0f}<extra></extra>",
    ), secondary_y=False)

    # ── Stacked CTL areas on RIGHT axis ────────────────────────────────
    fig.add_trace(go.Scatter(
        x=dates, y=pmc_ltp["ctl"].round(1),
        mode="lines", name="CTL Base",
        line=dict(color="seagreen", width=0.5),
        fillcolor="rgba(46,139,87,0.30)",
        stackgroup="ctl",
        hovertemplate="CTL Base: %{y:.1f}<extra></extra>",
    ), secondary_y=True)

    fig.add_trace(go.Scatter(
        x=dates, y=pmc_thresh["ctl"].round(1),
        mode="lines", name="CTL Threshold",
        line=dict(color="steelblue", width=0.5),
        fillcolor="rgba(70,130,180,0.30)",
        stackgroup="ctl",
        hovertemplate="CTL Thresh: %{y:.1f}<extra></extra>",
    ), secondary_y=True)

    fig.add_trace(go.Scatter(
        x=dates, y=pmc_awc["ctl"].round(1),
        mode="lines", name="CTL Anaerobic",
        line=dict(color="firebrick", width=0.5),
        fillcolor="rgba(220,80,30,0.30)",
        stackgroup="ctl",
        hovertemplate="CTL AWC: %{y:.1f}<extra></extra>",
    ), secondary_y=True)

    # ── TSB lines on RIGHT axis ────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=dates, y=pmc_awc["tsb"].round(1),
        mode="lines", name="TSB AWC",
        line=dict(color="red", width=2),
        hovertemplate="TSB AWC: %{y:.1f}<extra></extra>",
    ), secondary_y=True)

    fig.add_trace(go.Scatter(
        x=dates, y=pmc_map["tsb"].round(1),
        mode="lines", name="aTSB MAP",
        line=dict(color="green", width=2),
        hovertemplate="aTSB MAP: %{y:.1f}<extra></extra>",
    ), secondary_y=True)

    fig.add_hline(y=0, line=dict(color="grey", dash="dot", width=1),
                  secondary_y=True)

    # ── Align zero across both axes ────────────────────────────────────
    total_ctl = pmc_ltp["ctl"] + pmc_thresh["ctl"] + pmc_awc["ctl"]
    r_min = float(min(pmc_awc["tsb"].min(), pmc_map["tsb"].min(), 0))
    r_max = float(max(total_ctl.max(), pmc_map["tsb"].max(),
                      pmc_awc["tsb"].max(), 1))
    l_max = float(max(
        np.max(_tss_ltp_bars + _tss_thresh_bars + _tss_awc_bars), 1))
    pad   = 0.05
    r_min, r_max = r_min * (1 + pad), r_max * (1 + pad)
    l_max *= (1 + pad)
    r_span = r_max - r_min
    if r_span > 0 and r_min < 0:
        zero_frac = -r_min / r_span
        l_min = -l_max * zero_frac / (1 - zero_frac)
    else:
        l_min = 0.0

    fig.update_yaxes(title_text="Daily TSS", showgrid=False,
                     zeroline=False, range=[l_min, l_max],
                     secondary_y=False)
    fig.update_yaxes(title_text="CTL / TSB", showgrid=True,
                     gridcolor="lightgrey", zeroline=False,
                     range=[r_min, r_max],
                     secondary_y=True)

    today   = pd.Timestamp.today().normalize()
    x_start = (today - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
    x_end   = (today + pd.Timedelta(days=FUTURE_DAYS)).strftime("%Y-%m-%d")

    fig.update_xaxes(showgrid=True, gridcolor="lightgrey",
                     range=[x_start, x_end], title_text="Date")
    fig.update_layout(
        title=dict(text="Performance Management Chart — Combined", font=dict(size=14)),
        barmode="stack",
        height=380,
        margin=dict(t=50, b=50, l=70, r=20),
        template="plotly_white",
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig
