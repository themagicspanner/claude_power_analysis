"""graphs.py — Plotly figure builders for the cycling power analysis dashboard."""

import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from build_database import (
    _power_model, _power_model_extended, _fit_power_curve,
    _fit_with_endurance_tail, _normalized_power,
    PDC_K, PDC_INFLECTION, PDC_WINDOW,
)

LOG_TICK_S   = [1, 5, 10, 30, 60, 120, 300, 600, 1200, 1800, 3600,
                5400, 7200, 9000, 10800]
LOG_TICK_LBL = ["1s", "5s", "10s", "30s", "1min", "2min",
                 "5min", "10min", "20min", "30min", "1h",
                 "1h30", "2h", "2h30", "3h"]


# ── Zone colour palette (RAG: Red / Amber / Green) ───────────────────────────
_RGB_BASE   = (46, 139, 87)     # green  — Base (≤ LTP)
_RGB_THRESH = (235, 168, 36)    # amber  — Threshold (LTP → MAP)
_RGB_AWC    = (214, 55, 46)     # red    — Anaerobic (> MAP)


def _zc(rgb: tuple[int, int, int], a: float = 1.0) -> str:
    """Return an rgb()/rgba() CSS string for a zone colour at the given opacity."""
    if a >= 1.0:
        return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
    return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{a})"


# Convenience solid names (for line colours)
Z_BASE   = _zc(_RGB_BASE)
Z_THRESH = _zc(_RGB_THRESH)
Z_AWC    = _zc(_RGB_AWC)


# ── Figure builders ───────────────────────────────────────────────────────────

_SMOOTH = 6  # rolling-average window in seconds (1 Hz data)


def _fmt_hms(seconds: float) -> str:
    """Format seconds as H:MM:SS."""
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}"


def _time_axis_props(max_s: float) -> dict:
    """Return xaxis dict with hh:mm:ss tick labels for an elapsed-seconds axis."""
    if max_s <= 600:
        step = 60
    elif max_s <= 3600:
        step = 300
    elif max_s <= 10800:
        step = 600
    else:
        step = 1800
    vals = list(range(0, int(max_s) + 1, step))
    return dict(
        title_text="Elapsed Time",
        showgrid=True, gridcolor="lightgrey",
        tickvals=vals,
        ticktext=[_fmt_hms(v) for v in vals],
    )


def _hover_hms(seconds: np.ndarray) -> list[str]:
    """Return a list of H:MM:SS strings for use as customdata hover labels."""
    return [_fmt_hms(s) for s in seconds]


def fig_power_hr(records: pd.DataFrame, ride_name: str,
                 ltp: float | None = None,
                 map_power: float | None = None) -> go.Figure:
    """Power chart with zone-coloured line.

    When *ltp* and *map_power* are provided the line is coloured by zone
    (green ≤ LTP, amber LTP→MAP, red > MAP).
    """
    has_power = records["power"].notna().any()
    fig = go.Figure()

    zone_fill = (has_power and ltp is not None and map_power is not None
                 and ltp > 0 and map_power > 0)

    x_s = records["elapsed_s"].to_numpy(dtype=float)
    hms = _hover_hms(x_s)

    if has_power and zone_fill:
        power = records["power"].fillna(0).to_numpy(dtype=float)
        if len(power) >= _SMOOTH:
            kernel = np.ones(_SMOOTH) / _SMOOTH
            power = np.convolve(power, kernel, mode="same")

        # Assign each sample to a zone: 0=base, 1=threshold, 2=awc
        zones = np.where(power > map_power, 2, np.where(power > ltp, 1, 0))

        zone_cfg = [
            (0, "Base ≤LTP",    _RGB_BASE,   0.25, Z_BASE),
            (1, "Threshold",    _RGB_THRESH, 0.25, Z_THRESH),
            (2, "AWC",          _RGB_AWC,    0.22, Z_AWC),
        ]
        for z_id, z_name, z_rgb, z_alpha, z_line in zone_cfg:
            mask = zones == z_id
            extended = mask.copy()
            extended[:-1] |= mask[1:]
            extended[1:]  |= mask[:-1]
            y = np.where(extended, power, np.nan)
            if np.all(np.isnan(y)):
                continue
            fig.add_trace(go.Scatter(
                x=x_s, y=y,
                mode="lines", name=z_name,
                line=dict(color=z_line, width=1),
                showlegend=False,
                legendgroup="power",
                hoverinfo="skip",
            ))
        # Invisible trace for a single unified hover readout
        fig.add_trace(go.Scatter(
            x=x_s, y=power,
            mode="lines", name="Power",
            line=dict(color="rgba(0,0,0,0)", width=0),
            showlegend=False,
            customdata=hms,
            hovertemplate="%{customdata}  %{y:.0f} W<extra></extra>",
        ))
    elif has_power:
        power_s = records["power"].fillna(0).rolling(_SMOOTH, min_periods=1, center=True).mean()
        fig.add_trace(go.Scatter(
            x=x_s, y=power_s,
            mode="lines", name="Power",
            line=dict(color="darkorange", width=1),
            customdata=hms,
            hovertemplate="%{customdata}  %{y:.0f} W<extra></extra>",
        ))

    max_s = float(x_s[-1]) if len(x_s) else 3600
    power_color = "#444" if zone_fill else "darkorange"
    fig.update_layout(
        title=dict(text="Power", font=dict(size=15)),
        height=250, margin=dict(t=55, b=40, l=60, r=20),
        template="plotly_white",
        showlegend=False,
        hovermode="closest",
        xaxis=_time_axis_props(max_s),
        yaxis=dict(title=dict(text="Power (W)", font=dict(color=power_color)),
                   showgrid=True, gridcolor="lightgrey",
                   tickfont=dict(color=power_color),
                   fixedrange=True),
    )
    return fig


def fig_hr(records: pd.DataFrame) -> go.Figure:
    """Heart rate chart."""
    has_hr = "heart_rate" in records.columns and records["heart_rate"].notna().any()
    fig = go.Figure()

    x_s = records["elapsed_s"].to_numpy(dtype=float)
    hms = _hover_hms(x_s)
    if has_hr:
        hr_s = records["heart_rate"].rolling(_SMOOTH, min_periods=1, center=True).mean()
        fig.add_trace(go.Scatter(
            x=x_s, y=hr_s,
            mode="lines", name="Heart Rate",
            line=dict(color="crimson", width=1),
            customdata=hms,
            hovertemplate="%{customdata}  %{y:.0f} bpm<extra></extra>",
        ))

    max_s = float(x_s[-1]) if len(x_s) else 3600
    fig.update_layout(
        title=dict(text="Heart Rate", font=dict(size=14)),
        height=200, margin=dict(t=55, b=40, l=60, r=20),
        template="plotly_white",
        showlegend=False,
        hovermode="closest",
        xaxis=_time_axis_props(max_s),
        yaxis=dict(title=dict(text="Heart Rate (bpm)", font=dict(color="crimson")),
                   showgrid=True, gridcolor="lightgrey",
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

    x_s = alt["elapsed_s"].to_numpy(dtype=float)
    max_s = float(x_s[-1]) if len(x_s) else 3600
    fig = go.Figure(go.Scatter(
        x=x_s, y=alt["altitude_m"],
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
        xaxis=_time_axis_props(max_s),
    )
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
        tte = tte_b = None
        if live_pdc is not None:
            AWC, Pmax, MAP, tau2 = (
                live_pdc["AWC"], live_pdc["Pmax"],
                live_pdc["MAP"], live_pdc["tau2"],
            )
            tte   = live_pdc.get("tte")
            tte_b = live_pdc.get("tte_b")
            ok = True
        elif len(dur) >= 4:
            popt, ok, tte, tte_b = _fit_with_endurance_tail(dur, pwr)
            if ok:
                AWC, Pmax, MAP, tau2 = popt

        # Zone fills: base / threshold / anaerobic (drawn first = behind data)
        if ok:
            tau   = AWC / Pmax
            t_sm  = np.logspace(np.log10(dur.min()), np.log10(dur.max()), 400)
            p_aer_base = MAP * (1.0 - np.exp(-t_sm / tau2))
            # Beyond TtE aerobic is fully ramped, but total power declines
            if tte is not None:
                p_aer = np.where(t_sm <= tte, p_aer_base,
                                 np.minimum(p_aer_base, _power_model_extended(
                                     t_sm, AWC, Pmax, MAP, tau2, tte, tte_b)))
            else:
                p_aer = p_aer_base
            p_tot = _power_model_extended(t_sm, AWC, Pmax, MAP, tau2, tte, tte_b)
            ltp = float(MAP * (1.0 - (5.0 / 2.0) * ((AWC / 1000.0) / MAP)))
            ltp = max(ltp, 0.0)
            ltp_frac = ltp / MAP if MAP > 0 else 0.0
            p_base = p_aer * ltp_frac
            fig.add_trace(go.Scatter(
                x=t_sm, y=p_base,
                mode="lines", name="base (≤LTP)",
                fill="tozeroy", fillcolor=_zc(_RGB_BASE, 0.20),
                line=dict(color=_zc(_RGB_BASE, 0.7), width=1.2),
            ))
            fig.add_trace(go.Scatter(
                x=t_sm, y=p_aer,
                mode="lines", name="threshold (LTP→MAP)",
                fill="tonexty", fillcolor=_zc(_RGB_THRESH, 0.20),
                line=dict(color=_zc(_RGB_THRESH, 0.7), width=1.2),
            ))
            fig.add_trace(go.Scatter(
                x=t_sm, y=p_tot,
                mode="lines",
                name="model",
                fill="tonexty", fillcolor=_zc(_RGB_AWC, 0.18),
                line=dict(color=Z_AWC, width=2, dash="dash"),
            ))

        # Best aged MMP from other rides in the PDC window
        if not other.empty:
            fig.add_trace(go.Scatter(
                x=other["duration_s"], y=other["aged_power"],
                mode="lines", name="other rides (best)",
                line=dict(color="steelblue", width=1.5, dash="dot"),
            ))

    # This ride's MMP on top — black line, with a green overlay where
    # this ride beats the prior decayed MMP from other rides.
    if not this_ride.empty:
        tr_dur = this_ride["duration_s"].to_numpy()
        tr_pwr = this_ride["power"].to_numpy()

        # Build a lookup of the prior best aged MMP (excluding this ride)
        if not window.empty and not other.empty:
            prior_lookup = dict(zip(other["duration_s"], other["aged_power"]))
        else:
            prior_lookup = {}

        # Black line for the full ride MMP
        fig.add_trace(go.Scatter(
            x=tr_dur, y=tr_pwr,
            mode="lines", name=f"this ride ({ride_date})",
            line=dict(color="black", width=2.2),
        ))

        # Green line segments where this ride improved on prior best
        imp_dur = []
        imp_pwr = []
        for d, p in zip(tr_dur, tr_pwr):
            prior = prior_lookup.get(d)
            if prior is not None and p > prior:
                imp_dur.append(d)
                imp_pwr.append(p)
            else:
                # Insert None to break the line at non-improved points
                imp_dur.append(d)
                imp_pwr.append(None)
        if any(v is not None for v in imp_pwr):
            fig.add_trace(go.Scatter(
                x=imp_dur, y=imp_pwr,
                mode="lines", name="improved",
                line=dict(color="#00e676", width=3),
                connectgaps=False,
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


def fig_90day_mmp(mmp_all: pd.DataFrame,
                  reference_date: datetime.date | None = None) -> go.Figure:
    today     = reference_date or datetime.date.today()
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
            mode="lines", name="aged MMP",
            line=dict(color="black", width=2.5),
        ))

        # ── Model fit with aerobic / anaerobic contributions ──────────────
        dur = aged["duration_s"].to_numpy(dtype=float)
        pwr = aged["aged_power"].to_numpy(dtype=float)
        popt, ok, tte, tte_b = _fit_with_endurance_tail(dur, pwr)
        if ok:
            AWC, Pmax, MAP, tau2 = popt
            tau      = AWC / Pmax
            t_smooth = np.logspace(np.log10(dur.min()), np.log10(dur.max()), 400)
            p_aer_base = MAP * (1.0 - np.exp(-t_smooth / tau2))
            p_total   = _power_model_extended(t_smooth, AWC, Pmax, MAP, tau2, tte, tte_b)
            # Beyond TtE, aerobic is fully ramped; cap at total power
            if tte is not None:
                p_aerobic = np.minimum(p_aer_base, p_total)
            else:
                p_aerobic = p_aer_base
            ltp = float(MAP * (1.0 - (5.0 / 2.0) * ((AWC / 1000.0) / MAP)))
            ltp = max(ltp, 0.0)
            # Base component (0 → LTP/MAP proportion of aerobic curve)
            ltp_frac = ltp / MAP if MAP > 0 else 0.0
            p_base = p_aerobic * ltp_frac
            fig.add_trace(go.Scatter(
                x=t_smooth, y=p_base,
                mode="lines", name="base (≤LTP)",
                fill="tozeroy", fillcolor=_zc(_RGB_BASE, 0.20),
                line=dict(color=_zc(_RGB_BASE, 0.7), width=1.2),
            ))
            # Threshold component (LTP → aerobic curve)
            fig.add_trace(go.Scatter(
                x=t_smooth, y=p_aerobic,
                mode="lines", name="threshold (LTP→MAP)",
                fill="tonexty", fillcolor=_zc(_RGB_THRESH, 0.20),
                line=dict(color=_zc(_RGB_THRESH, 0.7), width=1.2),
            ))
            # Total model — fills from aerobic line, so the shaded band = anaerobic
            fig.add_trace(go.Scatter(
                x=t_smooth, y=p_total,
                mode="lines",
                name="model",
                fill="tonexty", fillcolor=_zc(_RGB_AWC, 0.18),
                line=dict(color=Z_AWC, width=2, dash="dash"),
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


def fig_pdc_params_history(daily_pdc: pd.DataFrame,
                           rides: pd.DataFrame,
                           reference_date: datetime.date | None = None) -> go.Figure:
    """History of the fitted two-component PDC parameters for every calendar day.

    *daily_pdc* is pre-computed and stored in the database by
    ``recompute_daily_pdc_params()`` in build_database.py, so this function
    is a cheap read — no curve fitting happens at render time.

    Left y-axis  : MAP, Pmax and LTP (W)
    Right y-axis : AWC (kJ)
    """
    df = daily_pdc
    if df.empty:
        return go.Figure()

    ride_dates = set(rides["ride_date"].dropna().unique())

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["MAP"],
        mode="lines", name="MAP (W)",
        line=dict(color=Z_THRESH, width=2, shape="hv"),
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["Pmax"],
        mode="lines", name="Pmax (W)",
        line=dict(color="mediumpurple", width=1.5, dash="dash", shape="hv"),
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["ltp"],
        mode="lines", name="LTP (W)",
        line=dict(color=Z_BASE, width=2, shape="hv"),
    ), secondary_y=False)

    # P(TtE) — normalizing power computed from the model at TtE (or 3600s fallback)
    if "tte" in df.columns and "tte_b" in df.columns:
        p_tte_vals = []
        tte_min_vals = []
        for _, row in df.iterrows():
            _AWC, _Pmax, _MAP, _tau2 = row["AWC"], row["Pmax"], row["MAP"], row["tau2"]
            _tte = float(row["tte"]) if pd.notna(row.get("tte")) else None
            _tte_b = float(row["tte_b"]) if pd.notna(row.get("tte_b")) else None
            _t = _tte if _tte is not None else 3600.0
            p_tte_vals.append(float(_power_model_extended(_t, _AWC, _Pmax, _MAP, _tau2, _tte, _tte_b)))
            tte_min_vals.append(_t / 60.0)
        fig.add_trace(go.Scatter(
            x=df["date"], y=p_tte_vals,
            mode="lines", name="P(TtE) (W)",
            line=dict(color="#e07020", width=2, dash="dashdot", shape="hv"),
            customdata=np.array(tte_min_vals),
            hovertemplate="P(TtE): %{y:.0f} W  (TtE=%{customdata:.0f} min)<extra></extra>",
        ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["AWC"] / 1000,
        mode="lines", name="AWC (kJ)",
        line=dict(color=Z_AWC, width=2, dash="dot", shape="hv"),
    ), secondary_y=True)

    # Ride-date markers on MAP to show when rides were recorded
    ride_rows = df[df["date"].isin(ride_dates)]
    if not ride_rows.empty:
        fig.add_trace(go.Scatter(
            x=ride_rows["date"], y=ride_rows["MAP"],
            mode="markers", name="Ride",
            marker=dict(color=Z_THRESH, size=7, symbol="circle-open",
                        line=dict(color=Z_THRESH, width=2)),
            hovertemplate="%{x}  MAP: %{y:.0f} W<extra>Ride</extra>",
        ), secondary_y=False)

    fig.update_xaxes(title_text="Date", showgrid=True, gridcolor="lightgrey")
    fig.update_yaxes(title_text="Power (W)", showgrid=True, gridcolor="lightgrey",
                     rangemode="tozero", secondary_y=False)
    fig.update_yaxes(title_text="AWC (kJ)", showgrid=False,
                     rangemode="tozero", secondary_y=True)
    if reference_date is not None:
        ref_iso = reference_date.isoformat()
        fig.add_shape(
            type="line", x0=ref_iso, x1=ref_iso, y0=0, y1=1,
            yref="paper", line=dict(color="#ff6b6b", width=2, dash="dash"),
        )
        fig.add_annotation(
            x=ref_iso, y=1, yref="paper",
            text=ref_iso, showarrow=False,
            font=dict(size=10, color="#ff6b6b"),
            yshift=10,
        )

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

def fig_zone_bars(zone_data: dict[int, float],
                  tss_ltp: float | None,
                  tss_map: float | None,
                  tss_awc: float | None,
                  ltp: float, map_: float) -> go.Figure:
    """Two horizontal stacked bars: time-in-zone (top) and TSS-by-zone (bottom)."""
    colors = [Z_BASE, Z_THRESH, Z_AWC]
    zone_names = [
        f"Z1  ≤{ltp:.0f} W",
        f"Z2  {ltp:.0f}–{map_:.0f} W",
        f"Z3  >{map_:.0f} W",
    ]

    def _fmt(seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        return f"{m}:{s:02d}"

    t_vals = [zone_data.get(z, 0.0) for z in (1, 2, 3)] if zone_data else [0, 0, 0]
    t_total = sum(t_vals)

    tss_vals = [
        float(v) if v is not None and pd.notna(v) else 0.0
        for v in (tss_ltp, tss_map, tss_awc)
    ]
    tss_total = sum(tss_vals)

    has_time = t_total > 0
    has_tss = tss_total > 0

    if not has_time and not has_tss:
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

    fig = go.Figure()
    rows = ["TSS by Zone", "Time in Zone"]  # bottom-up for plotly y ordering

    for i, (name, color) in enumerate(zip(zone_names, colors)):
        # Time bar
        t_pct = t_vals[i] / t_total * 100 if has_time else 0.0
        fig.add_trace(go.Bar(
            y=["Time in Zone"], x=[t_pct],
            name=name, legendgroup=name,
            showlegend=True,
            orientation="h", marker_color=color,
            hovertemplate=(f"{name}<br>{t_pct:.1f}%  ·  {_fmt(t_vals[i])}<extra></extra>"
                           if has_time else f"{name}<br>No data<extra></extra>"),
            text=f"{t_pct:.0f}%  ({_fmt(t_vals[i])})" if has_time and t_pct >= 8 else "",
            textposition="inside", textfont=dict(size=11, color="white"),
        ))

    for i, (name, color) in enumerate(zip(zone_names, colors)):
        # TSS bar
        tss_pct = tss_vals[i] / tss_total * 100 if has_tss else 0.0
        fig.add_trace(go.Bar(
            y=["TSS by Zone"], x=[tss_pct],
            name=name, legendgroup=name,
            showlegend=False,
            orientation="h", marker_color=color,
            hovertemplate=(f"{name}<br>{tss_pct:.1f}%  ·  TSS {tss_vals[i]:.1f}<extra></extra>"
                           if has_tss else f"{name}<br>No data<extra></extra>"),
            text=f"{tss_pct:.0f}%  ({tss_vals[i]:.1f})" if has_tss and tss_pct >= 8 else "",
            textposition="inside", textfont=dict(size=11, color="white"),
        ))

    fig.update_xaxes(range=[0, 100], showgrid=False, showticklabels=False)
    fig.update_yaxes(tickfont=dict(size=12))
    fig.update_layout(
        title=dict(text="Zone Distribution  (LTP / MAP 3-zone model)", font=dict(size=14)),
        barmode="stack",
        height=180,
        margin=dict(t=55, b=10, l=110, r=10),
        template="plotly_white",
        showlegend=True,
        legend=dict(
            orientation="h", x=0, y=-0.15,
            font=dict(size=11), traceorder="normal",
        ),
        hovermode="closest",
    )
    return fig


# ── TSS component series ──────────────────────────────────────────────────────

def _tss_rate_series(elapsed_s: np.ndarray, power: np.ndarray,
                     ftp: float, CP: float,
                     ltp: float | None = None,
                     AWC: float | None = None,
                     Pmax: float | None = None,
                     tau2: float | None = None) -> tuple:
    """Return TSS rate and cumulative LTP / Threshold / AWC series over the ride.

    Uses the recalculated-from-start method:
      TSS(t) = (t / 3600) * (NP(0..t) / P_norm)^2 * 100
    where NP(0..t) is the Normalized Power computed over all samples from
    the start up to time t.

    Zone fractions are derived from instantaneous power using the PDC model's
    time-dependent aerobic ramp-up when AWC/Pmax/tau2 are provided.  For power
    above MAP, the PDC is inverted to find the corresponding duration, and the
    aerobic contribution at that duration (which is small for sprint-level
    powers) determines the base/threshold split.  For power at or below MAP
    the static waterfall (LTP / MAP thresholds) is used.

    Returns (t_min, cum_ltp, cum_thresh, cum_awc,
             rate_ltp_ph, rate_thresh_ph, rate_awc_ph, rate_1h_avg)
    where rate_*_ph are the instantaneous TSS rates in TSS/hour and
    rate_1h_avg is the 1-hour time-weighted rolling average of total TSS rate.
    """
    p = np.where(np.isnan(power), 0.0, power.astype(float))
    n = len(p)

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

    # --- Cumulative TSS via recalculated-from-start NP method ---
    # At each time t: NP(0..t) = (mean(p_30s[0..t]^4))^0.25
    #                 TSS(t) = (t_s / 3600) * (NP(0..t) / FTP)^2 * 100
    cum_total = np.zeros(n)
    if ftp > 0:
        p_30s_4th  = p_30s ** 4
        cumsum_4th = np.cumsum(p_30s_4th)
        idx_arr    = np.arange(1, n, dtype=float)
        t_s_arr    = elapsed_s[1:] - elapsed_s[0]
        np_at_t    = (cumsum_4th[1:] / (idx_arr + 1)) ** 0.25
        cum_total[1:] = (t_s_arr / 3600.0) * (np_at_t / ftp) ** 2 * 100.0

    # --- Zone fractions from instantaneous power ---
    # When full PDC model parameters are available, use the model's
    # time-dependent aerobic ramp-up so that at sprint-level powers the
    # base/threshold contributions are appropriately small (matching the
    # sigmoidal shape visible on the PDC chart).
    use_pdc_model = (AWC is not None and Pmax is not None and tau2 is not None
                     and AWC > 0 and Pmax > 0 and tau2 > 0
                     and CP > 0 and ltp is not None and ltp > 0)

    if use_pdc_model:
        # Build a lookup table: for a fine grid of durations, compute
        # total power P(t) and aerobic power p_aer(t) from the PDC model.
        # P(t) is monotonically decreasing from ~Pmax to MAP, so we can
        # interpolate from power → aerobic contribution.
        tau = AWC / Pmax
        t_grid = np.logspace(-1, np.log10(7200), 2000)
        p_total_grid = _power_model(t_grid, AWC, Pmax, CP, tau2)
        p_aer_grid   = CP * (1.0 - np.exp(-t_grid / tau2))

        # np.interp needs increasing x, but P(t) is decreasing → reverse
        p_total_rev = p_total_grid[::-1]
        p_aer_rev   = p_aer_grid[::-1]

        ltp_frac = ltp / CP if CP > 0 else 0.0

        with np.errstate(invalid="ignore", divide="ignore"):
            # For p > MAP: look up the aerobic contribution from the PDC
            # For p <= MAP: aerobic system is fully ramped → static waterfall
            above_map = p > CP
            p_aer_at_p = np.where(
                above_map,
                np.interp(p, p_total_rev, p_aer_rev, left=CP, right=0.0),
                p,  # placeholder, overwritten below
            )
            # Base and threshold from the aerobic contribution
            p_base_at_p = np.where(above_map, p_aer_at_p * ltp_frac, 0.0)
            p_thresh_at_p = np.where(above_map, p_aer_at_p - p_base_at_p, 0.0)

            # Below MAP: static waterfall (aerobic fully engaged)
            p_base_at_p   = np.where(~above_map, np.minimum(p, ltp), p_base_at_p)
            p_thresh_at_p = np.where(
                ~above_map,
                np.maximum(np.minimum(p, CP) - ltp, 0.0),
                p_thresh_at_p,
            )

            f_ltp    = np.where(p > 0, p_base_at_p / p, 1.0)
            f_thresh = np.where(p > 0, p_thresh_at_p / p, 0.0)
            f_awc    = np.where(p > 0, np.maximum(p - p_base_at_p - p_thresh_at_p, 0.0) / p, 0.0)
    elif ltp is not None and ltp > 0 and CP > 0:
        # Fallback: static waterfall when PDC model params unavailable
        with np.errstate(invalid="ignore", divide="ignore"):
            f_awc    = np.where(p > 0, np.maximum(p - CP, 0.0) / p, 0.0)
            f_ltp    = np.where(p > 0, np.minimum(p, ltp) / p, 1.0)
            f_thresh = np.where(p > 0, np.maximum(np.minimum(p, CP) - ltp, 0.0) / p, 0.0)
    else:
        with np.errstate(invalid="ignore", divide="ignore"):
            f_awc = np.where(p > 0, np.maximum(p - CP, 0.0) / p, 0.0)
        f_ltp = 1.0 - f_awc
        f_thresh = np.zeros(n)

    # Weighted cumulative zone fractions (proportion of TSS rate at each point)
    tss_rate_local = (p_30s / ftp) ** 2 * 100.0 if ftp > 0 else np.zeros(n)
    rate_ltp_local    = tss_rate_local * f_ltp
    rate_thresh_local = tss_rate_local * f_thresh
    rate_awc_local    = tss_rate_local * f_awc
    rate_total_local  = rate_ltp_local + rate_thresh_local + rate_awc_local

    # Cumulative zone proportions (running weighted average of fraction)
    cum_rate_total = np.cumsum(rate_total_local * dt)
    cum_rate_ltp   = np.cumsum(rate_ltp_local * dt)
    cum_rate_thresh = np.cumsum(rate_thresh_local * dt)
    cum_rate_awc   = np.cumsum(rate_awc_local * dt)

    with np.errstate(invalid="ignore", divide="ignore"):
        frac_ltp    = np.where(cum_rate_total > 0, cum_rate_ltp / cum_rate_total, 0.0)
        frac_thresh = np.where(cum_rate_total > 0, cum_rate_thresh / cum_rate_total, 0.0)
        frac_awc    = np.where(cum_rate_total > 0, cum_rate_awc / cum_rate_total, 0.0)

    cum_ltp    = cum_total * frac_ltp
    cum_thresh = cum_total * frac_thresh
    cum_awc    = cum_total * frac_awc

    # --- Instantaneous rates (TSS/h) derived from cumulative derivative ---
    d_total = np.zeros(n)
    d_total[1:] = np.diff(cum_total)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate_total_ph = np.where(dt > 0, d_total / dt * 3600.0, 0.0)

    rate_ltp_ph    = rate_total_ph * f_ltp
    rate_thresh_ph = rate_total_ph * f_thresh
    rate_awc_ph    = rate_total_ph * f_awc
    t_s            = elapsed_s

    # 1-hour time-weighted rolling average of total TSS rate.
    cum_dt_ext      = np.empty(n + 1)
    cum_dt_ext[0]   = 0.0
    cum_dt_ext[1:]  = np.cumsum(dt)
    cum_rdt_ext     = np.empty(n + 1)
    cum_rdt_ext[0]  = 0.0
    cum_rdt_ext[1:] = np.cumsum(rate_total_ph * dt)
    left            = np.searchsorted(elapsed_s, elapsed_s - 3600.0, side="left")
    idx             = np.arange(n)
    window_dt       = cum_dt_ext[idx + 1] - cum_dt_ext[left]
    window_rdt      = cum_rdt_ext[idx + 1] - cum_rdt_ext[left]
    with np.errstate(invalid="ignore", divide="ignore"):
        rate_1h_avg = np.where(window_dt > 0, window_rdt / window_dt, 0.0)

    return (t_s, cum_ltp, cum_thresh, cum_awc,
            rate_ltp_ph, rate_thresh_ph, rate_awc_ph, rate_1h_avg)


def fig_tss_rate(records: pd.DataFrame, ride: pd.Series,
                 pdc_params: pd.DataFrame,
                 live_pdc: dict | None = None) -> go.Figure:
    """Total TSS rate (TSS/h) over the ride with 1-hour rolling average."""
    if records["power"].isna().all():
        return go.Figure()

    params_row = pdc_params[pdc_params["ride_id"] == ride["id"]]

    if live_pdc is not None:
        CP  = live_pdc["MAP"]
        ftp = live_pdc["ftp"]
        ltp = live_pdc.get("ltp")
        _awc = live_pdc.get("AWC"); _pmax = live_pdc.get("Pmax"); _tau2 = live_pdc.get("tau2")
    elif not params_row.empty:
        r   = params_row.iloc[0]
        CP  = float(r["MAP"])
        ftp = float(r["ftp"]) if pd.notna(r.get("ftp")) else CP
        ltp = float(r["ltp"]) if pd.notna(r.get("ltp")) else None
        _awc  = float(r["AWC"])  if pd.notna(r.get("AWC"))  else None
        _pmax = float(r["Pmax"]) if pd.notna(r.get("Pmax")) else None
        _tau2 = float(r["tau2"]) if pd.notna(r.get("tau2")) else None
    else:
        return go.Figure()

    elapsed = records["elapsed_s"].to_numpy(dtype=float)
    power   = records["power"].to_numpy(dtype=float)
    (t_min, cum_ltp, cum_thresh, cum_awc,
     rate_ltp, rate_thresh, rate_awc, rate_1h_avg) = _tss_rate_series(
        elapsed, power, ftp, CP, ltp=ltp,
        AWC=_awc, Pmax=_pmax, tau2=_tau2)

    rate_total = rate_ltp + rate_thresh + rate_awc

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_min, y=rate_total,
        mode="lines", name="TSS Rate",
        fill="tozeroy", fillcolor="rgba(100, 149, 237, 0.15)",
        line=dict(color="cornflowerblue", width=1),
    ))
    fig.add_trace(go.Scatter(
        x=t_min, y=rate_1h_avg,
        mode="lines", name="1h Rolling Avg",
        line=dict(color="midnightblue", width=1),
    ))

    wk_max_s = float(t_min[-1]) if len(t_min) else 3600
    fig.update_yaxes(title_text="TSS Rate (TSS/h)",
                     showgrid=True, gridcolor="lightgrey",
                     fixedrange=True)
    fig.update_layout(
        title=dict(text="TSS Rate", font=dict(size=14)),
        height=250,
        margin=dict(t=55, b=40, l=60, r=20),
        template="plotly_white",
        showlegend=False,
        hovermode="x unified",
        xaxis=_time_axis_props(wk_max_s),
    )
    return fig


def fig_tss_components(records: pd.DataFrame, ride: pd.Series,
                       pdc_params: pd.DataFrame,
                       live_pdc: dict | None = None) -> go.Figure:
    """TSS Rate (TSS/h) split into Base / Threshold / AWC stacked areas."""
    if records["power"].isna().all():
        return go.Figure()

    params_row = pdc_params[pdc_params["ride_id"] == ride["id"]]

    if live_pdc is not None:
        CP  = live_pdc["MAP"]
        ftp = live_pdc["ftp"]
        ltp = live_pdc.get("ltp")
        _awc = live_pdc.get("AWC"); _pmax = live_pdc.get("Pmax"); _tau2 = live_pdc.get("tau2")
    elif not params_row.empty:
        r   = params_row.iloc[0]
        CP  = float(r["MAP"])
        ftp = float(r["ftp"]) if pd.notna(r.get("ftp")) else CP
        ltp = float(r["ltp"]) if pd.notna(r.get("ltp")) else None
        _awc  = float(r["AWC"])  if pd.notna(r.get("AWC"))  else None
        _pmax = float(r["Pmax"]) if pd.notna(r.get("Pmax")) else None
        _tau2 = float(r["tau2"]) if pd.notna(r.get("tau2")) else None
    else:
        return go.Figure()

    elapsed = records["elapsed_s"].to_numpy(dtype=float)
    power   = records["power"].to_numpy(dtype=float)
    (t_min, cum_ltp, cum_thresh, cum_awc,
     rate_ltp, rate_thresh, rate_awc, rate_1h_avg) = _tss_rate_series(
        elapsed, power, ftp, CP, ltp=ltp,
        AWC=_awc, Pmax=_pmax, tau2=_tau2)

    final_ltp    = cum_ltp[-1]
    final_thresh = cum_thresh[-1]
    final_awc    = cum_awc[-1]
    rate_ltp_top    = rate_ltp
    rate_thresh_top = rate_ltp + rate_thresh
    rate_total      = rate_thresh_top + rate_awc

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_min, y=rate_ltp_top,
        mode="lines", name=f"Base ≤LTP ({final_ltp:.0f})",
        fill="tozeroy", fillcolor=_zc(_RGB_BASE, 0.25),
        line=dict(color=Z_BASE, width=1),
    ))
    fig.add_trace(go.Scatter(
        x=t_min, y=rate_thresh_top,
        mode="lines", name=f"Threshold ({final_thresh:.0f})",
        fill="tonexty", fillcolor=_zc(_RGB_THRESH, 0.25),
        line=dict(color=Z_THRESH, width=1),
    ))
    fig.add_trace(go.Scatter(
        x=t_min, y=rate_total,
        mode="lines", name=f"AWC ({final_awc:.0f})",
        fill="tonexty", fillcolor=_zc(_RGB_AWC, 0.22),
        line=dict(color=Z_AWC, width=1),
    ))
    fig.add_trace(go.Scatter(
        x=t_min, y=rate_1h_avg,
        mode="lines", name="Difficulty",
        line=dict(color="midnightblue", width=1),
    ))

    # Annotate the point of max difficulty (peak 1h rolling avg)
    if len(rate_1h_avg) > 0:
        peak_idx = int(np.argmax(rate_1h_avg))
        peak_x   = t_min[peak_idx]
        peak_y   = rate_1h_avg[peak_idx]
        if peak_y > 0:
            fig.add_annotation(
                x=peak_x, y=peak_y,
                text=f"Max Difficulty: {peak_y:.0f} TSS/h",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowcolor="midnightblue",
                ax=0, ay=-30,
                font=dict(size=10, color="midnightblue"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="midnightblue",
                borderwidth=1,
                borderpad=3,
            )

    max_s = float(t_min[-1]) if len(t_min) else 3600
    fig.update_yaxes(title_text="TSS Rate (TSS/h)",
                     showgrid=True, gridcolor="lightgrey",
                     fixedrange=True)
    fig.update_layout(
        title=dict(text="TSS Rate by Zone", font=dict(size=14)),
        height=250,
        margin=dict(t=55, b=40, l=60, r=20),
        template="plotly_white",
        showlegend=False,
        hovermode="x unified",
        xaxis=_time_axis_props(max_s),
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
        marker_color=Z_BASE,
        customdata=cd,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Base ≤LTP = %{y:.0f}<br>"
            "Threshold = %{customdata[4]:.0f}<br>"
            "AWC = %{customdata[5]:.0f}<br>"
            "NP = %{customdata[1]:.0f} W  "
            "IF = %{customdata[2]:.2f}<extra></extra>"
        ),
    ))
    fig.add_trace(go.Bar(
        x=df["ride_date"], y=df["tss_thresh"],
        name="Threshold (LTP→MAP)",
        marker_color=Z_THRESH,
        customdata=cd,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Threshold = %{y:.0f}<br>"
            "Base ≤LTP = %{customdata[3]:.0f}<br>"
            "AWC = %{customdata[5]:.0f}<br>"
            "NP = %{customdata[1]:.0f} W  "
            "IF = %{customdata[2]:.2f}<extra></extra>"
        ),
    ))
    fig.add_trace(go.Bar(
        x=df["ride_date"], y=df["tss_awc"],
        name="AWC (anaerobic)",
        marker_color=Z_AWC,
        customdata=cd,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "AWC = %{y:.0f}<br>"
            "Base ≤LTP = %{customdata[3]:.0f}<br>"
            "Threshold = %{customdata[4]:.0f}<br>"
            "NP = %{customdata[1]:.0f} W  "
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

    Panel 1 — Decayed MMP envelope + PDC curve:
        Black line shows the decayed MMP envelope.  Sections where the decay
        weight drops below 0.9 are highlighted in red (stale / needs retesting).
        The fitted PDC curve is overlaid for comparison.

    Panel 2 — Residuals (decayed envelope − PDC model):
        Line chart showing where the athlete is above or below the model.
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

    tte = tte_b = None
    popt, ok, tte, tte_b = (_fit_with_endurance_tail(dur_arr, pwr_arr)
                             if len(dur_arr) >= 4
                             else (None, False, None, None))

    if ok:
        AWC, Pmax, MAP, tau2 = popt
        model_vals             = _power_model_extended(dur_arr, AWC, Pmax, MAP, tau2, tte, tte_b)
        env_df["model_power"]  = model_vals
        env_df["residual"]     = env_df["aged_power"] - env_df["model_power"]
        env_df["residual_pct"] = (env_df["residual"] / env_df["model_power"] * 100).round(1)
        _tte_or_3600 = tte if tte is not None else 3600.0
        p_tte = float(_power_model_extended(_tte_or_3600, AWC, Pmax, MAP, tau2, tte, tte_b))
        ltp = float(MAP * (1.0 - (5.0 / 2.0) * ((AWC / 1000.0) / MAP)))

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.62, 0.38],
        vertical_spacing=0.10,
        subplot_titles=[
            "Decayed MMP vs PDC model  (red = decay < 0.9)",
            "Residual: aged MMP − PDC model",
        ],
    )

    # ── Panel 1: MMP envelope (black line, highlighted where decay < 0.9) ───
    # Split the envelope into "fresh" (weight >= 0.9) and "stale" (weight < 0.9)
    weights = env_df["weight"].to_numpy(dtype=float)
    stale_mask = weights < 0.9

    # Base envelope line in black
    fig.add_trace(go.Scatter(
        x=env_df["duration_s"],
        y=env_df["aged_power"],
        mode="lines",
        name="Decayed MMP envelope",
        line=dict(color="black", width=2),
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

    # Highlight segments where decay weight < 0.9 (stale data)
    if stale_mask.any():
        stale_x = env_df["duration_s"].to_numpy(dtype=float).copy()
        stale_y = env_df["aged_power"].to_numpy(dtype=float).copy()
        stale_y[~stale_mask] = np.nan
        fig.add_trace(go.Scatter(
            x=stale_x,
            y=stale_y,
            mode="lines",
            name="Stale (decay < 0.9)",
            line=dict(color="crimson", width=4),
            hoverinfo="skip",
            showlegend=True,
        ), row=1, col=1)

    # PDC model curve with zone splits (Base / Threshold / Anaerobic)
    if ok:
        t_sm  = np.logspace(np.log10(dur_arr.min()), np.log10(dur_arr.max()), 400)
        p_aer_base = MAP * (1.0 - np.exp(-t_sm / tau2))
        p_tot = _power_model_extended(t_sm, AWC, Pmax, MAP, tau2, tte, tte_b)
        # Beyond TtE, aerobic is fully ramped; cap at total power
        p_aer = np.minimum(p_aer_base, p_tot) if tte is not None else p_aer_base
        ltp_frac = ltp / MAP if MAP > 0 else 0.0
        p_base = p_aer * ltp_frac
        fig.add_trace(go.Scatter(
            x=t_sm, y=p_base,
            mode="lines", name="base (≤LTP)",
            fill="tozeroy", fillcolor=_zc(_RGB_BASE, 0.20),
            line=dict(color=_zc(_RGB_BASE, 0.7), width=1.2),
            hoverinfo="skip",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=t_sm, y=p_aer,
            mode="lines", name="threshold (LTP→MAP)",
            fill="tonexty", fillcolor=_zc(_RGB_THRESH, 0.20),
            line=dict(color=_zc(_RGB_THRESH, 0.7), width=1.2),
            hoverinfo="skip",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=t_sm, y=p_tot,
            mode="lines", name="PDC model",
            fill="tonexty", fillcolor=_zc(_RGB_AWC, 0.18),
            line=dict(color=Z_AWC, width=2.5, dash="dash"),
            hovertemplate="Model: %{y:.0f} W<extra></extra>",
        ), row=1, col=1)

    # ── Panel 2: residuals ────────────────────────────────────────────────────
    if ok:
        residuals_pct = env_df["residual_pct"].to_numpy(dtype=float)
        fig.add_trace(go.Scatter(
            x=dur_arr,
            y=residuals_pct,
            mode="lines",
            name="Residual",
            line=dict(color="steelblue", width=2),
            fill="tozeroy",
            fillcolor="rgba(70,130,180,0.15)",
            customdata=np.column_stack([
                env_df["residual"].to_numpy(),
                env_df["aged_power"].round(0),
                env_df["model_power"].round(0),
                env_df["weight"].round(3),
            ]),
            hovertemplate=(
                "<b>%{x:.0f} s</b><br>"
                "Residual: %{y:+.1f}% (%{customdata[0]:+.0f} W)<br>"
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
    fig.update_yaxes(title_text="Residual (%)", showgrid=True, gridcolor="lightgrey",
                     row=2, col=1)

    if ok:
        title_sub = (
            f"MAP {MAP:.0f} W  ·  P(TtE) {p_tte:.0f} W  ·  LTP {ltp:.0f} W  ·  "
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


def fig_pmc(pdc_params: pd.DataFrame, rides: pd.DataFrame) -> go.Figure:
    """Three-panel Performance Management Chart.

    Row 1 — Base (≤ LTP) TSS:        CTL (42d), ATL (7d), TSB
    Row 2 — Threshold (LTP→MAP) TSS: CTL (42d), ATL (7d), TSB
    Row 3 — Anaerobic (> MAP) TSS:   CTL (42d), ATL (7d), TSB

    TSB = CTL − ATL from the previous day (dotted line per panel).
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
            "Anaerobic (> MAP) — CTL / ATL / TSB",
            "Threshold (LTP → MAP) — CTL / ATL / TSB",
            "Base (≤ LTP) — CTL / ATL / TSB",
        ],
        vertical_spacing=0.07,
        specs=[[{"secondary_y": True}],
               [{"secondary_y": True}],
               [{"secondary_y": True}]],
    )

    panels = [
        # (row, pmc_df,     bar_data,         bar_color,                   ctl_col,   atl_col,                   label)
        (1, pmc_awc,    _tss_awc_bars,    _zc(_RGB_AWC, 0.45),     Z_AWC,     _zc(_RGB_AWC, 0.6),     "AWC"),
        (2, pmc_thresh, _tss_thresh_bars, _zc(_RGB_THRESH, 0.45),  Z_THRESH,  _zc(_RGB_THRESH, 0.6),  "Threshold"),
        (3, pmc_ltp,    _tss_ltp_bars,    _zc(_RGB_BASE, 0.45),    Z_BASE,    _zc(_RGB_BASE, 0.6),    "Base"),
    ]

    for row, pmc, bar_vals, bar_col, ctl_col, atl_col, label in panels:
        show  = row == 1   # show in legend only for the top panel
        dates = pmc["date"].dt.strftime("%Y-%m-%d")

        # TSS bars on LEFT axis (background context)
        fig.add_trace(go.Bar(
            x=dates, y=bar_vals,
            name=f"TSS {label}", marker_color=bar_col,
            showlegend=show,
            hovertemplate=f"TSS {label}: %{{y:.0f}}<extra></extra>",
        ), row=row, col=1, secondary_y=False)

        # TSB dotted line on RIGHT axis
        tsb_vals = pmc["tsb"].round(1)
        fig.add_trace(go.Scatter(
            x=dates, y=tsb_vals,
            mode="lines", name=f"TSB ({label})",
            line=dict(color=ctl_col, width=1.5, dash="dot"),
            showlegend=show,
            hovertemplate=f"TSB ({label}): %{{y:.1f}}<extra></extra>",
        ), row=row, col=1, secondary_y=True)

        # Readiness dots on the TSB line (red when below threshold)
        if label == "AWC":
            _cutoff_vals = pd.Series(0.0, index=pmc.index)
        elif label == "Threshold":
            _cutoff_vals = -0.30 * pmc["ctl"]
        else:  # Base
            _cutoff_vals = -0.50 * pmc["ctl"]
        _below = tsb_vals.values <= _cutoff_vals.values
        if _below.any():
            _x_red = dates[_below]
            _y_red = tsb_vals.values[_below]
            fig.add_trace(go.Scatter(
                x=_x_red, y=_y_red,
                mode="markers", name=f"TSB status ({label})",
                marker=dict(color="#dc2626", size=5),
                showlegend=False, hoverinfo="skip",
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
                         zeroline=False, autorange=False,
                         range=[l_min, l_max],
                         row=row, col=1, secondary_y=False)
        fig.update_yaxes(title_text="CTL / ATL / TSB", showgrid=True,
                         gridcolor="lightgrey", zeroline=False,
                         autorange=False, range=[r_min, r_max],
                         row=row, col=1, secondary_y=True)

    # Training cutoff shaded regions
    # Row 2 (Threshold): −0.3 × CTL
    # Row 3 (Base):      −0.5 × CTL
    _cutoff_panels = [
        (2, pmc_thresh, -0.30, "Training cutoff (−0.3 CTL)", "yaxis4", "yaxis3"),
        (3, pmc_ltp,    -0.50, "Training cutoff (−0.5 CTL)", "yaxis6", "yaxis5"),
    ]
    for row, pmc, frac, name, sec_ax, pri_ax in _cutoff_panels:
        _dates  = pmc["date"].dt.strftime("%Y-%m-%d")
        _cutoff = (frac * pmc["ctl"]).round(1)

        # Upper bound of shaded region (zero line)
        fig.add_trace(go.Scatter(
            x=_dates, y=np.zeros(len(_dates)),
            mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ), row=row, col=1, secondary_y=True)

        # Lower bound (cutoff) with fill back to zero
        fig.add_trace(go.Scatter(
            x=_dates, y=_cutoff,
            mode="lines", name=name,
            line=dict(color="grey", width=1, dash="dashdot"),
            fill="tonexty", fillcolor="rgba(180,180,180,0.18)",
            hovertemplate=f"Cutoff: %{{y:.1f}}<extra></extra>",
        ), row=row, col=1, secondary_y=True)

        # Widen axis if cutoff extends below current range
        c_min = float(_cutoff.min())
        if c_min < 0:
            cur = getattr(fig.layout, sec_ax).range
            if cur is not None and c_min < cur[0]:
                r_min_c = c_min * 1.05
                r_max_c = cur[1]
                l_range = getattr(fig.layout, pri_ax).range
                l_max_c = l_range[1] if l_range else 1.0
                r_span_c = r_max_c - r_min_c
                zf = -r_min_c / r_span_c if r_span_c > 0 else 0
                l_min_c = -l_max_c * zf / (1 - zf) if zf < 1 else 0
                fig.update_yaxes(range=[l_min_c, l_max_c], row=row, col=1, secondary_y=False)
                fig.update_yaxes(range=[r_min_c, r_max_c], row=row, col=1, secondary_y=True)

    # Default visible window: last 90 days + 7-day projection
    today = pd.Timestamp.today().normalize()
    x_start = (today - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
    x_end   = (today + pd.Timedelta(days=FUTURE_DAYS)).strftime("%Y-%m-%d")

    fig.update_xaxes(showgrid=True, gridcolor="lightgrey",
                     range=[x_start, x_end])
    fig.update_xaxes(title_text="Date", row=3, col=1)

    today_str = today.strftime("%Y-%m-%d")
    fig.add_vline(x=today_str, line=dict(color="grey", width=1.5))

    fig.update_layout(
        title=dict(text="Performance Management Chart", font=dict(size=14),
                   y=0.98, yanchor="top"),
        barmode="stack",
        height=780,
        margin=dict(t=90, b=50, l=70, r=20),
        template="plotly_white",
        hovermode="x unified",
        showlegend=False,
        dragmode="zoom",
    )
    fig.update_yaxes(fixedrange=True)
    return fig


def fig_pmc_combined(pdc_params: pd.DataFrame, rides: pd.DataFrame) -> go.Figure:
    """Single-panel PMC chart with stacked CTL components and zone TSB lines.

    TSS bars (left axis) are stacked by zone: Base (≤ LTP), Threshold
    (LTP→MAP), Anaerobic (> MAP).  CTL (right axis) is shown as stacked
    areas for the same three zones.  TSB lines for Base (green),
    Threshold (amber), and AWC (red) are plotted on the right axis.
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

    daily = df.groupby("ride_date")[["tss_ltp", "tss_thresh", "tss_awc"]].sum()

    FUTURE_DAYS = 7
    pmc_ltp    = _compute_pmc(daily["tss_ltp"],    future_days=FUTURE_DAYS)
    pmc_thresh = _compute_pmc(daily["tss_thresh"], future_days=FUTURE_DAYS)
    pmc_awc    = _compute_pmc(daily["tss_awc"],    future_days=FUTURE_DAYS)

    # Align daily TSS to the continuous date grid
    _idx             = pd.DatetimeIndex(pmc_ltp["date"])
    _tss_ltp_bars    = daily["tss_ltp"].reindex(_idx, fill_value=0.0).round(1).values
    _tss_thresh_bars = daily["tss_thresh"].reindex(_idx, fill_value=0.0).round(1).values
    _tss_awc_bars    = daily["tss_awc"].reindex(_idx, fill_value=0.0).round(1).values

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    dates = pmc_ltp["date"].dt.strftime("%Y-%m-%d")

    # ── Stacked TSS bars on LEFT axis (Base bottom, AWC top) ───────────
    fig.add_trace(go.Bar(
        x=dates, y=_tss_ltp_bars,
        name="TSS Base", marker_color=_zc(_RGB_BASE, 0.45),
        hovertemplate="TSS Base: %{y:.0f}<extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Bar(
        x=dates, y=_tss_thresh_bars,
        name="TSS Threshold", marker_color=_zc(_RGB_THRESH, 0.45),
        hovertemplate="TSS Thresh: %{y:.0f}<extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Bar(
        x=dates, y=_tss_awc_bars,
        name="TSS Anaerobic", marker_color=_zc(_RGB_AWC, 0.45),
        hovertemplate="TSS AWC: %{y:.0f}<extra></extra>",
    ), secondary_y=False)

    # ── Stacked CTL areas on RIGHT axis (Base bottom, AWC top) ───────
    fig.add_trace(go.Scatter(
        x=dates, y=pmc_ltp["ctl"].round(1),
        mode="lines", name="CTL Base",
        line=dict(color=Z_BASE, width=0.5),
        fillcolor=_zc(_RGB_BASE, 0.30),
        stackgroup="ctl",
        hovertemplate="CTL Base: %{y:.1f}<extra></extra>",
    ), secondary_y=True)

    fig.add_trace(go.Scatter(
        x=dates, y=pmc_thresh["ctl"].round(1),
        mode="lines", name="CTL Threshold",
        line=dict(color=Z_THRESH, width=0.5),
        fillcolor=_zc(_RGB_THRESH, 0.30),
        stackgroup="ctl",
        hovertemplate="CTL Thresh: %{y:.1f}<extra></extra>",
    ), secondary_y=True)

    fig.add_trace(go.Scatter(
        x=dates, y=pmc_awc["ctl"].round(1),
        mode="lines", name="CTL Anaerobic",
        line=dict(color=Z_AWC, width=0.5),
        fillcolor=_zc(_RGB_AWC, 0.30),
        stackgroup="ctl",
        hovertemplate="CTL AWC: %{y:.1f}<extra></extra>",
    ), secondary_y=True)

    # ── TSB lines on RIGHT axis ────────────────────────────────────────
    _tsb_awc_vals = pmc_awc["tsb"].round(1)
    fig.add_trace(go.Scatter(
        x=dates, y=_tsb_awc_vals,
        mode="lines", name="TSB AWC",
        line=dict(color=Z_AWC, width=2),
        hovertemplate="TSB AWC: %{y:.1f}<extra></extra>",
    ), secondary_y=True)
    _below_awc = _tsb_awc_vals.values <= 0
    if _below_awc.any():
        fig.add_trace(go.Scatter(
            x=dates[_below_awc], y=_tsb_awc_vals.values[_below_awc],
            mode="markers", name="TSB status AWC",
            marker=dict(color="#dc2626", size=5),
            showlegend=False, hoverinfo="skip",
        ), secondary_y=True)

    _tsb_thresh_vals = pmc_thresh["tsb"].round(1)
    fig.add_trace(go.Scatter(
        x=dates, y=_tsb_thresh_vals,
        mode="lines", name="TSB Threshold",
        line=dict(color=Z_THRESH, width=2),
        hovertemplate="TSB Thresh: %{y:.1f}<extra></extra>",
    ), secondary_y=True)
    _below_thresh = _tsb_thresh_vals.values <= (-0.30 * pmc_thresh["ctl"]).values
    if _below_thresh.any():
        fig.add_trace(go.Scatter(
            x=dates[_below_thresh], y=_tsb_thresh_vals.values[_below_thresh],
            mode="markers", name="TSB status Thresh",
            marker=dict(color="#dc2626", size=5),
            showlegend=False, hoverinfo="skip",
        ), secondary_y=True)

    _tsb_base_vals = pmc_ltp["tsb"].round(1)
    fig.add_trace(go.Scatter(
        x=dates, y=_tsb_base_vals,
        mode="lines", name="TSB Base",
        line=dict(color=Z_BASE, width=2),
        hovertemplate="TSB Base: %{y:.1f}<extra></extra>",
    ), secondary_y=True)
    _below_base = _tsb_base_vals.values <= (-0.50 * pmc_ltp["ctl"]).values
    if _below_base.any():
        fig.add_trace(go.Scatter(
            x=dates[_below_base], y=_tsb_base_vals.values[_below_base],
            mode="markers", name="TSB status Base",
            marker=dict(color="#dc2626", size=5),
            showlegend=False, hoverinfo="skip",
        ), secondary_y=True)

    fig.add_hline(y=0, line=dict(color="grey", dash="dot", width=1),
                  secondary_y=True)

    # ── Align zero across both axes ────────────────────────────────────
    total_ctl = pmc_ltp["ctl"] + pmc_thresh["ctl"] + pmc_awc["ctl"]
    r_min = float(min(pmc_ltp["tsb"].min(), pmc_thresh["tsb"].min(),
                      pmc_awc["tsb"].min(), 0))
    r_max = float(max(total_ctl.max(), pmc_ltp["tsb"].max(),
                      pmc_thresh["tsb"].max(), pmc_awc["tsb"].max(), 1))
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
                     zeroline=False, autorange=False,
                     range=[l_min, l_max],
                     secondary_y=False)
    fig.update_yaxes(title_text="CTL / TSB", showgrid=True,
                     gridcolor="lightgrey", zeroline=False,
                     autorange=False, range=[r_min, r_max],
                     secondary_y=True)

    today   = pd.Timestamp.today().normalize()
    x_start = (today - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
    x_end   = (today + pd.Timedelta(days=FUTURE_DAYS)).strftime("%Y-%m-%d")

    today_str = today.strftime("%Y-%m-%d")
    fig.add_vline(x=today_str, line=dict(color="grey", width=1.5))

    fig.update_xaxes(showgrid=True, gridcolor="lightgrey",
                     range=[x_start, x_end], title_text="Date")
    fig.update_layout(
        title=dict(text="Performance Management Chart — Combined",
                   font=dict(size=14), y=0.98, yanchor="top"),
        barmode="stack",
        height=380,
        margin=dict(t=80, b=50, l=70, r=20),
        template="plotly_white",
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        dragmode="zoom",
    )
    fig.update_yaxes(fixedrange=True)
    return fig
