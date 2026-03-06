"""Interactive training planner — standalone Dash app.

Run with:  python planner_app.py
Then open http://127.0.0.1:8051
"""

from __future__ import annotations

import os
from datetime import date, timedelta

import dash
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html
from plotly.subplots import make_subplots

from training_plan import (
    DayPlan,
    Phase,
    TrainingState,
    ZoneState,
    ZoneTSS,
    load_current_state,
    simulate_plan,
)

# ── Theme colours (match main app) ──────────────────────────────────────────

BG        = "#0d1117"
BG_CARD   = "#161b22"
BG_INPUT  = "#1e2a3a"
BORDER    = "#30363d"
TEXT      = "#e8edf5"
TEXT_DIM  = "#7a8fbb"
ACCENT    = "#4a9eff"

Z_BASE    = "rgb(46,139,87)"
Z_THRESH  = "rgb(235,168,36)"
Z_AWC     = "rgb(214,55,46)"

PHASE_COLORS = {
    Phase.BASE:  "rgba(46,139,87,0.15)",
    Phase.BUILD: "rgba(235,168,36,0.15)",
    Phase.PEAK:  "rgba(214,55,46,0.15)",
}
FRESHNESS_COLORS = {
    "green":  "#16a34a",
    "amber":  "#d97706",
    "red":    "#dc2626",
    "black":  "#555",
}
FRESHNESS_BG = {
    "green":  "rgba(22,163,74,0.08)",
    "amber":  "rgba(217,119,6,0.08)",
    "red":    "rgba(220,38,38,0.08)",
    "black":  "rgba(85,85,85,0.08)",
}

# ── Shared styles ────────────────────────────────────────────────────────────

_CARD = {
    "background": BG_CARD, "border": f"1px solid {BORDER}",
    "borderRadius": "8px", "padding": "16px 20px",
}
_LABEL = {
    "fontSize": "11px", "color": TEXT_DIM, "marginBottom": "2px",
    "textTransform": "uppercase", "letterSpacing": "0.05em",
}
_VALUE = {"fontSize": "22px", "fontWeight": "bold", "color": TEXT}
_UNIT  = {"fontSize": "12px", "color": TEXT_DIM, "marginLeft": "3px"}

_INPUT = {
    "width": "100%", "padding": "7px 10px", "borderRadius": "4px",
    "border": f"1px solid #555", "background": BG_INPUT,
    "color": TEXT, "fontSize": "14px",
}
_INPUT_LABEL = {"fontSize": "12px", "color": TEXT_DIM, "marginBottom": "4px"}

DB_PATH = os.path.join(os.path.dirname(__file__), "cycling.db")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _card(label: str, value: str, unit: str = "") -> html.Div:
    return html.Div(style=_CARD | {"textAlign": "center", "minWidth": "110px"}, children=[
        html.Div(label, style=_LABEL),
        html.Div([
            html.Span(value, style=_VALUE),
            html.Span(unit, style=_UNIT) if unit else None,
        ]),
    ])


def _input_group(label: str, input_id: str, value, **kwargs) -> html.Div:
    return html.Div(style={"flex": "1", "minWidth": "130px"}, children=[
        html.Div(label, style=_INPUT_LABEL),
        dcc.Input(id=input_id, type="number", value=value,
                  style=_INPUT, debounce=True, **kwargs),
    ])


# ── Plotly figure builders ───────────────────────────────────────────────────

_LAYOUT_BASE = dict(
    paper_bgcolor=BG, plot_bgcolor=BG,
    font=dict(color=TEXT_DIM, size=12),
    margin=dict(l=50, r=20, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    xaxis=dict(gridcolor="#1e2433", zerolinecolor="#1e2433"),
    yaxis=dict(gridcolor="#1e2433", zerolinecolor="#1e2433"),
)


def _fig_ctl_projection(plan: list[DayPlan], target: ZoneTSS) -> go.Figure:
    """CTL trajectories vs target lines, with phase-coloured background bands."""
    if not plan:
        fig = go.Figure()
        fig.update_layout(**_LAYOUT_BASE, title="No plan to display")
        return fig

    dates = [d.date.isoformat() for d in plan]
    ctl_b = [d.ctl_base for d in plan]
    ctl_t = [d.ctl_threshold for d in plan]
    ctl_a = [d.ctl_anaerobic for d in plan]

    fig = go.Figure()

    # Phase background bands
    i = 0
    while i < len(plan):
        phase = plan[i].phase
        j = i
        while j < len(plan) and plan[j].phase == phase:
            j += 1
        fig.add_vrect(
            x0=plan[i].date.isoformat(),
            x1=plan[min(j, len(plan) - 1)].date.isoformat(),
            fillcolor=PHASE_COLORS[phase], line_width=0,
            annotation_text=phase.name, annotation_position="top left",
            annotation=dict(font=dict(size=11, color=TEXT_DIM)),
        )
        i = j

    fig.add_trace(go.Scatter(x=dates, y=ctl_b, name="Base CTL",
                             line=dict(color=Z_BASE, width=2)))
    fig.add_trace(go.Scatter(x=dates, y=ctl_t, name="Threshold CTL",
                             line=dict(color=Z_THRESH, width=2)))
    fig.add_trace(go.Scatter(x=dates, y=ctl_a, name="Anaerobic CTL",
                             line=dict(color=Z_AWC, width=2)))

    # Target lines
    fig.add_hline(y=target.base, line=dict(color=Z_BASE, dash="dot", width=1),
                  annotation_text=f"Target {target.base:.0f}",
                  annotation=dict(font=dict(color=Z_BASE, size=10)))
    fig.add_hline(y=target.threshold, line=dict(color=Z_THRESH, dash="dot", width=1),
                  annotation_text=f"Target {target.threshold:.0f}",
                  annotation=dict(font=dict(color=Z_THRESH, size=10)))
    fig.add_hline(y=target.anaerobic, line=dict(color=Z_AWC, dash="dot", width=1),
                  annotation_text=f"Target {target.anaerobic:.0f}",
                  annotation=dict(font=dict(color=Z_AWC, size=10)))

    fig.update_layout(**_LAYOUT_BASE, title="Zone CTL Projection",
                      yaxis_title="CTL", height=350)
    return fig


def _fig_tsb_projection(plan: list[DayPlan]) -> go.Figure:
    """TSB trajectories with freshness-coloured background and TSS bars."""
    if not plan:
        fig = go.Figure()
        fig.update_layout(**_LAYOUT_BASE, title="No plan to display")
        return fig

    dates = [d.date.isoformat() for d in plan]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Freshness background strips (coloured by freshness status)
    i = 0
    while i < len(plan):
        fr = plan[i].freshness.value
        j = i
        while j < len(plan) and plan[j].freshness.value == fr:
            j += 1
        fig.add_vrect(
            x0=plan[i].date.isoformat(),
            x1=plan[min(j, len(plan) - 1)].date.isoformat(),
            fillcolor=FRESHNESS_BG[fr],
            line_width=0,
        )
        i = j

    # Stacked TSS bars on secondary y-axis (drawn first so lines sit on top)
    fig.add_trace(go.Bar(
        x=dates, y=[d.zone_tss.base for d in plan],
        name="Base TSS", marker_color=Z_BASE, opacity=0.25,
        legendgroup="tss",
    ), secondary_y=True)
    fig.add_trace(go.Bar(
        x=dates, y=[d.zone_tss.threshold for d in plan],
        name="Threshold TSS", marker_color=Z_THRESH, opacity=0.25,
        legendgroup="tss",
    ), secondary_y=True)
    fig.add_trace(go.Bar(
        x=dates, y=[d.zone_tss.anaerobic for d in plan],
        name="Anaerobic TSS", marker_color=Z_AWC, opacity=0.25,
        legendgroup="tss",
    ), secondary_y=True)

    # TSB lines on primary y-axis
    fig.add_trace(go.Scatter(
        x=dates, y=[d.tsb_base for d in plan],
        name="Base TSB", line=dict(color=Z_BASE, width=2),
        legendgroup="tsb",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=dates, y=[d.tsb_threshold for d in plan],
        name="Threshold TSB", line=dict(color=Z_THRESH, width=2),
        legendgroup="tsb",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=dates, y=[d.tsb_anaerobic for d in plan],
        name="Anaerobic TSB", line=dict(color=Z_AWC, width=2),
        legendgroup="tsb",
    ), secondary_y=False)
    fig.add_hline(y=0, line=dict(color="#555", dash="dot", width=1),
                  secondary_y=False)

    fig.update_layout(
        **_LAYOUT_BASE, barmode="stack",
        title="Zone TSB & Daily TSS", height=350,
        yaxis2=dict(gridcolor="#1e2433", zerolinecolor="#1e2433"),
    )
    fig.update_yaxes(title_text="TSB", secondary_y=False)
    fig.update_yaxes(title_text="TSS", secondary_y=True)
    return fig


def _fig_daily_tss(plan: list[DayPlan]) -> go.Figure:
    """Stacked bar chart of daily TSS by zone."""
    if not plan:
        fig = go.Figure()
        fig.update_layout(**_LAYOUT_BASE, title="No plan to display")
        return fig

    dates = [d.date.isoformat() for d in plan]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dates, y=[d.zone_tss.base for d in plan],
        name="Base", marker_color=Z_BASE))
    fig.add_trace(go.Bar(
        x=dates, y=[d.zone_tss.threshold for d in plan],
        name="Threshold", marker_color=Z_THRESH))
    fig.add_trace(go.Bar(
        x=dates, y=[d.zone_tss.anaerobic for d in plan],
        name="Anaerobic", marker_color=Z_AWC))

    fig.update_layout(**_LAYOUT_BASE, barmode="stack",
                      title="Daily TSS by Zone", yaxis_title="TSS",
                      height=250)
    return fig


def _fig_weekly_volume(plan: list[DayPlan]) -> go.Figure:
    """Weekly TSS volume stacked bars."""
    if not plan:
        fig = go.Figure()
        fig.update_layout(**_LAYOUT_BASE, title="No plan to display")
        return fig

    import pandas as pd
    df = pd.DataFrame([{
        "date": d.date,
        "base": d.zone_tss.base,
        "threshold": d.zone_tss.threshold,
        "anaerobic": d.zone_tss.anaerobic,
    } for d in plan])
    df["week"] = pd.to_datetime(df["date"]).dt.to_period("W").dt.start_time
    weekly = df.groupby("week")[["base", "threshold", "anaerobic"]].sum()

    fig = go.Figure()
    fig.add_trace(go.Bar(x=weekly.index, y=weekly["base"],
                         name="Base", marker_color=Z_BASE))
    fig.add_trace(go.Bar(x=weekly.index, y=weekly["threshold"],
                         name="Threshold", marker_color=Z_THRESH))
    fig.add_trace(go.Bar(x=weekly.index, y=weekly["anaerobic"],
                         name="Anaerobic", marker_color=Z_AWC))

    fig.update_layout(**_LAYOUT_BASE, barmode="stack",
                      title="Weekly TSS Volume", yaxis_title="TSS / week",
                      height=280)
    return fig


# ── Dash app ─────────────────────────────────────────────────────────────────

app = dash.Dash(__name__, title="Training Planner")

# Load initial state from DB
try:
    _init_state = load_current_state(DB_PATH)
except Exception:
    _init_state = TrainingState()

_default_event = (date.today() + timedelta(weeks=20)).isoformat()

app.layout = html.Div(style={
    "fontFamily": "sans-serif", "background": BG, "color": TEXT,
    "minHeight": "100vh", "padding": "24px 32px",
}, children=[

    # ── Header ───────────────────────────────────────────────────────────
    html.H1("Training Planner", style={
        "fontSize": "24px", "fontWeight": "600", "marginBottom": "20px"}),

    # ── Current state cards ──────────────────────────────────────────────
    html.Div(style={"display": "flex", "gap": "12px", "flexWrap": "wrap",
                     "marginBottom": "20px"}, children=[
        _card("Base CTL", f"{_init_state.base.ctl:.1f}"),
        _card("Threshold CTL", f"{_init_state.threshold.ctl:.1f}"),
        _card("Anaerobic CTL", f"{_init_state.anaerobic.ctl:.1f}"),
        _card("Base TSB", f"{_init_state.base.tsb:+.1f}"),
        _card("Threshold TSB", f"{_init_state.threshold.tsb:+.1f}"),
        _card("Anaerobic TSB", f"{_init_state.anaerobic.tsb:+.1f}"),
        _card("Freshness", _init_state.freshness().value.upper()),
    ]),

    # ── Controls ─────────────────────────────────────────────────────────
    html.Div(style={
        **_CARD, "display": "flex", "gap": "16px", "flexWrap": "wrap",
        "alignItems": "flex-end", "marginBottom": "20px",
    }, children=[
        html.Div(style={"flex": "1", "minWidth": "160px"}, children=[
            html.Div("Event date", style=_INPUT_LABEL),
            dcc.DatePickerSingle(
                id="event-date",
                date=_default_event,
                display_format="YYYY-MM-DD",
                style={"background": BG_INPUT},
            ),
        ]),
        _input_group("Target Base CTL", "target-base", 50, min=0, max=200),
        _input_group("Target Threshold CTL", "target-thresh", 25, min=0, max=100),
        _input_group("Target Anaerobic CTL", "target-ana", 12, min=0, max=60),
        _input_group("Base weeks", "base-weeks", 8, min=1, max=26),
        _input_group("Build weeks", "build-weeks", 8, min=1, max=26),
        _input_group("Peak weeks", "peak-weeks", 4, min=1, max=12),
        html.Div(style={"flex": "0 0 auto"}, children=[
            html.Button("Generate plan", id="btn-generate", n_clicks=0, style={
                "padding": "9px 24px", "cursor": "pointer", "borderRadius": "4px",
                "border": f"1px solid {ACCENT}", "background": ACCENT, "color": "#fff",
                "fontSize": "14px", "fontWeight": "600", "marginTop": "18px",
            }),
        ]),
    ]),

    # ── Today's suggestion ───────────────────────────────────────────────
    html.Div(id="today-suggestion", style={
        **_CARD, "marginBottom": "20px",
    }),

    # ── Charts ───────────────────────────────────────────────────────────
    dcc.Graph(id="graph-ctl", config={"displayModeBar": False}),
    dcc.Graph(id="graph-tsb", config={"displayModeBar": False}),
    dcc.Graph(id="graph-daily-tss", config={"displayModeBar": False}),
    dcc.Graph(id="graph-weekly", config={"displayModeBar": False}),

    # ── Day detail on click ──────────────────────────────────────────────
    html.Div(id="day-detail", style={**_CARD, "marginTop": "16px"}),

    # ── Hidden store for plan data ───────────────────────────────────────
    dcc.Store(id="plan-store", data=[]),
])


# ── Callbacks ────────────────────────────────────────────────────────────────

@app.callback(
    Output("graph-ctl", "figure"),
    Output("graph-tsb", "figure"),
    Output("graph-daily-tss", "figure"),
    Output("graph-weekly", "figure"),
    Output("today-suggestion", "children"),
    Output("plan-store", "data"),
    Input("btn-generate", "n_clicks"),
    State("event-date", "date"),
    State("target-base", "value"),
    State("target-thresh", "value"),
    State("target-ana", "value"),
    State("base-weeks", "value"),
    State("build-weeks", "value"),
    State("peak-weeks", "value"),
)
def generate_plan(n_clicks, event_str, t_base, t_thresh, t_ana,
                  base_w, build_w, peak_w):
    if not event_str:
        raise dash.exceptions.PreventUpdate

    event = date.fromisoformat(event_str[:10])
    target = ZoneTSS(
        base=float(t_base or 50),
        threshold=float(t_thresh or 25),
        anaerobic=float(t_ana or 12),
    )
    base_w = int(base_w or 8)
    build_w = int(build_w or 8)
    peak_w = int(peak_w or 4)

    state = load_current_state(DB_PATH)
    today = date.today()

    plan = simulate_plan(state, target, today, event, base_w, build_w, peak_w)

    fig_ctl = _fig_ctl_projection(plan, target)
    fig_tsb = _fig_tsb_projection(plan)
    fig_daily = _fig_daily_tss(plan)
    fig_weekly = _fig_weekly_volume(plan)

    # Today's suggestion
    if plan:
        d = plan[0]
        fresh_color = FRESHNESS_COLORS.get(d.freshness.value, TEXT)
        suggestion = html.Div(style={"display": "flex", "gap": "24px",
                                      "alignItems": "center", "flexWrap": "wrap"}, children=[
            html.Div([
                html.Div("TODAY'S WORKOUT", style=_LABEL),
                html.Div(d.workout_name, style={
                    "fontSize": "20px", "fontWeight": "bold", "color": TEXT}),
                html.Div(d.workout_description, style={
                    "fontSize": "13px", "color": TEXT_DIM, "marginTop": "4px"}),
            ]),
            html.Div([
                html.Div("PHASE", style=_LABEL),
                html.Div(d.phase.name, style={"fontSize": "16px", "fontWeight": "600"}),
            ]),
            html.Div([
                html.Div("FRESHNESS", style=_LABEL),
                html.Div(d.freshness.value.upper(), style={
                    "fontSize": "16px", "fontWeight": "600", "color": fresh_color}),
            ]),
            html.Div([
                html.Div("TSS", style=_LABEL),
                html.Div(f"{d.zone_tss.total:.0f}", style=_VALUE),
            ]),
            html.Div([
                html.Div("ZONE SPLIT", style=_LABEL),
                html.Div(style={"display": "flex", "gap": "8px"}, children=[
                    html.Span(f"B:{d.zone_tss.base:.0f}",
                              style={"color": Z_BASE, "fontWeight": "600"}),
                    html.Span(f"T:{d.zone_tss.threshold:.0f}",
                              style={"color": Z_THRESH, "fontWeight": "600"}),
                    html.Span(f"A:{d.zone_tss.anaerobic:.0f}",
                              style={"color": Z_AWC, "fontWeight": "600"}),
                ]),
            ]),
        ])
    else:
        suggestion = html.Div("No plan — event date is in the past.",
                              style={"color": TEXT_DIM})

    # Serialize plan for click interactions
    plan_data = [{
        "date": d.date.isoformat(),
        "phase": d.phase.name,
        "freshness": d.freshness.value,
        "workout_key": d.workout_key,
        "workout_name": d.workout_name,
        "workout_description": d.workout_description,
        "tss_total": round(d.zone_tss.total, 1),
        "tss_base": round(d.zone_tss.base, 1),
        "tss_threshold": round(d.zone_tss.threshold, 1),
        "tss_anaerobic": round(d.zone_tss.anaerobic, 1),
        "ctl_base": round(d.ctl_base, 1),
        "ctl_threshold": round(d.ctl_threshold, 1),
        "ctl_anaerobic": round(d.ctl_anaerobic, 1),
        "tsb_base": round(d.tsb_base, 1),
        "tsb_threshold": round(d.tsb_threshold, 1),
        "tsb_anaerobic": round(d.tsb_anaerobic, 1),
    } for d in plan]

    return fig_ctl, fig_tsb, fig_daily, fig_weekly, suggestion, plan_data


@app.callback(
    Output("day-detail", "children"),
    Input("graph-daily-tss", "clickData"),
    State("plan-store", "data"),
)
def show_day_detail(click_data, plan_data):
    if not click_data or not plan_data:
        return html.Div("Click a bar on the daily TSS chart to see workout details.",
                        style={"color": TEXT_DIM, "fontSize": "13px"})

    point = click_data["points"][0]
    clicked_date = point["x"]

    # Find the matching day
    day = None
    for d in plan_data:
        if d["date"] == clicked_date:
            day = d
            break

    if not day:
        return html.Div(f"No data for {clicked_date}", style={"color": TEXT_DIM})

    fresh_color = FRESHNESS_COLORS.get(day["freshness"], TEXT)
    return html.Div(style={"display": "flex", "gap": "24px",
                            "alignItems": "center", "flexWrap": "wrap"}, children=[
        html.Div([
            html.Div("DATE", style=_LABEL),
            html.Div(day["date"], style={"fontSize": "16px", "fontWeight": "600"}),
        ]),
        html.Div([
            html.Div("WORKOUT", style=_LABEL),
            html.Div(day["workout_name"], style={
                "fontSize": "16px", "fontWeight": "600"}),
            html.Div(day["workout_description"], style={
                "fontSize": "12px", "color": TEXT_DIM}),
        ]),
        html.Div([
            html.Div("PHASE", style=_LABEL),
            html.Div(day["phase"], style={"fontSize": "14px"}),
        ]),
        html.Div([
            html.Div("FRESHNESS", style=_LABEL),
            html.Div(day["freshness"].upper(), style={
                "fontSize": "14px", "color": fresh_color, "fontWeight": "600"}),
        ]),
        html.Div([
            html.Div("TOTAL TSS", style=_LABEL),
            html.Div(f"{day['tss_total']:.0f}", style=_VALUE),
        ]),
        html.Div([
            html.Div("ZONE TSS", style=_LABEL),
            html.Div(style={"display": "flex", "gap": "8px"}, children=[
                html.Span(f"B:{day['tss_base']:.0f}",
                          style={"color": Z_BASE, "fontWeight": "600"}),
                html.Span(f"T:{day['tss_threshold']:.0f}",
                          style={"color": Z_THRESH, "fontWeight": "600"}),
                html.Span(f"A:{day['tss_anaerobic']:.0f}",
                          style={"color": Z_AWC, "fontWeight": "600"}),
            ]),
        ]),
        html.Div([
            html.Div("ZONE CTL", style=_LABEL),
            html.Div(style={"display": "flex", "gap": "8px"}, children=[
                html.Span(f"B:{day['ctl_base']:.1f}",
                          style={"color": Z_BASE}),
                html.Span(f"T:{day['ctl_threshold']:.1f}",
                          style={"color": Z_THRESH}),
                html.Span(f"A:{day['ctl_anaerobic']:.1f}",
                          style={"color": Z_AWC}),
            ]),
        ]),
        html.Div([
            html.Div("ZONE TSB", style=_LABEL),
            html.Div(style={"display": "flex", "gap": "8px"}, children=[
                html.Span(f"B:{day['tsb_base']:+.1f}",
                          style={"color": Z_BASE}),
                html.Span(f"T:{day['tsb_threshold']:+.1f}",
                          style={"color": Z_THRESH}),
                html.Span(f"A:{day['tsb_anaerobic']:+.1f}",
                          style={"color": Z_AWC}),
            ]),
        ]),
    ])


if __name__ == "__main__":
    app.run(debug=True, port=8051)
