"""Interactive training planner — standalone Dash app.

Simple scenario planner: enter a starting CTL and a target CTL,
and see the 20-week (8 base / 8 build / 4 peak) projection.
Starting TSB is always 0 (ATL = CTL).

Run with:  python planner_app.py
Then open http://127.0.0.1:8051
"""

from __future__ import annotations

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


# ── Helpers ──────────────────────────────────────────────────────────────────

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

    weeks = [i / 7.0 for i in range(len(plan))]
    ctl_b = [d.ctl_base for d in plan]
    ctl_t = [d.ctl_threshold for d in plan]
    ctl_a = [d.ctl_anaerobic for d in plan]

    fig = go.Figure()

    # Phase background bands (in weeks)
    i = 0
    while i < len(plan):
        phase = plan[i].phase
        j = i
        while j < len(plan) and plan[j].phase == phase:
            j += 1
        fig.add_vrect(
            x0=i / 7.0, x1=min(j, len(plan) - 1) / 7.0,
            fillcolor=PHASE_COLORS[phase], line_width=0,
            annotation_text=phase.name, annotation_position="top left",
            annotation=dict(font=dict(size=11, color=TEXT_DIM)),
        )
        i = j

    fig.add_trace(go.Scatter(x=weeks, y=ctl_b, name="Base CTL",
                             line=dict(color=Z_BASE, width=2)))
    fig.add_trace(go.Scatter(x=weeks, y=ctl_t, name="Threshold CTL",
                             line=dict(color=Z_THRESH, width=2)))
    fig.add_trace(go.Scatter(x=weeks, y=ctl_a, name="Anaerobic CTL",
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
                      xaxis_title="Week", yaxis_title="CTL", height=350)
    return fig


def _fig_tsb_projection(plan: list[DayPlan]) -> go.Figure:
    """TSB trajectories with freshness-coloured background and TSS bars."""
    if not plan:
        fig = go.Figure()
        fig.update_layout(**_LAYOUT_BASE, title="No plan to display")
        return fig

    weeks = [i / 7.0 for i in range(len(plan))]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Freshness background strips
    i = 0
    while i < len(plan):
        fr = plan[i].freshness.value
        j = i
        while j < len(plan) and plan[j].freshness.value == fr:
            j += 1
        fig.add_vrect(
            x0=i / 7.0, x1=min(j, len(plan) - 1) / 7.0,
            fillcolor=FRESHNESS_BG[fr], line_width=0,
        )
        i = j

    # Stacked TSS bars on secondary y-axis
    fig.add_trace(go.Bar(
        x=weeks, y=[d.zone_tss.base for d in plan],
        name="Base TSS", marker_color=Z_BASE, opacity=0.25,
        legendgroup="tss",
    ), secondary_y=True)
    fig.add_trace(go.Bar(
        x=weeks, y=[d.zone_tss.threshold for d in plan],
        name="Threshold TSS", marker_color=Z_THRESH, opacity=0.25,
        legendgroup="tss",
    ), secondary_y=True)
    fig.add_trace(go.Bar(
        x=weeks, y=[d.zone_tss.anaerobic for d in plan],
        name="Anaerobic TSS", marker_color=Z_AWC, opacity=0.25,
        legendgroup="tss",
    ), secondary_y=True)

    # TSB lines on primary y-axis
    fig.add_trace(go.Scatter(
        x=weeks, y=[d.tsb_base for d in plan],
        name="Base TSB", line=dict(color=Z_BASE, width=2),
        legendgroup="tsb",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=weeks, y=[d.tsb_threshold for d in plan],
        name="Threshold TSB", line=dict(color=Z_THRESH, width=2),
        legendgroup="tsb",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=weeks, y=[d.tsb_anaerobic for d in plan],
        name="Anaerobic TSB", line=dict(color=Z_AWC, width=2),
        legendgroup="tsb",
    ), secondary_y=False)
    fig.add_hline(y=0, line=dict(color="#555", dash="dot", width=1),
                  secondary_y=False)

    fig.update_layout(
        **_LAYOUT_BASE, barmode="stack",
        title="Zone TSB & Daily TSS", height=350,
        xaxis_title="Week",
        yaxis2=dict(gridcolor="#1e2433", zerolinecolor="#1e2433"),
    )
    fig.update_yaxes(title_text="TSB", secondary_y=False)
    fig.update_yaxes(title_text="TSS", secondary_y=True)
    return fig


def _fig_weekly_volume(plan: list[DayPlan]) -> go.Figure:
    """Weekly TSS volume stacked bars."""
    if not plan:
        fig = go.Figure()
        fig.update_layout(**_LAYOUT_BASE, title="No plan to display")
        return fig

    import pandas as pd
    df = pd.DataFrame([{
        "day": i,
        "week": i // 7 + 1,
        "base": d.zone_tss.base,
        "threshold": d.zone_tss.threshold,
        "anaerobic": d.zone_tss.anaerobic,
    } for i, d in enumerate(plan)])
    weekly = df.groupby("week")[["base", "threshold", "anaerobic"]].sum()

    fig = go.Figure()
    fig.add_trace(go.Bar(x=weekly.index, y=weekly["base"],
                         name="Base", marker_color=Z_BASE))
    fig.add_trace(go.Bar(x=weekly.index, y=weekly["threshold"],
                         name="Threshold", marker_color=Z_THRESH))
    fig.add_trace(go.Bar(x=weekly.index, y=weekly["anaerobic"],
                         name="Anaerobic", marker_color=Z_AWC))

    fig.update_layout(**_LAYOUT_BASE, barmode="stack",
                      title="Weekly TSS Volume", xaxis_title="Week",
                      yaxis_title="TSS / week", height=280)
    return fig


# ── Dash app ─────────────────────────────────────────────────────────────────

app = dash.Dash(__name__, title="Training Planner")

# Ensure html/body don't clip — Dash's default styles can prevent scrolling
app.index_string = '''<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            html, body, #react-entry-point {
                margin: 0; padding: 0;
                height: auto; min-height: 100vh;
                overflow-y: auto;
                background: ''' + BG + ''';
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.layout = html.Div(style={
    "fontFamily": "sans-serif", "background": BG, "color": TEXT,
    "padding": "24px 32px 60px", "maxWidth": "1100px", "margin": "0 auto",
}, children=[

    # ── Header ───────────────────────────────────────────────────────────
    html.H1("Training Planner", style={
        "fontSize": "24px", "fontWeight": "600", "marginBottom": "4px"}),
    html.P("20-week scenario: set start and target CTL, TSB starts at 0.",
           style={"color": TEXT_DIM, "fontSize": "13px", "marginBottom": "20px"}),

    # ── Controls ─────────────────────────────────────────────────────────
    html.Div(style={
        **_CARD, "display": "flex", "gap": "16px", "flexWrap": "wrap",
        "alignItems": "flex-end", "marginBottom": "20px",
    }, children=[
        # Start CTL
        html.Div(style={"flex": "0 0 100%"}, children=[
            html.Div("START CTL", style={**_LABEL, "marginBottom": "8px"}),
        ]),
        _input_group("Base", "start-base", 20, min=0, max=200),
        _input_group("Threshold", "start-thresh", 10, min=0, max=100),
        _input_group("Anaerobic", "start-ana", 5, min=0, max=60),

        # Spacer
        html.Div(style={"flex": "0 0 100%", "height": "8px"}),

        # Target CTL
        html.Div(style={"flex": "0 0 100%"}, children=[
            html.Div("TARGET CTL", style={**_LABEL, "marginBottom": "8px"}),
        ]),
        _input_group("Base", "target-base", 50, min=0, max=200),
        _input_group("Threshold", "target-thresh", 25, min=0, max=100),
        _input_group("Anaerobic", "target-ana", 12, min=0, max=60),

        # Generate button
        html.Div(style={"flex": "0 0 auto"}, children=[
            html.Button("Generate plan", id="btn-generate", n_clicks=0, style={
                "padding": "9px 24px", "cursor": "pointer", "borderRadius": "4px",
                "border": f"1px solid {ACCENT}", "background": ACCENT, "color": "#fff",
                "fontSize": "14px", "fontWeight": "600", "marginTop": "8px",
            }),
        ]),
    ]),

    # ── Charts ───────────────────────────────────────────────────────────
    dcc.Graph(id="graph-ctl", config={"displayModeBar": False}),
    dcc.Graph(id="graph-tsb", config={"displayModeBar": False}),
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
    Output("graph-weekly", "figure"),
    Output("plan-store", "data"),
    Input("btn-generate", "n_clicks"),
    State("start-base", "value"),
    State("start-thresh", "value"),
    State("start-ana", "value"),
    State("target-base", "value"),
    State("target-thresh", "value"),
    State("target-ana", "value"),
)
def generate_plan(n_clicks, s_base, s_thresh, s_ana, t_base, t_thresh, t_ana):
    s_base = float(s_base or 0)
    s_thresh = float(s_thresh or 0)
    s_ana = float(s_ana or 0)
    t_base = float(t_base or 50)
    t_thresh = float(t_thresh or 25)
    t_ana = float(t_ana or 12)

    # Build starting state: TSB = 0 means ATL = CTL
    state = TrainingState(
        base=ZoneState(ctl=s_base, atl=s_base),
        threshold=ZoneState(ctl=s_thresh, atl=s_thresh),
        anaerobic=ZoneState(ctl=s_ana, atl=s_ana),
    )

    target = ZoneTSS(base=t_base, threshold=t_thresh, anaerobic=t_ana)

    # Fixed 20-week plan: use dummy dates (today + 140 days)
    today = date.today()
    event = today + timedelta(weeks=20)

    plan = simulate_plan(state, target, today, event,
                         base_weeks=8, build_weeks=8, peak_weeks=4)

    fig_ctl = _fig_ctl_projection(plan, target)
    fig_tsb = _fig_tsb_projection(plan)
    fig_weekly = _fig_weekly_volume(plan)

    # Serialize plan for click interactions
    plan_data = [{
        "day": i,
        "week": f"{i // 7 + 1}",
        "phase": d.phase.name,
        "freshness": d.freshness.value,
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
    } for i, d in enumerate(plan)]

    return fig_ctl, fig_tsb, fig_weekly, plan_data


@app.callback(
    Output("day-detail", "children"),
    Input("graph-weekly", "clickData"),
    State("plan-store", "data"),
)
def show_day_detail(click_data, plan_data):
    if not click_data or not plan_data:
        return html.Div("Click a bar on the weekly chart to see week details.",
                        style={"color": TEXT_DIM, "fontSize": "13px"})

    point = click_data["points"][0]
    clicked_week = int(point["x"])

    # Get all days in that week
    week_days = [d for d in plan_data if int(d["week"]) == clicked_week]
    if not week_days:
        return html.Div(f"No data for week {clicked_week}",
                        style={"color": TEXT_DIM})

    total_tss = sum(d["tss_total"] for d in week_days)
    last = week_days[-1]

    return html.Div(style={"display": "flex", "gap": "24px",
                            "alignItems": "center", "flexWrap": "wrap"}, children=[
        html.Div([
            html.Div("WEEK", style=_LABEL),
            html.Div(str(clicked_week), style=_VALUE),
        ]),
        html.Div([
            html.Div("PHASE", style=_LABEL),
            html.Div(week_days[0]["phase"], style={"fontSize": "16px", "fontWeight": "600"}),
        ]),
        html.Div([
            html.Div("WEEKLY TSS", style=_LABEL),
            html.Div(f"{total_tss:.0f}", style=_VALUE),
        ]),
        html.Div([
            html.Div("END-OF-WEEK CTL", style=_LABEL),
            html.Div(style={"display": "flex", "gap": "8px"}, children=[
                html.Span(f"B:{last['ctl_base']:.1f}",
                          style={"color": Z_BASE, "fontWeight": "600"}),
                html.Span(f"T:{last['ctl_threshold']:.1f}",
                          style={"color": Z_THRESH, "fontWeight": "600"}),
                html.Span(f"A:{last['ctl_anaerobic']:.1f}",
                          style={"color": Z_AWC, "fontWeight": "600"}),
            ]),
        ]),
        html.Div([
            html.Div("END-OF-WEEK TSB", style=_LABEL),
            html.Div(style={"display": "flex", "gap": "8px"}, children=[
                html.Span(f"B:{last['tsb_base']:+.1f}",
                          style={"color": Z_BASE}),
                html.Span(f"T:{last['tsb_threshold']:+.1f}",
                          style={"color": Z_THRESH}),
                html.Span(f"A:{last['tsb_anaerobic']:+.1f}",
                          style={"color": Z_AWC}),
            ]),
        ]),
    ])


if __name__ == "__main__":
    app.run(debug=True, port=8051)
