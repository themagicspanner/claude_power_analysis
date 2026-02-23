"""
app.py — Dash application for cycling power analysis.

Usage
─────
  python app.py          # starts on http://127.0.0.1:8050

Pages / sections
────────────────
  • Ride selector dropdown
  • Power + Heart Rate vs time for the selected ride
  • MMP for the selected ride vs 90-day best
  • All-rides MMP overview
  • Ride summary table
"""

import os
import sqlite3

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, dash_table, Input, Output

BASE_DIR = os.path.dirname(__file__)
DB_PATH  = os.path.join(BASE_DIR, "cycling.db")

# ── Duration formatting ────────────────────────────────────────────────────────

def fmt_dur(s: int) -> str:
    s = int(s)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        m, r = divmod(s, 60)
        return f"{m}min" if r == 0 else f"{m}:{r:02d}"
    h, r = divmod(s, 3600)
    return f"{h}h" if r == 0 else f"{h}h{r // 60}min"


LOG_TICK_S   = [1, 5, 10, 30, 60, 120, 300, 600, 1200, 1800, 3600]
LOG_TICK_LBL = ["1s", "5s", "10s", "30s", "1min", "2min",
                 "5min", "10min", "20min", "30min", "1h"]


# ── Data helpers ───────────────────────────────────────────────────────────────

def load_rides() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        """SELECT id, name, ride_date,
                  round(duration_s / 60.0, 1) AS duration_min,
                  avg_power, max_power, avg_hr, max_hr
           FROM rides ORDER BY ride_date, name""",
        conn,
    )
    conn.close()
    return df


def load_mmp_all(rides: pd.DataFrame) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    mmp = pd.read_sql("SELECT ride_id, duration_s, power FROM mmp", conn)
    conn.close()
    mmp = mmp.merge(rides.rename(columns={"id": "ride_id"})[["ride_id", "name", "ride_date"]],
                    on="ride_id")
    return mmp


def load_records(ride_id: int) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT elapsed_s, power, heart_rate FROM records WHERE ride_id = ? ORDER BY elapsed_s",
        conn,
        params=(ride_id,),
    )
    conn.close()
    df["elapsed_min"] = df["elapsed_s"] / 60.0
    return df


# ── Figures ───────────────────────────────────────────────────────────────────

def fig_power_hr(records: pd.DataFrame, ride_name: str) -> go.Figure:
    """Power and heart-rate vs elapsed time for a single ride."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=("Power (W)", "Heart Rate (bpm)"),
        vertical_spacing=0.08,
    )

    if records["power"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=records["elapsed_min"], y=records["power"],
                mode="lines", name="Power",
                line=dict(color="darkorange", width=1),
            ),
            row=1, col=1,
        )

    if records["heart_rate"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=records["elapsed_min"], y=records["heart_rate"],
                mode="lines", name="Heart Rate",
                line=dict(color="crimson", width=1),
            ),
            row=2, col=1,
        )

    fig.update_xaxes(title_text="Elapsed Time (min)", row=2, col=1)
    fig.update_layout(
        title=dict(text=ride_name.replace("_", " "), font=dict(size=15)),
        height=420,
        margin=dict(t=70, b=40, l=60, r=20),
        showlegend=False,
        template="plotly_white",
    )
    return fig


def fig_mmp_vs_90day(ride: pd.Series, mmp_all: pd.DataFrame) -> go.Figure:
    """MMP of a single ride vs the 90-day rolling best (excluding the ride itself)."""
    ride_date = ride["ride_date"]
    cutoff    = (pd.to_datetime(ride_date) - pd.Timedelta(days=90)).date().isoformat()

    window = mmp_all[
        mmp_all["ride_date"].between(cutoff, ride_date)
        & (mmp_all["ride_id"] != ride["id"])
    ]
    best_90 = (
        window.groupby("duration_s")["power"]
        .max()
        .reset_index()
        .sort_values("duration_s")
    )
    this_ride = (
        mmp_all[mmp_all["ride_id"] == ride["id"]]
        .sort_values("duration_s")
    )

    fig = go.Figure()

    if not best_90.empty:
        fig.add_trace(go.Scatter(
            x=best_90["duration_s"], y=best_90["power"],
            mode="lines", name="90-day best",
            fill="tozeroy", fillcolor="rgba(70,130,180,0.12)",
            line=dict(color="steelblue", width=2, dash="dash"),
        ))

    fig.add_trace(go.Scatter(
        x=this_ride["duration_s"], y=this_ride["power"],
        mode="lines+markers", name=f"This ride ({ride_date})",
        marker=dict(size=5),
        line=dict(color="tomato", width=2.2),
    ))

    fig.update_xaxes(
        type="log",
        tickvals=LOG_TICK_S,
        ticktext=LOG_TICK_LBL,
        title_text="Duration",
        showgrid=True, gridcolor="lightgrey",
    )
    fig.update_yaxes(title_text="Power (W)", showgrid=True, gridcolor="lightgrey")
    fig.update_layout(
        title=dict(
            text=f"MMP — {ride['name'].replace('_', ' ')}<br>"
                 f"<sup>vs 90-day best &nbsp;({cutoff} → {ride_date})</sup>",
            font=dict(size=14),
        ),
        height=380,
        margin=dict(t=80, b=50, l=60, r=20),
        template="plotly_white",
        legend=dict(x=0.98, xanchor="right", y=0.98),
    )
    return fig


def fig_all_mmp(mmp_all: pd.DataFrame, rides: pd.DataFrame) -> go.Figure:
    """All rides MMP on a single log-scale chart."""
    import plotly.express as px

    colours = px.colors.qualitative.Light24
    fig = go.Figure()
    for i, (_, ride) in enumerate(rides.iterrows()):
        rd = mmp_all[mmp_all["ride_id"] == ride["id"]].sort_values("duration_s")
        label = f"{ride['ride_date']}  {ride['name'][:28].replace('_', ' ')}"
        fig.add_trace(go.Scatter(
            x=rd["duration_s"], y=rd["power"],
            mode="lines+markers",
            name=label,
            marker=dict(size=3),
            line=dict(width=1.8, color=colours[i % len(colours)]),
        ))

    fig.update_xaxes(
        type="log",
        tickvals=LOG_TICK_S,
        ticktext=LOG_TICK_LBL,
        title_text="Duration",
        showgrid=True, gridcolor="lightgrey",
    )
    fig.update_yaxes(title_text="Power (W)", showgrid=True, gridcolor="lightgrey")
    fig.update_layout(
        title="Mean Maximal Power — all rides",
        height=480,
        margin=dict(t=60, b=50, l=60, r=200),
        template="plotly_white",
        legend=dict(font=dict(size=10), x=1.01, xanchor="left"),
    )
    return fig


# ── Layout ────────────────────────────────────────────────────────────────────

rides   = load_rides()
mmp_all = load_mmp_all(rides)

app = dash.Dash(__name__, title="Cycling Power Analysis")

app.layout = html.Div(
    style={"fontFamily": "sans-serif", "maxWidth": "1200px", "margin": "0 auto", "padding": "20px"},
    children=[
        html.H1("Cycling Power Analysis", style={"marginBottom": "4px"}),
        html.Hr(),

        # ── Ride selector ──────────────────────────────────────────────────
        html.Div([
            html.Label("Select a ride:", style={"fontWeight": "bold", "marginRight": "10px"}),
            dcc.Dropdown(
                id="ride-dropdown",
                options=[
                    {"label": f"{r['ride_date']}  {r['name'].replace('_', ' ')}", "value": r["id"]}
                    for _, r in rides.iterrows()
                ],
                value=int(rides.iloc[-1]["id"]),
                clearable=False,
                style={"width": "600px", "display": "inline-block", "verticalAlign": "middle"},
            ),
        ], style={"marginBottom": "20px"}),

        # ── Per-ride charts ────────────────────────────────────────────────
        dcc.Graph(id="graph-power-hr"),
        dcc.Graph(id="graph-mmp-vs-90"),

        html.Hr(),

        # ── All-rides overview ─────────────────────────────────────────────
        html.H2("All Rides — MMP Overview", style={"marginBottom": "4px"}),
        dcc.Graph(id="graph-all-mmp", figure=fig_all_mmp(mmp_all, rides)),

        html.Hr(),

        # ── Summary table ──────────────────────────────────────────────────
        html.H2("Ride Summary", style={"marginBottom": "8px"}),
        dash_table.DataTable(
            id="summary-table",
            columns=[
                {"name": "Name",           "id": "name"},
                {"name": "Date",           "id": "ride_date"},
                {"name": "Duration (min)", "id": "duration_min"},
                {"name": "Avg Power (W)",  "id": "avg_power"},
                {"name": "Max Power (W)",  "id": "max_power"},
                {"name": "Avg HR (bpm)",   "id": "avg_hr"},
                {"name": "Max HR (bpm)",   "id": "max_hr"},
            ],
            data=rides.rename(columns={"name": "name"}).to_dict("records"),
            sort_action="native",
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "padding": "6px 12px", "fontFamily": "sans-serif"},
            style_header={"backgroundColor": "#f0f0f0", "fontWeight": "bold"},
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "#fafafa"},
            ],
        ),

        html.Div(style={"height": "40px"}),
    ],
)


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("graph-power-hr",  "figure"),
    Output("graph-mmp-vs-90", "figure"),
    Input("ride-dropdown", "value"),
)
def update_ride_charts(ride_id: int):
    ride    = rides[rides["id"] == ride_id].iloc[0]
    records = load_records(ride_id)
    return (
        fig_power_hr(records, ride["name"]),
        fig_mmp_vs_90day(ride, mmp_all),
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
