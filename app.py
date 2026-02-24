"""
app.py — Dash application for cycling power analysis.

Usage
─────
  python app.py          # starts on http://127.0.0.1:8050

Auto-watch
──────────
  Dropping a new .fit file into the raw_data/ folder is all that is needed.
  A background watchdog thread detects it, processes it into cycling.db,
  and the UI refreshes automatically within a few seconds.

Pages / sections
────────────────
  • Ride selector dropdown (auto-updates when new rides appear)
  • Power + Heart Rate vs time for the selected ride
  • MMP for the selected ride vs 90-day best
  • All-rides MMP overview
  • Ride summary table
"""

import datetime
import os
import sqlite3
import threading
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import dash
from dash import dcc, html, dash_table, Input, Output, State
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from build_database import (
    init_db, process_ride, backfill_pdc_params,
    _power_model, _fit_power_curve,
    PDC_K, PDC_INFLECTION, PDC_WINDOW,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "cycling.db")
FIT_DIR  = os.path.join(BASE_DIR, "raw_data")

# ── Shared data store (updated by watcher, read by callbacks) ─────────────────

_lock         = threading.Lock()
_data_version = 0   # incremented every time new data lands in the DB
_rides        = None
_mmp_all      = None
_pdc_params   = None


def _reload():
    """Re-read rides, mmp, and pdc_params from the DB and bump the version counter."""
    global _rides, _mmp_all, _pdc_params, _data_version
    r = _load_rides()
    m = _load_mmp_all(r)
    p = _load_pdc_params()
    with _lock:
        _rides        = r
        _mmp_all      = m
        _pdc_params   = p
        _data_version += 1


def get_data():
    """Return a consistent (version, rides, mmp_all, pdc_params) snapshot."""
    with _lock:
        return _data_version, _rides.copy(), _mmp_all.copy(), _pdc_params.copy()


# ── Duration formatting ────────────────────────────────────────────────────────

LOG_TICK_S   = [1, 5, 10, 30, 60, 120, 300, 600, 1200, 1800, 3600]
LOG_TICK_LBL = ["1s", "5s", "10s", "30s", "1min", "2min",
                 "5min", "10min", "20min", "30min", "1h"]

COLOURS = px.colors.qualitative.Light24


# ── Database helpers ───────────────────────────────────────────────────────────

def _load_rides() -> pd.DataFrame:
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


def _load_mmp_all(rides: pd.DataFrame) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    mmp = pd.read_sql("SELECT ride_id, duration_s, power FROM mmp", conn)
    conn.close()
    mmp = mmp.merge(
        rides.rename(columns={"id": "ride_id"})[["ride_id", "name", "ride_date"]],
        on="ride_id",
    )
    return mmp


def _load_pdc_params() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT ride_id, AWC, Pmax, MAP, tau2 FROM pdc_params", conn)
    conn.close()
    return df


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


# ── Watchdog ──────────────────────────────────────────────────────────────────

class _FitHandler(FileSystemEventHandler):
    """Process any new .fit file that lands in FIT_DIR."""

    def _handle(self, path: str) -> None:
        if not path.lower().endswith(".fit"):
            return
        # Small delay to let the file finish copying before we read it
        time.sleep(1)
        print(f"[watcher] New file detected: {path}")
        try:
            conn = sqlite3.connect(DB_PATH)
            init_db(conn)
            process_ride(conn, path)
            conn.close()
            _reload()
            print(f"[watcher] Processed and reloaded data.")
        except Exception as exc:
            print(f"[watcher] Error processing {path}: {exc}")

    def on_created(self, event):
        if not event.is_directory:
            self._handle(event.src_path)

    def on_moved(self, event):
        if not event.is_directory:
            self._handle(event.dest_path)


def _start_watcher() -> None:
    os.makedirs(FIT_DIR, exist_ok=True)
    observer = Observer()
    observer.schedule(_FitHandler(), FIT_DIR, recursive=False)
    observer.daemon = True
    observer.start()
    print(f"[watcher] Watching {FIT_DIR} for new .fit files …")


# ── Figure builders ───────────────────────────────────────────────────────────

def fig_power_hr(records: pd.DataFrame, ride_name: str) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=("Power (W)", "Heart Rate (bpm)"),
        vertical_spacing=0.08,
    )
    if records["power"].notna().any():
        fig.add_trace(
            go.Scatter(x=records["elapsed_min"], y=records["power"],
                       mode="lines", name="Power",
                       line=dict(color="darkorange", width=1)),
            row=1, col=1,
        )
    if records["heart_rate"].notna().any():
        fig.add_trace(
            go.Scatter(x=records["elapsed_min"], y=records["heart_rate"],
                       mode="lines", name="Heart Rate",
                       line=dict(color="crimson", width=1)),
            row=2, col=1,
        )
    fig.update_xaxes(title_text="Elapsed Time (min)", row=2, col=1)
    fig.update_layout(
        title=dict(text=ride_name.replace("_", " "), font=dict(size=15)),
        height=420, margin=dict(t=70, b=40, l=60, r=20),
        showlegend=False, template="plotly_white",
    )
    return fig


def fig_mmp_vs_90day(ride: pd.Series, mmp_all: pd.DataFrame) -> go.Figure:
    ride_date = ride["ride_date"]
    cutoff    = (pd.to_datetime(ride_date) - pd.Timedelta(days=90)).date().isoformat()

    window = mmp_all[
        mmp_all["ride_date"].between(cutoff, ride_date)
        & (mmp_all["ride_id"] != ride["id"])
    ]
    best_90   = window.groupby("duration_s")["power"].max().reset_index().sort_values("duration_s")
    this_ride = mmp_all[mmp_all["ride_id"] == ride["id"]].sort_values("duration_s")

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
        marker=dict(size=5), line=dict(color="tomato", width=2.2),
    ))
    fig.update_xaxes(type="log", tickvals=LOG_TICK_S, ticktext=LOG_TICK_LBL,
                     title_text="Duration", showgrid=True, gridcolor="lightgrey")
    fig.update_yaxes(title_text="Power (W)", showgrid=True, gridcolor="lightgrey")
    fig.update_layout(
        title=dict(
            text=f"MMP — {ride['name'].replace('_', ' ')}<br>"
                 f"<sup>vs 90-day best &nbsp;({cutoff} → {ride_date})</sup>",
            font=dict(size=14),
        ),
        height=380, margin=dict(t=80, b=50, l=60, r=20),
        template="plotly_white",
        legend=dict(x=0.98, xanchor="right", y=0.98),
    )
    return fig


def fig_all_mmp(mmp_all: pd.DataFrame, rides: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for i, (_, ride) in enumerate(rides.iterrows()):
        rd    = mmp_all[mmp_all["ride_id"] == ride["id"]].sort_values("duration_s")
        label = f"{ride['ride_date']}  {ride['name'][:28].replace('_', ' ')}"
        fig.add_trace(go.Scatter(
            x=rd["duration_s"], y=rd["power"],
            mode="lines+markers", name=label,
            marker=dict(size=3),
            line=dict(width=1.8, color=COLOURS[i % len(COLOURS)]),
        ))
    fig.update_xaxes(type="log", tickvals=LOG_TICK_S, ticktext=LOG_TICK_LBL,
                     title_text="Duration", showgrid=True, gridcolor="lightgrey")
    fig.update_yaxes(title_text="Power (W)", showgrid=True, gridcolor="lightgrey")
    fig.update_layout(
        title="Mean Maximal Power — all rides",
        height=480, margin=dict(t=60, b=50, l=60, r=200),
        template="plotly_white",
        legend=dict(font=dict(size=10), x=1.01, xanchor="left"),
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
        raw = (
            window.groupby("duration_s")["power"]
            .max().reset_index().sort_values("duration_s")
        )

        # Raw hard-max as a faint reference
        fig.add_trace(go.Scatter(
            x=raw["duration_s"], y=raw["power"],
            mode="lines", name="raw max",
            line=dict(color="lightsteelblue", width=1.5, dash="dot"),
        ))
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
                name=f"model  AWC={AWC/1000:.1f} kJ  MAP={MAP:.0f} W",
                fill="tonexty", fillcolor="rgba(220,80,30,0.18)",
                line=dict(color="darkorange", width=2, dash="dash"),
            ))
            fig.add_annotation(
                xref="paper", yref="paper",
                x=0.02, y=0.08,
                text=(
                    f"AWC = {AWC/1000:.1f} kJ   Pmax = {Pmax:.0f} W<br>"
                    f"MAP = {MAP:.0f} W   τ₂ = {tau2:.0f} s   (τ = {tau:.0f} s)"
                ),
                showarrow=False, align="left",
                bgcolor="white", bordercolor="#bbb", borderwidth=1,
                font=dict(size=11),
            )

    fig.update_xaxes(type="log", tickvals=LOG_TICK_S, ticktext=LOG_TICK_LBL,
                     title_text="Duration", showgrid=True, gridcolor="lightgrey")
    fig.update_yaxes(title_text="Power (W)", showgrid=True, gridcolor="lightgrey")
    fig.update_layout(
        title=dict(
            text=(
                "90-Day Mean Maximal Power — S-curve aged<br>"
                f"<sup>Inflection {PDC_INFLECTION} days · K={PDC_K} · "
                f"reference date {today_str}</sup>"
            ),
            font=dict(size=14),
        ),
        height=440, margin=dict(t=90, b=50, l=60, r=20),
        template="plotly_white",
        legend=dict(x=0.98, xanchor="right", y=0.98),
    )
    return fig


def fig_pdc_at_date(ride: pd.Series, mmp_all: pd.DataFrame,
                    pdc_params: pd.DataFrame) -> go.Figure:
    """Power Duration Curve built from sigmoid-decayed MMP up to this ride.

    Shows the raw-max reference, the decayed MMP envelope, and the fitted
    two-component model with its key parameters annotated.
    """
    ride_date     = ride["ride_date"]
    ride_date_obj = datetime.date.fromisoformat(ride_date)
    cutoff        = (ride_date_obj - datetime.timedelta(days=PDC_WINDOW)).isoformat()

    window = mmp_all[mmp_all["ride_date"].between(cutoff, ride_date)].copy()

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
        raw = (
            window.groupby("duration_s")["power"]
            .max().reset_index().sort_values("duration_s")
        )

        fig.add_trace(go.Scatter(
            x=raw["duration_s"], y=raw["power"],
            mode="lines", name="raw max",
            line=dict(color="lightsteelblue", width=1.5, dash="dot"),
        ))
        fig.add_trace(go.Scatter(
            x=aged["duration_s"], y=aged["aged_power"],
            mode="lines+markers", name="decayed MMP",
            marker=dict(size=4),
            line=dict(color="steelblue", width=2.5),
        ))

        params_row = pdc_params[pdc_params["ride_id"] == ride["id"]]
        if not params_row.empty:
            r    = params_row.iloc[0]
            AWC, Pmax, MAP, tau2 = r["AWC"], r["Pmax"], r["MAP"], r["tau2"]
            tau  = AWC / Pmax
            dur  = aged["duration_s"].to_numpy(dtype=float)
            t_sm = np.logspace(np.log10(dur.min()), np.log10(dur.max()), 400)
            p_aerobic = MAP * (1.0 - np.exp(-t_sm / tau2))
            p_total   = _power_model(t_sm, AWC, Pmax, MAP, tau2)
            # Aerobic component — fills from zero
            fig.add_trace(go.Scatter(
                x=t_sm, y=p_aerobic,
                mode="lines", name="aerobic (MAP)",
                fill="tozeroy", fillcolor="rgba(46,139,87,0.20)",
                line=dict(color="rgba(46,139,87,0.7)", width=1.2),
            ))
            # Total model — fills from aerobic line, shaded band = anaerobic
            fig.add_trace(go.Scatter(
                x=t_sm, y=p_total,
                mode="lines",
                name=f"model  AWC={AWC/1000:.1f} kJ  MAP={MAP:.0f} W",
                fill="tonexty", fillcolor="rgba(220,80,30,0.18)",
                line=dict(color="darkorange", width=2, dash="dash"),
            ))
            fig.add_annotation(
                xref="paper", yref="paper",
                x=0.02, y=0.08,
                text=(
                    f"AWC = {AWC/1000:.1f} kJ   Pmax = {Pmax:.0f} W<br>"
                    f"MAP = {MAP:.0f} W   τ₂ = {tau2:.0f} s   (τ = {tau:.0f} s)"
                ),
                showarrow=False, align="left",
                bgcolor="white", bordercolor="#bbb", borderwidth=1,
                font=dict(size=11),
            )

    fig.update_xaxes(type="log", tickvals=LOG_TICK_S, ticktext=LOG_TICK_LBL,
                     title_text="Duration", showgrid=True, gridcolor="lightgrey")
    fig.update_yaxes(title_text="Power (W)", showgrid=True, gridcolor="lightgrey")
    fig.update_layout(
        title=dict(
            text=(
                f"Power Duration Curve — {ride['name'].replace('_', ' ')}<br>"
                f"<sup>Decayed MMP at {ride_date} "
                f"(inflection {PDC_INFLECTION} days · K={PDC_K})</sup>"
            ),
            font=dict(size=14),
        ),
        height=440, margin=dict(t=90, b=50, l=60, r=20),
        template="plotly_white",
        legend=dict(x=0.98, xanchor="right", y=0.98),
    )
    return fig


# ── App ───────────────────────────────────────────────────────────────────────

_boot_conn = sqlite3.connect(DB_PATH)
init_db(_boot_conn)
backfill_pdc_params(_boot_conn)
_boot_conn.close()

_reload()          # initial load (picks up freshly computed pdc_params)
_start_watcher()   # background thread

app = dash.Dash(__name__, title="Cycling Power Analysis")

app.layout = html.Div(
    style={"fontFamily": "sans-serif", "maxWidth": "1200px", "margin": "0 auto", "padding": "20px"},
    children=[
        # Hidden stores / ticker
        dcc.Store(id="known-version", data=0),
        dcc.Interval(id="poll-interval", interval=3000, n_intervals=0),  # check every 3 s

        html.H1("Cycling Power Analysis", style={"marginBottom": "4px"}),

        # Status bar
        html.Div(id="status-bar", style={
            "fontSize": "12px", "color": "#888", "marginBottom": "12px",
        }),

        dcc.Tabs(
            id="tabs",
            value="tab-activities",
            children=[

                # ── Activities tab ─────────────────────────────────────────
                dcc.Tab(label="Activities", value="tab-activities", children=[
                    html.Div(style={"paddingTop": "20px"}, children=[

                        # Ride selector
                        html.Div([
                            html.Label("Select a ride:", style={"fontWeight": "bold", "marginRight": "10px"}),
                            dcc.Dropdown(
                                id="ride-dropdown",
                                clearable=False,
                                style={"width": "600px", "display": "inline-block", "verticalAlign": "middle"},
                            ),
                        ], style={"marginBottom": "20px"}),

                        # Per-ride charts
                        dcc.Graph(id="graph-power-hr"),
                        dcc.Graph(id="graph-mmp-vs-90"),
                        dcc.Graph(id="graph-pdc"),

                        html.Hr(),

                        # Summary table
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
                            sort_action="native",
                            style_table={"overflowX": "auto"},
                            style_cell={"textAlign": "left", "padding": "6px 12px", "fontFamily": "sans-serif"},
                            style_header={"backgroundColor": "#f0f0f0", "fontWeight": "bold"},
                            style_data_conditional=[
                                {"if": {"row_index": "odd"}, "backgroundColor": "#fafafa"},
                            ],
                        ),

                        html.Div(style={"height": "40px"}),
                    ]),
                ]),

                # ── Fitness tab ────────────────────────────────────────────
                dcc.Tab(label="Fitness", value="tab-fitness", children=[
                    html.Div(style={"paddingTop": "20px"}, children=[

                        dcc.Graph(id="graph-90day-mmp"),

                        html.Hr(),

                        html.H2("Mean Maximal Power — all rides", style={"marginBottom": "4px"}),
                        dcc.Graph(id="graph-all-mmp"),

                        html.Div(style={"height": "40px"}),
                    ]),
                ]),
            ],
        ),
    ],
)


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("known-version",   "data"),
    Output("ride-dropdown",   "options"),
    Output("ride-dropdown",   "value"),
    Output("graph-90day-mmp", "figure"),
    Output("graph-all-mmp",   "figure"),
    Output("summary-table",   "data"),
    Output("status-bar",      "children"),
    Input("poll-interval",    "n_intervals"),
    State("known-version",    "data"),
    State("ride-dropdown",    "value"),
)
def poll_for_new_data(n_intervals, known_ver, current_ride_id):
    ver, rides, mmp_all, _pdc = get_data()

    if ver == known_ver and n_intervals > 0:
        # Nothing changed — return no-update for everything
        raise dash.exceptions.PreventUpdate

    options = [
        {"label": f"{r['ride_date']}  {r['name'].replace('_', ' ')}", "value": r["id"]}
        for _, r in rides.iterrows()
    ]
    # Keep the currently selected ride if it still exists, else pick the latest
    if current_ride_id in rides["id"].values:
        selected = current_ride_id
    else:
        selected = int(rides.iloc[-1]["id"])

    ride_count = len(rides)
    status = (
        f"Watching {FIT_DIR} for new .fit files — "
        f"{ride_count} ride{'s' if ride_count != 1 else ''} loaded"
    )

    return (
        ver,
        options,
        selected,
        fig_90day_mmp(mmp_all),
        fig_all_mmp(mmp_all, rides),
        rides.to_dict("records"),
        status,
    )


@app.callback(
    Output("graph-power-hr",  "figure"),
    Output("graph-mmp-vs-90", "figure"),
    Output("graph-pdc",       "figure"),
    Input("ride-dropdown",    "value"),
    State("known-version",    "data"),
)
def update_ride_charts(ride_id, _ver):
    if ride_id is None:
        raise dash.exceptions.PreventUpdate
    _, rides, mmp_all, pdc_params = get_data()
    ride    = rides[rides["id"] == ride_id].iloc[0]
    records = load_records(ride_id)
    return (
        fig_power_hr(records, ride["name"]),
        fig_mmp_vs_90day(ride, mmp_all),
        fig_pdc_at_date(ride, mmp_all, pdc_params),
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
