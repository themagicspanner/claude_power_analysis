"""
build_database.py

Parse all FIT files in 2026/, store records and mean-maximal-power (MMP)
curves in a SQLite database (cycling.db), then print a summary table.

Database schema
───────────────
  rides   – one row per ride with summary stats
  records – raw 1-Hz timestamp / power / heart_rate rows
  mmp     – best average power (W) for every (ride, duration) pair

Usage
─────
  python build_database.py          # build / update the database
  python build_database.py --show   # just print the MMP table
"""

import argparse
import glob
import os
import sqlite3

import fitdecode
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
FIT_DIR  = os.path.join(BASE_DIR, "2026")
DB_PATH  = os.path.join(BASE_DIR, "cycling.db")

# Standard MMP durations in seconds
MMP_DURATIONS = [
    1, 2, 3, 5, 8, 10, 12, 15, 20, 30,
    60, 90, 120, 180, 240, 300, 420, 600,
    900, 1200, 1800, 2400, 3600,
]


# ── Database ──────────────────────────────────────────────────────────────────

def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS rides (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            name          TEXT    UNIQUE NOT NULL,
            ride_date     TEXT,
            total_records INTEGER,
            duration_s    REAL,
            avg_power     REAL,
            max_power     INTEGER,
            avg_hr        REAL,
            max_hr        INTEGER
        );

        CREATE TABLE IF NOT EXISTS records (
            ride_id    INTEGER NOT NULL REFERENCES rides(id),
            timestamp  TEXT    NOT NULL,
            elapsed_s  REAL    NOT NULL,
            power      INTEGER,
            heart_rate INTEGER
        );

        CREATE INDEX IF NOT EXISTS idx_records_ride
            ON records(ride_id);

        CREATE TABLE IF NOT EXISTS mmp (
            ride_id    INTEGER NOT NULL REFERENCES rides(id),
            duration_s INTEGER NOT NULL,
            power      REAL    NOT NULL,
            PRIMARY KEY (ride_id, duration_s)
        );
    """)
    conn.commit()


# ── FIT parsing ───────────────────────────────────────────────────────────────

def read_fit(path: str) -> pd.DataFrame:
    """Return DataFrame with columns: timestamp, elapsed_s, power, heart_rate."""
    rows = []
    with fitdecode.FitReader(path) as fit:
        for frame in fit:
            if not isinstance(frame, fitdecode.FitDataMessage) or frame.name != "record":
                continue
            rows.append({
                "timestamp":  frame.get_value("timestamp",  fallback=None),
                "power":      frame.get_value("power",      fallback=None),
                "heart_rate": frame.get_value("heart_rate", fallback=None),
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["elapsed_s"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
    return df


# ── MMP calculation ───────────────────────────────────────────────────────────

def calculate_mmp(df: pd.DataFrame, durations: list[int]) -> dict[int, float]:
    """
    Return {duration_s: best_avg_power} for each requested duration.

    The records are already 1-second apart (verified).  For any duration d
    the MMP is simply the maximum of the d-sample rolling mean of the power
    series.  NaN power values (sensor dropout) are filled with 0 W, which
    is the convention used by most cycling analysis software.
    """
    if df.empty or df["power"].isna().all():
        return {}

    power = df["power"].fillna(0).to_numpy(dtype=float)
    n = len(power)

    # Build a cumulative-sum array for O(1) window sums
    cumsum = power.cumsum()

    result: dict[int, float] = {}
    for d in durations:
        if n < d:
            continue
        # window sums: sum[i..i+d-1] = cumsum[i+d-1] - cumsum[i-1]
        window_sums = cumsum[d - 1:].copy()
        window_sums[1:] -= cumsum[:n - d]
        result[d] = float(window_sums.max() / d)

    return result


# ── Per-ride processing ───────────────────────────────────────────────────────

def process_ride(conn: sqlite3.Connection, path: str) -> None:
    name = os.path.splitext(os.path.basename(path))[0]

    if conn.execute("SELECT 1 FROM rides WHERE name = ?", (name,)).fetchone():
        print(f"  {name}: already in database, skipping.")
        return

    df = read_fit(path)
    if df.empty:
        print(f"  {name}: no record data, skipping.")
        return

    # Ride metadata
    ride_date = df["timestamp"].iloc[0].date().isoformat()
    duration  = float(df["elapsed_s"].iloc[-1])
    has_power = df["power"].notna().any()
    has_hr    = df["heart_rate"].notna().any()

    cur = conn.execute(
        """INSERT INTO rides
               (name, ride_date, total_records, duration_s,
                avg_power, max_power, avg_hr, max_hr)
           VALUES (?,?,?,?,?,?,?,?)""",
        (
            name, ride_date, len(df), duration,
            round(float(df["power"].mean()),      1) if has_power else None,
            int(df["power"].max())                   if has_power else None,
            round(float(df["heart_rate"].mean()), 1) if has_hr    else None,
            int(df["heart_rate"].max())              if has_hr    else None,
        ),
    )
    ride_id = cur.lastrowid

    # Raw records
    conn.executemany(
        "INSERT INTO records (ride_id, timestamp, elapsed_s, power, heart_rate) "
        "VALUES (?,?,?,?,?)",
        (
            (
                ride_id,
                row.timestamp.isoformat(),
                row.elapsed_s,
                int(row.power)      if pd.notna(row.power)      else None,
                int(row.heart_rate) if pd.notna(row.heart_rate) else None,
            )
            for row in df.itertuples()
        ),
    )

    # MMP
    mmp = calculate_mmp(df, MMP_DURATIONS)
    conn.executemany(
        "INSERT INTO mmp (ride_id, duration_s, power) VALUES (?,?,?)",
        [(ride_id, d, round(p, 1)) for d, p in mmp.items()],
    )

    conn.commit()
    print(f"  {name}: {len(df)} records, {len(mmp)} MMP points stored.")


# ── Display helpers ───────────────────────────────────────────────────────────

def _fmt_duration(s: int) -> str:
    if s < 60:
        return f"{s}s"
    if s < 3600:
        m, rem = divmod(s, 60)
        return f"{m}min" if rem == 0 else f"{m}:{rem:02d}"
    h, rem = divmod(s, 3600)
    return f"{h}h" if rem == 0 else f"{h}h{rem // 60}min"


def print_mmp_table(db_path: str) -> None:
    """Print MMP pivot table: rows = durations, columns = rides."""
    conn = sqlite3.connect(db_path)
    rides = pd.read_sql(
        "SELECT id, name, ride_date FROM rides ORDER BY ride_date, name", conn
    )
    mmp = pd.read_sql(
        "SELECT ride_id, duration_s, power FROM mmp ORDER BY duration_s", conn
    )
    conn.close()

    if rides.empty:
        print("No rides in database yet.")
        return

    mmp = mmp.merge(rides.rename(columns={"id": "ride_id"}), on="ride_id")
    pivot = mmp.pivot(index="duration_s", columns="name", values="power")
    pivot.index = [_fmt_duration(int(d)) for d in pivot.index]
    pivot.columns.name = None

    # Truncate column names to keep the table readable
    pivot.columns = [c[:20] for c in pivot.columns]
    pivot = pivot.round(0).astype("Int64")

    print("\n── Mean Maximal Power per ride (W) ─────────────────────────────────")
    print(pivot.to_string())
    print()

    print("── Ride summary ────────────────────────────────────────────────────")
    conn = sqlite3.connect(db_path)
    summary = pd.read_sql(
        """SELECT name, ride_date,
                  duration_s / 60.0 AS duration_min,
                  avg_power, max_power, avg_hr, max_hr
           FROM rides ORDER BY ride_date, name""",
        conn,
    )
    conn.close()
    summary["duration_min"] = summary["duration_min"].round(1)
    summary["avg_power"]    = summary["avg_power"].round(1)
    print(summary.to_string(index=False))
    print()


# ── Chart ─────────────────────────────────────────────────────────────────────

def _apply_mmp_axes(ax: plt.Axes) -> None:
    """Apply shared log-scale x-axis formatting to an MMP axes."""
    tick_s   = [1, 5, 10, 30, 60, 120, 300, 600, 1200, 1800, 3600]
    tick_lbl = ["1s", "5s", "10s", "30s", "1min", "2min",
                "5min", "10min", "20min", "30min", "1h"]
    ax.set_xscale("log")
    ax.set_xticks(tick_s)
    ax.set_xticklabels(tick_lbl, fontsize=9)
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlabel("Duration", fontsize=12)
    ax.set_ylabel("Power (W)", fontsize=12)
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.grid(True, which="minor", linestyle=":",  linewidth=0.3, alpha=0.4)


def plot_ride_vs_90day_mmp(
    ride: pd.Series,
    mmp_all: pd.DataFrame,
    out_dir: str,
) -> None:
    """
    For a single ride, generate a chart comparing its MMP to the 90-day best
    MMP up to and including that ride's date.  Where the ride curve touches
    the 90-day curve the ride matched (or set) the rolling 90-day best.
    """
    ride_date = ride["ride_date"]
    cutoff    = (pd.to_datetime(ride_date) - pd.Timedelta(days=90)).date().isoformat()

    # Best at each duration across all rides in the 90-day window, excluding this ride
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

    fig, ax = plt.subplots(figsize=(11, 6))

    # 90-day envelope
    ax.fill_between(
        best_90["duration_s"], best_90["power"],
        alpha=0.12, color="steelblue", zorder=1,
    )
    ax.plot(
        best_90["duration_s"], best_90["power"],
        color="steelblue", linewidth=2.0, linestyle="--",
        label="90-day best", zorder=2,
    )

    # This ride
    ax.plot(
        this_ride["duration_s"], this_ride["power"],
        color="tomato", linewidth=2.2, marker="o", markersize=4,
        label=f"This ride  ({ride_date})", zorder=3,
    )

    _apply_mmp_axes(ax)
    ax.set_title(
        f"MMP — {ride['name']}\nvs 90-day best (window: {cutoff} → {ride_date})",
        fontsize=13,
    )
    ax.legend(fontsize=10, loc="upper right", framealpha=0.9)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{ride['name']}_mmp.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  {ride['name']} → {out_path}")


def plot_all_rides_vs_90day_mmp(db_path: str) -> None:
    """Generate per-ride MMP-vs-90-day-best charts for every ride in the DB."""
    conn  = sqlite3.connect(db_path)
    rides = pd.read_sql(
        "SELECT id, name, ride_date FROM rides ORDER BY ride_date, name", conn
    )
    mmp = pd.read_sql("SELECT ride_id, duration_s, power FROM mmp", conn)
    conn.close()

    if rides.empty:
        print("No rides in database.")
        return

    # Attach ride metadata to mmp rows for easy filtering
    mmp = mmp.merge(rides.rename(columns={"id": "ride_id"}), on="ride_id")

    out_dir = os.path.join(BASE_DIR, "plots")
    print(f"Generating per-ride MMP charts → {out_dir}/\n")
    for _, ride in rides.iterrows():
        plot_ride_vs_90day_mmp(ride, mmp, out_dir)
    print(f"\nDone — {len(rides)} chart(s) saved.")


def plot_mmp(db_path: str, out_path: str = "mmp_chart.png") -> None:
    """Plot MMP curves for every ride on a single log-scale chart."""
    conn = sqlite3.connect(db_path)
    rides = pd.read_sql(
        "SELECT id, name, ride_date FROM rides ORDER BY ride_date, name", conn
    )
    mmp = pd.read_sql("SELECT ride_id, duration_s, power FROM mmp", conn)
    conn.close()

    mmp = mmp.merge(rides.rename(columns={"id": "ride_id"}), on="ride_id")

    fig, ax = plt.subplots(figsize=(13, 7))

    colours = plt.cm.tab20(np.linspace(0, 1, len(rides)))
    for (_, ride), colour in zip(rides.iterrows(), colours):
        ride_data = mmp[mmp["ride_id"] == ride["id"]].sort_values("duration_s")
        ax.plot(
            ride_data["duration_s"],
            ride_data["power"],
            marker="o",
            markersize=3,
            linewidth=1.8,
            color=colour,
            label=f"{ride['ride_date']}  {ride['name'][:28]}",
        )

    _apply_mmp_axes(ax)
    ax.set_title("Mean Maximal Power curves — all rides", fontsize=14)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Chart saved → {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build cycling SQLite database from FIT files.")
    parser.add_argument("--show",       action="store_true", help="Print summary tables only, no processing.")
    parser.add_argument("--plot",       action="store_true", help="Save MMP chart to mmp_chart.png.")
    parser.add_argument("--plot-rides", action="store_true", help="Save per-ride MMP vs 90-day-best charts to plots/.")
    args = parser.parse_args()

    if args.show:
        print_mmp_table(DB_PATH)
        return

    if args.plot:
        plot_mmp(DB_PATH)
        return

    if args.plot_rides:
        plot_all_rides_vs_90day_mmp(DB_PATH)
        return

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    fit_files = sorted(glob.glob(os.path.join(FIT_DIR, "*.fit")))
    if not fit_files:
        print(f"No .fit files found in {FIT_DIR}")
        conn.close()
        return

    print(f"Processing {len(fit_files)} FIT file(s) → {DB_PATH}\n")
    for path in fit_files:
        process_ride(conn, path)

    conn.close()
    print_mmp_table(DB_PATH)


if __name__ == "__main__":
    main()
