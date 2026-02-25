"""
build_database.py

Parse all FIT files in raw_data/, store records and mean-maximal-power (MMP)
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
import datetime
import glob
import os
import sqlite3

import fitdecode
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

BASE_DIR = os.path.dirname(__file__)
FIT_DIR  = os.path.join(BASE_DIR, "raw_data")
DB_PATH  = os.path.join(BASE_DIR, "cycling.db")

# Sigmoid aging constants for decayed MMP / PDC fitting
PDC_K          = 0.15   # steepness of the S-curve
PDC_INFLECTION = 97     # days to midpoint (weight = 0.5)
PDC_WINDOW     = 150    # days of history to include

# Standard MMP durations in seconds
MMP_DURATIONS = [
    1, 2, 3, 5, 8, 10, 12, 15, 20, 30,
    60, 90, 120, 180, 240, 300, 420, 600,
    900, 1200, 1800, 2400, 3600,
]


# ── Power-duration model ──────────────────────────────────────────────────────

def _normalized_power(power: np.ndarray, sample_hz: float = 1.0) -> float:
    """Coggan normalized power — 4th-root of the mean 4th-power of the
    30-second rolling average. NaN samples are treated as 0 W."""
    p = np.where(np.isnan(power), 0.0, power.astype(float))
    window = max(1, int(30 * sample_hz))
    if len(p) < window:
        return float(np.mean(p))
    kernel  = np.ones(window) / window
    rolling = np.convolve(p, kernel, mode="valid")
    return float(np.mean(rolling ** 4) ** 0.25)


def _power_model(t, AWC, Pmax, MAP, tau2):
    """Two-component power-duration model.

    P(t) = AWC/t * (1 - exp(-t/tau))  +  MAP * (1 - exp(-t/tau2))

    where tau = AWC/Pmax  (Pmax is the instantaneous power limit as t → 0).
    """
    tau = AWC / Pmax
    return AWC / t * (1.0 - np.exp(-t / tau)) + MAP * (1.0 - np.exp(-t / tau2))


def _fit_power_curve(dur: np.ndarray, pwr: np.ndarray,
                     n_iter: int = 8, asymmetry: float = 10.0):
    """Skimming fit via iteratively reweighted least squares (IRLS).

    Points above the model (hard efforts) receive weight `asymmetry`; points
    below receive weight 1, so the curve rides the upper envelope.

    Returns (popt, True) on success or (None, False) on failure.
    popt = [AWC, Pmax, MAP, tau2]
    """
    p0      = [20_000, float(pwr.max()) * 1.1, float(np.percentile(pwr, 90)) * 0.9, 300.0]
    bounds  = ([0, 0, 0, 1], [500_000, 5_000, 3_000, 3_600])
    weights = np.ones(len(dur))
    popt    = None

    for i in range(n_iter):
        try:
            popt, _ = curve_fit(
                _power_model, dur, pwr,
                p0=p0 if popt is None else popt,
                bounds=bounds,
                sigma=1.0 / weights,
                absolute_sigma=False,
                maxfev=10_000,
            )
        except Exception as exc:
            print(f"[fit] IRLS iter {i} failed: {exc}")
            return None, False
        residuals = pwr - _power_model(dur, *popt)
        weights   = np.where(residuals > 0, asymmetry, 1.0)

    return popt, True


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

        CREATE TABLE IF NOT EXISTS pdc_params (
            ride_id          INTEGER PRIMARY KEY REFERENCES rides(id),
            AWC              REAL    NOT NULL,
            Pmax             REAL    NOT NULL,
            MAP              REAL    NOT NULL,
            tau2             REAL    NOT NULL,
            computed_at      TEXT    NOT NULL,
            ftp              REAL,
            normalized_power REAL,
            intensity_factor REAL,
            tss              REAL
        );
    """)
    conn.commit()

    # Idempotent migration: add TSS columns to databases created before this
    # version (ALTER TABLE silently fails if the column already exists via the
    # OperationalError catch).
    for col_def in [
        "ftp              REAL",
        "normalized_power REAL",
        "intensity_factor REAL",
        "tss              REAL",
    ]:
        try:
            conn.execute(f"ALTER TABLE pdc_params ADD COLUMN {col_def}")
        except sqlite3.OperationalError:
            pass  # column already present
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


# ── PDC fitting ───────────────────────────────────────────────────────────────

def compute_pdc_params(conn: sqlite3.Connection, ride_id: int) -> None:
    """Fit the power-duration curve to sigmoid-decayed MMP up to this ride.

    Loads all MMP rows for rides on or before this ride's date (within
    PDC_WINDOW days), applies sigmoid aging relative to this ride's date,
    fits the two-component model, and stores the parameters in pdc_params.
    """
    row = conn.execute("SELECT ride_date FROM rides WHERE id = ?", (ride_id,)).fetchone()
    if not row:
        return

    ride_date = datetime.date.fromisoformat(row[0])
    cutoff    = (ride_date - datetime.timedelta(days=PDC_WINDOW)).isoformat()
    end_date  = ride_date.isoformat()

    mmp = pd.read_sql(
        """SELECT m.duration_s, m.power, r.ride_date
           FROM mmp m JOIN rides r ON m.ride_id = r.id
           WHERE r.ride_date BETWEEN ? AND ?""",
        conn,
        params=(cutoff, end_date),
    )
    if mmp.empty:
        return

    mmp["age_days"]   = mmp["ride_date"].apply(
        lambda d: (ride_date - datetime.date.fromisoformat(d)).days
    )
    mmp["weight"]     = 1.0 / (1.0 + np.exp(PDC_K * (mmp["age_days"] - PDC_INFLECTION)))
    mmp["aged_power"] = mmp["power"] * mmp["weight"]

    aged = (
        mmp.groupby("duration_s")["aged_power"]
        .max().reset_index().sort_values("duration_s")
    )
    dur = aged["duration_s"].to_numpy(dtype=float)
    pwr = aged["aged_power"].to_numpy(dtype=float)

    if len(dur) < 4:
        return

    popt, ok = _fit_power_curve(dur, pwr)
    if not ok:
        return

    AWC, Pmax, MAP, tau2 = popt

    # ── TSS metrics ───────────────────────────────────────────────────────────
    ftp = float(_power_model(3600.0, AWC, Pmax, MAP, tau2))

    rec = pd.read_sql(
        "SELECT elapsed_s, power FROM records WHERE ride_id = ? ORDER BY elapsed_s",
        conn, params=(ride_id,),
    )
    np_val = if_val = tss = 0.0
    if not rec.empty and rec["power"].notna().any():
        elapsed = rec["elapsed_s"].to_numpy(dtype=float)
        power   = rec["power"].to_numpy(dtype=float)
        dt      = np.diff(elapsed)
        hz      = 1.0 / float(np.median(dt[dt > 0])) if dt[dt > 0].size > 0 else 1.0
        np_val  = _normalized_power(power, hz)
        if_val  = np_val / ftp if ftp > 0 else 0.0
        dur_s   = float(elapsed[-1] - elapsed[0]) if len(elapsed) > 1 else 0.0
        tss     = (dur_s / 3600.0) * (if_val ** 2) * 100.0

    conn.execute(
        """INSERT OR REPLACE INTO pdc_params
               (ride_id, AWC, Pmax, MAP, tau2, computed_at,
                ftp, normalized_power, intensity_factor, tss)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            ride_id,
            round(float(AWC),  1),
            round(float(Pmax), 1),
            round(float(MAP),  1),
            round(float(tau2), 1),
            datetime.date.today().isoformat(),
            round(ftp,    1),
            round(np_val, 1),
            round(if_val, 3),
            round(tss,    1),
        ),
    )
    conn.commit()


def backfill_pdc_params(conn: sqlite3.Connection) -> None:
    """Compute PDC params for any rides that don't have them yet."""
    rows = conn.execute(
        """SELECT id FROM rides
           WHERE id NOT IN (SELECT ride_id FROM pdc_params)
           ORDER BY ride_date, id"""
    ).fetchall()
    for (ride_id,) in rows:
        compute_pdc_params(conn, ride_id)
    if rows:
        print(f"[pdc] Computed PDC params for {len(rows)} ride(s).")


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
    compute_pdc_params(conn, ride_id)


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


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build cycling SQLite database from FIT files.")
    parser.add_argument("--show", action="store_true", help="Print summary tables only, no processing.")
    args = parser.parse_args()

    if args.show:
        print_mmp_table(DB_PATH)
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

    backfill_pdc_params(conn)
    conn.close()
    print_mmp_table(DB_PATH)


if __name__ == "__main__":
    main()
