"""
build_database.py

Parse all FIT files in raw_data/, store records and mean-maximal-power (MMP)
curves in a SQLite database (cycling.db), then print a summary table.

Database schema
───────────────
  rides   – one row per ride with summary stats
  records – raw 1-Hz timestamp / power rows
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
FIT_DIR      = os.path.join(BASE_DIR, "raw_data")
_SEMI_TO_DEG = 180.0 / 2**31   # FIT semicircle → decimal degree
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


def _tss_components(elapsed_s: np.ndarray, power: np.ndarray,
                    ftp: float, CP: float, tss_total: float,
                    sample_hz: float = 1.0) -> tuple[float, float]:
    """Split the NP-based TSS into aerobic (MAP) and anaerobic (AWC) parts.

    Uses p_30s² as a time-weighting kernel (same as NP methodology) to decide
    how much of each second's training stress should be credited to MAP vs AWC.
    The per-sample AWC fraction comes from instantaneous power so that even
    brief above-CP efforts register:

        f_AWC(t) = max(0, P(t) − CP) / P(t)   [instantaneous]
        f_MAP(t) = 1 − f_AWC(t)

    The final values are scaled so that TSS_MAP + TSS_AWC = tss_total exactly
    (the NP-based TSS).

    Returns (tss_map, tss_awc).
    """
    p = np.where(np.isnan(power), 0.0, power.astype(float))
    window = max(1, int(30 * sample_hz))
    kernel = np.ones(window) / window
    p_30s  = np.convolve(p, kernel, mode="same")   # same length as input

    dt = np.empty_like(elapsed_s)
    dt[0]  = 0.0
    dt[1:] = np.diff(elapsed_s)
    dt     = np.clip(dt, 0.0, None)

    # Split fractions from instantaneous power
    with np.errstate(invalid="ignore", divide="ignore"):
        f_awc = np.where(p > 0, np.maximum(p - CP, 0.0) / p, 0.0)
    f_map = 1.0 - f_awc

    # p_30s² weights — same basis as NP; used as split ratio only
    weights = (p_30s ** 2) * dt if ftp > 0 else np.zeros_like(p_30s)
    w_total = float(np.sum(weights))
    if w_total > 0 and tss_total > 0:
        awc_frac = float(np.sum(weights * f_awc)) / w_total
        tss_awc  = tss_total * awc_frac
        tss_map  = tss_total - tss_awc
    else:
        tss_awc = 0.0
        tss_map = float(tss_total)
    return tss_map, tss_awc


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
            max_power     INTEGER
        );

        CREATE TABLE IF NOT EXISTS records (
            ride_id    INTEGER NOT NULL REFERENCES rides(id),
            timestamp  TEXT    NOT NULL,
            elapsed_s  REAL    NOT NULL,
            power      INTEGER
        );

        CREATE INDEX IF NOT EXISTS idx_records_ride
            ON records(ride_id);

        CREATE TABLE IF NOT EXISTS mmp (
            ride_id    INTEGER NOT NULL REFERENCES rides(id),
            duration_s INTEGER NOT NULL,
            power      REAL    NOT NULL,
            PRIMARY KEY (ride_id, duration_s)
        );

        CREATE TABLE IF NOT EXISTS mmh (
            ride_id    INTEGER NOT NULL REFERENCES rides(id),
            duration_s INTEGER NOT NULL,
            heart_rate REAL    NOT NULL,
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
            tss              REAL,
            tss_map          REAL,
            tss_awc          REAL,
            ltp              REAL
        );

        CREATE TABLE IF NOT EXISTS db_meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
    """)
    conn.commit()

    # Idempotent migration: add columns to databases created before this
    # version (ALTER TABLE silently fails if the column already exists via the
    # OperationalError catch).
    for col_def in [
        "ftp              REAL",
        "normalized_power REAL",
        "intensity_factor REAL",
        "tss              REAL",
        "tss_map          REAL",
        "tss_awc          REAL",
        "ltp              REAL",
    ]:
        try:
            conn.execute(f"ALTER TABLE pdc_params ADD COLUMN {col_def}")
        except sqlite3.OperationalError:
            pass  # column already present

    for col_def in ["heart_rate INTEGER", "latitude REAL", "longitude REAL", "altitude_m REAL"]:
        try:
            conn.execute(f"ALTER TABLE records ADD COLUMN {col_def}")
        except sqlite3.OperationalError:
            pass

    for col_def in ["avg_heart_rate REAL", "max_heart_rate INTEGER"]:
        try:
            conn.execute(f"ALTER TABLE rides ADD COLUMN {col_def}")
        except sqlite3.OperationalError:
            pass

    # Purge any rows that pre-date the ltp column so backfill recomputes them.
    conn.execute("DELETE FROM pdc_params WHERE ltp IS NULL")
    conn.commit()


# ── FIT parsing ───────────────────────────────────────────────────────────────

def read_fit(path: str) -> pd.DataFrame:
    """Return DataFrame with columns: timestamp, elapsed_s, power, heart_rate,
    latitude, longitude, altitude_m."""
    rows = []
    with fitdecode.FitReader(path) as fit:
        for frame in fit:
            if not isinstance(frame, fitdecode.FitDataMessage) or frame.name != "record":
                continue
            raw_lat = frame.get_value("position_lat",  fallback=None)
            raw_lon = frame.get_value("position_long", fallback=None)
            rows.append({
                "timestamp":  frame.get_value("timestamp",        fallback=None),
                "power":      frame.get_value("power",            fallback=None),
                "heart_rate": frame.get_value("heart_rate",       fallback=None),
                "latitude":   round(raw_lat * _SEMI_TO_DEG, 7) if raw_lat is not None else None,
                "longitude":  round(raw_lon * _SEMI_TO_DEG, 7) if raw_lon is not None else None,
                "altitude_m": frame.get_value("enhanced_altitude", fallback=None),
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


def calculate_mmh(df: pd.DataFrame, durations: list[int]) -> dict[int, float]:
    """Return {duration_s: best_avg_heart_rate} for each requested duration.

    Uses the same rolling-window algorithm as calculate_mmp.
    Rides without heart rate data return an empty dict.
    """
    if df.empty or "heart_rate" not in df.columns or df["heart_rate"].isna().all():
        return {}

    hr = df["heart_rate"].fillna(0).to_numpy(dtype=float)
    n  = len(hr)
    cumsum = hr.cumsum()

    result: dict[int, float] = {}
    for d in durations:
        if n < d:
            continue
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

    # Lower threshold power (first lactate turn point)
    ltp = float(MAP * (1.0 - (5.0 / 2.0) * ((AWC / 1000.0) / MAP)))

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
        tss_map, tss_awc = _tss_components(elapsed, power, ftp, float(MAP), tss, hz)

    conn.execute(
        """INSERT OR REPLACE INTO pdc_params
               (ride_id, AWC, Pmax, MAP, tau2, computed_at,
                ftp, normalized_power, intensity_factor, tss,
                tss_map, tss_awc, ltp)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            ride_id,
            round(float(AWC),  1),
            round(float(Pmax), 1),
            round(float(MAP),  1),
            round(float(tau2), 1),
            datetime.date.today().isoformat(),
            round(ftp,     1),
            round(np_val,  1),
            round(if_val,  3),
            round(tss,     1),
            round(tss_map, 1),
            round(tss_awc, 1),
            round(ltp,     1),
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


def backfill_mmh(conn: sqlite3.Connection) -> None:
    """Compute MMH for any ride that has no MMH rows yet.

    Re-reads the original .fit file so that rides processed before heart rate
    support was added get their curves populated retrospectively.
    Also updates rides.avg_heart_rate / max_heart_rate where missing.
    """
    rows = conn.execute(
        """SELECT r.id, r.name FROM rides r
           WHERE NOT EXISTS (SELECT 1 FROM mmh m WHERE m.ride_id = r.id)
             AND EXISTS (SELECT 1 FROM records rec
                         WHERE rec.ride_id = r.id AND rec.heart_rate IS NOT NULL)
           ORDER BY r.ride_date, r.id"""
    ).fetchall()
    if not rows:
        return

    print(f"[mmh] Backfilling MMH for {len(rows)} ride(s) …")
    for ride_id, name in rows:
        fit_path = os.path.join(FIT_DIR, name + ".fit")
        if not os.path.exists(fit_path):
            continue
        df = read_fit(fit_path)
        if df.empty or "heart_rate" not in df.columns or df["heart_rate"].isna().all():
            continue

        mmh = calculate_mmh(df, MMP_DURATIONS)
        if mmh:
            conn.executemany(
                "INSERT OR IGNORE INTO mmh (ride_id, duration_s, heart_rate) VALUES (?,?,?)",
                [(ride_id, d, round(h, 1)) for d, h in mmh.items()],
            )
        conn.execute(
            "UPDATE rides SET avg_heart_rate = ?, max_heart_rate = ? WHERE id = ?",
            (
                round(float(df["heart_rate"].mean()), 1),
                int(df["heart_rate"].max()),
                ride_id,
            ),
        )
        conn.commit()
    print("[mmh] Backfill complete.")


def backfill_gps_elevation(conn: sqlite3.Connection) -> None:
    """Populate latitude/longitude/altitude_m for rides processed before GPS support."""
    rows = conn.execute(
        """SELECT r.id, r.name FROM rides r
           WHERE NOT EXISTS (
               SELECT 1 FROM records rec
               WHERE rec.ride_id = r.id AND rec.latitude IS NOT NULL
           )
           ORDER BY r.ride_date, r.id"""
    ).fetchall()
    if not rows:
        return

    print(f"[gps] Backfilling GPS/elevation for {len(rows)} ride(s) …")
    for ride_id, name in rows:
        fit_path = os.path.join(FIT_DIR, name + ".fit")
        if not os.path.exists(fit_path):
            continue
        df = read_fit(fit_path)
        if df.empty or "latitude" not in df.columns or df["latitude"].isna().all():
            continue
        conn.executemany(
            """UPDATE records
               SET latitude = ?, longitude = ?, altitude_m = ?
               WHERE ride_id = ? AND elapsed_s = ?""",
            (
                (
                    row.latitude  if pd.notna(row.latitude)  else None,
                    row.longitude if pd.notna(row.longitude) else None,
                    round(float(row.altitude_m), 1) if pd.notna(row.altitude_m) else None,
                    ride_id,
                    row.elapsed_s,
                )
                for row in df.itertuples()
            ),
        )
        conn.commit()
    print("[gps] Backfill complete.")


def recompute_all_pdc_params(conn: sqlite3.Connection) -> None:
    """Delete and recompute PDC params for all rides in chronological order.

    Use this to bring stored params in sync after a bulk import or code change.
    """
    conn.execute("DELETE FROM pdc_params")
    conn.commit()
    rows = conn.execute(
        "SELECT id FROM rides ORDER BY ride_date, id"
    ).fetchall()
    for (ride_id,) in rows:
        compute_pdc_params(conn, ride_id)
    if rows:
        print(f"[pdc] Recomputed PDC params for {len(rows)} ride(s).")


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
    has_hr    = "heart_rate" in df.columns and df["heart_rate"].notna().any()

    cur = conn.execute(
        """INSERT OR IGNORE INTO rides
               (name, ride_date, total_records, duration_s, avg_power, max_power,
                avg_heart_rate, max_heart_rate)
           VALUES (?,?,?,?,?,?,?,?)""",
        (
            name, ride_date, len(df), duration,
            round(float(df["power"].mean()),      1) if has_power else None,
            int(df["power"].max())                   if has_power else None,
            round(float(df["heart_rate"].mean()), 1) if has_hr    else None,
            int(df["heart_rate"].max())              if has_hr    else None,
        ),
    )
    if cur.rowcount == 0:
        # Another concurrent call already inserted this ride (race condition).
        print(f"  {name}: already in database, skipping.")
        return
    ride_id = cur.lastrowid

    # Raw records
    conn.executemany(
        "INSERT INTO records (ride_id, timestamp, elapsed_s, power, heart_rate,"
        " latitude, longitude, altitude_m) VALUES (?,?,?,?,?,?,?,?)",
        (
            (
                ride_id,
                row.timestamp.isoformat(),
                row.elapsed_s,
                int(row.power)      if pd.notna(row.power)      else None,
                int(row.heart_rate) if pd.notna(row.heart_rate) else None,
                row.latitude   if pd.notna(row.latitude)   else None,
                row.longitude  if pd.notna(row.longitude)  else None,
                round(float(row.altitude_m), 1) if pd.notna(row.altitude_m) else None,
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

    # MMH
    mmh = calculate_mmh(df, MMP_DURATIONS)
    if mmh:
        conn.executemany(
            "INSERT INTO mmh (ride_id, duration_s, heart_rate) VALUES (?,?,?)",
            [(ride_id, d, round(h, 1)) for d, h in mmh.items()],
        )

    conn.commit()
    hr_note = f", {len(mmh)} MMH points" if mmh else ""
    print(f"  {name}: {len(df)} records, {len(mmp)} MMP points{hr_note} stored.")

    # Recompute PDC params for this ride AND all subsequent rides whose PDC
    # window now includes this newly added ride's data.
    # A ride at date D uses MMP from [D − PDC_WINDOW, D], so any ride at date
    # D ∈ [ride_date, ride_date + PDC_WINDOW] is now stale.
    ride_date_obj = datetime.date.fromisoformat(ride_date)
    end_affected  = (ride_date_obj + datetime.timedelta(days=PDC_WINDOW)).isoformat()
    conn.execute(
        "DELETE FROM pdc_params WHERE ride_id IN "
        "(SELECT id FROM rides WHERE ride_date BETWEEN ? AND ?)",
        (ride_date, end_affected),
    )
    conn.commit()
    for (rid,) in conn.execute(
        "SELECT id FROM rides WHERE ride_date BETWEEN ? AND ? ORDER BY ride_date, id",
        (ride_date, end_affected),
    ).fetchall():
        compute_pdc_params(conn, rid)


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
                  avg_power, max_power
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
