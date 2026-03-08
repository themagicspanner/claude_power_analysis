"""
build_database.py

Core database schema, MMP/MMH calculation, PDC fitting, and ride ingestion.
Called by strava_import.py and app.py to insert rides and recompute metrics.

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
import os
import sqlite3

import numpy as np
import pandas as pd

from helpers import (
    PDC_K, PDC_INFLECTION, PDC_WINDOW,
    calculate_ltp, apply_sigmoid_aging, aged_envelope,
    calculate_rolling_max_avg, fmt_duration,
)
from pdc_fitting import (
    power_model, fit_power_curve, normalized_power,
    tss_components, aerobic_decoupling,
)

BASE_DIR = os.path.dirname(__file__)
DB_PATH  = os.path.join(BASE_DIR, "cycling.db")

# Standard MMP durations in seconds
MMP_DURATIONS = sorted(set(
    list(range(1, 61))                        # 1–60 s, every 1 s
    + list(range(60, 121, 5))                 # 60–120 s, every 5 s
    + list(range(120, 301, 10))               # 120–300 s, every 10 s
    + list(range(300, 601, 30))               # 300–600 s, every 30 s
    + list(range(600, 3601, 60))              # 600–3600 s, every 60 s
))


def mmp_durations_for_ride(n_samples: int) -> list[int]:
    """Return MMP durations list extended in 5-min intervals to the ride length."""
    durations = list(MMP_DURATIONS)
    t = 3600 + 300
    while t <= n_samples:
        durations.append(t)
        t += 300
    return durations


# ── Backwards-compatible aliases for external importers ──────────────────────
# app.py and graphs.py previously imported these as private names from here.
_power_model         = power_model
_fit_power_curve     = fit_power_curve
_normalized_power    = normalized_power
_tss_components      = tss_components
_aerobic_decoupling  = aerobic_decoupling


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
            ride_id               INTEGER PRIMARY KEY REFERENCES rides(id),
            AWC                   REAL    NOT NULL,
            Pmax                  REAL    NOT NULL,
            MAP                   REAL    NOT NULL,
            tau2                  REAL    NOT NULL,
            computed_at           TEXT    NOT NULL,
            ftp                   REAL,
            normalized_power      REAL,
            intensity_factor      REAL,
            tss                   REAL,
            tss_map               REAL,
            tss_awc               REAL,
            tss_ltp               REAL,
            ltp                   REAL,
            variability_index     REAL,
            aerobic_decoupling_pct REAL
        );

        CREATE TABLE IF NOT EXISTS zone_distribution (
            ride_id  INTEGER NOT NULL REFERENCES rides(id),
            zone     INTEGER NOT NULL,
            seconds  REAL    NOT NULL,
            PRIMARY KEY (ride_id, zone)
        );

        CREATE TABLE IF NOT EXISTS daily_pdc_params (
            date  TEXT PRIMARY KEY,
            MAP   REAL NOT NULL,
            Pmax  REAL NOT NULL,
            AWC   REAL NOT NULL,
            tau2  REAL NOT NULL,
            ltp   REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS db_meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS pdc_overrides (
            ride_id      INTEGER PRIMARY KEY REFERENCES rides(id),
            delta_map_W  REAL    NOT NULL,
            delta_awc_J  REAL    NOT NULL,
            eff_AWC      REAL    NOT NULL,
            eff_MAP      REAL    NOT NULL,
            eff_ftp      REAL    NOT NULL,
            eff_ltp      REAL    NOT NULL,
            eff_tss      REAL    NOT NULL,
            eff_tss_map  REAL    NOT NULL,
            eff_tss_awc  REAL    NOT NULL,
            applied_at   TEXT    NOT NULL
        );
    """)
    conn.commit()

    # Idempotent migration: add columns to databases created before this
    # version (ALTER TABLE silently fails if the column already exists via the
    # OperationalError catch).
    for col_def in [
        "ftp                    REAL",
        "normalized_power       REAL",
        "intensity_factor       REAL",
        "tss                    REAL",
        "tss_map                REAL",
        "tss_awc                REAL",
        "tss_ltp                REAL",
        "ltp                    REAL",
        "variability_index      REAL",
        "aerobic_decoupling_pct REAL",
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

    # Purge stale rows so backfill recomputes them:
    #   • ltp IS NULL          — pre-dates the ltp column
    #   • tss_awc IS NULL      — pre-dates the tss split columns
    #   • components mismatch  — tss_map+tss_awc were written with a different
    #                            FTP than tss (old bug where the two calculations
    #                            used inconsistent FTP values)
    conn.execute("""
        DELETE FROM pdc_params
        WHERE ltp IS NULL
           OR tss_awc IS NULL
           OR tss_ltp IS NULL
           OR ABS(tss - (tss_map + tss_awc)) > 0.01
    """)
    # Remove zone rows whose PDC params were just purged (zones depend on LTP/MAP)
    conn.execute("""
        DELETE FROM zone_distribution
        WHERE ride_id NOT IN (SELECT ride_id FROM pdc_params)
    """)
    conn.commit()


# ── MMP calculation ───────────────────────────────────────────────────────────

def calculate_mmp(df: pd.DataFrame, durations: list[int]) -> dict[int, float]:
    """Return {duration_s: best_avg_power} for each requested duration."""
    return calculate_rolling_max_avg(df, "power", durations)


def calculate_mmh(df: pd.DataFrame, durations: list[int]) -> dict[int, float]:
    """Return {duration_s: best_avg_heart_rate} for each requested duration."""
    return calculate_rolling_max_avg(df, "heart_rate", durations)


def calculate_zones(df: pd.DataFrame, ltp: float, map_: float) -> dict[int, float]:
    """Return seconds spent in each of three physiological zones.

    Zone 1 : ≤ LTP          — base / below first lactate threshold
    Zone 2 : LTP < P ≤ MAP  — threshold / sweet-spot
    Zone 3 : > MAP           — high intensity / VO₂ max+

    NaN power values are treated as 0 W (Zone 1).
    """
    power = df["power"].fillna(0).to_numpy(dtype=float)
    z1 = float(np.sum(power <= ltp))
    z3 = float(np.sum(power > map_))
    z2 = float(len(power) - z1 - z3)
    return {1: z1, 2: z2, 3: z3}


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

    aged = aged_envelope(mmp, ride_date)
    dur = aged["duration_s"].to_numpy(dtype=float)
    pwr = aged["aged_power"].to_numpy(dtype=float)

    if len(dur) < 4:
        return

    popt, ok = _fit_power_curve(dur, pwr)
    if not ok:
        return

    AWC, Pmax, MAP, tau2 = popt

    ltp = calculate_ltp(AWC, MAP)

    # ── TSS metrics ───────────────────────────────────────────────────────────
    ftp = float(_power_model(3600.0, AWC, Pmax, MAP, tau2))

    rec = pd.read_sql(
        "SELECT elapsed_s, power, heart_rate FROM records WHERE ride_id = ? ORDER BY elapsed_s",
        conn, params=(ride_id,),
    )
    np_val = if_val = tss = tss_ltp = tss_map = tss_awc = 0.0
    vi = aedec = None
    if not rec.empty and rec["power"].notna().any():
        elapsed = rec["elapsed_s"].to_numpy(dtype=float)
        power   = rec["power"].to_numpy(dtype=float)
        dt      = np.diff(elapsed)
        hz      = 1.0 / float(np.median(dt[dt > 0])) if dt[dt > 0].size > 0 else 1.0
        np_val  = _normalized_power(power, hz)
        if_val  = np_val / ftp if ftp > 0 else 0.0
        dur_s   = float(elapsed[-1] - elapsed[0]) if len(elapsed) > 1 else 0.0
        tss     = (dur_s / 3600.0) * (if_val ** 2) * 100.0
        tss_ltp, tss_map, tss_awc = _tss_components(
            elapsed, power, ftp, float(MAP), tss, hz, ltp=float(ltp),
            AWC_val=float(AWC), Pmax_val=float(Pmax), tau2_val=float(tau2))
        ap  = float(rec["power"].fillna(0).mean())
        vi  = round(np_val / ap, 3) if ap > 0 else None
        aedec = _aerobic_decoupling(rec)

    conn.execute(
        """INSERT OR REPLACE INTO pdc_params
               (ride_id, AWC, Pmax, MAP, tau2, computed_at,
                ftp, normalized_power, intensity_factor, tss,
                tss_map, tss_awc, tss_ltp, ltp, variability_index, aerobic_decoupling_pct)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            ride_id,
            round(float(AWC),  1),
            round(float(Pmax), 1),
            float(MAP),               # full precision — re-used in split computation
            round(float(tau2), 1),
            datetime.date.today().isoformat(),
            float(ftp),               # full precision — re-used in TSS computation
            round(np_val,  1),
            round(if_val,  3),
            tss,                      # full precision — tss_map + tss_awc == tss exactly
            tss_map,
            tss_awc,
            tss_ltp,
            round(ltp,     1),
            vi,
            aedec,
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


def recompute_daily_pdc_params(conn: sqlite3.Connection,
                               from_date: str | None = None) -> None:
    """(Re)compute daily PDC parameters and store in daily_pdc_params.

    If *from_date* (ISO ``YYYY-MM-DD``) is given, recompute from that date to
    today.  Otherwise, recompute from the earliest ride date.  Uses warm-
    starting from the previous day's stored popt when available.
    """
    today = datetime.date.today()

    if from_date is None:
        row = conn.execute("SELECT MIN(ride_date) FROM rides").fetchone()
        if not row or not row[0]:
            return
        start = datetime.date.fromisoformat(row[0])
        conn.execute("DELETE FROM daily_pdc_params")
        prev_popt: list | None = None
    else:
        start = datetime.date.fromisoformat(from_date)
        conn.execute("DELETE FROM daily_pdc_params WHERE date >= ?", (from_date,))
        # Warm-start from the most recent stored day before start
        prev_row = conn.execute(
            "SELECT AWC, Pmax, MAP, tau2 FROM daily_pdc_params "
            "WHERE date < ? ORDER BY date DESC LIMIT 1",
            (from_date,),
        ).fetchone()
        prev_popt = list(prev_row) if prev_row else None

    # Load all MMP data with ride dates
    mmp = pd.read_sql(
        "SELECT m.duration_s, m.power, r.ride_date "
        "FROM mmp m JOIN rides r ON m.ride_id = r.id",
        conn,
    )
    if mmp.empty:
        conn.commit()
        return

    mmp["date_obj"] = pd.to_datetime(mmp["ride_date"]).dt.date
    ride_dates = set(mmp["date_obj"].unique())
    n_days = (today - start).days + 1

    rows: list[tuple] = []
    for i in range(n_days):
        ref    = start + datetime.timedelta(days=i)
        cutoff = ref - datetime.timedelta(days=PDC_WINDOW)

        w = mmp[(mmp["date_obj"] >= cutoff) & (mmp["date_obj"] <= ref)]
        if w.empty:
            prev_popt = None
            continue

        # Convert date_obj to ISO strings for apply_sigmoid_aging compatibility
        w_iso = w.assign(ride_date=w["date_obj"].apply(lambda d: d.isoformat()))
        aged = aged_envelope(w_iso, ref)
        if len(aged) < 4:
            prev_popt = None
            continue

        dur = aged["duration_s"].to_numpy(dtype=float)
        pwr = aged["aged_power"].to_numpy(dtype=float)

        has_new_ride = ref in ride_dates
        n_iter = 8 if (has_new_ride or prev_popt is None) else 4
        popt, ok = _fit_power_curve(dur, pwr, n_iter=n_iter, p0_init=prev_popt)
        if not ok:
            prev_popt = None
            continue

        prev_popt = list(popt)
        AWC, Pmax, MAP, tau2 = popt
        ltp = calculate_ltp(AWC, MAP)

        rows.append((
            ref.isoformat(),
            round(float(MAP), 1),
            round(float(Pmax), 1),
            round(float(AWC), 1),
            round(float(tau2), 1),
            round(ltp, 1),
        ))

    if rows:
        conn.executemany(
            "INSERT OR REPLACE INTO daily_pdc_params "
            "(date, MAP, Pmax, AWC, tau2, ltp) VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
    conn.commit()


def ensure_daily_pdc_current(conn: sqlite3.Connection) -> None:
    """Ensure daily_pdc_params is populated up to today.

    If the table is empty, does a full recompute.  Otherwise, only fills in
    days after the last stored date (typically just today on a new calendar
    day with no new rides).
    """
    row = conn.execute("SELECT MAX(date) FROM daily_pdc_params").fetchone()
    if not row or not row[0]:
        recompute_daily_pdc_params(conn)
        return
    last_date = datetime.date.fromisoformat(row[0])
    today = datetime.date.today()
    if last_date < today:
        next_date = last_date + datetime.timedelta(days=1)
        recompute_daily_pdc_params(conn, from_date=next_date.isoformat())


def backfill_vi_aedec(conn: sqlite3.Connection) -> None:
    """Compute variability_index and aerobic_decoupling_pct for rides missing them.

    Reuses the already-stored normalized_power and rides.avg_power for VI so that
    only a records read (for AeDec%) is needed — no expensive curve refitting.
    """
    rows = conn.execute(
        """SELECT p.ride_id, p.normalized_power, r.avg_power
           FROM pdc_params p
           JOIN rides r ON r.id = p.ride_id
           WHERE p.variability_index IS NULL
             AND p.normalized_power IS NOT NULL
             AND r.avg_power > 0
           ORDER BY p.ride_id"""
    ).fetchall()
    if not rows:
        return
    print(f"[vi] Backfilling VI/AeDec for {len(rows)} ride(s) …")
    for ride_id, np_val, avg_pwr in rows:
        vi = round(float(np_val) / float(avg_pwr), 3)
        rec = pd.read_sql(
            "SELECT power, heart_rate FROM records WHERE ride_id = ? ORDER BY elapsed_s",
            conn, params=(ride_id,),
        )
        aedec = _aerobic_decoupling(rec) if not rec.empty else None
        conn.execute(
            "UPDATE pdc_params SET variability_index = ?, aerobic_decoupling_pct = ?"
            " WHERE ride_id = ?",
            (vi, aedec, ride_id),
        )
    conn.commit()
    print("[vi] Backfill complete.")


def backfill_zones(conn: sqlite3.Connection) -> None:
    """Compute zone_distribution for rides that have PDC params but no zone data."""
    rows = conn.execute(
        """SELECT p.ride_id, p.ltp, p.MAP
           FROM pdc_params p
           WHERE p.ltp IS NOT NULL AND p.MAP IS NOT NULL
             AND NOT EXISTS (
                 SELECT 1 FROM zone_distribution z WHERE z.ride_id = p.ride_id
             )
           ORDER BY p.ride_id"""
    ).fetchall()
    if not rows:
        return
    print(f"[zones] Backfilling zone distribution for {len(rows)} ride(s) …")
    for ride_id, ltp, map_ in rows:
        rec = pd.read_sql(
            "SELECT power FROM records WHERE ride_id = ? ORDER BY elapsed_s",
            conn, params=(ride_id,),
        )
        if rec.empty:
            continue
        zones = calculate_zones(rec, float(ltp), float(map_))
        conn.executemany(
            "INSERT OR IGNORE INTO zone_distribution (ride_id, zone, seconds) VALUES (?,?,?)",
            [(ride_id, z, s) for z, s in zones.items()],
        )
        conn.commit()
    print("[zones] Backfill complete.")


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


def backfill_missing_mmp(conn: sqlite3.Connection) -> None:
    """Backfill any MMP (and MMH) durations missing from the current resolution.

    Compares each ride's stored durations against the full set from
    mmp_durations_for_ride() and computes any that are absent.
    """
    rows = conn.execute(
        "SELECT id, name, total_records FROM rides ORDER BY id"
    ).fetchall()
    if not rows:
        return

    backfilled = 0
    for ride_id, name, n_records in rows:
        durations = mmp_durations_for_ride(n_records)

        existing = {r[0] for r in conn.execute(
            "SELECT duration_s FROM mmp WHERE ride_id = ?",
            (ride_id,),
        ).fetchall()}
        missing = [d for d in durations if d not in existing]
        if not missing:
            continue

        df = pd.read_sql(
            "SELECT elapsed_s, power, heart_rate FROM records WHERE ride_id = ? ORDER BY elapsed_s",
            conn, params=(ride_id,),
        )
        if df.empty or df["power"].isna().all():
            continue

        mmp = calculate_mmp(df, missing)
        if mmp:
            conn.executemany(
                "INSERT OR IGNORE INTO mmp (ride_id, duration_s, power) VALUES (?,?,?)",
                [(ride_id, d, round(p, 1)) for d, p in mmp.items()],
            )

        if "heart_rate" in df.columns and df["heart_rate"].notna().any():
            mmh = calculate_mmh(df, missing)
            if mmh:
                conn.executemany(
                    "INSERT OR IGNORE INTO mmh (ride_id, duration_s, heart_rate) VALUES (?,?,?)",
                    [(ride_id, d, round(h, 1)) for d, h in mmh.items()],
                )

        backfilled += 1

    conn.commit()
    if backfilled:
        print(f"[mmp] Backfilled missing MMP durations for {backfilled} ride(s).")


# ── Per-ride processing ───────────────────────────────────────────────────────

def ingest_ride(conn: sqlite3.Connection, name: str, df: pd.DataFrame) -> None:
    """Store a ride DataFrame in the database (records, MMP, MMH, PDC).

    *name* is the unique ride identifier (e.g. FIT filename stem or
    ``strava_<id>``).  *df* must contain at least ``timestamp`` and
    ``elapsed_s`` columns; ``power``, ``heart_rate``, ``latitude``,
    ``longitude``, and ``altitude_m`` are optional.
    """
    if conn.execute("SELECT 1 FROM rides WHERE name = ?", (name,)).fetchone():
        print(f"  {name}: already in database, skipping.")
        return

    if df.empty:
        print(f"  {name}: no record data, skipping.")
        return

    # Ride metadata
    ride_date = df["timestamp"].iloc[0].date().isoformat()
    duration  = float(df["elapsed_s"].iloc[-1])
    has_power = "power" in df.columns and df["power"].notna().any()
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
        print(f"  {name}: already in database, skipping.")
        return
    ride_id = cur.lastrowid

    # Ensure optional columns exist
    for col in ("power", "heart_rate", "latitude", "longitude", "altitude_m"):
        if col not in df.columns:
            df[col] = None

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

    # MMP — extend durations in 30-min intervals to the ride length
    ride_durations = mmp_durations_for_ride(len(df))
    mmp = calculate_mmp(df, ride_durations)
    conn.executemany(
        "INSERT INTO mmp (ride_id, duration_s, power) VALUES (?,?,?)",
        [(ride_id, d, round(p, 1)) for d, p in mmp.items()],
    )

    # MMH
    mmh = calculate_mmh(df, ride_durations)
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
        "DELETE FROM zone_distribution WHERE ride_id IN "
        "(SELECT id FROM rides WHERE ride_date BETWEEN ? AND ?)",
        (ride_date, end_affected),
    )
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

    # Recompute daily PDC history from this ride's date onward
    recompute_daily_pdc_params(conn, from_date=ride_date)


# ── Display helpers ───────────────────────────────────────────────────────────



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
    pivot.index = [fmt_duration(int(d)) for d in pivot.index]
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
    parser = argparse.ArgumentParser(description="Build cycling SQLite database.")
    parser.add_argument("--show", action="store_true", help="Print summary tables only, no processing.")
    args = parser.parse_args()

    if args.show:
        print_mmp_table(DB_PATH)
        return

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    backfill_pdc_params(conn)
    backfill_vi_aedec(conn)
    backfill_zones(conn)
    conn.close()
    print_mmp_table(DB_PATH)


if __name__ == "__main__":
    main()
