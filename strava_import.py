"""
strava_import.py

Import cycling activity data from the Strava API into the local SQLite
database (cycling.db).  Reuses the same ingestion pipeline as FIT file
imports so that MMP, MMH, and PDC calculations are applied automatically.

Setup (one-time)
────────────────
  1. Register an app at https://developers.strava.com
  2. Run:  python strava_import.py --setup
     (walks you through the OAuth flow and saves credentials locally)

Usage
─────
  python strava_import.py --after 2026-01-01              # import rides after date
  python strava_import.py --after 2026-01-01 --before 2026-03-01
"""

import argparse
import datetime
import json
import os
import sqlite3
import time

import pandas as pd

os.environ.setdefault("SILENCE_TOKEN_WARNINGS", "true")
from stravalib import Client

from build_database import DB_PATH, init_db, ingest_ride

BASE_DIR = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(BASE_DIR, "strava_config.json")


# ── Config helpers ────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def save_config(cfg: dict) -> None:
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


# ── OAuth ─────────────────────────────────────────────────────────────────────

def setup_oauth() -> None:
    """Interactive one-time OAuth setup."""
    print("── Strava API Setup ────────────────────────────────────────────")
    print()
    print("1. Go to https://www.strava.com/settings/api")
    print("2. Create an application (or use an existing one).")
    print("3. Set the 'Authorization Callback Domain' to 'localhost'.")
    print("4. Note your Client ID and Client Secret.")
    print()

    client_id = int(input("Client ID: ").strip())
    client_secret = input("Client Secret: ").strip()

    client = Client()
    auth_url = client.authorization_url(
        client_id=client_id,
        redirect_uri="http://localhost",
        scope=["activity:read_all"],
    )

    print()
    print("5. Open this URL in your browser and authorize the app:")
    print()
    print(f"   {auth_url}")
    print()
    print("6. After authorizing, you'll be redirected to a localhost URL.")
    print("   Copy the 'code' parameter from the URL.")
    print("   Example: http://localhost?state=&code=COPY_THIS_PART&scope=...")
    print()

    code = input("Authorization code: ").strip()

    token_response = client.exchange_code_for_token(
        client_id=client_id,
        client_secret=client_secret,
        code=code,
    )

    cfg = {
        "client_id": client_id,
        "client_secret": client_secret,
        "access_token": token_response["access_token"],
        "refresh_token": token_response["refresh_token"],
        "expires_at": token_response["expires_at"],
    }
    save_config(cfg)
    print()
    print(f"Credentials saved to {CONFIG_PATH}")
    print("You can now import rides with: python strava_import.py --after YYYY-MM-DD")


def get_client() -> Client:
    """Return an authenticated stravalib Client, refreshing the token if needed."""
    cfg = load_config()
    client = Client()

    # Refresh if token has expired (or will expire within 60 seconds)
    if time.time() >= cfg["expires_at"] - 60:
        token_response = client.refresh_access_token(
            client_id=cfg["client_id"],
            client_secret=cfg["client_secret"],
            refresh_token=cfg["refresh_token"],
        )
        cfg["access_token"] = token_response["access_token"]
        cfg["refresh_token"] = token_response["refresh_token"]
        cfg["expires_at"] = token_response["expires_at"]
        save_config(cfg)

    client.access_token = cfg["access_token"]
    return client


# ── Import logic ──────────────────────────────────────────────────────────────

STREAM_TYPES = ["time", "watts", "heartrate", "latlng", "altitude"]


def _activity_to_df(client: Client, activity) -> pd.DataFrame:
    """Fetch streams for a Strava activity and return a DataFrame matching
    the format expected by ``ingest_ride``."""
    streams = client.get_activity_streams(
        activity.id,
        types=STREAM_TYPES,
    )

    if not streams or "time" not in streams:
        return pd.DataFrame()

    n = len(streams["time"].data)
    start_dt = activity.start_date  # UTC datetime

    elapsed = streams["time"].data
    timestamps = [start_dt + datetime.timedelta(seconds=s) for s in elapsed]

    data: dict = {
        "timestamp": pd.to_datetime(timestamps, utc=True),
        "elapsed_s": [float(s) for s in elapsed],
    }

    if "watts" in streams:
        data["power"] = streams["watts"].data
    else:
        data["power"] = [None] * n

    if "heartrate" in streams:
        data["heart_rate"] = streams["heartrate"].data
    else:
        data["heart_rate"] = [None] * n

    if "latlng" in streams:
        latlng = streams["latlng"].data
        data["latitude"] = [pt[0] for pt in latlng]
        data["longitude"] = [pt[1] for pt in latlng]
    else:
        data["latitude"] = [None] * n
        data["longitude"] = [None] * n

    if "altitude" in streams:
        data["altitude_m"] = streams["altitude"].data
    else:
        data["altitude_m"] = [None] * n

    return pd.DataFrame(data)


def fetch_and_import(client: Client, conn: sqlite3.Connection,
                     after: str, before: str | None = None) -> None:
    """List Strava cycling activities in the date range and import each one."""
    after_dt = datetime.datetime.fromisoformat(after)

    kwargs: dict = {"after": after_dt}
    if before:
        kwargs["before"] = datetime.datetime.fromisoformat(before)

    print(f"Fetching Strava activities after {after}"
          + (f" and before {before}" if before else "")
          + " ...")

    activities = list(client.get_activities(**kwargs))

    # Filter to cycling only
    rides = [a for a in activities if getattr(a, "type", None) == "Ride"
             or getattr(a, "sport_type", None) == "Ride"]

    if not rides:
        print("No cycling activities found in the specified date range.")
        return

    print(f"Found {len(rides)} cycling activity/activities.\n")

    imported = 0
    for activity in rides:
        name = f"strava_{activity.id}"

        # Quick dedup check before fetching streams
        if conn.execute("SELECT 1 FROM rides WHERE name = ?", (name,)).fetchone():
            print(f"  {name}: already in database, skipping.")
            continue

        print(f"  Fetching streams for {activity.name} ({activity.start_date.date()}) ...")
        df = _activity_to_df(client, activity)
        if df.empty:
            print(f"  {name}: no stream data, skipping.")
            continue

        ingest_ride(conn, name, df)
        imported += 1

        # Brief pause to stay within rate limits
        time.sleep(1.5)

    print(f"\nDone. {imported} ride(s) imported.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import cycling data from Strava into cycling.db."
    )
    parser.add_argument(
        "--setup", action="store_true",
        help="Run one-time OAuth setup to connect your Strava account.",
    )
    parser.add_argument(
        "--after",
        help="Import activities after this date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--before",
        help="Import activities before this date (YYYY-MM-DD, optional).",
    )
    args = parser.parse_args()

    if args.setup:
        setup_oauth()
        return

    if not args.after:
        parser.error("--after is required (or use --setup for first-time configuration).")

    if not os.path.exists(CONFIG_PATH):
        print(f"No Strava config found at {CONFIG_PATH}.")
        print("Run 'python strava_import.py --setup' first.")
        return

    client = get_client()
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    fetch_and_import(client, conn, args.after, args.before)

    conn.close()


if __name__ == "__main__":
    main()
