"""
Extract power and heart rate data vs time from FIT files in the raw_data folder.

Outputs:
  - raw_data_fit_data.csv   : combined data from all rides
  - plots/<ride>.png    : power + heart rate vs time for each ride
"""

import os
import glob
import fitdecode
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


FIT_DIR = os.path.join(os.path.dirname(__file__), "raw_data")
PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
CSV_OUT  = os.path.join(os.path.dirname(__file__), "raw_data_fit_data.csv")


def read_fit_file(path: str) -> pd.DataFrame:
    """Return a DataFrame with columns: timestamp, power, heart_rate."""
    records = []
    with fitdecode.FitReader(path) as fit:
        for frame in fit:
            if not isinstance(frame, fitdecode.FitDataMessage):
                continue
            if frame.name != "record":
                continue

            row = {}
            for field in ("timestamp", "power", "heart_rate"):
                row[field] = frame.get_value(field, fallback=None)
            records.append(row)

    df = pd.DataFrame(records)
    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Derive elapsed seconds from ride start
    df["elapsed_s"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()

    return df


def plot_ride(df: pd.DataFrame, ride_name: str, out_path: str) -> None:
    """Plot power and heart rate vs elapsed time and save to out_path."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle(ride_name, fontsize=13, fontweight="bold")

    elapsed_min = df["elapsed_s"] / 60.0

    # Power
    if df["power"].notna().any():
        ax1.plot(elapsed_min, df["power"], color="tab:orange", linewidth=0.8, label="Power (W)")
        ax1.set_ylabel("Power (W)")
        ax1.legend(loc="upper right", fontsize=8)
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "No power data", ha="center", va="center", transform=ax1.transAxes)
        ax1.set_ylabel("Power (W)")

    # Heart rate
    if df["heart_rate"].notna().any():
        ax2.plot(elapsed_min, df["heart_rate"], color="tab:red", linewidth=0.8, label="Heart Rate (bpm)")
        ax2.set_ylabel("Heart Rate (bpm)")
        ax2.legend(loc="upper right", fontsize=8)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No heart rate data", ha="center", va="center", transform=ax2.transAxes)
        ax2.set_ylabel("Heart Rate (bpm)")

    ax2.set_xlabel("Elapsed Time (min)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    fit_files = sorted(glob.glob(os.path.join(FIT_DIR, "*.fit")))
    if not fit_files:
        print(f"No .fit files found in {FIT_DIR}")
        return

    all_dfs = []

    for path in fit_files:
        ride_name = os.path.splitext(os.path.basename(path))[0]
        print(f"Processing: {ride_name}")

        df = read_fit_file(path)
        if df.empty:
            print(f"  -> No record data found, skipping.")
            continue

        df.insert(0, "ride", ride_name)
        all_dfs.append(df)

        # Per-ride plot
        plot_path = os.path.join(PLOT_DIR, f"{ride_name}.png")
        plot_ride(df, ride_name.replace("_", " "), plot_path)
        print(f"  -> {len(df)} records | plot saved to {plot_path}")

    if not all_dfs:
        print("No data extracted.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(CSV_OUT, index=False)
    print(f"\nCombined CSV saved to: {CSV_OUT}")
    print(f"Total records: {len(combined)}")
    print(f"\nSummary per ride:")
    summary = (
        combined.groupby("ride")
        .agg(
            records=("timestamp", "count"),
            avg_power=("power", "mean"),
            max_power=("power", "max"),
            avg_hr=("heart_rate", "mean"),
            max_hr=("heart_rate", "max"),
        )
        .round(1)
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
