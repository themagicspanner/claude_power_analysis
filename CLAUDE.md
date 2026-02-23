# CLAUDE.md

This file provides guidance for AI assistants working in this repository.

## Project Overview

A cycling power analysis tool that parses Garmin/ANT+ FIT files, extracts
power and heart rate data, stores everything in a SQLite database, and
generates Mean Maximal Power (MMP) charts and per-ride visualisations.

## Repository Structure

```
claude_power_analysis/
├── 2026/                    # Source FIT files (one per ride, ~12 files)
├── plots/                   # Generated PNG charts (created on first run)
│   ├── <ride>.png           # Power + heart rate vs elapsed time
│   └── <ride>_mmp.png       # Ride MMP vs rolling 90-day best
├── extract_fit_data.py      # Script 1: extract data → CSV + per-ride plots
├── build_database.py        # Script 2: build SQLite DB, MMP calc, charts
├── 2026_fit_data.csv        # Combined raw data (generated artefact)
├── cycling.db               # SQLite database (generated artefact)
└── mmp_chart.png            # All-rides MMP overlay chart (generated artefact)
```

## Scripts

### `extract_fit_data.py`

First-generation extraction script. Reads every `.fit` file in `2026/`,
extracts `timestamp`, `power`, and `heart_rate` from `record` messages,
writes a combined `2026_fit_data.csv`, and saves a two-panel
(power / HR vs elapsed time) PNG per ride into `plots/`.

```bash
python extract_fit_data.py
```

Outputs:
- `2026_fit_data.csv` — one row per 1-second record across all rides
- `plots/<ride_name>.png` — per-ride power + HR chart

### `build_database.py`

Full pipeline: SQLite storage, MMP calculation, and chart generation.
Rides already present in the database are skipped (idempotent).

```bash
# Build / update the database (default action)
python build_database.py

# Print the MMP pivot table + ride summary without re-processing
python build_database.py --show

# Save an all-rides MMP overlay chart to mmp_chart.png
python build_database.py --plot

# Save per-ride MMP vs 90-day-best charts to plots/
python build_database.py --plot-rides
```

## Database Schema (`cycling.db`)

```sql
rides   (id, name, ride_date, total_records, duration_s,
         avg_power, max_power, avg_hr, max_hr)

records (ride_id → rides.id, timestamp, elapsed_s, power, heart_rate)
        INDEX idx_records_ride ON records(ride_id)

mmp     (ride_id → rides.id, duration_s, power)
        PRIMARY KEY (ride_id, duration_s)
```

- `ride_date` is stored as ISO-8601 text (`YYYY-MM-DD`).
- `timestamp` in `records` is stored as ISO-8601 text with UTC timezone.
- `power` and `heart_rate` are `NULL` when the sensor had no reading.
- MMP durations (seconds): 1, 2, 3, 5, 8, 10, 12, 15, 20, 30, 60, 90,
  120, 180, 240, 300, 420, 600, 900, 1200, 1800, 2400, 3600.

## MMP Calculation

`calculate_mmp()` in `build_database.py` uses a **cumulative-sum sliding
window** for O(n) computation:

- Input: 1-Hz power series (NaN gaps filled with 0 W, matching most
  cycling analysis software conventions).
- For each duration `d`, the MMP is `max(rolling_mean over d samples)`.
- Rides shorter than `d` seconds produce no entry for that duration.

## Dependencies

No `requirements.txt` is committed. Install manually:

```bash
pip install fitdecode pandas matplotlib numpy
```

| Package      | Role                              |
|--------------|-----------------------------------|
| `fitdecode`  | Parse binary FIT files            |
| `pandas`     | DataFrame manipulation, CSV I/O   |
| `matplotlib` | Chart generation (Agg backend)    |
| `numpy`      | Cumulative-sum MMP calculation    |
| `sqlite3`    | Built-in; no extra install needed |

Python 3.11+ is required (uses `list[int]` / `dict[int, float]` type hints).

## Development Conventions

- **No test suite** exists. Validate changes by running the scripts against
  the `2026/` FIT files and inspecting outputs.
- **Idempotency**: `build_database.py` skips rides already in the database
  by name. To reprocess a ride, delete its row from `rides` (cascading
  deletes are not configured — also delete from `records` and `mmp`).
- **matplotlib backend**: `Agg` is set in `extract_fit_data.py` so charts
  can be generated without a display. `build_database.py` does not set it
  explicitly — add `matplotlib.use("Agg")` before the import if running
  headlessly.
- **Paths**: All paths are derived from `__file__` / `os.path.dirname`,
  so both scripts work correctly when called from any working directory.
- **Column naming**: Use `snake_case` for all DataFrame columns and
  database columns (`elapsed_s`, `heart_rate`, `avg_power`, etc.).

## Adding New Rides

1. Drop the `.fit` file(s) into `2026/`.
2. Run `python build_database.py` — new rides are processed; existing ones
   are skipped.
3. Optionally regenerate charts: `python build_database.py --plot-rides`.

To extend to future years, update `FIT_DIR` / `BASE_DIR` constants or
parameterise them via `argparse`.

## Key Implementation Notes

- FIT file parsing uses `fitdecode.FitDataMessage` with `frame.name == "record"`.
  Other frame types (file headers, device info, session summaries) are ignored.
- `elapsed_s` is computed as `(timestamp - first_timestamp).total_seconds()`,
  not taken from a FIT field directly.
- The 90-day MMP comparison in `plot_ride_vs_90day_mmp()` **excludes the
  current ride** from the baseline window to avoid self-comparison.
- Chart x-axes use a log scale for duration (standard in cycling literature).
