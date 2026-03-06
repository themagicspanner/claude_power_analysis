"""Training plan generator — constrained forward simulation.

Given current zone CTLs (from real ride history), a target zone-CTL profile,
and an event date, produces a day-by-day training prescription where:

1. Zone-specific TSB *gates* what kind of work is allowed (freshness).
2. The current periodisation phase *prioritises* what work is preferred.
3. Daily TSS targets steer zone CTLs toward the target by event day.

The three training zones match the rest of the codebase:
  • Base      (≤ LTP)
  • Threshold (LTP → MAP)
  • Anaerobic (> MAP)

Terminology:
  CTL  — Chronic Training Load  (τ = 42 days)
  ATL  — Acute Training Load    (τ = 7 days)
  TSB  — Training Stress Balance (CTL_prev − ATL_prev)
"""

from __future__ import annotations

import math
import sqlite3
import os
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum, auto

import numpy as np
import pandas as pd

# ── Constants ────────────────────────────────────────────────────────────────

TAU_CTL = 42.0
TAU_ATL = 7.0
K_CTL = 1.0 - math.exp(-1.0 / TAU_CTL)
K_ATL = 1.0 - math.exp(-1.0 / TAU_ATL)

DB_PATH = os.path.join(os.path.dirname(__file__), "cycling.db")


# ── Enums & data classes ─────────────────────────────────────────────────────

class Phase(Enum):
    BASE = auto()
    BUILD = auto()
    PEAK = auto()


class Freshness(Enum):
    """Zone-aware readiness — mirrors app.py _compute_freshness_status."""
    GREEN = "green"    # all zones OK → anything allowed
    AMBER = "amber"    # base & threshold OK, AWC depleted → aerobic only
    RED = "red"        # threshold depleted → base / easy only
    BLACK = "black"    # base depleted → rest


@dataclass
class ZoneTSS:
    """TSS split across the three training zones."""
    base: float = 0.0
    threshold: float = 0.0
    anaerobic: float = 0.0

    @property
    def total(self) -> float:
        return self.base + self.threshold + self.anaerobic


@dataclass
class Workout:
    """A workout template."""
    name: str
    zone_tss: ZoneTSS
    description: str = ""
    requires: Freshness = Freshness.GREEN  # minimum freshness to attempt

    @property
    def total_tss(self) -> float:
        return self.zone_tss.total


@dataclass
class ZoneState:
    """CTL / ATL / TSB for a single zone."""
    ctl: float = 0.0
    atl: float = 0.0

    @property
    def tsb(self) -> float:
        return self.ctl - self.atl

    def step(self, tss: float) -> None:
        """Advance one day with the given TSS load."""
        self.atl += K_ATL * (tss - self.atl)
        self.ctl += K_CTL * (tss - self.ctl)


@dataclass
class TrainingState:
    """Full three-zone training state for a single day."""
    base: ZoneState = field(default_factory=ZoneState)
    threshold: ZoneState = field(default_factory=ZoneState)
    anaerobic: ZoneState = field(default_factory=ZoneState)

    def freshness(self) -> Freshness:
        """Determine freshness status from zone TSBs — matches app.py logic."""
        base_cutoff = -0.50 * self.base.ctl if self.base.ctl > 0 else -10.0
        thresh_cutoff = -0.30 * self.threshold.ctl if self.threshold.ctl > 0 else -5.0

        if self.base.tsb <= base_cutoff:
            return Freshness.BLACK
        if self.threshold.tsb <= thresh_cutoff:
            return Freshness.RED
        if self.anaerobic.tsb <= 0:
            return Freshness.AMBER
        return Freshness.GREEN

    def step(self, zone_tss: ZoneTSS) -> None:
        """Advance all zones by one day."""
        self.base.step(zone_tss.base)
        self.threshold.step(zone_tss.threshold)
        self.anaerobic.step(zone_tss.anaerobic)

    def copy(self) -> "TrainingState":
        return TrainingState(
            base=ZoneState(self.base.ctl, self.base.atl),
            threshold=ZoneState(self.threshold.ctl, self.threshold.atl),
            anaerobic=ZoneState(self.anaerobic.ctl, self.anaerobic.atl),
        )


# ── Workout library ─────────────────────────────────────────────────────────
# TSS values are *typical* for each workout type.  The planner scales them
# to hit CTL targets, but the *ratio* between zones is fixed per template.

WORKOUTS: dict[str, Workout] = {
    # ── Rest / recovery ──
    "rest": Workout(
        "Rest day", ZoneTSS(0, 0, 0),
        "Full rest day", Freshness.BLACK,
    ),
    "recovery": Workout(
        "Recovery spin", ZoneTSS(25, 0, 0),
        "Easy 45-60 min Z1 spin", Freshness.BLACK,
    ),

    # ── Base / endurance ──
    "endurance_short": Workout(
        "Endurance (short)", ZoneTSS(50, 5, 0),
        "1.5 h steady Z2 ride", Freshness.RED,
    ),
    "endurance_long": Workout(
        "Endurance (long)", ZoneTSS(90, 10, 0),
        "3 h Z2 endurance ride", Freshness.RED,
    ),

    # ── Threshold ──
    "sweet_spot": Workout(
        "Sweet spot", ZoneTSS(30, 45, 0),
        "2×20 min sweet spot (88-93% FTP)", Freshness.AMBER,
    ),
    "threshold_intervals": Workout(
        "Threshold intervals", ZoneTSS(25, 55, 0),
        "3×15 min at FTP", Freshness.AMBER,
    ),

    # ── VO2max / Anaerobic ──
    "vo2max": Workout(
        "VO2max intervals", ZoneTSS(20, 20, 40),
        "5×4 min at 110-120% FTP", Freshness.GREEN,
    ),
    "anaerobic": Workout(
        "Anaerobic repeats", ZoneTSS(15, 10, 50),
        "8×1 min all-out / 2 min rest", Freshness.GREEN,
    ),
}

# ── Phase-ordered workout preferences ────────────────────────────────────────
# For each phase, workouts in order of preference (best → fallback).
# The planner picks the first one whose freshness requirement is met.

PHASE_PREFERENCES: dict[Phase, list[str]] = {
    # Base: build aerobic foundation only — no threshold / anaerobic
    Phase.BASE: [
        "endurance_long",
        "endurance_short",
        "recovery",
        "rest",
    ],
    # Build: introduce threshold stimulus, maintain base volume
    Phase.BUILD: [
        "threshold_intervals",
        "sweet_spot",
        "endurance_short",
        "recovery",
        "rest",
    ],
    # Peak: add anaerobic / VO2max on top of threshold, reduce volume
    Phase.PEAK: [
        "vo2max",
        "anaerobic",
        "sweet_spot",
        "threshold_intervals",
        "recovery",
        "rest",
    ],
}

# Zone weights per phase: controls how much of the ideal daily TSS goes to
# each zone.  During base, nearly all load is base; build adds threshold;
# peak adds anaerobic.  The weights are normalised so they sum to 1.
PHASE_ZONE_WEIGHTS: dict[Phase, tuple[float, float, float]] = {
    #                      base   thresh  anaerobic
    Phase.BASE:  (1.0,   0.0,    0.0),
    Phase.BUILD: (0.50,  0.45,   0.05),
    Phase.PEAK:  (0.25,  0.35,   0.40),
}

# Freshness hierarchy: GREEN > AMBER > RED > BLACK
_FRESHNESS_RANK = {
    Freshness.GREEN: 3,
    Freshness.AMBER: 2,
    Freshness.RED: 1,
    Freshness.BLACK: 0,
}


def _freshness_allows(current: Freshness, required: Freshness) -> bool:
    """True if current freshness meets or exceeds the workout requirement."""
    return _FRESHNESS_RANK[current] >= _FRESHNESS_RANK[required]


# ── CTL target ramp ─────────────────────────────────────────────────────────

def _daily_tss_for_ctl_ramp(ctl_now: float, ctl_target: float,
                             days: int) -> float:
    """Constant daily TSS needed to move CTL from ctl_now → ctl_target in N days.

    Derived from the exponential filter:  CTL(n) = CTL(0)·α^n + T·(1−α^n)
    where α = 1−K_CTL and T = constant daily TSS.
    Solving for T:  T = (ctl_target − ctl_now·α^n) / (1 − α^n)
    """
    if days <= 0:
        return 0.0
    alpha = 1.0 - K_CTL
    alpha_n = alpha ** days
    denom = 1.0 - alpha_n
    if denom < 1e-9:
        return ctl_target  # effectively infinite horizon
    return (ctl_target - ctl_now * alpha_n) / denom


def _phase_for_day(day_offset: int, total_days: int,
                   base_weeks: int = 8, build_weeks: int = 8,
                   peak_weeks: int = 4) -> Phase:
    """Determine which periodisation phase a given day falls in.

    Phases are sized proportionally if total_days doesn't match exactly
    base_weeks + build_weeks + peak_weeks.
    """
    total_phase_weeks = base_weeks + build_weeks + peak_weeks
    scale = total_days / (total_phase_weeks * 7)
    base_end = int(base_weeks * 7 * scale)
    build_end = base_end + int(build_weeks * 7 * scale)

    if day_offset < base_end:
        return Phase.BASE
    if day_offset < build_end:
        return Phase.BUILD
    return Phase.PEAK


# ── Workout scaling ─────────────────────────────────────────────────────────

def _scale_workout(workout: Workout, target_total_tss: float,
                    max_factor: float = 2.0) -> ZoneTSS:
    """Scale a workout template's zone TSS to hit a target total TSS.

    Preserves the zone ratio of the template.  Clamps minimum at 0.
    max_factor caps how far the workout can be scaled up.
    """
    if workout.total_tss <= 0:
        return ZoneTSS(0, 0, 0)
    factor = max(0.0, target_total_tss / workout.total_tss)
    factor = min(factor, max_factor)
    return ZoneTSS(
        base=workout.zone_tss.base * factor,
        threshold=workout.zone_tss.threshold * factor,
        anaerobic=workout.zone_tss.anaerobic * factor,
    )


# ── Core simulator ──────────────────────────────────────────────────────────

@dataclass
class DayPlan:
    """A single day's training prescription."""
    date: date
    phase: Phase
    freshness: Freshness
    workout_key: str
    workout_name: str
    workout_description: str
    zone_tss: ZoneTSS
    ctl_base: float
    ctl_threshold: float
    ctl_anaerobic: float
    tsb_base: float
    tsb_threshold: float
    tsb_anaerobic: float


def simulate_plan(
    state: TrainingState,
    target_ctl: ZoneTSS,
    start_date: date,
    event_date: date,
    base_weeks: int = 8,
    build_weeks: int = 8,
    peak_weeks: int = 4,
) -> list[DayPlan]:
    """Run the forward simulation from start_date to event_date.

    Parameters
    ----------
    state : TrainingState
        Current zone CTL/ATL (typically loaded from real ride history).
    target_ctl : ZoneTSS
        Desired CTL in each zone on event day.
    start_date, event_date : date
        Planning window.
    base_weeks, build_weeks, peak_weeks : int
        Periodisation phase durations (scaled proportionally to fit window).

    Returns
    -------
    list[DayPlan]
        One entry per day from start_date to event_date (inclusive).
    """
    total_days = (event_date - start_date).days + 1
    if total_days <= 0:
        return []

    sim = state.copy()
    plan: list[DayPlan] = []

    for day_i in range(total_days):
        current_date = start_date + timedelta(days=day_i)
        days_remaining = total_days - day_i

        # 1. Determine phase and freshness
        phase = _phase_for_day(day_i, total_days, base_weeks, build_weeks,
                               peak_weeks)
        freshness = sim.freshness()

        # 2. Compute ideal daily TSS, gated by phase zone weights.
        #    Only ramp zones the current phase trains — base phase asks
        #    for base TSS only, build adds threshold, peak adds anaerobic.
        w_b, w_t, w_a = PHASE_ZONE_WEIGHTS[phase]
        ideal_base = (_daily_tss_for_ctl_ramp(
            sim.base.ctl, target_ctl.base, days_remaining) if w_b > 0 else 0)
        ideal_thresh = (_daily_tss_for_ctl_ramp(
            sim.threshold.ctl, target_ctl.threshold, days_remaining) if w_t > 0 else 0)
        ideal_anaerobic = (_daily_tss_for_ctl_ramp(
            sim.anaerobic.ctl, target_ctl.anaerobic, days_remaining) if w_a > 0 else 0)
        ideal_total = max(0, ideal_base + ideal_thresh + ideal_anaerobic)

        # 3. Safety cap: don't exceed 1.5× zone CTL in a single day
        max_tss = 1.5 * (sim.base.ctl + sim.threshold.ctl + sim.anaerobic.ctl)
        max_tss = max(max_tss, 40.0)  # floor for very low CTL athletes
        ideal_total = min(ideal_total, max_tss)

        # 4. Pick the best workout that freshness allows, scored by
        #    how well it matches the phase's zone weight distribution.
        prefs = PHASE_PREFERENCES[phase]
        chosen_key = "rest"
        chosen = WORKOUTS["rest"]
        best_score = -1e9

        for idx, key in enumerate(prefs):
            w = WORKOUTS[key]
            if not _freshness_allows(freshness, w.requires):
                continue
            if w.total_tss == 0:
                # Rest/recovery: only chosen if nothing else is allowed
                if best_score < -100:
                    best_score = -100
                    chosen_key, chosen = key, w
                continue
            # Score = how well this workout's zone mix matches the
            # phase's target zone distribution
            w_total = w.zone_tss.total
            zone_match = (
                (w.zone_tss.base / w_total) * w_b
                + (w.zone_tss.threshold / w_total) * w_t
                + (w.zone_tss.anaerobic / w_total) * w_a
            )
            # Phase preference bonus (first in list = highest)
            pref_bonus = 0.3 * (1.0 - idx / len(prefs))
            score = zone_match + pref_bonus
            if score > best_score:
                best_score = score
                chosen_key, chosen = key, w

        # 5. Scale the chosen workout to the ideal total TSS.
        #    Recovery rides are capped at 1× (don't inflate easy days).
        if chosen.total_tss > 0:
            cap = 1.0 if chosen_key == "recovery" else 2.0
            zone_tss = _scale_workout(chosen, ideal_total, max_factor=cap)
        else:
            zone_tss = ZoneTSS(0, 0, 0)

        # Record the plan for this day
        plan.append(DayPlan(
            date=current_date,
            phase=phase,
            freshness=freshness,
            workout_key=chosen_key,
            workout_name=chosen.name,
            workout_description=chosen.description,
            zone_tss=zone_tss,
            ctl_base=sim.base.ctl,
            ctl_threshold=sim.threshold.ctl,
            ctl_anaerobic=sim.anaerobic.ctl,
            tsb_base=sim.base.tsb,
            tsb_threshold=sim.threshold.tsb,
            tsb_anaerobic=sim.anaerobic.tsb,
        ))

        # 6. Advance the simulation
        sim.step(zone_tss)

    return plan


# ── Load current state from the database ─────────────────────────────────────

def load_current_state(db_path: str = DB_PATH) -> TrainingState:
    """Build a TrainingState from the real ride history in the database.

    Replays zone-specific TSS through the same exponential filter used
    everywhere else in the codebase.
    """
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("""
            SELECT r.ride_date, p.tss_ltp, p.tss_map, p.tss_awc
            FROM pdc_params p
            JOIN rides r ON r.id = p.ride_id
            WHERE p.tss_map IS NOT NULL AND p.tss_awc IS NOT NULL
            ORDER BY r.ride_date
        """, conn)
    finally:
        conn.close()

    if df.empty:
        return TrainingState()

    df["ride_date"] = pd.to_datetime(df["ride_date"])
    # Derive threshold component (same as app.py)
    df["tss_ltp"] = df["tss_ltp"].fillna(df["tss_map"])
    df["tss_thresh"] = (df["tss_map"] - df["tss_ltp"]).clip(lower=0)

    daily = df.groupby("ride_date")[["tss_ltp", "tss_thresh", "tss_awc"]].sum()

    # Replay through exponential filter day by day
    if daily.empty:
        return TrainingState()

    all_dates = pd.date_range(daily.index.min(),
                              pd.Timestamp.today().normalize(), freq="D")
    daily = daily.reindex(all_dates, fill_value=0.0)

    state = TrainingState()
    for _, row in daily.iterrows():
        state.base.step(row["tss_ltp"])
        state.threshold.step(row["tss_thresh"])
        state.anaerobic.step(row["tss_awc"])

    return state


# ── Suggest today's workout ──────────────────────────────────────────────────

def suggest_today(
    target_ctl: ZoneTSS,
    event_date: date,
    db_path: str = DB_PATH,
    base_weeks: int = 8,
    build_weeks: int = 8,
    peak_weeks: int = 4,
) -> DayPlan | None:
    """Suggest a single workout for today based on real training history.

    Loads the current zone state from the database, determines freshness and
    phase, then picks the best available workout.
    """
    state = load_current_state(db_path)
    today = date.today()

    if today > event_date:
        return None

    plan = simulate_plan(
        state, target_ctl, today, event_date,
        base_weeks, build_weeks, peak_weeks,
    )
    return plan[0] if plan else None


# ── Pretty-print helper ─────────────────────────────────────────────────────

def format_plan(plan: list[DayPlan]) -> str:
    """Return a human-readable text summary of a training plan."""
    lines = []
    lines.append(f"{'Date':<12} {'Phase':<6} {'Fresh':<6} "
                 f"{'Workout':<24} {'TSS':>5} "
                 f"{'Base':>5} {'Thr':>5} {'Ana':>5}  "
                 f"{'CTLb':>5} {'CTLt':>5} {'CTLa':>5}")
    lines.append("-" * 100)

    for d in plan:
        lines.append(
            f"{d.date.isoformat():<12} "
            f"{d.phase.name:<6} "
            f"{d.freshness.value:<6} "
            f"{d.workout_name:<24} "
            f"{d.zone_tss.total:5.0f} "
            f"{d.zone_tss.base:5.0f} {d.zone_tss.threshold:5.0f} "
            f"{d.zone_tss.anaerobic:5.0f}  "
            f"{d.ctl_base:5.1f} {d.ctl_threshold:5.1f} "
            f"{d.ctl_anaerobic:5.1f}"
        )
    return "\n".join(lines)


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a periodised training plan")
    parser.add_argument("--event", required=True,
                        help="Event date (YYYY-MM-DD)")
    parser.add_argument("--target-base", type=float, default=40.0,
                        help="Target base CTL (default: 40)")
    parser.add_argument("--target-threshold", type=float, default=20.0,
                        help="Target threshold CTL (default: 20)")
    parser.add_argument("--target-anaerobic", type=float, default=10.0,
                        help="Target anaerobic CTL (default: 10)")
    parser.add_argument("--base-weeks", type=int, default=8)
    parser.add_argument("--build-weeks", type=int, default=8)
    parser.add_argument("--peak-weeks", type=int, default=4)
    args = parser.parse_args()

    event = date.fromisoformat(args.event)
    target = ZoneTSS(args.target_base, args.target_threshold,
                     args.target_anaerobic)

    state = load_current_state()
    today = date.today()

    print(f"Current state:")
    print(f"  Base CTL:      {state.base.ctl:5.1f}  TSB: {state.base.tsb:+.1f}")
    print(f"  Threshold CTL: {state.threshold.ctl:5.1f}  TSB: "
          f"{state.threshold.tsb:+.1f}")
    print(f"  Anaerobic CTL: {state.anaerobic.ctl:5.1f}  TSB: "
          f"{state.anaerobic.tsb:+.1f}")
    print(f"  Freshness:     {state.freshness().value}")
    print()
    print(f"Target CTL at {event}: base={target.base}, "
          f"threshold={target.threshold}, anaerobic={target.anaerobic}")
    print(f"Days to event: {(event - today).days}")
    print()

    plan = simulate_plan(state, target, today, event,
                         args.base_weeks, args.build_weeks, args.peak_weeks)
    print(format_plan(plan))
