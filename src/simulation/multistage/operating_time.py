"""
Operating-time clock — the single definition of "one monthly planning horizon" used by every
simulation engine, evaluator, and adaptive search in the project.

Problem this replaces: the simulators used to run `env.run(until=done_event)` — i.e. until
every order was processed, however long that took — while `hours_per_worker_month` (the
economic capacity a worker is actually paid for) was a separate, disconnected number. Because
generated/historical arrivals were spread across the full calendar month (~744h for a 31-day
month), the emergent simulated horizon landed close to 744h, while labour cost was computed
against 160h. One worker therefore got ~584h of free physical capacity every month.

Fix: PHYSICAL CAPACITY HOURS = ECONOMICALLY PAID HOURS, always.

  operating_horizon_minutes(hours_per_worker_month) — the only definition of "how long the
  monthly SimPy clock runs for". A worker resource is only ever "on the clock" for this long,
  matching exactly the hours used in `labour_cost = workers * worker_cost_per_hour *
  hours_per_worker_month`.

  compress_to_operating_time() — maps each order's real calendar position within ITS OWN
  calendar month onto [0, operating_horizon_minutes), preserving relative ordering and
  burstiness (spec: "the workload that must be processed during the available monthly
  productive hours"). Used identically for:
    - Future Planning: the generator still produces realistic calendar-shaped arrival
      timestamps spread across the full month (seasonal burst pattern, intraday weights) —
      this function then compresses that shape onto the operating horizon.
    - Historical Analysis: real order timestamps are compressed onto the operating horizon of
      the month they fall in, so historical demand is replayed against the same monthly FTE
      capacity basis as Future Planning.

  slice_month_operating_time() — the one place that both filters `orders_all` down to a single
  month AND compresses it; used by every caller that needs "this month's orders, ready for
  simulation" (evaluate_rl3_monthly_capacity_cost.py, bottleneck_report.py) so there is exactly
  one implementation of that combined step.

SIM_EPOCH is an arbitrary fixed reference timestamp — NOT derived from the data — so the
simulated horizon always starts at operating-minute 0 regardless of which order happens to
arrive first. This keeps the internal SimPy engines working in pandas Timestamps (minimal
change to stage_metrics.py, which just diffs two timestamps) while decoupling the simulated
clock entirely from wall-clock/calendar time.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

SIM_EPOCH = pd.Timestamp("2000-01-01")


def operating_horizon_minutes(hours_per_worker_month: float) -> float:
    """The only definition of the monthly SimPy horizon: one worker's paid hours, in minutes."""
    return float(hours_per_worker_month) * 60.0


def equivalent_operating_days(hours_per_worker_month: float, hours_per_operating_day: float) -> float:
    """Explanatory-only conversion (spec §6) — never a second capacity source of truth."""
    if hours_per_operating_day <= 0:
        return 0.0
    return float(hours_per_worker_month) / float(hours_per_operating_day)


def with_operating_horizon(sim_cfg: dict, hours_per_worker_month: float) -> dict:
    """Return a copy of sim_cfg with operating_horizon_minutes set — call this at every site
    that loads sim_multistage.yaml, before the config reaches run_simulation_multistage /
    FullStageRLRunner. There are exactly two such sites (evaluate_rl3_monthly_capacity_cost.py,
    bottleneck_report.py) plus the adaptive-search / capacity_search.py callers that receive
    sim_cfg already built by one of those two."""
    return {**sim_cfg, "operating_horizon_minutes": operating_horizon_minutes(hours_per_worker_month)}


def compress_to_operating_time(
    df: pd.DataFrame,
    horizon_minutes: float,
    calendar_col: str = "arrival_time",
) -> pd.DataFrame:
    """Map each row's calendar timestamp onto its own calendar month's operating-time window.

    calendar_fraction = (timestamp - month_start) / (month_end - month_start)
    operating_arrival_time = calendar_fraction * operating_horizon_minutes

    Adds/overwrites:
      - 'calendar_arrival_time': the original calendar timestamp (kept for display / demand
        composition — NEVER used for simulation scheduling after this point).
      - 'operating_arrival_min': float minutes from the operating horizon start (for
        debugging/validation).
      - 'arrival_time': overwritten with a synthetic SIM_EPOCH-anchored timestamp — this is
        what the SimPy engines actually schedule against.

    Relative ordering and intra-month dispersion (burstiness) are preserved exactly, since the
    mapping is a per-row linear rescale within each row's own month.
    """
    df = df.copy()
    ts = pd.to_datetime(df[calendar_col])
    month_start = ts.dt.to_period("M").dt.to_timestamp()
    month_end = month_start + pd.DateOffset(months=1)

    calendar_fraction = (ts - month_start) / (month_end - month_start)
    calendar_fraction = calendar_fraction.clip(lower=0.0, upper=0.999999)
    operating_min = (calendar_fraction * float(horizon_minutes)).astype(float)

    df["calendar_arrival_time"] = ts
    df["operating_arrival_min"] = operating_min
    df["arrival_time"] = SIM_EPOCH + pd.to_timedelta(operating_min, unit="m")
    return df


def slice_month_operating_time(
    orders_all: pd.DataFrame,
    month_num: int,
    horizon_minutes: float,
    month_col: str = "month",
) -> pd.DataFrame:
    """Filter `orders_all` to one month (sorted by real calendar arrival) and compress that
    slice onto the operating horizon. The one shared implementation of "this month's orders,
    ready for simulation" — used by evaluate_monthly_capacity_cost and build_bottleneck_report
    so both always see the same operating-time-normalised data for a given month."""
    month_orders = (
        orders_all[orders_all[month_col] == month_num]
        .sort_values("arrival_time")
        .reset_index(drop=True)
    )
    return compress_to_operating_time(month_orders, horizon_minutes)


def horizon_end_timestamp(horizon_minutes: float) -> pd.Timestamp:
    return SIM_EPOCH + pd.to_timedelta(float(horizon_minutes), unit="m")


def rebase_to_sim_clock(df: pd.DataFrame, calendar_col: str = "arrival_time") -> "tuple[pd.DataFrame, float]":
    """Diagnostic/legacy helper for standalone research scripts that evaluate an arbitrary
    order window (not a single real calendar month) — e.g. evaluate_rl3.py,
    evaluate_rl3_multiseed.py. Rebases arrival_time to a SIM_EPOCH-anchored clock starting at
    the window's own first arrival (matching the pre-operating-time-model "run until done"
    behaviour those scripts were written around), and returns a horizon spanning the window
    plus a margin — NOT a finite monthly capacity horizon. Business-path code (Future/Historical
    evaluation) must use compress_to_operating_time / slice_month_operating_time instead, which
    map onto the real, finite, paid monthly horizon.
    """
    df = df.copy()
    ts = pd.to_datetime(df[calendar_col])
    t0 = ts.min()
    offset_min = (ts - t0).dt.total_seconds() / 60.0
    df[calendar_col] = SIM_EPOCH + pd.to_timedelta(offset_min, unit="m")
    horizon_minutes = float(offset_min.max()) + 240.0  # span + 4h margin so the last arrivals can drain
    return df, horizon_minutes
