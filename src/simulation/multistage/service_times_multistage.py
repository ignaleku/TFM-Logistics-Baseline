from __future__ import annotations
from typing import Dict
import numpy as np


def sample_service_minutes(
    rng: np.random.Generator,
    units: float,
    cfg: Dict,
) -> float:
    """
    Sample service time in minutes from pre-computed workload units.

    units — picking_units / packing_units / dispatch_units (stage-specific)
    cfg   — must contain: base_minutes, minutes_per_unit,
             optionally: noise_clip_lo (default 0.80), noise_clip_hi (default 1.25),
             min_minutes (default 0.2)
    """
    base     = float(cfg["base_minutes"])
    per_unit = float(cfg["minutes_per_unit"])
    lo       = float(cfg.get("noise_clip_lo", 0.80))
    hi       = float(cfg.get("noise_clip_hi", 1.25))
    min_min  = float(cfg.get("min_minutes", 0.2))

    noise = float(np.clip(rng.normal(1.0, 0.12), lo, hi))
    return max(min_min, (base + per_unit * float(units)) * noise)


def minutes_to_seconds_int(minutes: float) -> int:
    return max(1, int(round(float(minutes) * 60.0)))
