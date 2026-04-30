from __future__ import annotations
from typing import Dict
import numpy as np


def sample_service_minutes(
    rng: np.random.Generator,
    num_items: int,
    product_class: str,
    cfg: Dict,
) -> float:
    base = float(cfg["base_minutes"])
    per_item = float(cfg["minutes_per_item"])
    sigma = float(cfg["noise_lognormal_sigma"])
    min_minutes = float(cfg["min_minutes"])

    mult_map = cfg["class_multiplier"]
    mult = float(mult_map.get(product_class, 1.0))

    deterministic = (base + per_item * float(num_items)) * mult
    noise = rng.lognormal(mean=0.0, sigma=sigma)
    return max(deterministic * noise, min_minutes)


def minutes_to_seconds_int(minutes: float) -> int:
    return max(1, int(round(float(minutes) * 60.0)))
