from __future__ import annotations

from typing import Dict
import numpy as np


def sample_service_time_minutes(
    rng: np.random.Generator,
    num_items: int,
    product_class: str,
    cfg: Dict,
) -> float:
    """
    Tiempo de servicio simple pero realista:
      base + alpha * items, multiplicado por complejidad de clase + ruido lognormal.
    """
    base = float(cfg["base_minutes"])
    per_item = float(cfg["minutes_per_item"])
    mult_map = cfg["class_multiplier"]
    sigma = float(cfg["noise_lognormal_sigma"])
    min_minutes = float(cfg["min_minutes"])

    mult = float(mult_map.get(product_class, 1.0))
    deterministic = (base + per_item * float(num_items)) * mult

    # Ruido multiplicativo (lognormal centrado aprox en 1)
    noise = rng.lognormal(mean=0.0, sigma=sigma)

    service = deterministic * noise
    return max(service, min_minutes)
