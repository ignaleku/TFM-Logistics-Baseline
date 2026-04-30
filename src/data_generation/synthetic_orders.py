"""Generador de pedidos sintéticos basado en configuración YAML.

Fase 1 (inputs):
- Genera pedidos con estacionalidad mensual + patrón semanal + patrón horario.
- Soporta escenarios:
  - Multiplicador anual (sin ventana temporal)
  - Campaña con ventana temporal (redistribución dentro del año, SIN duplicar el total anual)
  - Multiplicador de urgencias (p.ej., escenario stress)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class ScenarioConfig:
    """Configuración específica de un escenario."""

    name: str
    demand_multiplier: float
    start: str | None = None
    end: str | None = None
    urgent_multiplier: float | None = None


def _load_config(config_path: str | Path) -> Dict:
    """Carga el YAML en un diccionario."""
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _build_date_index(year: int) -> pd.DatetimeIndex:
    """Construye el índice de fechas para un año completo (diario)."""
    start = datetime(year, 1, 1)
    end = datetime(year, 12, 31)
    return pd.date_range(start=start, end=end, freq="D")


def _monthly_weights(monthly_config: Dict[str, float]) -> Dict[int, float]:
    """Convierte el diccionario mensual a pesos por número de mes (1-12)."""
    month_order = [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ]
    weights = {idx + 1: float(monthly_config[name]) for idx, name in enumerate(month_order)}
    return weights


def _weekday_name(date_value: datetime) -> str:
    """Devuelve el nombre de día en inglés para mapear el patrón semanal."""
    return date_value.strftime("%A").lower()


def _scenario_from_config(config: Dict, scenario_name: str) -> ScenarioConfig:
    """Extrae la configuración del escenario solicitado."""
    scenario_config = config["scenarios"][scenario_name]
    return ScenarioConfig(
        name=scenario_name,
        demand_multiplier=float(scenario_config.get("demand_multiplier", 1.0)),
        start=scenario_config.get("start"),
        end=scenario_config.get("end"),
        urgent_multiplier=scenario_config.get("urgent_multiplier"),
    )


def _validate_prob_sum(name: str, probs: List[float], tol: float = 1e-6) -> None:
    """Valida que una lista de probabilidades suma ~1."""
    s = float(np.sum(probs))
    if not np.isfinite(s) or abs(s - 1.0) > tol:
        raise ValueError(f"{name} debe sumar 1.0 (suma actual={s:.6f}).")


def _validate_inputs(config: Dict) -> None:
    """Validaciones ligeras para evitar configs peligrosos."""
    # Monthly: suma ~1
    month_weights = _monthly_weights(config["monthly_seasonality"])
    _validate_prob_sum("monthly_seasonality", list(month_weights.values()), tol=1e-3)

    # Product class: suma ~1
    pc = config["product_class"]
    _validate_prob_sum(
        "product_class",
        [float(pc["A"]["probability"]), float(pc["B"]["probability"]), float(pc["C"]["probability"])],
        tol=1e-6,
    )

    # Order types: suma ~1
    ot = config["order_types"]
    _validate_prob_sum(
        "order_types",
        [float(ot["urgent"]["probability"]), float(ot["normal"]["probability"])],
        tol=1e-6,
    )

    # Hourly pattern: suma > 0
    hourly = config["hourly_pattern"]
    # claves pueden venir como int o str -> normalizamos al leer en _hourly_weights
    total = float(np.sum([float(v) for v in hourly.values()]))
    if total <= 0:
        raise ValueError("hourly_pattern debe sumar > 0.")


def _apply_campaign_multiplier(
    dates: pd.DatetimeIndex,
    weights: np.ndarray,
    scenario: ScenarioConfig,
) -> np.ndarray:
    """Aplica multiplicadores del escenario SOLO dentro de ventana temporal (campaña)."""
    adjusted = weights.copy()
    if scenario.start and scenario.end:
        start = pd.to_datetime(scenario.start)
        end = pd.to_datetime(scenario.end)
        mask = (dates >= start) & (dates <= end)
        adjusted[mask] *= scenario.demand_multiplier
    return adjusted


def _daily_weights(config: Dict, scenario: ScenarioConfig) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    """Calcula pesos diarios combinando estacionalidad y patrón semanal, y campaña si aplica."""
    dates = _build_date_index(int(config["global"]["year"]))
    month_weights = _monthly_weights(config["monthly_seasonality"])
    weekly_pattern = config["weekly_pattern"]

    weights = np.array(
        [month_weights[date.month] * float(weekly_pattern[_weekday_name(date)]) for date in dates],
        dtype=float,
    )

    # Campaña: repondera SOLO la ventana y luego normaliza
    weights = _apply_campaign_multiplier(dates, weights, scenario)

    total = float(weights.sum())
    if total <= 0:
        raise ValueError("La combinación de patrones produce cero pedidos (revisa weekly_pattern/monthly).")

    return dates, weights / total


def _hourly_weights(hourly_pattern: Dict) -> Tuple[List[int], np.ndarray]:
    """Normaliza los pesos por hora y devuelve horas y probabilidades (robusto a claves int/str)."""
    hours = sorted(int(h) for h in hourly_pattern.keys())

    weights_list: List[float] = []
    for h in hours:
        if str(h) in hourly_pattern:
            weights_list.append(float(hourly_pattern[str(h)]))
        else:
            weights_list.append(float(hourly_pattern[h]))  # clave int

    weights = np.array(weights_list, dtype=float)
    s = float(weights.sum())
    if s <= 0:
        raise ValueError("hourly_pattern suma 0 o negativo.")
    return hours, weights / s


def _allocate_orders(rng: np.random.Generator, total_orders: int, weights: np.ndarray) -> np.ndarray:
    """Distribuye el total de pedidos según pesos (multinomial)."""
    if total_orders < 0:
        raise ValueError("total_orders no puede ser negativo.")
    return rng.multinomial(int(total_orders), weights)


def _sample_order_types(
    rng: np.random.Generator,
    order_types: Dict,
    scenario: ScenarioConfig,
    size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Muestrea tipo de pedido y SLA asociado, aplicando urgent_multiplier si existe."""
    urgent_prob = float(order_types["urgent"]["probability"])
    normal_prob = float(order_types["normal"]["probability"])

    if scenario.urgent_multiplier:
        urgent_prob *= float(scenario.urgent_multiplier)

    total = urgent_prob + normal_prob
    if total <= 0:
        raise ValueError("Probabilidades de tipos de pedido inválidas (suma <= 0).")

    urgent_prob /= total
    normal_prob /= total

    types = rng.choice(["urgent", "normal"], size=size, p=[urgent_prob, normal_prob])
    sla_minutes = np.where(
        types == "urgent",
        int(order_types["urgent"]["sla_minutes"]),
        int(order_types["normal"]["sla_minutes"]),
    )
    return types, sla_minutes


def _sample_order_sizes(rng: np.random.Generator, size_config: Dict, size: int) -> np.ndarray:
    """Muestrea el número de ítems por pedido."""
    dist = str(size_config.get("distribution", "lognormal")).lower()
    mean = float(size_config["mean"])
    sigma = float(size_config["sigma"])
    min_items = int(size_config["min_items"])
    max_items = int(size_config["max_items"])

    if dist != "lognormal":
        raise ValueError(f"Distribución no soportada aún: {dist}. Usa 'lognormal' en Fase 1.")

    items = rng.lognormal(mean=mean, sigma=sigma, size=size)
    items = np.rint(items).astype(int)
    return np.clip(items, min_items, max_items)


def _sample_product_class(rng: np.random.Generator, product_class: Dict, size: int) -> np.ndarray:
    """Muestrea la clase de producto según probabilidades."""
    classes = ["A", "B", "C"]
    probabilities = [
        float(product_class["A"]["probability"]),
        float(product_class["B"]["probability"]),
        float(product_class["C"]["probability"]),
    ]
    # Si el YAML viniera mal, aquí fallará con ValueError de numpy
    return rng.choice(classes, size=size, p=probabilities)


def _expand_daily_orders(
    rng: np.random.Generator,
    dates: pd.DatetimeIndex,
    daily_orders: np.ndarray,
    hourly_pattern: Dict,
) -> List[datetime]:
    """Expande pedidos diarios a timestamps por hora con minutos/segundos aleatorios."""
    hours, hour_weights = _hourly_weights(hourly_pattern)
    arrival_times: List[datetime] = []

    for date, orders in zip(dates, daily_orders):
        if orders == 0:
            continue
        hourly_counts = _allocate_orders(rng, int(orders), hour_weights)
        for hour, count in zip(hours, hourly_counts):
            if count == 0:
                continue
            minutes = rng.integers(0, 60, size=int(count))
            seconds = rng.integers(0, 60, size=int(count))
            for minute, second in zip(minutes, seconds):
                arrival_times.append(
                    datetime(
                        int(date.year),
                        int(date.month),
                        int(date.day),
                        int(hour),
                        int(minute),
                        int(second),
                    )
                )

    return arrival_times


def generate_orders(config_path: str | Path, scenario: str, seed: int) -> pd.DataFrame:
    """Genera pedidos sintéticos basados en configuración y escenario.

    Reglas de escenarios:
    - Si el escenario NO tiene ventana (start/end): total anual = base_total * demand_multiplier
    - Si el escenario SÍ tiene ventana (campaña): total anual = base_total, y demand_multiplier
      se aplica SOLO como reponderación dentro de la ventana (sin duplicar el total anual).
    """

    config = _load_config(config_path)
    _validate_inputs(config)

    scenario_config = _scenario_from_config(config, scenario)
    rng = np.random.default_rng(int(seed))

    base_total = int(round(float(config["global"]["total_orders_year"])))

    # Evitamos el “doble multiplicador” en campañas con ventana:
    if scenario_config.start and scenario_config.end:
        total_orders = base_total
    else:
        total_orders = int(round(base_total * float(scenario_config.demand_multiplier)))

    dates, daily_weights = _daily_weights(config, scenario_config)
    daily_orders = _allocate_orders(rng, total_orders, daily_weights)

    arrival_times = _expand_daily_orders(
        rng,
        dates,
        daily_orders,
        config["hourly_pattern"],
    )

    num_orders = len(arrival_times)
    order_types, sla_minutes = _sample_order_types(
        rng,
        config["order_types"],
        scenario_config,
        num_orders,
    )
    num_items = _sample_order_sizes(rng, config["order_size"], num_orders)
    product_class = _sample_product_class(rng, config["product_class"], num_orders)

    arrival_series = pd.to_datetime(pd.Series(arrival_times), errors="raise")

    dataframe = pd.DataFrame(
        {
            "order_id": np.arange(1, num_orders + 1),
            "arrival_time": arrival_series,
            "month": arrival_series.dt.month,
            "weekday": arrival_series.dt.day_name().str.lower(),
            "hour": arrival_series.dt.hour,
            "order_type": order_types,
            "sla_minutes": sla_minutes,
            "num_items": num_items,
            "product_class": product_class,
            "scenario": scenario_config.name,
        }
    )

    # Recomendado para análisis y para SimPy (entradas ordenadas en el tiempo)
    dataframe = dataframe.sort_values("arrival_time").reset_index(drop=True)
    dataframe["order_id"] = np.arange(1, len(dataframe) + 1)

    return dataframe
