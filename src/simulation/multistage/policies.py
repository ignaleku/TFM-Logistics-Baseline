from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Optional


@dataclass
class Job:
    order_id: int
    order_type: str  # "urgent" | "normal"


PolicyFn = Callable[[List[Job]], Optional[int]]
# Devuelve el índice del job elegido en la lista (o None si vacío)


def fifo_policy(queue: List[Job]) -> Optional[int]:
    if not queue:
        return None
    return 0


def urgent_first_policy(queue: List[Job]) -> Optional[int]:
    if not queue:
        return None
    for i, job in enumerate(queue):
        if job.order_type == "urgent":
            return i
    return 0


def get_policy(name: str) -> PolicyFn:
    name = name.strip().lower()
    if name == "fifo":
        return fifo_policy
    if name in {"urgent_first", "urgent-first", "urgent"}:
        return urgent_first_policy
    raise ValueError(f"Política desconocida: {name}")
