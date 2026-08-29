"""
THESIS-ONLY. Derivation of Table 5.8 (policy divergence by capacity-pressure band).

Read-only: consumes data/api_runs/latest/historical/rl3_monthly_capacity_cost_results.csv
and writes nothing to the production system. Added during the final correction pass to make
the signed and per-1,000-order columns of Table 5.8 reproducible.

Three statistics are computed per picking-utilisation band, over the 192 (month, regime)
pairs at which both urgency-aware policies were evaluated at equal workforce:

  absolute gap   |C_RL3 - C_UF|          -- separation between the policies
  per 1,000 ord. |C_RL3 - C_UF| / n * 1e3 -- separation normalised for month volume
  signed gap     C_UF - C_RL3            -- positive where RL-3 is cheaper,
                                            split by whether any feasible plan existed

The signed split is what distinguishes divergence from advantage: the absolute gap rises
monotonically across the bands, but the sign reverses where neither policy reaches
feasibility.

Run:  python thesis_support/analysis/verify_pressure_bands.py
"""
from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "data" / "api_runs" / "latest" / "historical" / "rl3_monthly_capacity_cost_results.csv"

BANDS = [
    (0.00, 0.70, "below 70%"),
    (0.70, 0.80, "70-80%"),
    (0.80, 0.90, "80-90%"),
    (0.90, 0.95, "90-95%"),
    (0.95, 2.00, "above 95%"),
]


def _truthy(value: str) -> bool:
    return value.strip().lower() == "true"


def main() -> None:
    rows = list(csv.DictReader(RESULTS.open(encoding="utf-8")))
    by_config: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_config[(row["month"], row["regime"])][row["policy"]] = row

    absolute: dict[str, list[float]] = defaultdict(list)
    per_thousand: dict[str, list[float]] = defaultdict(list)
    signed_feasible: dict[str, list[float]] = defaultdict(list)
    signed_infeasible: dict[str, list[float]] = defaultdict(list)

    for policies in by_config.values():
        rl3, uf = policies.get("rl3_dqn"), policies.get("urgent_first")
        if rl3 is None or uf is None:
            continue

        utilisation = float(rl3["picking_utilisation"])
        cost_rl3 = float(rl3["estimated_total_cost"])
        cost_uf = float(uf["estimated_total_cost"])
        orders = float(rl3["total_orders"])
        signed = cost_uf - cost_rl3  # positive => RL-3 cheaper
        any_feasible = _truthy(rl3["feasible"]) or _truthy(uf["feasible"])

        for low, high, label in BANDS:
            if low <= utilisation < high:
                absolute[label].append(abs(signed))
                per_thousand[label].append(abs(signed) / orders * 1000.0)
                (signed_feasible if any_feasible else signed_infeasible)[label].append(signed)
                break

    header = f"{'band':<12}{'n':>5}{'abs mean':>12}{'per 1000':>11}{'signed|feas':>14}{'signed|infeas':>16}"
    print(header)
    print("-" * len(header))
    for _, _, label in BANDS:
        n = len(absolute[label])
        feas, infeas = signed_feasible[label], signed_infeasible[label]
        mean_feas = f"{statistics.mean(feas):+,.1f} ({len(feas)})" if feas else "--"
        mean_infeas = f"{statistics.mean(infeas):+,.0f} ({len(infeas)})" if infeas else "--"
        print(
            f"{label:<12}{n:>5}{statistics.mean(absolute[label]):>12,.0f}"
            f"{statistics.mean(per_thousand[label]):>11,.0f}{mean_feas:>14}{mean_infeas:>16}"
        )


if __name__ == "__main__":
    main()
