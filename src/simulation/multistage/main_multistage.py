from __future__ import annotations

from pathlib import Path
import yaml
import pandas as pd

from src.simulation.multistage.sim_multistage import run_simulation_multistage
from src.simulation.multistage.operating_time import slice_month_operating_time, with_operating_horizon
from src.data.planning_profile import load_planning_profile


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    cfg_path = root / "configs" / "sim_multistage.yaml"
    orders_path = root / "data" / "orders_base_seasonal.csv"

    print("▶️  Fase 3 — Multi-etapa (Ticket 1)")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    hours_per_worker_month = float(load_planning_profile()["cost_defaults"]["hours_per_worker_month"])
    sim_cfg = with_operating_horizon(cfg["simulation"], hours_per_worker_month)
    resources_cfg = cfg["resources"]
    service_cfg = cfg["service_time"]

    policy_name = sim_cfg.get("policy", "fifo")

    orders = pd.read_csv(orders_path, parse_dates=["arrival_time"])
    orders = orders.sort_values("arrival_time").reset_index(drop=True)
    month_num = int(orders["month"].iloc[0])
    orders = slice_month_operating_time(
        orders, month_num, sim_cfg["operating_horizon_minutes"],
    )

    df, summary = run_simulation_multistage(
        orders=orders,
        sim_cfg=sim_cfg,
        resources_cfg=resources_cfg,
        service_cfg=service_cfg,
    )

    out_path = root / "data" / f"sim_results_multistage_{policy_name}.csv"
    df.to_csv(out_path, index=False)

    print("✅ Resultados guardados en:", out_path)
    print("📌 Summary:")
    for k, v in summary.items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
