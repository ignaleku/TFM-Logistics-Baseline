from __future__ import annotations

from pathlib import Path
import yaml
import pandas as pd

from src.simulation.multistage.sim_multistage import run_simulation_multistage


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    cfg_path = root / "configs" / "sim_multistage.yaml"
    orders_path = root / "data" / "orders_base.csv"

    print("▶️  Fase 3 — Multi-etapa (Ticket 1)")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    sim_cfg = cfg["simulation"]
    resources_cfg = cfg["resources"]
    service_cfg = cfg["service_time"]

    policy_name = sim_cfg.get("policy", "fifo")

    orders = pd.read_csv(orders_path, parse_dates=["arrival_time"])
    orders = orders.sort_values("arrival_time").reset_index(drop=True)
    orders = orders.head(10_000).copy()

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
