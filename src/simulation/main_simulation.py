from __future__ import annotations

from pathlib import Path
import yaml
import pandas as pd

from src.simulation.sim_mvp import run_simulation_mvp


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SIM_CFG_PATH = PROJECT_ROOT / "configs" / "sim_mvp.yaml"
ORDERS_PATH = PROJECT_ROOT / "data" / "orders_base.csv"


def main() -> None:
    print("▶️  Fase 2 — SimPy MVP (1 etapa)")

    with open(SIM_CFG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    sim_cfg = cfg["simulation"]
    service_cfg = cfg["service_time"]

    workers = int(sim_cfg["workers"])
    seed = int(sim_cfg["random_seed"])

    orders = pd.read_csv(ORDERS_PATH, parse_dates=["arrival_time"])
    orders = orders.sort_values("arrival_time").reset_index(drop=True)

    results_df, summary = run_simulation_mvp(orders, sim_cfg, service_cfg)

    # Output con nombre único
    out_path = PROJECT_ROOT / "data" / f"sim_results_base_mvp_workers_{workers}_seed_{seed}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)

    print("✅ Resultados guardados en:", out_path)
    print("📌 Summary:")
    for k, v in summary.items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
