from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data_generation.synthetic_orders import generate_orders


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera pedidos sintéticos y los guarda en CSV.")
    parser.add_argument("--config", type=str, default="configs/demand_base.yaml", help="Ruta al YAML de configuración.")
    parser.add_argument("--scenario", type=str, default="base", help="Escenario: base | peak_campaign | stress (según YAML).")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para reproducibilidad.")
    parser.add_argument("--out", type=str, default="data/orders.csv", help="Ruta CSV de salida.")
    args = parser.parse_args()

    df = generate_orders(config_path=args.config, scenario=args.scenario, seed=args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"✅ Generado: {out_path} | filas={len(df):,} | escenario={args.scenario} | seed={args.seed}")


if __name__ == "__main__":
    main()
