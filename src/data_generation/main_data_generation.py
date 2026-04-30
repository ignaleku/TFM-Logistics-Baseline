"""
Main clásico para Fase 1: generación y validación de pedidos sintéticos.
"""

from pathlib import Path

from src.data_generation.synthetic_orders import generate_orders
from src.data_generation.validate_orders import validate

# ─────────────────────────────────────────────────────────────
# Rutas robustas (independientes del working directory)
# ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "demand_base.yaml"
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"


def run_generation() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    scenarios = ["base", "peak_campaign", "stress"]
    seed = 42

    print(f"Usando config: {CONFIG_PATH}")

    for scenario in scenarios:
        df = generate_orders(
            config_path=CONFIG_PATH,
            scenario=scenario,
            seed=seed,
        )
        out_path = DATA_DIR / f"orders_{scenario}.csv"
        df.to_csv(out_path, index=False)
        print(f"✅ Generado {out_path} ({len(df):,} pedidos)")


def run_validation() -> None:
    validate(
        config_path=str(CONFIG_PATH),
        fig_dir=str(FIG_DIR),
    )


def main() -> None:
    print("▶️  Fase 1 — Generación de datos sintéticos")
    run_generation()

    print("\n▶️  Fase 1 — Validación")
    run_validation()

    print("\n🏁 Fase 1 completada.")


if __name__ == "__main__":
    main()
