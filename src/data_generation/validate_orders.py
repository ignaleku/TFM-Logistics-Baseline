from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .synthetic_orders import generate_orders


def _assert_columns(df: pd.DataFrame) -> None:
    expected = [
        "order_id", "arrival_time", "month", "weekday", "hour",
        "order_type", "sla_minutes", "num_items", "product_class", "scenario"
    ]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise AssertionError(f"Faltan columnas: {missing}")


def _assert_no_nans(df: pd.DataFrame) -> None:
    if df.isna().any().any():
        cols = df.columns[df.isna().any()].tolist()
        raise AssertionError(f"Hay NaNs en columnas: {cols}")


def _assert_reproducible(config_path: str, scenario: str) -> None:
    df1 = generate_orders(config_path=config_path, scenario=scenario, seed=42).reset_index(drop=True)
    df2 = generate_orders(config_path=config_path, scenario=scenario, seed=42).reset_index(drop=True)
    pd.testing.assert_frame_equal(df1, df2)

    df3 = generate_orders(config_path=config_path, scenario=scenario, seed=43).reset_index(drop=True)
    if df1.equals(df3):
        raise AssertionError("Con distinta semilla sale idéntico. Revisa el RNG.")


def _plot_distributions(df: pd.DataFrame, fig_dir: Path, tag: str) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Mes
    month_counts = df["month"].value_counts().sort_index()
    plt.figure()
    month_counts.plot(kind="bar")
    plt.title(f"Pedidos por mes — {tag}")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{tag}_orders_by_month.png")
    plt.close()

    # Semana
    weekday_order = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    wd_counts = df["weekday"].value_counts().reindex(weekday_order).fillna(0)
    plt.figure()
    wd_counts.plot(kind="bar")
    plt.title(f"Pedidos por día de la semana — {tag}")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{tag}_orders_by_weekday.png")
    plt.close()

    # Hora
    hour_counts = df["hour"].value_counts().sort_index()
    plt.figure()
    hour_counts.plot(kind="bar")
    plt.title(f"Pedidos por hora — {tag}")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{tag}_orders_by_hour.png")
    plt.close()


def _print_summary(df: pd.DataFrame, tag: str) -> None:
    urgent_rate = (df["order_type"] == "urgent").mean()
    print(
        f"[{tag}] total={len(df):,} | %urgent={urgent_rate:.3f} | "
        f"items_mean={df['num_items'].mean():.2f} | items_p90={df['num_items'].quantile(0.9):.2f}"
    )


def validate(config_path: str, fig_dir: str) -> None:
    fig_dir_path = Path(fig_dir)

    for scenario in ["base", "peak_campaign", "stress"]:
        df = generate_orders(config_path=config_path, scenario=scenario, seed=42)

        _assert_columns(df)
        _assert_no_nans(df)
        if (df["num_items"] < 1).any():
            raise AssertionError(f"[{scenario}] num_items tiene valores < 1")

        # arrival_time ordenado (esperable)
        at = pd.to_datetime(df["arrival_time"])
        if not at.is_monotonic_increasing:
            raise AssertionError(f"[{scenario}] arrival_time NO está ordenado. (Se esperaba ordenado)")

        _print_summary(df, scenario)
        _plot_distributions(df, fig_dir_path, scenario)

    _assert_reproducible(config_path=config_path, scenario="base")

    print(f"\n✅ Validación completada. Figuras en: {fig_dir_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Valida el generador de pedidos (Fase 1) y genera gráficas.")
    parser.add_argument("--config", type=str, default="configs/demand_base.yaml", help="Ruta al YAML de configuración.")
    parser.add_argument("--figdir", type=str, default="reports/figures", help="Carpeta de salida para figuras.")
    args = parser.parse_args()

    validate(config_path=args.config, fig_dir=args.figdir)


if __name__ == "__main__":
    main()
