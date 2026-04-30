from __future__ import annotations

from pathlib import Path
import pandas as pd


def project_root() -> Path:
    # .../TFM-Logistic-Process/src/data_generation/quick_check_campaign.py
    # parents[2] -> .../TFM-Logistic-Process
    return Path(__file__).resolve().parents[2]


def read_orders_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe el CSV: {path}")
    return pd.read_csv(path, parse_dates=["arrival_time"])


def main() -> None:
    root = project_root()
    data_dir = root / "data"

    base_path = data_dir / "orders_base.csv"
    peak_path = data_dir / "orders_peak_campaign.csv"

    print("📌 Project root:", root)
    print("📄 Leyendo base:", base_path)
    print("📄 Leyendo peak:", peak_path)

    # Lectura robusta
    try:
        base = read_orders_csv(base_path)
        peak = read_orders_csv(peak_path)
    except FileNotFoundError as e:
        print("\n❌", e)
        print("   Revisa que hayas generado los CSV en:", data_dir)
        return

    # --- Check campaña ---
    base_daily = base.groupby(base["arrival_time"].dt.date).size()
    peak_daily = peak.groupby(peak["arrival_time"].dt.date).size()

    start = pd.to_datetime("2026-11-15").date()
    end = pd.to_datetime("2026-11-30").date()

    # Medias dentro/fuera de ventana
    base_campaign_mean = base_daily.loc[start:end].mean()
    peak_campaign_mean = peak_daily.loc[start:end].mean()

    base_non = base_daily.drop(base_daily.loc[start:end].index, errors="ignore")
    peak_non = peak_daily.drop(peak_daily.loc[start:end].index, errors="ignore")

    base_non_mean = base_non.mean()
    peak_non_mean = peak_non.mean()

    print("\n=== CHECK CAMPAÑA ===")
    print("BASE campaña media/día:", base_campaign_mean)
    print("PEAK campaña media/día:", peak_campaign_mean)
    print("BASE fuera campaña media/día:", base_non_mean)
    print("PEAK fuera campaña media/día:", peak_non_mean)
    print("Ratio campaña (peak/base):", peak_campaign_mean / base_campaign_mean)
    print("Ratio fuera campaña (peak/base):", peak_non_mean / base_non_mean)

    # --- Check fin de semana (y weekdays en general) ---
    wd_counts = base["arrival_time"].dt.day_name().str.lower().value_counts()
    saturday = int(wd_counts.get("saturday", 0))
    sunday = int(wd_counts.get("sunday", 0))

    print("\n=== CHECK WEEKDAYS (BASE) ===")
    print(wd_counts)
    print(f"\nSábado={saturday} | Domingo={sunday}")

    # Extra: check orden temporal (útil)
    is_sorted = base["arrival_time"].is_monotonic_increasing
    print("\nOrdenado por arrival_time (base):", is_sorted)


if __name__ == "__main__":
    main()
