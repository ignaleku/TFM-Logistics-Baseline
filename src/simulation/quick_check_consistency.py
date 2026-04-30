from __future__ import annotations

from pathlib import Path
import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    csv_path = root / "data" / "sim_results_base_mvp_workers_8_seed_123.csv"

    print("Leyendo:", csv_path)
    df = pd.read_csv(
        csv_path,
        parse_dates=["arrival_time", "start_service_time", "end_service_time"],
    )

    assert (df["start_service_time"] >= df["arrival_time"]).all(), "start < arrival en alguna fila"
    assert (df["end_service_time"] >= df["start_service_time"]).all(), "end < start en alguna fila"
    assert (df["waiting_time_min"] >= 0).all(), "waiting < 0 en alguna fila"
    assert (df["service_time_min"] > 0).all(), "service <= 0 en alguna fila"

    err = (df["system_time_min"] - (df["waiting_time_min"] + df["service_time_min"])).abs().max()
    print("max abs error (system - (waiting + service)):", err)

    print("✅ Consistencia OK.")


if __name__ == "__main__":
    main()
