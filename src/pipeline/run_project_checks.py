# src/pipeline/run_project_checks.py
"""
Reproducibility check: verifies that all required project artefacts exist
and prints the next command to run if any are missing.

Does NOT execute training, simulation, or data generation automatically.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CHECKS = [
    {
        "label": "Config: demand_base.yaml",
        "path": "configs/demand_base.yaml",
        "command": None,
    },
    {
        "label": "Config: sim_multistage.yaml",
        "path": "configs/sim_multistage.yaml",
        "command": None,
    },
    {
        "label": "Config: rl.yaml",
        "path": "configs/rl.yaml",
        "command": None,
    },
    {
        "label": "Data:   orders_base.csv",
        "path": "data/orders_base.csv",
        "command": "python -m src.data_generation.main_data_generation",
    },
    {
        "label": "Model:  dqn_final.pt",
        "path": "data/dqn_final.pt",
        "command": "python -m src.rl.main_train_rl",
    },
    {
        "label": "Eval:   rl_eval_results.csv",
        "path": "data/rl_eval_results.csv",
        "command": "python -m src.rl.evaluate_dqn",
    },
    {
        "label": "Eval:   rl_eval_multiseed_results.csv",
        "path": "data/rl_eval_multiseed_results.csv",
        "command": "python -m src.rl.evaluate_dqn_multiseed",
    },
]


def main() -> None:
    print("=" * 60)
    print("  Project reproducibility check")
    print(f"  Root: {ROOT}")
    print("=" * 60)

    first_missing_command: str | None = None
    all_ok = True

    for check in CHECKS:
        path = ROOT / check["path"]
        exists = path.exists()
        status = "OK     " if exists else "MISSING"
        print(f"  [{status}]  {check['label']}")
        if not exists:
            all_ok = False
            if first_missing_command is None:
                first_missing_command = check["command"]

    print("-" * 60)

    if all_ok:
        print("  All artefacts present. Pipeline is complete.")
    else:
        if first_missing_command:
            print("  Next step:")
            print(f"    {first_missing_command}")
        else:
            print("  Missing config files. Check the configs/ directory.")

    print("=" * 60)


if __name__ == "__main__":
    main()
