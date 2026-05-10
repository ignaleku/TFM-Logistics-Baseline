"""
Project pipeline runner.

Orchestrates existing modules via subprocess. Does not contain any logic
beyond sequencing commands.

Usage:
    python -m src.pipeline.run_all [flags]
"""

import argparse
import subprocess
import sys
import time


def run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    start = time.time()
    subprocess.run(cmd, check=True)
    elapsed = time.time() - start
    print(f"    Done in {elapsed:.1f}s")


def print_help_and_examples() -> None:
    print(
        """
TFM Logistics — Project Runner
===============================

No flags provided. Nothing was run.

Available flags:

  Data & simulation
  --data                Generate synthetic order data
  --simulation          Run 3-stage multistage simulation (RL-3 baseline)

  RL-3 pipeline
  --train-rl3           Train RL-3 agent  (slow)
  --eval-rl3            Evaluate RL-3 agent (7 regimes)
  --multiseed-rl3       Multi-seed RL-3 evaluation  (slow)

  RL-3 decision-support analytics
  --monthly-capacity-rl3  Monthly capacity-cost optimisation (7 regimes × 12 months × 3 policies)
  --export-app-data       Export webapp-ready recommendation CSVs

  Reporting & checks
  --plots               Generate final result plots (RL-3)
  --cost-analysis       SLA business penalty-cost analysis
  --checks              Run project sanity checks

Composite flags:

  --base                --data --simulation --checks
  --rl3                 --train-rl3 --eval-rl3 --multiseed-rl3
  --decision-support    --monthly-capacity-rl3 --export-app-data
  --all                 Everything (training steps can take a long time)

Examples:

  1. python -m src.pipeline.run_all --base
  2. python -m src.pipeline.run_all --train-rl3
  3. python -m src.pipeline.run_all --eval-rl3 --multiseed-rl3
  4. python -m src.pipeline.run_all --decision-support
  5. python -m src.pipeline.run_all --all
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.pipeline.run_all",
        description="Orchestrate the TFM Logistics pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data & simulation
    parser.add_argument("--data",       action="store_true", help="Generate synthetic order data")
    parser.add_argument("--simulation", action="store_true", help="Run 3-stage multistage simulation")

    # RL-3
    parser.add_argument("--train-rl3",     action="store_true", help="Train RL-3 agent (slow)")
    parser.add_argument("--eval-rl3",      action="store_true", help="Evaluate RL-3 agent")
    parser.add_argument("--multiseed-rl3", action="store_true", help="Multi-seed RL-3 evaluation (slow)")

    # RL-3 decision-support analytics
    parser.add_argument("--monthly-capacity-rl3", action="store_true",
                        help="Monthly capacity-cost optimisation (RL-3)")
    parser.add_argument("--export-app-data", action="store_true",
                        help="Export webapp-ready recommendation CSVs (RL-3)")

    # Reporting & checks
    parser.add_argument("--plots",         action="store_true", help="Generate final result plots")
    parser.add_argument("--cost-analysis", action="store_true", help="SLA business penalty-cost analysis")
    parser.add_argument("--checks",        action="store_true", help="Run project sanity checks")

    # Composite
    parser.add_argument("--base",             action="store_true", help="--data --simulation --checks")
    parser.add_argument("--rl3",              action="store_true", help="--train-rl3 --eval-rl3 --multiseed-rl3")
    parser.add_argument("--decision-support", action="store_true",
                        help="--monthly-capacity-rl3 --export-app-data")
    parser.add_argument("--all",              action="store_true",
                        help="Run every step (training can take a long time)")

    args = parser.parse_args()

    # Expand composite flags
    if args.base:
        args.data = args.simulation = args.checks = True
    if args.rl3:
        args.train_rl3 = args.eval_rl3 = args.multiseed_rl3 = True
    if args.decision_support:
        args.monthly_capacity_rl3 = args.export_app_data = True
    if args.all:
        args.data = args.simulation = True
        args.train_rl3 = args.eval_rl3 = args.multiseed_rl3 = True
        args.monthly_capacity_rl3 = args.export_app_data = True
        args.plots = args.cost_analysis = args.checks = True

    py = sys.executable

    steps_requested = any([
        args.data, args.simulation,
        args.train_rl3, args.eval_rl3, args.multiseed_rl3,
        args.monthly_capacity_rl3, args.export_app_data,
        args.plots, args.cost_analysis, args.checks,
    ])

    if not steps_requested:
        print_help_and_examples()
        return

    completed: list[str] = []
    pipeline_start = time.time()

    if args.data:
        run([py, "-m", "src.data_generation.main_data_generation"])
        completed.append("data")

    if args.simulation:
        run([py, "-m", "src.simulation.multistage.main_multistage"])
        completed.append("simulation")

    if args.train_rl3:
        run([py, "-m", "src.rl.main_train_rl3"])
        completed.append("train-rl3")

    if args.eval_rl3:
        run([py, "-m", "src.rl.evaluate_rl3"])
        completed.append("eval-rl3")

    if args.multiseed_rl3:
        run([py, "-m", "src.rl.evaluate_rl3_multiseed"])
        completed.append("multiseed-rl3")

    if args.monthly_capacity_rl3:
        run([py, "-m", "src.rl.evaluate_rl3_monthly_capacity_cost"])
        completed.append("monthly-capacity-rl3")

    if args.export_app_data:
        run([py, "-m", "src.reporting.export_rl3_monthly_recommendations"])
        completed.append("export-app-data")

    if args.plots:
        run([py, "-m", "src.reporting.plot_final_results"])
        completed.append("plots")

    if args.cost_analysis:
        run([py, "-m", "src.reporting.sla_cost_calculator"])
        completed.append("cost-analysis")

    if args.checks:
        run([py, "-m", "src.pipeline.run_project_checks"])
        run([py, "-m", "src.validation.quick_project_checks"])
        completed.append("checks")

    total = time.time() - pipeline_start
    print(f"\nCompleted steps: {', '.join(completed)}")
    print(f"Total elapsed time: {total:.1f}s")


if __name__ == "__main__":
    main()
