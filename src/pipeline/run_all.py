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
  --sim-5stage          Run 5-stage simulation baselines (FIFO / urgent_first)

  RL-3 pipeline
  --train-rl3           Train RL-3 agent  (slow)
  --eval-rl3            Evaluate RL-3 agent (7 regimes)
  --multiseed-rl3       Multi-seed RL-3 evaluation  (slow)

  RL-5 pipeline
  --train-rl5           Train RL-5 DQN agent  (slow)
  --eval-rl5            Evaluate RL-5 across 7 regimes
  --multiseed-rl5       Multi-window RL-5 evaluation (5 windows × 7 regimes)

  RL-5 decision-support analytics
  --monthly-rl5         Monthly RL-5 evaluation (per-month SLA + cost)
  --capacity-cost-rl5   Monthly capacity-cost optimisation (15 regimes × 12 months)
  --worker-cost-sensitivity  Worker-cost sensitivity sweep (pure pandas, fast)
  --calibrate-costs     Economic assumption calibration sweep (180 combos, fast)
  --export-app-data     Export webapp-ready recommendation CSVs

  Reporting & checks
  --plots               Generate final result plots (RL-3)
  --cost-analysis       SLA business penalty-cost analysis
  --checks              Run project sanity checks

Composite flags:

  --base                --data --simulation --checks
  --rl3                 --train-rl3 --eval-rl3 --multiseed-rl3
  --rl5                 --train-rl5 --eval-rl5 --multiseed-rl5
  --decision-support    --capacity-cost-rl5 --worker-cost-sensitivity
                        --calibrate-costs --export-app-data
  --all                 Everything (training steps can take a long time)

Examples:

  1. python -m src.pipeline.run_all --base
  2. python -m src.pipeline.run_all --train-rl5
  3. python -m src.pipeline.run_all --eval-rl5 --multiseed-rl5
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
    parser.add_argument("--sim-5stage", action="store_true", help="Run 5-stage simulation baselines")

    # RL-3
    parser.add_argument("--train-rl3",     action="store_true", help="Train RL-3 agent (slow)")
    parser.add_argument("--eval-rl3",      action="store_true", help="Evaluate RL-3 agent")
    parser.add_argument("--multiseed-rl3", action="store_true", help="Multi-seed RL-3 evaluation (slow)")

    # RL-5
    parser.add_argument("--train-rl5",     action="store_true", help="Train RL-5 DQN agent (slow)")
    parser.add_argument("--eval-rl5",      action="store_true", help="Evaluate RL-5 across 7 regimes")
    parser.add_argument("--multiseed-rl5", action="store_true", help="Multi-window RL-5 evaluation (slow)")

    # RL-5 decision-support analytics
    parser.add_argument("--monthly-rl5",           action="store_true", help="Monthly RL-5 evaluation")
    parser.add_argument("--capacity-cost-rl5",     action="store_true", help="Monthly capacity-cost optimisation")
    parser.add_argument("--worker-cost-sensitivity", action="store_true", help="Worker-cost sensitivity sweep (fast)")
    parser.add_argument("--calibrate-costs",        action="store_true", help="Economic assumption calibration (fast)")
    parser.add_argument("--export-app-data",        action="store_true", help="Export webapp-ready recommendation CSVs")

    # Reporting & checks
    parser.add_argument("--plots",        action="store_true", help="Generate final result plots")
    parser.add_argument("--cost-analysis",action="store_true", help="SLA business penalty-cost analysis")
    parser.add_argument("--checks",       action="store_true", help="Run project sanity checks")

    # Composite
    parser.add_argument("--base",             action="store_true", help="--data --simulation --checks")
    parser.add_argument("--rl3",              action="store_true", help="--train-rl3 --eval-rl3 --multiseed-rl3")
    parser.add_argument("--rl5",              action="store_true", help="--train-rl5 --eval-rl5 --multiseed-rl5")
    parser.add_argument("--decision-support", action="store_true",
                        help="--capacity-cost-rl5 --worker-cost-sensitivity --calibrate-costs --export-app-data")
    parser.add_argument("--all",              action="store_true",
                        help="Run every step (training + evaluation can take a long time)")

    args = parser.parse_args()

    # Expand composite flags
    if args.base:
        args.data = args.simulation = args.checks = True
    if args.rl3:
        args.train_rl3 = args.eval_rl3 = args.multiseed_rl3 = True
    if args.rl5:
        args.train_rl5 = args.eval_rl5 = args.multiseed_rl5 = True
    if args.decision_support:
        args.capacity_cost_rl5 = args.worker_cost_sensitivity = True
        args.calibrate_costs   = args.export_app_data         = True
    if args.all:
        args.data = args.simulation = args.sim_5stage = True
        args.train_rl3 = args.eval_rl3 = args.multiseed_rl3 = True
        args.train_rl5 = args.eval_rl5 = args.multiseed_rl5 = True
        args.monthly_rl5 = args.capacity_cost_rl5 = True
        args.worker_cost_sensitivity = args.calibrate_costs = args.export_app_data = True
        args.plots = args.cost_analysis = args.checks = True

    py = sys.executable

    steps_requested = any([
        args.data, args.simulation, args.sim_5stage,
        args.train_rl3, args.eval_rl3, args.multiseed_rl3,
        args.train_rl5, args.eval_rl5, args.multiseed_rl5,
        args.monthly_rl5, args.capacity_cost_rl5,
        args.worker_cost_sensitivity, args.calibrate_costs, args.export_app_data,
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

    if args.sim_5stage:
        run([py, "-m", "src.simulation.multistage.sim_5stage"])
        completed.append("sim-5stage")

    if args.train_rl3:
        run([py, "-m", "src.rl.main_train_rl3"])
        completed.append("train-rl3")

    if args.eval_rl3:
        run([py, "-m", "src.rl.evaluate_rl3"])
        completed.append("eval-rl3")

    if args.multiseed_rl3:
        run([py, "-m", "src.rl.evaluate_rl3_multiseed"])
        completed.append("multiseed-rl3")

    if args.train_rl5:
        run([py, "-m", "src.rl.main_train_rl5"])
        completed.append("train-rl5")

    if args.eval_rl5:
        run([py, "-m", "src.rl.evaluate_rl5"])
        completed.append("eval-rl5")

    if args.multiseed_rl5:
        run([py, "-m", "src.rl.evaluate_rl5_multiseed"])
        completed.append("multiseed-rl5")

    if args.monthly_rl5:
        run([py, "-m", "src.rl.evaluate_rl5_monthly"])
        completed.append("monthly-rl5")

    if args.capacity_cost_rl5:
        run([py, "-m", "src.rl.evaluate_rl5_monthly_capacity_cost"])
        completed.append("capacity-cost-rl5")

    if args.worker_cost_sensitivity:
        run([py, "-m", "src.rl.evaluate_rl5_worker_cost_sensitivity"])
        completed.append("worker-cost-sensitivity")

    if args.calibrate_costs:
        run([py, "-m", "src.analysis.calibrate_capacity_cost_assumptions"])
        completed.append("calibrate-costs")

    if args.export_app_data:
        run([py, "-m", "src.reporting.export_rl5_monthly_recommendations"])
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