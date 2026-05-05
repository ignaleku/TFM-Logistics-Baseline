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

  --data            Generate synthetic data
  --simulation      Run multistage simulation
  --train-rl3       Train RL-3 agent  (slow)
  --eval-rl3        Evaluate RL-3 agent
  --multiseed-rl3   Multi-seed RL-3 evaluation  (slow)
  --plots           Generate final result plots
  --checks          Run project sanity checks

Composite flags:

  --base            --data --simulation --checks
  --rl3             --train-rl3 --eval-rl3 --multiseed-rl3
  --all             Everything (training + evaluation can take a long time)

Examples:

  1. python -m src.pipeline.run_all --base
  2. python -m src.pipeline.run_all --train-rl3
  3. python -m src.pipeline.run_all --eval-rl3 --plots
  4. python -m src.pipeline.run_all --all
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.pipeline.run_all",
        description="Orchestrate the TFM Logistics pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--data", action="store_true", help="Generate synthetic data")
    parser.add_argument("--simulation", action="store_true", help="Run multistage simulation")
    parser.add_argument("--train-rl3", action="store_true", help="Train RL-3 agent (slow)")
    parser.add_argument("--eval-rl3", action="store_true", help="Evaluate RL-3 agent")
    parser.add_argument("--multiseed-rl3", action="store_true", help="Multi-seed RL-3 evaluation (slow)")
    parser.add_argument("--plots", action="store_true", help="Generate final result plots")
    parser.add_argument("--checks", action="store_true", help="Run project sanity checks")

    parser.add_argument("--base", action="store_true", help="Equivalent to --data --simulation --checks")
    parser.add_argument("--rl3", action="store_true", help="Equivalent to --train-rl3 --eval-rl3 --multiseed-rl3")
    parser.add_argument("--all", action="store_true", help="Run every step (training + evaluation can take a long time)")

    args = parser.parse_args()

    # Expand composite flags
    if args.base:
        args.data = args.simulation = args.checks = True
    if args.rl3:
        args.train_rl3 = args.eval_rl3 = args.multiseed_rl3 = True
    if args.all:
        args.data = args.simulation = True
        args.train_rl3 = args.eval_rl3 = args.multiseed_rl3 = True
        args.plots = args.checks = True

    py = sys.executable

    # Resolve hyphenated flag names to their argparse attribute equivalents
    steps_requested = any([
        args.data,
        args.simulation,
        args.train_rl3,
        args.eval_rl3,
        args.multiseed_rl3,
        args.plots,
        args.checks,
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

    if args.plots:
        run([py, "-m", "src.reporting.plot_final_results"])
        completed.append("plots")

    if args.checks:
        run([py, "-m", "src.pipeline.run_project_checks"])
        run([py, "-m", "src.validation.quick_project_checks"])
        completed.append("checks")

    total = time.time() - pipeline_start
    print(f"\nCompleted steps: {', '.join(completed)}")
    print(f"Total elapsed time: {total:.1f}s")


if __name__ == "__main__":
    main()