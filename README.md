# TFM Logistics Baseline — Discrete-Event Simulation + RL-3 Prioritisation

A discrete-event simulation environment for analysing logistics operations and training
reinforcement learning policies for operational prioritisation decisions.

This project generates synthetic order demand, runs a multi-stage SimPy simulation with
FIFO and `urgent_first` baseline policies, and trains an RL-3 DQN agent that decides
order priority at Picking, Packing, and Dispatch simultaneously. Results are evaluated
via SLA compliance and system-time KPIs and compared against the baselines across
multiple capacity regimes.

> Preferred terms: discrete-event simulation, simulation-based operational analysis,
> simulation environment for decision learning. Not a "digital twin".

---

## Final Pipeline Overview

```
Synthetic data
→ Multi-stage simulation  (SimPy, Picking → Packing → Dispatch)
→ Baselines               (FIFO / urgent_first)
→ RL-3 full-stage DQN     (one shared agent acting at all three stages)
→ Evaluation              (single-window + multi-window robustness)
→ Reporting               (plots / checks)
```

---

## Repository Structure

```text
TFM-Logistics-Baseline/
├── configs/
│   ├── demand_base.yaml              # data generation parameters
│   ├── sim_multistage.yaml           # simulation resources and service times
│   ├── rl3.yaml                      # RL-3 training config (final model)
│   └── legacy/                       # RL-1, RL-2, MVP configs (archived)
│
├── src/
│   ├── data_generation/              # synthetic order generator
│   │   └── legacy/                   # earlier generation scripts (archived)
│   ├── simulation/
│   │   ├── multistage/               # SimPy Picking → Packing → Dispatch model
│   │   └── legacy/                   # single-stage MVP (archived)
│   ├── rl/
│   │   ├── dqn_agent.py              # DQN agent and Q-network
│   │   ├── replay_buffer.py          # experience replay buffer
│   │   ├── env_fullstage_rl.py       # RL-3 environment (Pick + Pack + Dispatch)
│   │   ├── main_train_rl3.py         # RL-3 training entry point
│   │   ├── evaluate_rl3.py           # single-window evaluation
│   │   ├── evaluate_rl3_multiseed.py # multi-window robustness evaluation
│   │   └── legacy/                   # RL-1 and RL-2 scripts (archived)
│   ├── pipeline/
│   │   ├── run_all.py                # general pipeline runner (entry point)
│   │   └── run_project_checks.py     # reproducibility check
│   ├── validation/
│   │   └── quick_project_checks.py   # data and eval sanity checks
│   └── reporting/
│       └── plot_final_results.py     # final thesis plots
│
├── data/                             # generated — not committed (see .gitignore)
├── reports/
│   ├── figures/final/                # generated plots — not committed
│   └── legacy/                       # RL-1 summary and handoff docs
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks script execution, use the batch activator instead:

```powershell
.venv\Scripts\activate.bat
```

Or configure the interpreter directly in PyCharm (`File → Settings → Python Interpreter`).

Verify the installation:

```powershell
python -c "import torch; print(torch.__version__)"
python -c "import simpy, pandas, yaml; print('ok')"
```

If PyTorch installation fails, install the CPU build explicitly:

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## Quick Start — General Runner

The project has a single entry point that orchestrates all pipeline steps.

Show available options:

```powershell
python -m src.pipeline.run_all
```

Run base setup (data + simulation + checks):

```powershell
python -m src.pipeline.run_all --base
```

Train RL-3 *(takes time)*:

```powershell
python -m src.pipeline.run_all --train-rl3
```

Evaluate RL-3 (single window):

```powershell
python -m src.pipeline.run_all --eval-rl3
```

Multi-window robustness evaluation *(takes time)*:

```powershell
python -m src.pipeline.run_all --multiseed-rl3
```

Generate final plots:

```powershell
python -m src.pipeline.run_all --plots
```

Run sanity and reproducibility checks:

```powershell
python -m src.pipeline.run_all --checks
```

Full pipeline *(training + multi-seed evaluation can take significant time)*:

```powershell
python -m src.pipeline.run_all --all
```

---

## Manual Commands

The same steps can be run directly without the runner:

```powershell
python -m src.data_generation.main_data_generation
python -m src.simulation.multistage.main_multistage
python -m src.rl.main_train_rl3
python -m src.rl.evaluate_rl3
python -m src.rl.evaluate_rl3_multiseed
python -m src.reporting.plot_final_results
python -m src.pipeline.run_project_checks
python -m src.validation.quick_project_checks
```

---

## Main Outputs

All generated files are excluded from git and must be recreated locally.

| File | Description |
| --- | --- |
| `data/orders_base.csv` | Synthetic orders, base demand scenario |
| `data/orders_peak_campaign.csv` | Synthetic orders, peak campaign scenario |
| `data/orders_stress.csv` | Synthetic orders, stress scenario |
| `data/dqn_rl3_final.pt` | Trained RL-3 model weights |
| `data/rl3_train_history.csv` | Per-episode training metrics |
| `data/rl3_eval_results.csv` | Single-window evaluation results |
| `data/rl3_eval_multiseed_results.csv` | Multi-window robustness results |
| `reports/figures/final/` | Final thesis plots (PNG) |

---

## RL-3 Agent

RL-3 uses a single shared DQN that acts at all three stages: Picking, Packing, and Dispatch.

At each decision point, the agent chooses between:
- **Action 0** — process the urgent order next
- **Action 1** — process the normal order next

The agent only acts when both an urgent and a normal order are simultaneously available
at the same stage. When only one type is present, no decision is made.

This addresses the main limitation of RL-1, where the agent could only intervene at
Picking. Moving urgent orders forward at Picking does not help if they then queue behind
normal orders at Packing or Dispatch.

Results should be interpreted as comparisons against FIFO and `urgent_first` baselines.
Regime-dependent behaviour is expected: the agent's impact is largest where the system
is most congested and decisions most constrained.

---

## Capacity Regimes

| Regime | Picking workers | Packing workers | Dispatch workers |
| --- | --- | --- | --- |
| s111 | 1 | 1 | 1 |
| s211 | 2 | 1 | 1 |
| s221 | 2 | 2 | 1 |

**s211** is the most informative regime: adding a second Picking worker shifts the
bottleneck to Packing, making prioritization decisions at Packing and Dispatch more
consequential. The training mix gives special weight to this regime because it is the most informative for this 
bottleneck behaviour.

---

## Legacy Experiments

Earlier experiments are archived under `legacy/` subdirectories for traceability.
They are not part of the main pipeline.

| Version | Scope | Status |
| --- | --- | --- |
| RL-1 | DQN deciding only at Picking | archived (`src/rl/legacy/`) |
| RL-2 | Reward sensitivity analysis on RL-1 | archived (`src/rl/legacy/`) |
| MVP simulation | Earlier single-stage SimPy model | archived (`src/simulation/legacy/`) |

RL-3 supersedes these. RL-1 and RL-2 results are referenced in the thesis to motivate
the design evolution.

---

## Git-Ignored Files

The following are generated at runtime and excluded from version control:

| Pattern            | Contents                                    |
|--------------------|---------------------------------------------|
| `data/*.csv`       | Orders, simulation results, training history, eval results |
| `data/*.pt`        | Model checkpoints and final weights         |
| `reports/figures/` | All generated plots                         |
| `.venv/`           | Virtual environment                         |
| `__pycache__/`     | Python bytecode                             |
| `.claude/`         | Claude Code local/internal files            |

Source code, configs, and markdown reports are committed. Data and model artefacts
are not. After cloning or switching branches, regenerate them:

```powershell
python -m src.pipeline.run_all --base
python -m src.pipeline.run_all --train-rl3
```

---

## Recommended Workflow (Clean Clone)

```powershell
# 1. Create and activate the virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1          # or activate.bat if PS is blocked

# 2. Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 3. Generate data and run baseline simulation
python -m src.pipeline.run_all --base

# 4. Train RL-3  (takes time)
python -m src.pipeline.run_all --train-rl3

# 5. Evaluate RL-3 (single window)
python -m src.pipeline.run_all --eval-rl3

# 6. Multi-window robustness evaluation  (takes time)
python -m src.pipeline.run_all --multiseed-rl3

# 7. Generate plots
python -m src.pipeline.run_all --plots

# 8. Run sanity checks
python -m src.pipeline.run_all --checks
```

---

## Current Status

The final technical pipeline is centred on RL-3. All evaluation results from RL-3
should be used as the main results for the thesis.
