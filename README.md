# TFM Logistic Process

A discrete-event simulation environment for analysing logistics operations and training reinforcement learning policies for operational decision-making.

The project should not be described as a "digital twin". Preferred terms: discrete-event simulation, simulation-based operational analysis, simulation environment for decision learning.

---

## Project Structure

```text
TFM-Logistic-Process/
├── configs/
│   ├── demand_base.yaml          # data generation parameters
│   ├── sim_multistage.yaml       # simulation resources and service times
│   ├── rl3.yaml                  # RL-3 training config (final model)
│   └── legacy/                   # RL-1, RL-2 and MVP configs (archived)
│
├── src/
│   ├── data_generation/          # synthetic order generator
│   ├── simulation/
│   │   ├── multistage/           # SimPy Picking → Packing → Dispatch model
│   │   └── legacy/               # single-stage MVP (archived)
│   ├── rl/
│   │   ├── dqn_agent.py          # DQN agent and Q-network
│   │   ├── replay_buffer.py      # experience replay buffer
│   │   ├── env_fullstage_rl.py   # RL-3 environment (Pick + Pack + Dispatch)
│   │   ├── main_train_rl3.py     # RL-3 training entry point
│   │   ├── evaluate_rl3.py       # single-window evaluation
│   │   ├── evaluate_rl3_multiseed.py  # multi-window robustness evaluation
│   │   └── legacy/               # RL-1 and RL-2 scripts (archived)
│   ├── pipeline/
│   │   └── run_project_checks.py # reproducibility check
│   ├── validation/
│   │   └── quick_project_checks.py  # data and eval sanity checks
│   └── reporting/
│       └── plot_final_results.py # final thesis plots
│
├── data/                         # generated — not committed (see .gitignore)
├── reports/
│   ├── figures/final/            # generated plots — not committed
│   └── legacy/                   # RL-1 summary and project handoff docs
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PyTorch installation fails, install the CPU build explicitly:

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Verify:

```powershell
python -c "import torch; print(torch.__version__)"
python -c "import simpy, pandas, yaml; print('ok')"
```

---

## Execution Pipeline

Run each step in order. All commands are executed from the repository root.

### Step 1 — Generate synthetic data

```bash
python -m src.data_generation.main_data_generation
```

Writes three order files to `data/`:

```text
data/orders_base.csv
data/orders_peak_campaign.csv
data/orders_stress.csv
```

Configuration: `configs/demand_base.yaml`

---

### Step 2 — Run multi-stage baseline simulation

```bash
python -m src.simulation.multistage.main_multistage
```

Runs the Picking → Packing → Dispatch SimPy simulator with FIFO and urgent_first policies.

Configuration: `configs/sim_multistage.yaml`

Key parameters:

```yaml
resources:
  picking_workers: 1
  packing_workers: 1
  dispatch_workers: 1
```

A congested regime (1-1-1) is recommended for RL training because the prioritisation policy has the most impact there.

---

### Step 3 — Train RL-3

```bash
python -m src.rl.main_train_rl3
```

Trains a single DQN agent that decides at Picking, Packing, and Dispatch simultaneously.

Configuration: `configs/rl3.yaml`

Per-episode output (stdout):

```text
[EP 001] scen=s211 W=2-1-1 SLA=... U=... N=... ma5=... | eps=... loss=... buf=... upd=... grad=... | dec=... %U=... [pick:.../... pack:.../... disp:.../....]
```

Saved files:

```text
data/rl3_train_history.csv          # per-episode metrics
data/dqn_rl3_ckpt_ep010.pt         # periodic checkpoints
data/dqn_rl3_final.pt              # final model weights
```

---

### Step 4 — Evaluate RL-3 (single-window)

```bash
python -m src.rl.evaluate_rl3
```

Evaluates the trained RL-3 agent against FIFO and urgent_first on three resource regimes (s111, s211, s221).

Output: `data/rl3_eval_results.csv`

---

### Step 5 — Evaluate RL-3 (multi-window robustness)

```bash
python -m src.rl.evaluate_rl3_multiseed
```

Repeats evaluation across 5 order windows. Low standard deviation confirms the single-window result is stable.

Output: `data/rl3_eval_multiseed_results.csv`

---

### Step 6 — Generate final plots

```bash
python -m src.reporting.plot_final_results
```

Reads `data/rl3_eval_multiseed_results.csv` (falls back to single-window if not present) and `data/rl3_train_history.csv`.

Writes plots to `reports/figures/final/`:

```text
monthly_order_volume.png
hourly_order_profile.png
order_type_distribution.png
rl_total_sla_comparison.png
rl_urgent_normal_sla_comparison.png
rl_p90_comparison.png
dqn_training_sla_ma5.png
dqn_urgent_decision_rate_training.png
```

---

### Reproducibility check

```bash
python -m src.pipeline.run_project_checks
```

Verifies that all required artefacts exist and prints the next command to run if any are missing.

### Sanity checks

```bash
python -m src.validation.quick_project_checks
```

Validates schema and value ranges in order files and RL-3 evaluation CSVs.

---

## Git-ignored Files

The following are generated at runtime and are excluded from version control via `.gitignore`:

| Pattern | Contents |
|---|---|
| `data/*.csv` | Generated orders, simulation results, training history, eval results |
| `data/*.pt` | Model checkpoints and final weights |
| `data/*.pth` | Alternative PyTorch save format |
| `reports/figures/` | All generated plots |
| `.venv/` | Virtual environment |
| `__pycache__/` | Python bytecode |
| `.claude/` | Claude Code internal files |

Source code, configs, and markdown reports are committed. Data and artefacts are not.

---

## Reward Mode — RL-3

RL-3 uses the `rl1_current` reward mode: a continuous reward deferred to dispatch completion.

| Outcome | Reward |
|---|---|
| Urgent on time | +5.0 |
| Urgent late | −proportional to lateness |
| Normal on time | +2.0 |
| Normal late | −proportional to lateness |

The reward is assigned to all decision-point transitions of the same order (across Picking, Packing, and Dispatch). This is safe because the SLA outcome is fully determined at dispatch.

---

## Prior Experiments — Legacy

`src/rl/legacy/`, `configs/legacy/`, and `reports/legacy/` contain the RL-1 and RL-2 experiments used to justify the evolution to RL-3.

| Version | Scope | Config | Status |
|---|---|---|---|
| RL-1 | DQN decides only at Picking | `configs/legacy/rl.yaml` | archived |
| RL-2 | RL-1 with binary reward sensitivity | `configs/legacy/rl2.yaml` | archived |
| RL-3 | DQN decides at Picking + Packing + Dispatch | `configs/rl3.yaml` | **current** |

To reproduce RL-1 or RL-2 results, restore the legacy scripts and configs from `src/rl/legacy/` and run `python -m src.rl.legacy.main_train_rl --config configs/legacy/rl.yaml`.

---

## Development Principles

- Keep the baseline simulation separate from RL code.
- Compare every learned policy against FIFO and urgent_first.
- Prefer reproducible scripts over notebook-only workflows.
- Keep hyperparameters in YAML config files, not hardcoded.
