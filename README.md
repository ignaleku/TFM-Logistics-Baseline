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
│   ├── rl.yaml                   # RL-1 training config (reward_mode: rl1_current)
│   └── rl2.yaml                  # RL-2 training config (reward_mode: urgent_protection)
│
├── src/
│   ├── data_generation/          # synthetic order generator
│   ├── simulation/
│   │   └── multistage/           # SimPy Picking → Packing → Dispatch model
│   └── rl/
│       ├── env_pick_rl.py        # RL environment (PickRLRunner)
│       ├── dqn_agent.py          # DQN agent and Q-network
│       ├── replay_buffer.py      # experience replay buffer
│       ├── main_train_rl.py      # training entry point
│       ├── evaluate_dqn.py       # single-window evaluation
│       ├── evaluate_dqn_multiseed.py  # multi-window robustness evaluation
│       └── plot_rl_results.py    # result plots
│
├── data/                         # generated — not committed (see .gitignore)
├── reports/
│   ├── figures/rl/               # generated plots — not committed
│   └── rl_results_summary.md
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

Increasing workers reduces congestion. A congested regime (1-1-1) is recommended for RL training because the prioritisation policy has the most impact there.

---

### Step 3 — Train RL-1

```bash
python -m src.rl.main_train_rl
```

Trains a DQN agent to prioritise urgent or normal orders at the Picking stage.

Configuration: `configs/rl.yaml`

Per-episode output (stdout):

```text
[EP 001] scen=s211 W=2-1-1 SLA=... | U=... | N=... | eps=... | loss=... | buf=... | upd=...
```

Saved files:

```text
data/rl_train_history.csv      # per-episode metrics
data/dqn_ckpt_ep010.pt         # periodic checkpoints (every ckpt_every episodes)
data/dqn_final.pt              # final model weights
```

---

### Step 4 — Evaluate RL-1

#### Single-window evaluation (FIFO vs urgent_first vs DQN)

```bash
python -m src.rl.evaluate_dqn
```

Output: `data/rl_eval_results.csv`

#### Multi-window robustness evaluation (5 order windows)

```bash
python -m src.rl.evaluate_dqn_multiseed
```

Output: `data/rl_eval_multiseed_results.csv`

Prints mean ± std per regime and policy across all windows. Low standard deviation confirms the single-window result is stable.

#### Generate plots

```bash
python -m src.rl.plot_rl_results
```

Reads `data/rl_eval_results.csv`. Writes four plots to `reports/figures/rl/`:

```text
sla_total_by_regime_policy.png
sla_by_order_type.png
p90_system_time.png
dqn_urgent_decision_rate.png
```

---

### Step 5 — Train and evaluate RL-2 (sensitivity analysis)

RL-2 uses a different reward function (`urgent_protection`) that assigns a hard binary reward instead of the continuous lateness penalty used in RL-1:

| Outcome | RL-1 reward | RL-2 reward |
|---|---|---|
| Urgent on time | +5.0 | +10.0 |
| Urgent late | −proportional | −5.0 |
| Normal on time | +2.0 | +1.0 |
| Normal late | −proportional | 0.0 |

**Train:**

```bash
python -m src.rl.main_train_rl --config configs/rl2.yaml --run-name rl2
```

Saved files:

```text
data/rl2_train_history.csv
data/dqn_rl2_ckpt_ep010.pt
data/dqn_rl2_final.pt
```

**Evaluate (single-window):**

```bash
python -m src.rl.evaluate_dqn \
  --checkpoint data/dqn_rl2_final.pt \
  --output data/rl2_eval_results.csv
```

**Evaluate (multi-window):**

```bash
python -m src.rl.evaluate_dqn_multiseed \
  --checkpoint data/dqn_rl2_final.pt \
  --output data/rl2_eval_multiseed_results.csv
```

RL-1 can be re-evaluated at any time without retraining:

```bash
python -m src.rl.evaluate_dqn
python -m src.rl.evaluate_dqn_multiseed
```

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

Source code, configs, and markdown reports are committed. Data and artefacts are not.

---

## Reward Modes

The reward function is controlled by `reward_mode` in the `reward` section of the RL config:

| Mode | Description |
|---|---|
| `rl1_current` | Continuous reward: +w if on time, −proportional penalty if late |
| `urgent_protection` | Binary reward: +10/−5 for urgent, +1/0 for normal |

To add a new reward mode, implement the branch in `src/rl/env_pick_rl.py` → `PickRLRunner._reward` and create a corresponding config file.

---

## Development Principles

- Work in small tickets. Do not attempt to complete the full thesis in one change.
- Do not modify `src/data_generation/` or `src/simulation/multistage/sim_multistage.py` unless strictly necessary.
- Keep the baseline simulation separate from RL code.
- Compare every learned policy against FIFO and urgent_first.
- Prefer reproducible scripts over notebook-only workflows.
- Keep hyperparameters in YAML config files, not hardcoded.
