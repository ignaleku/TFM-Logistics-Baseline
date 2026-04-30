# TFM Logistic Process

This repository contains the code for a Master's Thesis project focused on simulation-based operational analysis and Reinforcement Learning for logistics processes.

The project combines:

1. Synthetic data generation.
2. Discrete-event simulation with SimPy.
3. Multi-stage logistics process modelling.
4. Baseline policy comparison.
5. Reinforcement Learning for operational prioritisation.

The project should not be framed as a “digital twin”. A more accurate description is:

> A discrete-event simulation environment for analysing logistics operations and training reinforcement learning policies for operational decision-making.

---

## Project Structure

```text
TFM-Logistic-Process/
├── configs/
│   ├── demand_base.yaml
│   ├── sim_multistage.yaml
│   └── rl.yaml
│
├── src/
│   ├── data_generation/
│   ├── simulation/
│   │   └── multistage/
│   └── rl/
│
├── data/
├── reports/
├── CLAUDE_CODE_HANDOFF.md
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Current Project Status

### Phase 1 — Synthetic Data Generation

Implemented.

The generator creates synthetic orders with:

- monthly seasonality,
- weekly pattern,
- hourly demand profile,
- urgent and normal orders,
- SLA targets,
- order sizes,
- product classes,
- different demand scenarios.

Generated outputs include:

```text
data/orders_base.csv
data/orders_peak_campaign.csv
data/orders_stress.csv
```

---

### Phase 2 — Single-Stage Simulation

Implemented and validated.

The single-stage SimPy MVP was used to validate:

- queues,
- capacity,
- waiting time,
- system time,
- SLA compliance,
- utilisation,
- backlog.

---

### Phase 3 — Multi-Stage Simulation

Implemented.

The current multi-stage process is:

```text
Picking → Packing → Dispatch
```

Two baseline policies are supported:

- FIFO
- urgent_first

The baseline simulator is used as the reference environment for comparing operational policies.

---

### Phase 4 — Reinforcement Learning

In progress.

The current RL goal is to train a DQN agent to decide whether to prioritise urgent or normal orders in the Picking stage.

Actions:

```text
0 = serve urgent
1 = serve normal
```

Reward:

```text
+5 if urgent order meets SLA
+1 if normal order meets SLA
0 otherwise
```

The reward is delayed and assigned when the order completes Dispatch.

---

## Installation

Create a clean virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PyTorch installation is problematic, install CPU PyTorch explicitly:

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Verify installation:

```powershell
python -c "import torch; print(torch.__version__)"
python -c "import simpy, pandas, yaml; print('ok')"
```

---

## Running the Project

### Synthetic data generation

Run the data generation main script from the IDE or terminal.

Expected outputs are written to:

```text
data/
reports/figures/
```

---

### Multi-stage simulation

Run:

```powershell
python -m src.simulation.multistage.main_multistage
```

This runs the multi-stage SimPy simulator using the current configuration in:

```text
configs/sim_multistage.yaml
```

---

### RL training

Run:

```powershell
python -m src.rl.main_train_rl
```

The current priority is to ensure that the DQN training loop:

- runs end-to-end,
- performs gradient updates,
- prints epsilon,
- prints mean loss,
- prints replay buffer size,
- reports SLA metrics per episode.

---

## Claude Code Instructions

Before making changes, Claude Code should read:

```text
CLAUDE_CODE_HANDOFF.md
```

The first task should be:

```text
Fix RL-1 training so that python -m src.rl.main_train_rl actually trains the DQN.
```

Claude should not modify:

```text
src/data_generation/
src/simulation/multistage/sim_multistage.py
```

unless strictly necessary.

---

## Development Principles

- Work in small tickets.
- Do not refactor stable modules unnecessarily.
- Keep configs YAML-driven.
- Keep baseline simulation separate from RL code.
- Prefer reproducible scripts over notebook-only workflows.
- Compare every learned policy against baselines.

---

## Next Steps

1. Fix RL training loop.
2. Validate DQN training metrics.
3. Add action distribution metrics.
4. Create evaluation script:
   - FIFO vs urgent_first vs DQN.
5. Generate comparison plots and result tables.
6. Iterate reward design if necessary.

---

## Notes

Generated data, model checkpoints and figures should not be committed to Git by default.

Use `.gitignore` to avoid committing:

```text
data/*.csv
data/*.pt
reports/figures/
.venv/
```
