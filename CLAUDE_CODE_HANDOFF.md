# Claude Code Handoff — TFM Logistic Process

## 1. Project Overview

This repository belongs to a Master's Thesis project focused on logistics/operations simulation and Reinforcement Learning.

The project builds:

1. A synthetic order/data generator.
2. A discrete-event simulation model using SimPy.
3. A multi-stage operational process: Picking → Packing → Dispatch.
4. Baseline policies: FIFO and urgent-first.
5. A Reinforcement Learning path using DQN, where the agent learns a prioritisation decision in Picking.

The project should not be described as a “digital twin”. Prefer terms such as:

- discrete-event simulation
- simulation-based operational analysis
- simulation environment for decision learning
- reinforcement learning for operational prioritisation

The current focus is **RL-1**: a DQN agent that decides whether to prioritise urgent or normal orders in Picking.

---

## 2. Current Status

### Phase 1 — Synthetic Data Generation

Status: completed.

Synthetic order generation is already implemented.

Main characteristics:

- Monthly seasonality.
- Weekly demand pattern.
- Hourly demand pattern.
- Order types: urgent / normal.
- SLA per order type.
- Order size generated using a lognormal distribution.
- Product class A/B/C.
- Scenarios:
  - base
  - peak_campaign
  - stress

Expected output files:

```text
data/orders_base.csv
data/orders_peak_campaign.csv
data/orders_stress.csv
```

Important validation results already achieved:

- Saturday and Sunday demand is zero by design.
- The campaign scenario increases demand during the campaign window.
- Arrival timestamps are ordered.
- The generated data is reproducible through seeds.

Relevant folder:

```text
src/data_generation/
```

Do not refactor this folder unless explicitly requested.

---

### Phase 2 — Single-Stage SimPy MVP

Status: completed.

A one-stage SimPy simulation was implemented and validated.

It includes:

- arrivals from generated orders,
- one processing resource,
- waiting time,
- service time,
- system time,
- SLA compliance,
- estimated utilisation,
- max backlog,
- consistency checks.

Important technical fix already applied:

- The simulation clock uses integer seconds internally to avoid floating point timestamp errors.

Relevant folder:

```text
src/simulation/
```

Do not refactor the MVP unless explicitly requested.

---

### Phase 3 — Multi-Stage Simulation

Status: working.

A multi-stage SimPy model exists:

```text
Picking → Packing → Dispatch
```

It supports:

- FIFO policy.
- urgent_first policy.
- SLA metrics by order type.
- total system time.
- p90 system time.

Important design decisions:

- FIFO should be true FIFO.
- urgent_first should genuinely prioritise urgent jobs.
- The baseline simulator should remain stable and should not be modified unnecessarily.

Important issue already solved:

- A previous implementation using manual SimPy Events caused deadlocks.
- The current approach uses `simpy.Store`.
- Avoid reintroducing manual event logic unless necessary.

Relevant folder:

```text
src/simulation/multistage/
```

Important file:

```text
src/simulation/multistage/sim_multistage.py
```

Do not break this baseline simulator. It is the reference for comparisons.

---

## 3. Baseline Behaviour Already Observed

With enough capacity, FIFO and urgent_first give very similar results because the system is not congested.

With congested capacity, especially:

```yaml
resources:
  picking_workers: 1
  packing_workers: 1
  dispatch_workers: 1
```

the policy matters.

Observed behaviour:

- urgent_first strongly improves urgent SLA.
- urgent_first can harm normal SLA.
- FIFO distributes performance differently.
- This creates the trade-off needed for Reinforcement Learning.

This is expected and important.

---

## 4. RL Objective

The current RL phase is called RL-1.

The goal is to train a DQN agent to decide, in the Picking stage, whether to serve an urgent or normal order when both are available.

### Action space

```text
0 = serve urgent
1 = serve normal
```

### Reward v1

Reward is assigned when the order completes Dispatch.

```text
+5 if urgent order meets SLA
+1 if normal order meets SLA
0 otherwise
```

This is delayed reward.

No tardiness penalty or backlog penalty should be added yet unless explicitly requested.

---

## 5. RL Architecture

Relevant folder:

```text
src/rl/
```

Expected files:

```text
src/rl/
  __init__.py
  replay_buffer.py
  dqn_agent.py
  env_pick_rl.py
  main_train_rl.py
```

### `env_pick_rl.py`

This should contain the RL environment / runner.

Expected behaviour:

- Runs a SimPy episode.
- Uses the same process structure:
  - Picking
  - Packing
  - Dispatch
- The agent controls only Picking prioritisation.
- Packing and Dispatch can remain FIFO for RL-1.
- When both urgent and normal jobs are available in Picking, call:

```python
agent.act(state)
```

- If the selected queue is empty, fall back safely.
- Track which transition belongs to each order_id.
- Assign delayed reward when the order completes Dispatch.

### State vector v1

The initial state vector should be small and normalised.

Suggested features:

```text
q_urgent_pick
q_normal_pick
wip_pack
wip_disp
time_norm
```

Expected input dimension for DQN:

```text
input_dim = 5
```

---

## 6. Current RL Issue

The RL path is not fully validated yet.

A previous training run produced episode metrics, but likely did not perform real DQN training because:

- there was no printed loss,
- there was no printed epsilon,
- it was unclear whether `agent.train_step(...)` was actually called.

The next priority is to fix the RL training loop.

---

## 7. Immediate Task

First task for Claude Code:

Make `python -m src.rl.main_train_rl` run end-to-end and actually train the DQN.

It must print per episode:

- total SLA
- urgent SLA
- normal SLA
- epsilon
- mean loss
- replay buffer size
- episode runtime if convenient

Do not modify:

```text
src/data_generation/
src/simulation/multistage/sim_multistage.py
```

unless absolutely necessary.

Prefer minimal patches over full rewrites.

---

## 8. Expected RL Config

There should be a config file:

```text
configs/rl.yaml
```

Recommended starting configuration:

```yaml
training:
  episodes: 20
  episode_orders: 10000
  lr: 0.001
  gamma: 0.99
  batch_size: 256
  target_update_steps: 2000
  train_start_size: 3000
  train_every_steps: 4
  updates_per_episode: 500

epsilon:
  start: 1.0
  end: 0.05
  decay_steps: 50000

network:
  hidden_dim: 64

buffer:
  capacity: 200000
```

---

## 9. Recommended Simulation Config for RL-1

A congested regime is useful for RL-1 because the policy matters.

Recommended initial setup in `configs/sim_multistage.yaml`:

```yaml
simulation:
  random_seed: 123
  policy: urgent_first
  time_unit: seconds

resources:
  picking_workers: 1
  packing_workers: 1
  dispatch_workers: 1
```

If learning is too noisy, try:

```yaml
resources:
  picking_workers: 2
  packing_workers: 1
  dispatch_workers: 1
```

or:

```yaml
resources:
  picking_workers: 2
  packing_workers: 2
  dispatch_workers: 1
```

---

## 10. Evaluation Goal

After training works, create an evaluation script that compares:

- FIFO
- urgent_first
- trained DQN

on the same order subset and seeds.

Metrics to compare:

- total SLA
- urgent SLA
- normal SLA
- mean system time
- p90 system time
- optionally action distribution:
  - percentage of urgent choices
  - percentage of normal choices

The goal is not necessarily for DQN to beat urgent_first in every metric.

The goal is to analyse:

- whether DQN learns a policy,
- whether it improves over FIFO,
- how it trades off urgent vs normal orders,
- whether reward design needs improvement.

---

## 11. Important Project Philosophy

Work in small tickets.

Do not attempt to “finish the whole TFM” in one change.

Preferred workflow:

1. Inspect relevant files.
2. Explain the issue.
3. Make minimal changes.
4. Run or describe the expected command.
5. Report what should be checked next.

Avoid broad refactors.

Avoid modifying stable modules.

---

## 12. Suggested First Prompt for Claude Code

Use this prompt:

```text
Read CLAUDE_CODE_HANDOFF.md first.

Inspect only:
- src/rl/
- configs/
- src/simulation/multistage/ only if needed for imports

Do not edit files yet.

Report:
1. Which RL files exist.
2. Whether main_train_rl.py actually trains the DQN.
3. Whether dqn_agent.py and replay_buffer.py interfaces match main_train_rl.py.
4. The minimum code changes needed to make python -m src.rl.main_train_rl run end-to-end and print SLA, urgent SLA, normal SLA, epsilon, mean loss and replay buffer size.
```

Then, after the report:

```text
Now implement only the minimum changes required so that:

python -m src.rl.main_train_rl

runs end-to-end and actually trains the DQN.

Do not modify data generation or baseline simulation.
Prefer small patches over rewrites.
```

---

## 13. Environment Notes

Use a clean virtual environment.

Recommended commands:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PyTorch causes issues, install CPU PyTorch explicitly:

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then verify:

```powershell
python -c "import torch; print(torch.__version__)"
python -c "import simpy, pandas, yaml; print('ok')"
```

---

## 14. Do Not Commit

Do not commit:

```text
.venv/
__pycache__/
data/*.csv
data/*.pt
data/*.pth
models/
reports/figures/
```

Use `.gitignore`.

---

## 15. Current Priority

Current priority:

```text
Fix RL-1 training loop and validate that DQN is actually learning.
```

Only after that:

```text
Create evaluation script comparing FIFO vs urgent_first vs DQN.
```
