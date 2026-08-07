# TFM Logistics Baseline — Prescriptive Warehouse Planning

A discrete-event simulation + reinforcement learning decision-support system for warehouse operations.

This project is a **prescriptive warehouse planning tool**. It supports retrospective analysis of
historical order data and prospective planning from aggregate demand forecasts. It simulates
heterogeneous orders through **Picking → Packing → Dispatch**, compares FIFO, urgent-first, and
RL-3 sequencing policies, identifies bottlenecks, evaluates workforce configurations, and estimates
whether adding capacity is economically justified.

> Preferred terms: discrete-event simulation, simulation-based operational analysis, decision support.
> Not a "digital twin".

---

## Two Modes, One Simulator

```
Historical order CSV                    Aggregate demand forecast
        │                                          │
        ▼                                          ▼
Canonical enriched order DataFrame ◄── Future scenario generator
        │                          (src/data/future_scenario.py)
        └──────────────────┬───────────────────────┘
                            ▼
                Same SimPy 3-stage engine
       (src/simulation/multistage · src/rl/env_fullstage_rl.py)
                            ▼
     FIFO / Urgent-First / RL-3 DQN, common random numbers
                            ▼
      Bottleneck ranking · break-even · adaptive capacity search
                            ▼
                  Decision support (webapp)
```

**Historical Analysis** — upload a real order-level CSV. The system performs counterfactual
analysis: which policy/workforce would have worked best, where the bottleneck was, whether an
extra worker would have paid for itself.

**Future Planning** — the user does **not** need to provide every future order individually.
A small set of aggregate inputs (planning month, expected annual orders, uncertainty level) is
combined with a calibrated client planning profile (`configs/planning_profile.yaml`) to generate
plausible simulated orders. The system does not predict each individual future order — it
transforms an aggregate forecast into simulated operational scenarios and prescribes workforce
capacity and policy. A small number of scenario **replications** (default 3, different seeds)
avoid presenting one random realisation as certainty; results report mean and p90 cost, mean
SLA, and probability of meeting SLA targets.

Both modes feed the **same** SimPy simulation engine after input preparation — there is exactly
one simulator.

---

## Order Heterogeneity

Orders carry three dimensions that drive differentiated workloads at each stage:

| Dimension        | Values                    | Effect                                                |
|------------------|---------------------------|-------------------------------------------------------|
| `order_type`     | urgent / normal           | SLA (4h / 24h), dispatch urgency multiplier ×1.3     |
| `product_family` | standard / fragile / bulky | Packing time: ×1.0 / ×1.8 / ×1.6                  |
| `complexity_level` | low / medium / high      | Packing time: ×0.8 / ×1.2 / ×1.7                  |

```
picking_units  = num_items × family_mult × complexity_mult
packing_units  = (1 + 0.25 × num_items) × family_mult × complexity_mult
dispatch_units = urgency_mult × family_mult × complexity_mult
```

All multipliers, monthly seasonal shares, urgency shares, item-count targets, uncertainty
mappings, SLA targets, cost defaults, workforce regimes, and bottleneck-score weights live in
**`configs/planning_profile.yaml`** — the single source of truth read by the seasonal generator,
the future-planning generator, uploaded-CSV enrichment, and the API. RL reward coefficients stay
in `configs/rl3.yaml` (internal training config, not a client-facing planning assumption).

---

## Fair Policy Comparison (Common Random Numbers)

FIFO, Urgent-First and RL-3 DQN are compared under **common random numbers**
(`src/simulation/multistage/service_time_map.py`): for a given scenario seed, every order's
service time at every stage is sampled **once**, sorted by `order_id` — independent of which
policy processes it or in what order it gets dequeued. All three engines look up the same map
instead of drawing independently. This is what makes "RL-3 is 10% cheaper than Urgent-First"
a statement about the policy, not about which one got luckier service times. It's used
everywhere: historical evaluation, future-planning replications, adaptive capacity search, and
the RL-3 audit.

---

## Bottleneck Instrumentation

Each stage (`src/simulation/multistage/stage_metrics.py`) tracks, event-based (no polling):
worker count, busy-worker time → utilisation, processed count → throughput, service-time and
wait-time distributions (mean/p95), time-weighted average and max queue length, and — for late
orders only — accumulated waiting time per stage and its share of all late-order waiting.

A transparent heuristic (`src/analysis/bottleneck.py`) combines these into one **pressure
score** per stage:

```
pressure_score = 0.40·utilisation + 0.25·normalised_p95_wait
               + 0.20·late_wait_share + 0.15·normalised_avg_queue
```

weights configurable in `planning_profile.yaml::bottleneck_score`. The top-ranked stage is the
primary bottleneck, with a plain-language explanation. This is an operational indicator, not a
formal causal proof.

## Extra-Worker Break-Even & Adaptive Capacity Search

`src/analysis/capacity_search.py` computes the theoretical break-even (late orders an extra
worker's monthly cost would need to prevent to pay for itself — urgent-only, normal-only, and
current-mix lenses), and runs a **bottleneck-directed adaptive search**: starting from the base
16-regime recommendation, if it's infeasible (or near max base capacity with real late-order
cost), add one worker to the top bottleneck stage (or the top two, if their pressure scores are
close) and re-evaluate all three policies — the best policy can change as capacity changes.
Every candidate is logged accepted/rejected with a reason (`src/analysis/bottleneck_report.py`).

## SLA Feasibility

A candidate is **feasible** only if it meets both SLA floors (`planning_profile.yaml::sla`,
default urgent ≥ 95%, normal ≥ 80%). Recommendations prefer the cheapest *feasible* candidate; if
none exists, the system explicitly labels its pick "best available — SLA targets not fully met"
and ranks by SLA violation first. A pathological 100%-urgent/2%-normal result is never presented
as an unqualified winner.

---

## RL-3 Audit & Generalisation

A December result was flagged as suspicious (RL-3 near-perfect urgent SLA, collapsed normal
SLA). `src/rl/rl_audit.py` reproduces it under common random numbers and checks, in order:
urgent-first's queue semantics (synthetic overtaking test), state information leakage (static
code inspection — the 13-then-16-feature state only reads already-arrived orders and elapsed
time), the reward's on-time/late swing per class, and aggregate RL decision diagnostics
(urgent-selection rate, streaks, wait distributions) on both a training-seen and an unseen
workforce regime. Findings are written to `data/api_runs/latest/rl3_audit_report.json`.

The evidence pointed at **RL generalisation failure** as the primary mechanism (urgent-selection
rate flips from ~1% to ~97% between two regimes while normal SLA collapses in both — inconsistent
with a pure "reward always favours urgent" story), compounded by a real reward imbalance. The
justified fix, applied together in one retrain:

1. **Capacity features added to the state** (13 → 16 dims): normalised picking/packing/dispatch
   worker counts, so the agent can condition on current capacity (`env_fullstage_rl.py::_state`).
2. **Training regime mix widened** from 3 scenarios to a uniform draw over the 12 stratified
   `rl_generalisation.train_regimes` in `planning_profile.yaml` (low/medium/high capacity).
3. **Reward rebalanced**: `w_normal` 2.0→3.0, `late_penalty_normal` 1.0→2.0, narrowing the
   on-time↔late swing ratio between classes from 2.33x to 1.4x (`configs/rl3.yaml`).

`src/rl/evaluate_rl3_generalisation.py` evaluates the retrained checkpoint on the 12 training
regimes vs. the 4 exact-held-out `holdout_regimes` (`s112`, `s231`, `s322`, `s432`), reporting
mean cost/SLA/feasibility for both groups against FIFO and Urgent-First on the same regimes —
the goal is to characterise generalisation, not force RL to win everywhere.

---

## Repository Structure

```text
TFM-Logistics-Baseline/
├── configs/
│   ├── planning_profile.yaml         # single source of truth: seasonal/operational assumptions
│   ├── sim_multistage.yaml           # 3-stage simulation config (service_time keys)
│   ├── rl3.yaml                      # RL-3 training config (network, reward, episodes)
│   └── legacy/                       # archived RL-1, RL-2, RL-5 configs
│
├── src/
│   ├── data/
│   │   ├── planning_profile.py       # loader/validator for planning_profile.yaml
│   │   ├── order_generation_core.py  # shared order-generation logic (family/complexity/units)
│   │   ├── generate_orders_seasonal.py  # 240k heterogeneous orders, full year
│   │   └── future_scenario.py        # aggregate-forecast → simulated future orders + preview
│   ├── simulation/
│   │   ├── multistage/
│   │   │   ├── sim_multistage.py     # FIFO / urgent_first SimPy engine
│   │   │   ├── service_time_map.py   # common-random-number service-time sampling
│   │   │   └── stage_metrics.py      # bottleneck instrumentation (queue area, wait, utilisation)
│   │   └── legacy/                   # archived 5-stage and MVP simulators
│   ├── analysis/
│   │   ├── bottleneck.py             # pressure-score ranking
│   │   ├── capacity_search.py        # break-even + adaptive capacity search
│   │   ├── sla_feasibility.py        # feasibility / violation scoring
│   │   ├── bottleneck_report.py      # ties the above into one API-ready report
│   │   └── replication_aggregation.py  # future-planning replication mean/p90 aggregation
│   ├── rl/
│   │   ├── dqn_agent.py, replay_buffer.py
│   │   ├── env_fullstage_rl.py       # RL-3 environment (16-feature state incl. capacity)
│   │   ├── main_train_rl3.py         # training entry point (12-regime stratified sampling)
│   │   ├── evaluate_rl3_monthly_capacity_cost.py  # monthly grid (16 regimes × 3 policies)
│   │   ├── evaluate_rl3_generalisation.py  # seen vs. held-out regime evaluation
│   │   ├── rl_audit.py               # December anomaly audit
│   │   └── legacy/                   # archived RL-5 scripts
│   ├── reporting/
│   │   └── export_rl3_monthly_recommendations.py
│   ├── api/                          # FastAPI backend
│   │   ├── main.py, runners.py, schemas.py, utils.py
│   └── validation/
│       └── quick_project_checks.py
│
├── webapp/                           # React + Vite frontend (6 tabs)
│   └── src/components/tabs/
│       ├── UploadRunTab.tsx          # mode selector: Historical Analysis / Future Planning
│       ├── WorkforcePlannerTab.tsx   # monthly capacity recommendations (Recommendations tab)
│       ├── DemandComplexityTab.tsx   # seasonal demand + heterogeneity charts
│       ├── PolicyComparisonTab.tsx   # FIFO vs Urgent-First vs RL-3, feasibility/starvation badges
│       ├── CapacityBottlenecksTab.tsx  # bottleneck ranking, break-even, adaptive search trail
│       └── MethodTab.tsx
│
├── data/                             # generated — not committed
│   ├── orders_base_seasonal.csv, orders_base_seasonal_summary.csv
│   ├── dqn_rl3_final.pt              # active RL-3 checkpoint (16-dim state)
│   ├── rl3_train_history.csv
│   └── api_runs/latest/              # status.json, results CSVs, bottleneck_analysis.json,
│                                      # future_planning_summary.json, rl3_audit_report.json
├── requirements.txt
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

Verify:

```powershell
python -c "import torch, simpy, pandas, yaml; print('ok')"
```

---

## Validation Pipeline

```powershell
# 1. Generate heterogeneous seasonal dataset (240,000 orders), from planning_profile.yaml
python -m src.data.generate_orders_seasonal

# 2. Sanity checks (data columns, value ranges, file presence)
python -m src.validation.quick_project_checks

# 3. Train RL-3 DQN (uniform sampling over the 12 stratified training regimes)
python -m src.rl.main_train_rl3

# 4. Audit the RL-3 policy (fair comparison, urgent_first validation, leakage check, diagnostics)
python -m src.rl.rl_audit --month December --regimes s221,s432

# 5. Seen vs. held-out regime generalisation
python -m src.rl.evaluate_rl3_generalisation --months December

# 6. Monthly capacity-cost optimisation (bottleneck metrics + feasibility flattened into the CSV)
python -m src.rl.evaluate_rl3_monthly_capacity_cost ^
    --orders data/orders_base_seasonal.csv ^
    --months January,June,October,December ^
    --cost-late-urgent 15 --cost-late-normal 10 ^
    --worker-cost-per-hour 18 --hours-per-worker-month 160 ^
    --output data/rl3_monthly_capacity_cost_results.csv

# 7. Export webapp-ready CSVs
python -m src.reporting.export_rl3_monthly_recommendations ^
    --input data/rl3_monthly_capacity_cost_results.csv ^
    --output-summary data/app_exports/rl3_monthly_recommendations_summary.csv ^
    --output-full data/app_exports/rl3_monthly_capacity_cost_results_app.csv
```

---

## Capacity Regimes (16 base regimes)

`s{picking}{packing}{dispatch}` — from minimal (`s111`) to heavy (`s432`). The adaptive capacity
search can generate additional regimes beyond these 16 (e.g. `s532`) when needed to reach
feasibility. Full list: `configs/planning_profile.yaml::regimes`.

## Policies

| Policy        | Description                                                       |
|---------------|---------------------------------------------------------------------|
| `fifo`        | First-in first-out (arrival order, no differentiation)           |
| `urgent_first`| Always serve urgent orders before normal                          |
| `rl3_dqn`     | Learned DQN policy — acts at all 3 stages, capacity-aware state  |

---

## Webapp — Local Development

### Backend (FastAPI)

```powershell
python -m uvicorn src.api.main:app --reload --port 8000
```

Endpoints:

| Method | Path                              | Description                                    |
|--------|-----------------------------------|-------------------------------------------------|
| GET    | /health                           | Service status                                 |
| GET    | /planning/profile                 | Read-only client-profile assumptions for the UI |
| POST   | /planning/preview                 | Derived future-planning assumptions (no run)   |
| POST   | /upload-orders                    | Upload orders CSV (auto-enriched)              |
| POST   | /run/monthly-capacity-cost        | Historical: run RL-3 monthly optimisation      |
| POST   | /run/future-planning              | Future: generate scenario(s) + optimise        |
| GET    | /run/status                       | Poll run progress (status.json)                |
| GET    | /results/latest/recommendations   | Monthly recommendation summary                 |
| GET    | /results/latest/full              | Full results CSV                               |
| GET    | /results/latest/bottlenecks       | Bottleneck ranking, break-even, adaptive trail |
| GET    | /recommend/month/{month_name}     | Month-specific recommendation                  |
| GET    | /files/status                     | Check file availability                        |
| GET    | /data/order-summary               | Monthly order statistics (for charts)          |

### Frontend (React + Vite)

```powershell
cd webapp
npm install
npm run dev
```

Tabs: **Run** (Historical / Future Planning mode selector) · **Recommendations** ·
**Demand & Complexity** · **Policy Comparison** · **Capacity & Bottlenecks** · **Method**.

### Environment variables (optional)

```
VITE_API_BASE_URL=http://localhost:8000   # frontend → backend URL
FRONTEND_ORIGIN=http://localhost:5173     # backend CORS allowlist
```

---

## Sanity Checks

```powershell
python -m src.validation.quick_project_checks
```

---

## Legacy (RL-5 / 5-stage)

The earlier RL-5 experiment (5-stage: Picking → QC → Packing → Labelling → Dispatch) is archived
and **not part of the current pipeline**: `configs/legacy/rl5/`, `src/rl/legacy/rl5/`,
`src/simulation/legacy/5stage/`, `src/reporting/legacy/`.

## Limitations & Honest Caveats

- Future-planning results are scenario-based estimates over a handful of replicated seeds, not
  guarantees — wider uncertainty settings widen the true range further than 3 replications show.
- Bottleneck attribution (late-order wait share per stage) is operational/explanatory, not a
  formal causal decomposition.
- The RL-3 audit's root-cause interpretation is evidence-based on the regimes actually tested;
  it is not an exhaustive proof for every possible regime/month combination.
