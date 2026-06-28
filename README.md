# TFM Logistics Baseline — Decision Support System

A discrete-event simulation + reinforcement learning decision-support system for logistics operations.

The system generates synthetic heterogeneous order demand, runs a **3-stage SimPy simulation** with FIFO
and `urgent_first` baseline policies, and evaluates an **RL-3 DQN agent** that dynamically prioritises
orders at **Picking → Packing → Dispatch**. Order heterogeneity (product family × complexity level)
creates shifting stage bottlenecks that make intelligent sequencing measurably more valuable than fixed
policies. Results drive a monthly workforce planning tool that recommends the staffing configuration
minimising total SLA penalty cost + labour cost across 16 capacity regimes.

> Preferred terms: discrete-event simulation, simulation-based operational analysis, decision support.
> Not a "digital twin".

---

## Final Pipeline Overview

```
Synthetic orders (heterogeneous: product_family × complexity_level)
→ 3-stage simulation     (SimPy: Picking → Packing → Dispatch)
→ Baseline policies      (FIFO / Urgent-First)
→ RL-3 DQN agent         (one shared agent acting at all three stages)
→ Monthly analysis        (per-month SLA + cost across 16 capacity regimes)
→ Decision support        (capacity-cost optimisation + monthly recommendations)
→ Reporting               (webapp-ready CSVs)
→ Webapp                  (React + FastAPI — upload orders → get recommendation)
```

---

## Order Heterogeneity

Orders carry three dimensions that drive differentiated workloads at each stage:

| Dimension        | Values                    | Effect                                                |
|------------------|---------------------------|-------------------------------------------------------|
| `order_type`     | urgent / normal           | SLA (4h / 24h), dispatch urgency multiplier ×1.3     |
| `product_family` | standard / fragile / bulky | Packing time: ×1.0 / ×1.8 / ×1.6                  |
| `complexity_level` | low / medium / high      | Packing time: ×0.8 / ×1.2 / ×1.7                  |

A fragile + high-complexity order requires **3×** more packing time than a standard + low-complexity one,
even at the same item count. This creates shifting bottlenecks that make the RL-3 sequencing problem
non-trivial across months.

Pre-computed workload units are stored per order and consumed by the simulation:

```
picking_units  = num_items × family_mult × complexity_mult
packing_units  = (1 + 0.25 × num_items) × family_mult × complexity_mult
dispatch_units = 1 × urgency_mult × family_mult × complexity_mult
```

---

## Repository Structure

```text
TFM-Logistics-Baseline/
├── configs/
│   ├── sim_multistage.yaml           # 3-stage simulation config (service_time keys)
│   ├── rl3.yaml                      # RL-3 training config (network, reward, episodes)
│   └── legacy/                       # archived RL-1, RL-2, RL-5 configs
│
├── src/
│   ├── data/
│   │   └── generate_orders_seasonal.py  # 240k heterogeneous orders with seasonal demand
│   ├── simulation/
│   │   ├── multistage/               # 3-stage SimPy model (sim_multistage.py)
│   │   └── legacy/                   # archived 5-stage and MVP simulators
│   ├── rl/
│   │   ├── dqn_agent.py              # DQN agent and Q-network
│   │   ├── replay_buffer.py          # experience replay buffer
│   │   ├── env_fullstage_rl.py       # RL-3 environment (3 stages, workload-unit service times)
│   │   ├── main_train_rl3.py         # RL-3 training entry point
│   │   ├── evaluate_rl3.py           # RL-3 single-window evaluation
│   │   ├── evaluate_rl3_monthly_capacity_cost.py  # monthly analysis (16 regimes × months × 3 policies)
│   │   └── legacy/                   # archived RL-5 scripts
│   ├── reporting/
│   │   └── export_rl3_monthly_recommendations.py  # webapp-ready CSV export
│   ├── api/                          # FastAPI backend
│   │   ├── main.py                   # endpoints (v3.0.0)
│   │   ├── utils.py                  # CSV enrichment for uploaded files
│   │   ├── runners.py                # subprocess orchestration + status tracking
│   │   └── schemas.py                # Pydantic models
│   └── validation/
│       └── quick_project_checks.py   # data and eval sanity checks
│
├── webapp/                           # React + Vite frontend (5 tabs)
│   └── src/
│       ├── App.tsx                   # main app (tabs, header)
│       ├── api.ts                    # API client
│       ├── types.ts                  # TypeScript interfaces
│       └── components/tabs/
│           ├── UploadRunTab.tsx      # upload + run + progress bar
│           ├── WorkforcePlannerTab.tsx  # monthly capacity recommendations
│           ├── DemandComplexityTab.tsx  # seasonal demand + order heterogeneity charts
│           ├── PolicyComparisonTab.tsx  # FIFO vs Urgent-First vs RL-3
│           └── MethodTab.tsx            # static methodology explanation
│
├── data/                             # generated — not committed
│   ├── orders_base_seasonal.csv      # 240,000 heterogeneous synthetic orders
│   ├── orders_base_seasonal_summary.csv  # monthly statistics
│   ├── dqn_rl3_final.pt             # trained RL-3 model weights
│   ├── rl3_monthly_capacity_cost_results.csv  # full evaluation results
│   └── app_exports/                  # webapp-ready CSVs
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

If PowerShell blocks script execution:

```powershell
.venv\Scripts\activate.bat
```

Verify:

```powershell
python -c "import torch, simpy, pandas, yaml; print('ok')"
```

---

## Full Validation Pipeline

```powershell
# 1. Generate heterogeneous seasonal dataset (240,000 orders)
python -m src.data.generate_orders_seasonal

# 2. Sanity checks (data columns, value ranges, file presence)
python -m src.validation.quick_project_checks

# 3. Train RL-3 DQN (60 episodes on seasonal dataset)
python -m src.rl.main_train_rl3

# 4. Evaluate RL-3 vs FIFO vs Urgent-First
python -m src.rl.evaluate_rl3

# 5. Monthly capacity-cost optimisation (5 months × 16 regimes × 3 policies = 240 runs)
python -m src.rl.evaluate_rl3_monthly_capacity_cost ^
    --orders data/orders_base_seasonal.csv ^
    --months January,June,August,October,December ^
    --cost-late-urgent 15 ^
    --cost-late-normal 10 ^
    --worker-cost-per-hour 18 ^
    --hours-per-worker-month 160 ^
    --output data/rl3_monthly_capacity_cost_results.csv

# 6. Export webapp-ready CSVs
python -m src.reporting.export_rl3_monthly_recommendations ^
    --input data/rl3_monthly_capacity_cost_results.csv ^
    --output-summary data/app_exports/rl3_monthly_recommendations_summary.csv ^
    --output-full data/app_exports/rl3_monthly_capacity_cost_results_app.csv
```

---

## Capacity Regimes (16 total)

| Regime | Pick | Pack | Dispatch | Total |
|--------|------|------|----------|-------|
| s111   | 1    | 1    | 1        | 3     |
| s211   | 2    | 1    | 1        | 4     |
| s121   | 1    | 2    | 1        | 4     |
| s112   | 1    | 1    | 2        | 4     |
| s221   | 2    | 2    | 1        | 5     |
| s212   | 2    | 1    | 2        | 5     |
| s122   | 1    | 2    | 2        | 5     |
| s311   | 3    | 1    | 1        | 5     |
| s231   | 2    | 3    | 1        | 6     |
| s312   | 3    | 1    | 2        | 6     |
| s222   | 2    | 2    | 2        | 6     |
| s321   | 3    | 2    | 1        | 6     |
| s322   | 3    | 2    | 2        | 7     |
| s331   | 3    | 3    | 1        | 7     |
| s332   | 3    | 3    | 2        | 8     |
| s432   | 4    | 3    | 2        | 9     |

---

## Policies

| Policy        | Description                                                       |
|---------------|-------------------------------------------------------------------|
| `fifo`        | First-in first-out (arrival order, no differentiation)           |
| `urgent_first`| Always serve urgent orders before normal                          |
| `rl3_dqn`     | Learned DQN policy — acts at all 3 stages when a worker is free  |

---

## Seasonal Demand Pattern

240,000 orders across a full year (2026):

| Season      | Months    | Share  | Urgent % |
|-------------|-----------|--------|----------|
| Winter peak | Jan, Feb  | 24%    | 18-20%   |
| Low         | May-Aug   | 15.5%  | 8-9%     |
| Autumn ramp | Sep-Oct   | 15.5%  | 12-15%   |
| Pre-Xmas    | Nov       | 14%    | 22%      |
| Christmas   | Dec       | 17%    | 25%      |

Campaign months (Jan, Feb, Nov, Dec) also have bursty within-month day distributions.

---

## Webapp — Local Development

### Backend (FastAPI)

```powershell
python -m uvicorn src.api.main:app --reload --port 8000
```

Endpoints:

| Method | Path                              | Description                           |
|--------|-----------------------------------|---------------------------------------|
| GET    | /health                           | Service status                        |
| POST   | /upload-orders                    | Upload orders CSV (auto-enriched)     |
| POST   | /run/monthly-capacity-cost        | Run RL-3 monthly optimisation         |
| GET    | /run/status                       | Poll run progress (status.json)       |
| GET    | /results/latest/recommendations   | Monthly recommendation summary        |
| GET    | /results/latest/full              | Full results CSV                      |
| GET    | /recommend/month/{month_name}     | Month-specific recommendation         |
| GET    | /files/status                     | Check file availability               |
| GET    | /data/order-summary               | Monthly order statistics (for charts) |

**Upload enrichment**: CSVs missing `product_family`, `complexity_level`, or workload unit columns
are automatically enriched using the same distributional assumptions as the generator.

### Frontend (React + Vite)

```powershell
cd webapp
npm install
npm run dev
```

Open `http://localhost:5173`.

Tabs:
- **Run** — upload orders CSV and trigger monthly optimisation
- **Recommendations** — monthly workforce planning cards
- **Demand & Complexity** — seasonal demand charts + heterogeneity breakdown
- **Policy Comparison** — FIFO vs Urgent-First vs RL-3 across regimes
- **Method** — methodology explanation (simulation, policies, cost formula)

### Environment variables (optional)

```
VITE_API_BASE_URL=http://localhost:8000   # frontend → backend URL
FRONTEND_ORIGIN=http://localhost:5173     # backend CORS allowlist
```

---

## Recommendation Modes

For each month, the export computes four recommendation categories:

| Card                                   | Logic                                                   |
|----------------------------------------|---------------------------------------------------------|
| **Cheapest Option**                    | Min total cost across all regimes and policies          |
| **Best RL-3 Option**                   | Min total cost using only `rl3_dqn` policy              |
| **Min Workforce for Urgent SLA ≥ 95%** | Fewest workers where `urgent_sla ≥ 0.95`               |
| **Min Workforce for Total SLA ≥ 80%**  | Fewest workers where `total_sla ≥ 0.80`                |

---

## Sanity Checks

```powershell
python -m src.validation.quick_project_checks
```

---

## Legacy (RL-5 / 5-stage)

The earlier RL-5 experiment (5-stage: Picking → QC → Packing → Labelling → Dispatch) has been
archived and is **not part of the current pipeline**:

```
configs/legacy/rl5/
src/rl/legacy/rl5/
src/simulation/legacy/5stage/
src/reporting/legacy/
```
