# TFM Logistics Baseline — Decision Support System

A discrete-event simulation + reinforcement learning decision-support system for logistics operations.

The system generates synthetic order demand, runs a **3-stage SimPy simulation** with FIFO and `urgent_first`
baseline policies, and evaluates an **RL-3 DQN agent** that dynamically prioritises orders at
**Picking → Packing → Dispatch**. Results drive a monthly workforce planning tool that recommends the
staffing configuration minimising total SLA penalty cost + labour cost.

> Preferred terms: discrete-event simulation, simulation-based operational analysis, decision support.
> Not a "digital twin".

---

## Final Pipeline Overview

```
Synthetic orders
→ 3-stage simulation     (SimPy: Picking → Packing → Dispatch)
→ Baseline policies      (FIFO / Urgent-First)
→ RL-3 DQN agent         (one shared agent acting at all three stages)
→ Evaluation             (single-window + multi-window robustness)
→ Monthly analysis        (per-month SLA + cost across 7 capacity regimes)
→ Decision support        (capacity-cost optimisation + monthly recommendations)
→ Reporting               (plots / sanity checks / webapp-ready CSVs)
→ Webapp                  (React + FastAPI — upload orders → get recommendation)
```

---

## Repository Structure

```text
TFM-Logistics-Baseline/
├── configs/
│   ├── demand_base.yaml              # data generation parameters
│   ├── sim_multistage.yaml           # 3-stage simulation config
│   ├── rl3.yaml                      # RL-3 training config
│   └── sensitivity_scenarios.yaml    # bottleneck service-time multipliers
│
├── src/
│   ├── data_generation/              # synthetic order generator
│   ├── simulation/
│   │   ├── multistage/               # 3-stage SimPy model (sim_multistage.py)
│   │   └── legacy/                   # archived 5-stage and MVP simulators
│   ├── rl/
│   │   ├── dqn_agent.py              # DQN agent and Q-network
│   │   ├── replay_buffer.py          # experience replay buffer
│   │   ├── env_fullstage_rl.py       # RL-3 environment (3 stages)
│   │   ├── main_train_rl3.py         # RL-3 training entry point
│   │   ├── evaluate_rl3.py           # RL-3 single-window evaluation (12 regimes)
│   │   ├── evaluate_rl3_multiseed.py # RL-3 multi-window robustness
│   │   ├── evaluate_rl3_monthly_capacity_cost.py  # monthly capacity-cost (16 regimes × 12 months × 3 policies)
│   │   └── legacy/                   # archived RL-5 scripts + evaluate_rl3_sensitivity
│   ├── reporting/
│   │   ├── export_rl3_monthly_recommendations.py  # webapp-ready CSV export
│   │   ├── plot_final_results.py     # result plots (RL-3)
│   │   ├── plot_bottleneck_sensitivity.py         # bottleneck sensitivity plots (RL-3)
│   │   ├── sla_cost_calculator.py    # SLA cost analysis
│   │   └── legacy/                   # archived RL-5 reporting scripts
│   ├── api/                          # FastAPI backend
│   │   ├── main.py                   # endpoints
│   │   ├── runners.py                # subprocess orchestration + status tracking
│   │   └── schemas.py                # Pydantic models
│   ├── pipeline/
│   │   └── run_all.py                # general pipeline runner
│   └── validation/
│       └── quick_project_checks.py   # data and eval sanity checks
│
├── webapp/                           # React + Vite frontend
│   └── src/
│       ├── App.tsx                   # main app (tabs, header)
│       ├── api.ts                    # API client
│       ├── types.ts                  # TypeScript interfaces
│       └── components/
│           └── tabs/
│               ├── OverviewTab.tsx
│               ├── UploadRunTab.tsx  # upload + run + progress bar
│               ├── WorkforcePlannerTab.tsx
│               ├── MonthlyResultsTab.tsx
│               ├── PolicyComparisonTab.tsx
│               ├── RL3PolicyTab.tsx  # RL-3 DQN visualisation
│               └── DataExplorerTab.tsx
│
├── data/                             # generated — not committed
│   ├── app_exports/                  # webapp-ready CSVs
│   └── api_runs/latest/              # API run outputs (status.json + CSVs)
├── legacy/                           # archived experiments (RL-5, 5-stage, old outputs)
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

## Capacity Regimes

The monthly capacity-cost optimisation evaluates 16 worker configurations (Picking × Packing × Dispatch).
The single-window evaluation (`evaluate_rl3.py`) uses a subset of 12 regimes.

| Regime | Pick | Pack | Dispatch | Total | Notes |
|--------|------|------|----------|-------|-------|
| s111   | 1    | 1    | 1        | 3     | minimal |
| s211   | 2    | 1    | 1        | 4     | picking-focused |
| s121   | 1    | 2    | 1        | 4     | packing-focused |
| s112   | 1    | 1    | 2        | 4     | dispatch-focused |
| s221   | 2    | 2    | 1        | 5     | picking + packing |
| s212   | 2    | 1    | 2        | 5     | picking + dispatch |
| s122   | 1    | 2    | 2        | 5     | packing + dispatch |
| s311   | 3    | 1    | 1        | 5     | strong picking |
| s231   | 2    | 3    | 1        | 6     | strong packing |
| s312   | 3    | 1    | 2        | 6     | strong picking + dispatch |
| s222   | 2    | 2    | 2        | 6     | balanced medium |
| s321   | 3    | 2    | 1        | 6     | high picking + packing |
| s322   | 3    | 2    | 2        | 7     | high picking, balanced downstream |
| s331   | 3    | 3    | 1        | 7     | strong picking + packing |
| s332   | 3    | 3    | 2        | 8     | strong all-round |
| s432   | 4    | 3    | 2        | 9     | maximum throughput |

---

## Policies

| Policy      | Description                                                  |
|-------------|--------------------------------------------------------------|
| fifo        | First-in first-out (arrival order, no differentiation)      |
| urgent_first | Always serve urgent orders before normal                    |
| rl3_dqn     | Learned DQN policy — acts at each stage when worker is free |

---

## Quick Start — Pipeline Runner

```powershell
# Show all options
python -m src.pipeline.run_all

# Generate data + run 3-stage simulation
python -m src.pipeline.run_all --base

# Train RL-3 (slow)
python -m src.pipeline.run_all --train-rl3

# Evaluate RL-3
python -m src.pipeline.run_all --eval-rl3 --multiseed-rl3

# Run monthly capacity-cost optimisation and export
python -m src.pipeline.run_all --decision-support

# Everything
python -m src.pipeline.run_all --all
```

---

## CLI — Decision Support

Run the full monthly analysis directly:

```powershell
# Step 0 (optional): generate seasonal orders (preferred input for capacity evaluation)
python -m src.data.generate_orders_seasonal

# Step 1: monthly capacity-cost simulation (16 regimes × 12 months × 3 policies = 576 runs)
python -m src.rl.evaluate_rl3_monthly_capacity_cost
  # or with overrides:
  python -m src.rl.evaluate_rl3_monthly_capacity_cost ^
      --orders data/orders_base_seasonal.csv ^
      --checkpoint data/dqn_rl3_final.pt ^
      --cost-late-urgent 15 ^
      --cost-late-normal 10 ^
      --worker-cost-per-hour 18 ^
      --hours-per-worker-month 160 ^
      --output data/rl3_monthly_capacity_cost_results.csv

# Step 2: export webapp-ready CSVs
python -m src.reporting.export_rl3_monthly_recommendations
  # or with overrides:
  python -m src.reporting.export_rl3_monthly_recommendations ^
      --input data/rl3_monthly_capacity_cost_results.csv ^
      --output-summary data/app_exports/rl3_monthly_recommendations_summary.csv ^
      --output-full data/app_exports/rl3_monthly_capacity_cost_results_app.csv
```

---

## Webapp — Local Development

### Backend (FastAPI)

```powershell
python -m uvicorn src.api.main:app --reload --port 8000
```

Endpoints:

| Method | Path                          | Description                        |
|--------|-------------------------------|------------------------------------|
| GET    | /health                       | Service status                     |
| POST   | /upload-orders                | Upload historical orders CSV       |
| POST   | /run/monthly-capacity-cost    | Run RL-3 monthly optimisation      |
| GET    | /run/status                   | Poll run progress (status.json)    |
| GET    | /results/latest/recommendations | Monthly recommendation summary   |
| GET    | /results/latest/full          | Full results CSV                   |
| GET    | /recommend/month/{month_name} | Month-specific recommendation      |
| GET    | /files/status                 | Check file availability            |

### Frontend (React + Vite)

```powershell
cd webapp
npm install
npm run dev
```

Open `http://localhost:5173`.

The frontend shows a **progress bar** when "Run Monthly Optimisation" is clicked. It polls `/run/status`
every 2 seconds and shows elapsed time, step name, and progress percentage.

### Environment variables (optional)

```
VITE_API_BASE_URL=http://localhost:8000   # frontend → backend URL
FRONTEND_ORIGIN=http://localhost:5173     # backend CORS allowlist
```

---

## Expected Output Files

After running the decision-support pipeline:

```
data/
├── orders_base_seasonal.csv                    # seasonal synthetic orders (preferred input)
├── orders_base.csv                             # base synthetic orders
├── dqn_rl3_final.pt                            # trained RL-3 model weights
├── rl3_eval_results.csv                        # single-window evaluation (12 regimes)
├── rl3_eval_multiseed_results.csv              # multi-window robustness
├── rl3_monthly_capacity_cost_results.csv       # 576 simulation runs (16×12×3)
└── app_exports/
    ├── rl3_monthly_recommendations_summary.csv # 1 row per month, 4 recommendation modes
    └── rl3_monthly_capacity_cost_results_app.csv  # full results for webapp
```

---

## Recommendation Modes

For each month, the export computes four recommendation categories:

| Card                                  | Logic                                                   |
|---------------------------------------|---------------------------------------------------------|
| **Cheapest Option**                   | Min total cost across all regimes and policies          |
| **Best RL-3 Option**                  | Min total cost using only rl3_dqn policy                |
| **Min Workforce for Urgent SLA ≥ 95%**| Fewest workers where urgent_sla ≥ 0.95                 |
| **Min Workforce for Total SLA ≥ 80%** | Fewest workers where total_sla ≥ 0.80                  |

---

## Deployment

**Frontend** — deploy the `webapp/dist` build to Vercel or Netlify.

```powershell
cd webapp && npm run build
```

**Backend** — deploy to Render or Railway. Set env var `FRONTEND_ORIGIN` to the deployed frontend URL.

---

## Sanity Checks

```powershell
python -m src.validation.quick_project_checks
```

Checks:
- Order files (columns, uniqueness, SLA values)
- RL-3 eval files (required columns, value ranges)
- Monthly capacity-cost results
- App export CSVs

---

## Legacy (RL-5 / 5-stage)

The earlier RL-5 experiment (5-stage: Picking → QC → Packing → Labelling → Dispatch) has been
archived and is **not part of the current pipeline**. Legacy locations:

```
legacy/old_models/        # dqn_rl5_v2_*.pt and RL-3 intermediate checkpoints
legacy/old_outputs/       # RL-5 CSVs and old data/plots/
legacy/graphic_design/    # warehouse animation scripts and GIFs
legacy/misc/              # other archived files
src/rl/legacy/rl5/        # RL-5 training and evaluation scripts
src/simulation/legacy/5stage/  # 5-stage SimPy simulator
src/reporting/legacy/     # RL-5 reporting and calibration scripts
webapp/src/components/tabs/legacy/  # RL5PolicyTab (not imported)
```

These files remain for reference and are not imported or executed by the current codebase.
