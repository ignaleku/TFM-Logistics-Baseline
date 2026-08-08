# TFM Logistics Baseline — Prescriptive Warehouse Planning

A discrete-event simulation + reinforcement learning decision-support system for warehouse operations.

This project is a **prescriptive warehouse planning tool** with three top-level areas: **Run**
(execution), **Future Planning** (latest prospective result), and **Historical Analysis** (latest
retrospective result). It simulates heterogeneous orders through **Picking → Packing → Dispatch**,
compares FIFO, urgent-first, and RL-3 sequencing policies, identifies bottlenecks, evaluates
workforce configurations, and estimates whether adding capacity is economically justified — all
against a **finite monthly operating-time capacity horizon** (see below), not an unbounded clock.

> Preferred terms: discrete-event simulation, simulation-based operational analysis, decision support.
> Not a "digital twin".

---

## Operating-Time Capacity Model

**One worker = one monthly FTE.** Its physical productive capacity and its economic cost are the
same figure — `hours_per_worker_month` (160 by default, `configs/planning_profile.yaml::cost_defaults`)
— so a worker can never be simulated as busy longer than it is actually paid for:

```
operating_horizon_minutes = hours_per_worker_month × 60        (160h → 9,600 minutes)
labour_cost                = workers × worker_cost_per_hour × hours_per_worker_month
```

This was **not previously true**. The simulator used to run `env.run(until=done_event)` — until
every order was processed, however long that took — while `hours_per_worker_month` only fed the
economic formula. Because generated/historical arrivals were spread across the full calendar month
(~744h for a 31-day month), the emergent simulated horizon landed close to 744h while labour cost
was computed against 160h: **one worker got ~584 free physical hours every month.** For a December
scenario with a 5,000-order monthly override this showed as "Operating Days: 31 · Orders/Operating
Hour: 6.72" — implying ~744h of assumed capacity — while the economics still charged for 160h.

Fixed in `src/simulation/multistage/operating_time.py`:

- **`compress_to_operating_time`** maps each order's calendar position within its own calendar
  month onto `[0, operating_horizon_minutes)`, preserving relative order and burstiness. Applied
  identically to Future Planning's generated (seasonal-shaped) arrivals and to Historical
  Analysis's real timestamps — both replay against the same monthly FTE capacity basis.
- **`sim_multistage.py` / `env_fullstage_rl.py`** now run `env.run(until=operating_horizon_seconds)`
  — a fixed horizon, never `until=done_event`. Orders still queued or in service when the horizon
  ends are monthly **backlog**: reported separately (`completed_orders`, `unfinished_orders`,
  `unfinished_urgent_orders`, `unfinished_normal_orders`, `backlog_share`) and always counted as
  SLA failures — the model never lets a worker "catch up" for free.
- **Stage utilisation** = `busy_worker_minutes / (workers × operating_horizon_minutes)`, now
  structurally bounded to ≤ 1.0 (a `RuntimeWarning` fires if it isn't — see `stage_metrics.py`).
- **SLA/due-time** was already relative (`sla_minutes` added to arrival, never an absolute
  wall-clock deadline), so it stays internally consistent on the compressed clock unchanged.
- The RL-3 environment's idle-wait loop was refactored from `while empty: yield env.timeout(1)`
  polling to an event-driven `_StageSignal` wake — required once the horizon is a fixed 9,600
  minutes rather than "however long the queue took to drain" (polling every second for a long
  idle tail after work completes is prohibitively expensive; event-driven waiting isn't).

Before/after, December, 5,000-order override:

| | Old (buggy) | New (corrected) |
|---|---|---|
| Assumed capacity window | 31 × 24 = 744h (implicit) | 160h (`hours_per_worker_month`) |
| Orders / operating hour | 6.72 | **31.25** |
| Worker physical vs. paid hours | Free ~584h/worker/month | Identical — physically capped at 160h |

`hours_per_operating_day` (`calendar_profile.hours_per_operating_day`, default 8) is a **separate,
explanatory-only** figure for an "Equivalent Operating Days" display (`160 / 8 = 20`) — it is never
a second capacity source of truth.

---

## Analytical Capacity Estimate + Dynamic Workforce Candidates

Business workforce search no longer relies on the static 16-regime research grid alone
(`configs/planning_profile.yaml::regimes`, `s111`…`s432`, 1–9 total workers), which the corrected
operating-time model can make far too small at real demand volumes (see validation results below).

1. **Analytical estimate** (`src/analysis/capacity_estimate.py`): deterministic expected service
   workload per stage (same service-time formula, noise fixed at 1.0 instead of sampled), divided
   by paid capacity per worker at a target utilisation (`capacity_planning.target_utilisation`,
   0.85 default) and ceil'd. A screening anchor, not the final answer.
2. **Dynamic candidates** (`src/analysis/candidate_generation.py`): ~16
   (`capacity_planning.candidate_count`) workforce candidates around that centre — single-stage
   ±1/±2 perturbations, leaner (−15%) / safer (+15%) total-workforce variants, balanced two-stage
   combinations, and the client's current workforce if given. Never a full cubic grid, never a
   stage below 1.
3. SimPy (via the existing screening+validation / adaptive-search machinery) determines the true
   recommendation from there, exactly as before.

The original static 16 regimes remain the RL research/benchmark/generalisation grid
(`rl_generalisation.train_regimes` / `holdout_regimes`) — no longer the business-planning search
space.

**Regime naming** (`src/analysis/regime_naming.py`): `s{picking}{packing}{dispatch}` when every
stage is under 10 workers (unchanged, e.g. `s432`); `s{picking}_{packing}_{dispatch}` otherwise
(e.g. `s26_14_7`) — dynamic candidates and adaptive search can legitimately need 10+ workers per
stage at peak demand, and `s1063` would be ambiguous. One implementation, used by the evaluator,
the API, the frontend (`webapp/src/utils/regime.ts`), the adaptive search, and reporting.

**Adaptive search limits** (`configs/planning_profile.yaml::adaptive_search.max_extra_workers_per_stage`,
default 4): relative to *that month's* analytical estimate per stage, not a single global cap
sized for the old small regimes.

### Validation evidence (December)

| Monthly demand | Old model implied | Analytical estimate | SimPy recommendation | Orders/operating hour |
|---|---|---|---|---|
| 5,000 (override) | ~s111-scale | picking 4 / packing 2 / dispatch 1 | **s321** (6 workers), Urgent-First, feasible | 31.25 |
| 40,800 (240k annual forecast) | ~s432 (9 workers) | picking 27 / packing 14 / dispatch 7 | **s26_14_7** (47 workers), Urgent-First, feasible, 100.0%/91.8% SLA | 255.0 |

The peak-month recommendation is materially larger than the old static grid could ever produce —
**expected and correct**, not a regression: the old grid topped out at 9 total workers regardless
of demand.

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
      compress_to_operating_time (per-month operating horizon)
                            ▼
                Same SimPy 3-stage engine
       (src/simulation/multistage · src/rl/env_fullstage_rl.py)
                            ▼
     FIFO / Urgent-First / RL-3 DQN, common random numbers
                            ▼
      Bottleneck ranking · break-even · adaptive capacity search
                            ▼
          Decision support (webapp) — mode-separated results
```

**Historical Analysis** — upload a real order-level CSV. For each analysed month independently:
real timestamps are compressed onto that month's own operating horizon, an analytical capacity
estimate and dynamic candidate set are generated from that month's actual workload, and the system
performs counterfactual analysis: which policy/workforce would have performed best, where the
bottleneck was, whether an extra worker would have paid for itself. Retrospective language
throughout ("recommended historical configuration", "counterfactual estimated cost").

**Future Planning** — the user does **not** need to provide every future order individually.
A small set of aggregate inputs (planning month, expected annual orders, uncertainty level,
optional current workforce) is combined with a calibrated client planning profile
(`configs/planning_profile.yaml`) to generate plausible simulated orders, compressed onto that
month's operating horizon. A small number of scenario **replications** (default 3, different
seeds) avoid presenting one random realisation as certainty; results report mean and p90 cost,
mean SLA, and probability of meeting SLA targets. Uses a screening+validation strategy (§ below)
over the dynamic candidate set rather than the exhaustive 16×3×3 grid.

Both modes feed the **same** SimPy simulation engine after input preparation — there is exactly
one simulator — and their results are **persisted separately**
(`data/api_runs/latest/{future,historical}/`) so running one mode never overwrites the other's
results (see Mode-Separated Persistence below).

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
mappings, SLA targets, cost defaults, workforce regimes, capacity-planning parameters, and
bottleneck-score weights live in **`configs/planning_profile.yaml`** — the single source of truth
read by the seasonal generator, the future-planning generator, uploaded-CSV enrichment, and the
API. RL reward coefficients stay in `configs/rl3.yaml` (internal training config, not a
client-facing planning assumption).

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
worker count, busy-worker time → utilisation (bounded by the finite operating horizon), processed
count → throughput, service-time and wait-time distributions (mean/p95), time-weighted average
and max queue length, and — for late orders only — accumulated waiting time per stage and its
share of all late-order waiting.

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
current-mix lenses: +1 worker = +1 monthly FTE = +`hours_per_worker_month` operating hours at
`worker_cost_per_hour × hours_per_worker_month`), and runs a **bottleneck-directed adaptive
search**: starting from the recommended dynamic candidate, if it's infeasible (or under high
pressure with real late-order cost), add one worker to the top bottleneck stage (or the top two,
if their pressure scores are close) and re-evaluate all three policies — the best policy can
change as capacity changes. The per-stage ceiling is that month's analytical estimate plus
`adaptive_search.max_extra_workers_per_stage` (default 4), not a single global cap. Every
candidate is logged accepted/rejected with a reason (`src/analysis/bottleneck_report.py`).

## SLA Feasibility

A candidate is **feasible** only if it meets both SLA floors (`planning_profile.yaml::sla`,
default urgent ≥ 95%, normal ≥ 80%). Recommendations prefer the cheapest *feasible* candidate; if
none exists, the system explicitly labels its pick "best available — SLA targets not fully met"
and ranks by SLA violation first. A pathological 100%-urgent/2%-normal result is never presented
as an unqualified winner. The "Minimum Feasible Workforce" card/column requires **both** floors —
a config with total SLA 88% but urgent SLA 64% is never promoted as a minimum workforce.

---

## RL-3: Operating-Time Retrain

Switching to a finite 9,600-minute monthly horizon with real backlog, and to dynamic candidates
that can need 10+ workers per stage, changes the RL-3 environment's state/capacity distribution
enough that the pre-existing checkpoint (trained under the old unbounded-horizon, 1–4-worker-per-stage
regime) is no longer valid. This was verified empirically before touching anything: manual
policies (FIFO, Urgent-First) were validated first under the corrected model — both produce
coherent results (see the December 5k/40.8k table above) — then RL-3 was compared at the actual
peak-month recommended capacity (December, 47 workers, `s26_14_7`): **RL-3 collapsed to 0.8% total
SLA / 0.4% urgent SLA**, versus Urgent-First's 93.9% / 100.0% at the same workforce.

### A second, sharper bug hiding inside that failure

Part of that collapse was a genuine checkpoint/distribution mismatch (expected). But investigating
it turned up a second, more serious bug in the RL-3 environment itself, introduced earlier in this
same change (the polling→event-driven idle-wait rewrite needed once the horizon became fixed —
see Operating-Time Capacity Model above): `_StageSignal.reset()` was called by *each worker*
immediately before it waited, and unconditionally replaced the shared `self.event`. With W workers
idle at the same stage, the first W−1 to call `reset()` were silently left holding a reference to
an **orphaned** event object that `notify()` would never touch again — only the *last* worker to
call `reset()` before the next `notify()` was ever reachable. Net effect: **only ~1/W of a stage's
workers ever did any work**, regardless of backlog. Diagnosed by running the retrained-so-far
checkpoint on December's `s26_14_7` (26 picking workers) and finding picking utilisation of
**3.8%** (≈ 1/26) with a queue of 38,867 unpicked orders sitting right there — not a policy-quality
problem, a concurrency bug. `sim_multistage.py` (FIFO/Urgent-First) never used this mechanism (it
blocks directly on `simpy.Store.get()`, which has no such issue), so **only RL-3 was affected** —
every FIFO/Urgent-First number in this document is unaffected by it.

Fix: `notify()` now owns the reset — it succeeds the current event and immediately replaces
`self.event` with a fresh one, exactly once per firing, never per-waiter. Workers read
`signal.event` fresh at each wait instead of caching a `reset()`-returned reference. Verified
immediately: the same December `s26_14_7` regime under a random policy went from 3.8% picking
utilisation / 95.3% backlog to **78.5% utilisation / 4.4% backlog** — in line with FIFO/Urgent-First
on the same regime — confirming the fix, not the retrain, was what mattered most.

The RL-3 numbers below are trained and evaluated **after** this fix (`src/rl/env_fullstage_rl.py`);
the 0.8%/0.4% figures above predate it and should be read as "the environment was still broken",
not as a clean measurement of checkpoint generalisation.

Retrain (`src/rl/main_train_rl3.py`), per the corrected model:

- **Representative months**: June (low demand), October (medium), December (peak) — each month's
  real seasonal orders, compressed onto its own 9,600-minute operating horizon exactly as
  Future/Historical evaluation does (not an arbitrary cross-month order window as before).
- **Dynamic training pool** (`src/analysis/capacity_estimate.py` +
  `src/analysis/candidate_generation.py`): for each representative month, an analytical capacity
  centre plus ~9 dynamically generated candidates spanning under-/near-/over-capacity workforce;
  2 candidates per month held out from training for exact-configuration generalisation testing.
  The realised pool (regime labels, per-month order counts) is written to
  `data/rl3_train_pool.json` by every training run.
- **Capacity feature normalisation**: `rl_generalisation.capacity_feature_scale` (20) replaces
  reusing the unrelated adaptive-search worker-limit config — a fixed, documented scale wide
  enough that October/December-scale candidates (9–27 workers/stage) don't saturate the feature
  the way a scale of 6 would.
- **Backlog-aware reward**: orders still unresolved at the horizon end previously left their
  buffered transitions at the placeholder `reward=0.0` (never reached dispatch, so `_reward()` was
  never called) — a spuriously neutral signal for "left this order in backlog at month end".
  Fixed to assign the same maximal late penalty a completed-but-very-late order would receive.
- 200 episodes, uniform sampling over the training pool's (month, regime) pairs — with the
  `_StageSignal` fix, each episode now does genuinely more simulated work than the pre-fix 400
  planned, so 200 was recalibrated after confirming per-episode wall-clock cost (~5s June,
  ~11s October, ~25s December) and that the replay buffer (200,000 capacity) and epsilon decay
  (`decay_steps=200000`) both saturate well before episode 200.

`src/rl/rl_audit.py` and `src/rl/evaluate_rl3_generalisation.py` were updated to run on the
operating-time model (finite horizon, `slice_month_operating_time`) and both accept dynamic
(non-`REGIME_LOOKUP`) regime labels, not just the static 16.

### Results (post-fix, post-retrain)

Training converged to high SLA on all three representative months (mean of the last 50 training
episodes, greedy-dominant since epsilon reaches its 0.05 floor by ~episode 20):

| Month | Mean total SLA | Mean urgent SLA | Mean normal SLA | Mean backlog share |
|---|---|---|---|---|
| June (low) | 97.2% | 100.0% | 97.0% | 1.5% |
| October (medium) | 99.4% | 99.95% | 99.3% | 0.6% |
| December (peak) | 95.4% | 99.96% | 93.8% | 4.6% |

`rl_audit.py --month December --regimes s22_11_5,s26_14_7` (one December holdout regime never
seen in training, plus the analytically-centred one actually recommended by Future Planning):

| Regime | Seen in training | RL-3 urgent / normal SLA | Urgent-First urgent / normal SLA |
|---|---|---|---|
| `s22_11_5` (holdout) | No | 99.22% / **85.40%** | 99.27% / 73.88% |
| `s26_14_7` (near-centre) | No (this exact seed) | 99.93% / **93.94%** | 99.99% / 93.28% |

`anomaly_reproduced: False`, `starvation_detected: False`, `code_bug_detected: False` — RL-3 now
matches Urgent-First on urgent SLA and **beats it on normal SLA** on both a held-out and the
actually-recommended December regime, with no starvation pattern. `evaluate_rl3_generalisation.py`
against the *static* 16-regime grid (1–9 workers) shows all three policies — FIFO, Urgent-First,
and RL-3 alike — at ~0% feasibility for December's full 40,800-order demand: expected and correct,
not an RL failure, since that grid is now far too small for peak demand regardless of policy (the
entire reason the dynamic candidate system replaced it, §12–§14) — the meaningful comparison is
against dynamically-sized candidates, not the obsolete static grid.

**Caveat**: this is a genuinely positive result, not a guarantee. The training pool covers three
representative months and a bounded candidate spread per month; RL-3's behaviour on months or
capacity levels well outside that pool is untested. The SLA-feasibility gate
(`src/analysis/sla_feasibility.py`) remains the safety net regardless: if RL-3 is ever infeasible
for a given month/regime, the recommendation falls back to the best feasible manual policy, so
this limitation cannot silently produce a bad business recommendation.

### End-to-end confirmation: full peak-December Future Planning run

Re-running the actual business flow (December, 240,000-order annual forecast → 40,800 expected
orders, `POST /run/future-planning`, full screening+validation over 16 dynamic candidates) with
the corrected environment and retrained checkpoint — not a hand-picked diagnostic regime, the
real recommendation pipeline:

| Policy | Feasible | Total SLA | Urgent SLA | Normal SLA | Total cost |
|---|---|---|---|---|---|
| FIFO | No | 85.7% | 60.9% | 94.0% | €212,595 |
| Urgent-First | Yes | 92.7% | 99.97% | 90.2% | €163,262 |
| **RL-3 DQN** | **Yes** | **93.9%** | **99.93%** | **91.9%** | **€158,103** |

Recommended: **RL-3 DQN, `s25_14_7`** (25 picking / 14 packing / 7 dispatch = 46 workers) —
feasible, and **€5,159/month cheaper than Urgent-First** while also posting the better normal
SLA. No adaptive search needed (already optimal among the tested candidates). This is the same
scenario that produced the original 0.8%/0.4% catastrophic failure at the very start of this
investigation — the full journey (buggy capacity model → fixed model exposes a broken RL
environment → fixed environment + retrain → RL-3 outperforms both manual policies) is the honest
before/after, not a cherry-picked one.

---

## Repository Structure

```text
TFM-Logistics-Baseline/
├── configs/
│   ├── planning_profile.yaml         # single source of truth: seasonal/operational/capacity assumptions
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
│   │   │   ├── sim_multistage.py     # FIFO / urgent_first SimPy engine, finite operating horizon
│   │   │   ├── operating_time.py     # THE operating-time clock: horizon, compression, slicing
│   │   │   ├── service_time_map.py   # common-random-number service-time sampling
│   │   │   └── stage_metrics.py      # bottleneck instrumentation (queue area, wait, utilisation)
│   │   └── legacy/                   # archived 5-stage and MVP simulators
│   ├── analysis/
│   │   ├── bottleneck.py             # pressure-score ranking
│   │   ├── capacity_estimate.py      # analytical pre-simulation workforce estimate
│   │   ├── candidate_generation.py   # dynamic workforce candidate generation
│   │   ├── regime_naming.py          # sPKD / sP_K_D formatting+parsing (single implementation)
│   │   ├── capacity_search.py        # break-even + adaptive capacity search
│   │   ├── sla_feasibility.py        # feasibility / violation scoring
│   │   ├── bottleneck_report.py      # ties the above into one API-ready report
│   │   ├── future_screening.py       # screening+validation regime ranking
│   │   ├── order_summary.py          # run-scoped demand/complexity summary
│   │   └── replication_aggregation.py  # future-planning replication mean/p90 aggregation
│   ├── rl/
│   │   ├── dqn_agent.py, replay_buffer.py
│   │   ├── env_fullstage_rl.py       # RL-3 environment (16-feature state, finite horizon)
│   │   ├── main_train_rl3.py         # training entry point (dynamic-candidate representative months)
│   │   ├── evaluate_rl3_monthly_capacity_cost.py  # shared evaluation core (dynamic or static regimes)
│   │   ├── evaluate_rl3_generalisation.py  # seen vs. held-out regime evaluation
│   │   ├── rl_audit.py               # generalisation / starvation audit
│   │   └── legacy/                   # archived RL-5 scripts
│   ├── reporting/
│   │   └── export_rl3_monthly_recommendations.py
│   ├── api/                          # FastAPI backend
│   │   ├── main.py                   # mode-aware endpoints (?mode=future|historical)
│   │   ├── runners.py                # mode-separated persistence, dynamic candidate wiring
│   │   ├── schemas.py, utils.py
│   └── validation/
│       └── quick_project_checks.py
│
├── webapp/                           # React + Vite frontend — 3 top-level areas
│   └── src/
│       ├── App.tsx                   # Run | Future Planning | Historical Analysis + Methodology modal
│       └── components/
│           ├── tabs/UploadRunTab.tsx # Run: Historical/Future execution forms (unchanged UX)
│           ├── tabs/MethodTab.tsx    # methodology content (shown via modal, not a top-level tab)
│           ├── MethodologyModal.tsx
│           └── results/              # mode-aware result content, shared by both top-level areas
│               ├── ModeResultsTab.tsx        # subnav shell (Recommendation(s)/Demand/Policy/Capacity)
│               ├── ContextBanner.tsx         # persistent run-context banner
│               ├── RecommendationsContent.tsx  # single-month card OR multi-month table+charts
│               ├── DemandComplexityContent.tsx
│               ├── PolicyComparisonContent.tsx
│               └── CapacityBottlenecksContent.tsx
│
├── data/                             # generated — not committed
│   ├── orders_base_seasonal.csv, orders_base_seasonal_summary.csv
│   ├── dqn_rl3_final.pt              # active RL-3 checkpoint (16-dim state)
│   ├── rl3_train_pool.json           # last training run's (month, regime) pool + holdout
│   ├── rl3_train_history.csv
│   └── api_runs/latest/
│       ├── status.json               # global background-job status
│       ├── future/                   # latest Future Planning result (survives a Historical run)
│       └── historical/               # latest Historical Analysis result (survives a Future run)
├── requirements.txt
└── README.md
```

---

## Mode-Separated Persistence

`data/api_runs/latest/{future,historical}/` each hold their own
`rl3_monthly_capacity_cost_results*.csv`, `bottleneck_analysis.json`, `run_manifest.json`, and
(future only) `future_planning_summary.json` / (historical only) `historical_analysis_summary.json`.
Running Historical Analysis never overwrites the latest Future Planning result and vice versa —
verified by running Future (December) then Historical (May+June) and confirming both remained
independently readable afterward. The background-job status (`status.json`) stays global/shared
since only one run executes at a time.

API endpoints take an explicit `mode=future|historical` query parameter rather than duplicating
every route: `GET /results/latest/{recommendations,full,bottlenecks,context}?mode=...`,
`GET /data/order-summary?mode=...` (omit `mode` for the static annual client-profile baseline).

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

# 3. Train RL-3 DQN (dynamic-candidate representative-month sampling)
python -m src.rl.main_train_rl3

# 4. Audit the RL-3 policy (fair comparison, urgent_first validation, leakage check, diagnostics)
python -m src.rl.rl_audit --month December --regimes s221,s26_14_7

# 5. Seen vs. held-out regime generalisation (static 16-regime benchmark grid)
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

## Capacity Regimes

Business planning (Future/Historical) uses **dynamic candidates** generated per month around an
analytical capacity estimate (`src/analysis/candidate_generation.py`) — not a fixed list.

The original **16 static base regimes** (`s111` … `s432`, `configs/planning_profile.yaml::regimes`)
remain available as a fixed research/benchmark/generalisation grid (RL train/holdout split,
`--regimes` override for manual/CLI evaluation) but are no longer the business-planning search
space, which now scales with actual expected demand.

Naming: `s{picking}{packing}{dispatch}` when every stage is under 10 workers, else
`s{picking}_{packing}_{dispatch}` (`src/analysis/regime_naming.py`).

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

Key endpoints (`mode` is `future` or `historical`):

| Method | Path                                    | Description                                    |
|--------|------------------------------------------|-------------------------------------------------|
| GET    | /health                                 | Service status                                 |
| GET    | /planning/profile                       | Read-only client-profile assumptions for the UI |
| POST   | /planning/preview                       | Derived future-planning assumptions (no run)   |
| POST   | /upload-orders                          | Upload orders CSV (auto-enriched)              |
| POST   | /run/monthly-capacity-cost              | Historical: run analysis (dynamic candidates)  |
| POST   | /run/future-planning                    | Future: generate scenario(s) + optimise        |
| GET    | /run/status                             | Poll run progress (status.json)                |
| GET    | /results/latest/recommendations?mode=   | Monthly recommendation summary                 |
| GET    | /results/latest/full?mode=              | Full results CSV                               |
| GET    | /results/latest/bottlenecks?mode=       | Bottleneck ranking, break-even, adaptive trail |
| GET    | /results/latest/context?mode=           | Run-scoped context (demand summary, banner)    |
| GET    | /files/status                           | Check file availability                        |
| GET    | /data/order-summary?mode=               | Order statistics — run-scoped, or annual if `mode` omitted |

### Frontend (React + Vite)

```powershell
cd webapp
npm install
npm run dev
```

**Top-level**: Run · Future Planning · Historical Analysis, plus a header **Methodology** button
(modal — not a top-level tab).

**Future Planning / Historical Analysis subnav**: Recommendation(s) · Demand & Complexity ·
Policy Comparison · Capacity & Bottlenecks — each shows only that mode's latest result, persisted
across refresh via the backend `run_manifest.json`, never ephemeral React state.

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
- Backlog accounting is conservative: an order mid-service exactly at horizon end is excluded
  from that stage's busy-minutes (undercounting utilisation slightly) rather than given partial
  credit — a deliberate, documented approximation, never an overcount.
- The RL-3 audit's root-cause interpretation is evidence-based on the regimes actually tested;
  it is not an exhaustive proof for every possible regime/month combination.
- The retrained RL-3 checkpoint is trained on three representative months (June/October/December)
  and a bounded dynamic candidate pool per month — not every month or every possible workforce
  size; the feasibility layer (SLA floors) remains the safety net regardless of what RL-3 proposes.
