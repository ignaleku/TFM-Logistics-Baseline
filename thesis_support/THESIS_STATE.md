# THESIS STATE

Living control document for the IMIM Master's Thesis production.
Source commit at thesis start: `c42c8e4` (branch `thesis-branch`).
Last updated: Phase A/C complete (system reconstruction + architecture lock).

---

## 1. Title (locked)

**Simulation-Based Decision Support for Warehouse Workforce Planning and Order Sequencing
under Differentiated Service-Level Constraints**

RL is deliberately absent from the title: RL-3 is one of three compared sequencing policies,
not the object of the thesis.

## 2. Author / metadata

| Field | Value |
|---|---|
| Author | Ignacio González |
| Programme | International Master in Industrial Management (IMIM), edition 2025–27 |
| Institutions | UPM / Heriot-Watt University |
| Tutor | `TODO_TUTOR_FULL_NAME` |
| Submission | `TODO_SUBMISSION_MONTH_YEAR` |
| Modality | Academic (topic *motivated* by an internship, developed independently) |

## 3. Research design (locked)

**Applied simulation-based experimental research.**
Not Design Science Research — there were no formal design cycles or stakeholder evaluation
rounds that would justify that framing (§43 of the master prompt).

## 4. Research questions (locked)

| RQ | Question |
|---|---|
| **RQ1** | How can discrete-event simulation support monthly warehouse workforce-capacity planning under heterogeneous demand, and what does it add over an analytical capacity estimate alone? |
| **RQ2** | How do alternative order-sequencing policies (FIFO, Urgent-First, RL-3 DQN) affect differentiated service-level attainment and operating cost, at equal workforce and at each policy's own feasible workforce? |
| **RQ3** | How can stage-level bottleneck diagnostics and adaptive capacity evaluation improve the explainability and economic interpretation of workforce recommendations? |
| **RQ4** | How can one simulation framework support both retrospective (counterfactual) and forecast-based capacity planning, and how do their uncertainty structures differ? |

## 5. Objectives

**General objective.** To design, implement and evaluate a discrete-event simulation-based
decision-support framework that recommends monthly warehouse workforce capacity and an order
sequencing policy under differentiated service-level and economic constraints.

**Specific objectives.**
1. Formalise a three-stage (Picking → Packing → Dispatch) monthly capacity-planning problem
   with heterogeneous, stage-differentiated order workload.
2. Implement an analytical capacity anchor and a bounded dynamic candidate search evaluated by
   simulation under common random numbers.
3. Compare three sequencing policies under class-specific SLA feasibility and an explicit
   economic model.
4. Provide stage-level bottleneck diagnostics and an adaptive capacity procedure that tests
   whether additional capacity is economically justified.
5. Demonstrate the framework in both retrospective and forecast-based planning modes.

---

## 6. LOCKED IMPLEMENTATION FACTS

Every fact below was verified against the current code/outputs at commit `c42c8e4`.
**These are the facts the thesis must state. Do not restate from convention or memory.**

### 6.1 Scope and flow
- Three active stages: **Picking → Packing → Dispatch**, in series, each with its own queue
  and its own worker pool. No batching, no waves, no cross-stage worker sharing, no blocking
  (queues are unbounded).
- 5-stage / RL-5 code exists under `src/**/legacy/` and is **not** on the active execution
  path. Historical artefact only.

### 6.2 The three sequencing policies — AS IMPLEMENTED
Both baselines live inside `src/simulation/multistage/sim_multistage.py`.
`src/simulation/multistage/policies.py` is **dead code** — nothing imports it (verified by
grep). Do not describe it as the policy implementation.

| Policy | Data structure | Within-class ordering |
|---|---|---|
| `fifo` | one `simpy.Store` per stage | FIFO **on stage-entry order**, i.e. arrival order at Picking, upstream-completion order at Packing/Dispatch |
| `urgent_first` | one `simpy.PriorityStore` per stage, `PriorityItem(0=urgent, 1=normal)` | **Unspecified / binary-heap structural — NOT FIFO** |
| `rl3_dqn` | two `simpy.Store` per stage (urgent, normal), agent selects the class | FIFO **within each class** |

**Critical verified fact (RQ2 / §13 / §82).** SimPy's `PriorityItem.__lt__` compares *only*
`priority`, so equal-priority items are ordered by the binary heap's internal structure.
Empirically demonstrated: inserting equal-priority ids 1…10 dequeues as
`1, 3, 7, 10, 9, 6, 8, 5, 2, 4`. Therefore **Urgent-First is a strict two-class priority rule
with no guaranteed within-class ordering** — it is *not* "urgent first, FIFO within class".
This is a genuine, documentable property of the implemented baseline, and it plausibly
contributes to Urgent-First's much larger p95 waiting times for normal orders. Document it;
do not correct it. Reproduced by `thesis_support/analysis/verify_priority_store.py`.

**Asymmetry between engines is a design fact, not a defect (§54).** RL-3 keeps two FIFO
sub-queues per stage; the Urgent-First baseline keeps one heap. They therefore differ in
within-class mechanics. Policy comparison concerns the implemented decision rules.

### 6.3 Operating time (§20–24)
- `H = hours_per_worker_month × 60 = 160 × 60 = **9,600 operating minutes/month**`
  (`operating_time.py::operating_horizon_minutes`).
- One worker unit = **one monthly Full-Time Equivalent**, i.e. 160 productive hours.
  **Never** "N employees on one shift simultaneously." Shift rostering is out of scope.
- `compress_to_operating_time` maps each order's position *within its own calendar month*
  linearly onto `[0, H)`. Relative ordering and intra-month burstiness are preserved; literal
  calendar clock time is not. `SIM_EPOCH = 2000-01-01`, arbitrary and data-independent.
- **Finite horizon.** `env.run(until=horizon_seconds)`. The simulation does **not** run until
  all orders finish. Orders unfinished at H are **backlog**: `system_time` is `NaN`, so
  `met_sla` is `False`, and they are additionally reported via `unfinished_orders` /
  `backlog_share`. This prevents insufficient capacity being hidden by unlimited
  post-period processing.
- SLA is evaluated on this **operating-time clock**. Urgent 240 min / normal 1,440 min are
  productive elapsed minutes, **not** literal wall-clock customer delivery times.

### 6.4 Service times and order heterogeneity
`minutes = max(min_minutes, (base + per_unit × units) × noise)`,
`noise ~ clip(N(1.0, 0.12), 0.80, 1.25)`.

| Stage | base | per unit |
|---|---|---|
| Picking | 0.5 | 0.90 |
| Packing | 0.5 | 0.70 |
| Dispatch | 0.5 | 0.65 |

`picking_units / packing_units / dispatch_units` are precomputed per order from
`num_items × family multiplier × complexity multiplier` (and, for dispatch only, an
**urgency** multiplier of 1.3 for urgent orders). Because the multipliers differ by stage
(e.g. `fragile` costs 1.1× at picking but **1.8×** at packing), **one order imposes different
workload intensity at different stages** — this is what defeats a naive 1:1:1 stage model and
is a core modelling argument of the thesis.

### 6.5 Common random numbers (§40)
`service_time_map.py::build_service_time_map` precomputes service minutes for every
`(order_id, stage)` once per scenario seed, iterating orders **sorted by `order_id`** and
stages in fixed order. All three policies look up the same map, so identical orders receive
identical service times regardless of dequeue order. Without it, "same seed" would not mean
"same service times", because reordering policies advance the RNG stream differently.

### 6.6 SLA targets and feasibility (§27)
- Thresholds: urgent 240 min, normal 1,440 min.
- Targets: `urgent_target = 0.95`, `normal_target = 0.80`.
- Feasible **iff both** floors are met (`sla_feasibility.py::check_feasibility`).
  `sla_violation` = sum of the two shortfalls. Class-specific by design: an aggregate SLA rate
  would hide catastrophic urgent-class failure (see October `s753`, FIFO: total SLA 0.81 but
  urgent SLA **0.245**).

### 6.7 Economic model (§28, §85)
`Total Cost = labour + urgent-late + normal-late`, where
`labour = total_workers × worker_cost_per_hour × hours_per_worker_month`.

| Run | urgent late | normal late | worker €/h | hours/month |
|---|---|---|---|---|
| Historical (`latest/historical`) | **20.0** | **5.0** | **15.0** | 160 |
| Future (`latest/future`) | **15.0** | **10.0** | **18.0** | 160 |

**HARD RULE:** the two runs use different economic assumptions. **No absolute monetary
comparison across modes.** Within a mode, comparisons are valid under that run's common
parameter set. Report each run *as run*.

### 6.8 Analytical capacity estimate (§25)
`workers_stage = max(1, ceil( Σ deterministic stage minutes / (H × target_utilisation) ))`
with `target_utilisation = 0.85` and noise fixed at 1.0. It is a **screening anchor**, never
the final recommendation.

### 6.9 Dynamic candidate generation (§26)
`candidate_generation.py` builds `candidate_count = 16` distinct candidates around the
analytical centre, in this fixed order: centre → client's current workforce (if given) →
single-stage ±1 → ±15% total-workforce variants → balanced two-stage combinations →
single-stage ±2. Every stage clamped to ≥1. **A bounded structured neighbourhood — never a
full grid, never global optimisation.**
Labels: `sPKD` when all stages are single-digit (`s432`), else `sP_K_D` (`s22_11_5`).

### 6.10 Stage Pressure / bottleneck score (§29, §86)
`pressure = 0.40·utilisation + 0.25·p95_wait_norm + 0.20·late_wait_share + 0.15·avg_queue_norm`
- `utilisation` is an absolute ratio clipped to [0,1] — **not** cross-stage normalised.
- `p95_wait_norm` and `avg_queue_norm` are normalised **by the maximum across the three
  stages** (`_normalise`), so they are *relative* signals; the top stage always scores 1.0 on
  these two components.
- `late_wait_share` is already a share of late orders' waiting time across stages (sums to 1).
- Heuristic, transparent, diagnostic. **Not** causal, **not** statistically calibrated,
  **not** expert-elicited. Say "identified as the primary pressure location", never
  "causes every late order".

### 6.11 RL-3 specification (§33–39) — `src/rl/env_fullstage_rl.py`
- Single **DQN**, shared across all three stages; the stage is supplied to the network as a
  state feature (`stage_id ∈ {0.0, 0.5, 1.0}`).
- **State: 16 features, in this exact order** —
  `[0] pick_urgent_q/200, [1] pick_normal_q/500, [2] pack_urgent_q/200, [3] pack_normal_q/500,`
  `[4] disp_urgent_q/200, [5] disp_normal_q/500, [6] wip_pick/5, [7] wip_pack/5,`
  `[8] wip_disp/5, [9] time_norm, [10] slack_urgent_head, [11] slack_normal_head,`
  `[12] stage_id, [13] cap_pick/20, [14] cap_pack/20, [15] cap_disp/20]`
  all clipped to [0,1]. Slack is `(clip((sla − elapsed)/sla, −1, 1) + 1)/2`, defaulting to
  0.5 on an empty queue. **Do not use the obsolete 13-feature description.**
- **Actions: 2.** `0 = serve the urgent queue`, `1 = serve the normal queue`.
- **A decision is only taken when both queues at that stage are non-empty.** If exactly one
  queue is occupied the action is forced and *not* recorded as a training transition.
- **Reward** (`rl1_current`), deferred to dispatch completion and applied to **every**
  transition of that order: on time → `w` (`urgent 5.0`, `normal 3.0`); late →
  `−p × min(1, lateness / (sla × 2.0))` with `p = 2.0` for both classes. Orders left as
  backlog at H receive `−p` (treated as maximally late).
- **RL-3 does not observe the future.** It sees only the current state; nothing in the 16
  features encodes future arrivals.
- Training: 200 episodes, each sampling one (month, regime) pair from
  `data/rl3_train_pool.json`; `lr 1e-3`, `gamma 0.99`, batch 256, hidden 64, buffer 200,000,
  target update every 2,000 steps, ε 1.0 → 0.05 over 200,000 steps. Training deliberately
  omits the CRN map so each episode has independent service times.
- **Training pool — CORRECTED (final pass).** Training samples uniformly from
  `build_training_pool` (`main_train_rl3.py`), persisted at `data/rl3_train_pool.json`:
  3 representative months (June/October/December) x 9 generated candidates each, last 2 held
  out -> **21 training pairs, 6 holdout pairs**. Consequences: 9 of 12 months are untrained by
  construction; `s22_11_5` (Dec) and `s753` (Oct) are **holdouts** (`rl3_audit_report.json`
  records `seen_during_training: false`).
- `planning_profile.yaml::rl_generalisation` (12 `train_regimes` / 4 `holdout_regimes`
  `s112, s231, s322, s432`) is a **stale, month-agnostic list** used only by
  `evaluate_rl3_generalisation.py` — see the comment at `rl_audit.py:271-276`. It is **not**
  what the trained agent was trained on. Do not describe it as the training design.

### 6.12 Checkpoint status — REPRODUCIBILITY LIMITATION
`data/dqn_rl3_final.pt` is **absent from the working tree** (matched by the `.gitignore`
pattern `data/*.pt`, so it was never tracked). `data/diagnostics/rl3_historical/
checkpoint_provenance.json` records its sha256 `2002aaab…0f2eed`, 24,669 bytes, and confirms
it loaded cleanly into the current 16→64→2 architecture.
**Consequence:** RL-3 results cannot be regenerated in this environment. All RL evidence in
the thesis comes from the persisted run outputs. This must be stated as an explicit
limitation, not glossed over. (`torch` is also not installed; `pandas/numpy/simpy/matplotlib`
are, under Python 3.12.)

### 6.13 Data provenance (§17) — VERIFIED
- `data/uploads/orders_uploaded.csv`: **240,000 orders**, `scenario = seasonal_base`,
  spanning 2026-01-01 → 2026-12-31, overall urgent share 0.1717.
- Monthly counts match `planning_profile.yaml::months.annual_share × 240,000` **exactly**
  (Jan 0.130 → 31,200; Dec 0.170 → 40,800). This is the output of
  `src/data/generate_orders_seasonal.py`.
- **Therefore: the Historical Analysis experiments use structured SYNTHETIC data in
  historical order-level format.** Distinguish clearly:
  *system capability* = real order-level history may be uploaded;
  *experimental provenance* = synthetic.
  Never claim real customer, client or Baobab operational data.

### 6.14 Experimental design as executed
| | Historical | Future |
|---|---|---|
| Scope | **12 months** | **December only** |
| Rows | 576 = 12 × 16 candidates × 3 policies | 48 = 16 × 3 |
| Replications | 1 scenario seed per month | 3 (`screening` on rep 1 → top 4 finalists `validated` on all 3) |
| Uncertainty | none across seeds; single realisation per month | demand_cv 0.10, arrival_cv 0.30 (`standard`) |
| Extra outputs | — | `p90_total_cost`, `prob_meets_sla_targets` |

RQ4 must be framed honestly: **Future is a peak-month planning demonstration, not an annual
counterpart to Historical.** The two modes have genuinely different uncertainty structures
(§42) and different economics (§6.7).

---

## 7. Headline results (verified, `latest/historical`, 576 rows)

- **Recommended configuration by month:** RL-3 in **7/12** months
  (Jan, Feb, May, Jun, Jul, Nov, Dec), Urgent-First in **5/12** (Mar, Apr, Aug, Sep, Oct),
  **FIFO in 0/12**. **CORRECTED (final pass):** the framework's recommendation rule is
  `bottleneck_report.py::_select_recommendation`, which is **lexicographic** — validated-rows-only
  (forecast mode) → cheapest among rows meeting BOTH floors → fallback to min `sla_violation`
  then min cost, reported infeasible. The unconstrained `idxmin(total_cost)` is a *descriptive*
  export column (`best_total_*` in `export_rl3_monthly_recommendations.py`), not the selector.
  Verified row-by-row: the two coincide in all 12 months of this run; the fallback was never
  reached. Do not restate the old "unconstrained idxmin" description.
- **Feasibility rate:** FIFO **84/192**, Urgent-First **166/192**, RL-3 **167/192**.
- **★ FIFO is infeasible everywhere in the peak season.** In January, February, November and
  December — the four campaign months — FIFO meets the class-specific targets at **none** of
  the 16 candidate workforces, while Urgent-First and RL-3 do at almost all of them. Sequencing
  is not a marginal refinement in peak season; it decides whether the plan is viable at all.
- **★ December `s22_11_5` (38 FTE) — the single strongest configuration:**
  FIFO urgent SLA **0.197**; Urgent-First urgent 0.993 but normal **0.738** → *infeasible*;
  RL-3 urgent 0.992 and normal 0.854 → **feasible**, at **115,040** vs Urgent-First's 132,570.
  RL-3 attains feasibility at a workforce where Urgent-First cannot, 13% cheaper.
- **★ Picking is the primary bottleneck in all 12 months** (pressure score 0.91–1.00),
  with `late_wait_share` 0.84–1.00 — waiting is overwhelmingly concentrated at one stage.
- **★ The December adaptive search rejects hiring.** It tested `s23_11_5` (+1 picking FTE at
  the identified bottleneck) and rejected it: labour cost **+2,400** against a late-penalty
  reduction of **−20**. Break-even would require preventing 120 urgent-late orders; the
  observed average penalty per late order is **5.27**. A bottleneck is a diagnosis, not a
  hiring instruction.
- **Representative under-capacity case — October `s753` (15 FTE):**
  FIFO urgent SLA **0.245** (infeasible, cost 90,460);
  Urgent-First urgent 1.000 / normal 0.842 (feasible, 49,635);
  RL-3 urgent 0.997 / normal **0.894** (feasible, **45,400**).
  Sequencing alone moves feasibility and cost dramatically at fixed workforce.
- **Well-resourced months compress policy differences** (e.g. October `s11_7_5`: all three
  policies ≥ 0.9995 total SLA, costs within 30 of each other) — a genuine finding: policy
  choice matters most under capacity pressure.
- **Stage-differentiated RL behaviour** (Future December `s25_14_7`): urgent-action share
  **0.267 at picking, 0.732 at packing, 1.000 at dispatch** — the shared agent learned
  markedly different behaviour per stage. Strong RQ2/RQ3 material.

## 8. Claim discipline — never write these
Global optimum · RL always wins · real client data · industrial validation · FTE = simultaneous
headcount · 160 h = warehouse opening hours · shifts are modelled · Future predicts individual
orders · RL sees the future · Stage Pressure proves causality · pressure weights were
calibrated · operating-time SLA = customer wall-clock SLA · cross-mode cost comparison.

## 9. Chapter progress — COMPLETE

| Chapter | Words | Manual minimum |
|---|---|---|
| Front matter (incl. Extended Abstract, 1,164 w) | 1,659 | — |
| 1 Introduction | 2,078 | — |
| 2 Theoretical foundations | 4,145 | 3,000 met |
| 3 Problem & framework | 1,926 | — |
| 4 Methodology | 5,458 | 3,000 met |
| 5 Results | 5,237 | 4,000 met |
| 6 Discussion | 4,723 | 3,000 met |
| 7 Conclusions | 2,012 | 1,000 met |
| Appendices A–F | 2,529 | — |
| **Total** | **~29,800** | |

Compiled PDF: **105 pages** (61 main body). 0 errors, 0 LaTeX warnings, 0 undefined references
or citations, 0 missing characters, 2 overfull boxes of 2–3 pt. 35 verified references, all
cited. 17 figures, 33 tables.

## 10. Remaining actions
1. Replace `thesis/frontmatter/titlepage.tex` with the official IMIM cover page (author holds
   it). Two placeholders remain inside that file only.
2. Optional: frontend screenshots as an Appendix G — not captured, see COMPLETION_REPORT §8.
