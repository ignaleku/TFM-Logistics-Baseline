# EVIDENCE LEDGER

Every substantive quantitative claim in the thesis, traced to its source.
Evidence types: **LIT** (literature) · **IMPL** (current code/config) · **EXP** (current
persisted output) · **INTERP** (qualified analysis) · **LIM** (explicit caveat).

Source paths are relative to the repository root at commit `c42c8e4`.
`HIST` = `data/api_runs/latest/historical/rl3_monthly_capacity_cost_results.csv` (576 rows)
`FUT`  = `data/api_runs/latest/future/rl3_monthly_capacity_cost_results.csv` (48 rows)
`REC`  = `data/api_runs/latest/historical/rl3_monthly_recommendations_summary.csv` (12 rows)

> **Note on the "Where used" column.** These section references were assigned while the
> architecture was being locked, before the chapters were written, and are indicative rather
> than exact. The authoritative mapping from claim to location is the compiled document itself;
> every claim below was checked against the text during the final accuracy pass.

| # | Claim | Type | Source | Where used | Confidence |
|---|---|---|---|---|---|
| E1 | Three stages in series: Picking → Packing → Dispatch | IMPL | `sim_multistage.py` | 3, 4 | High |
| E2 | Operating horizon H = 160 h × 60 = 9,600 min | IMPL | `operating_time.py`, `planning_profile.yaml::cost_defaults` | 4 | High |
| E3 | Simulation halts at H; unfinished orders are backlog counted as SLA failures | IMPL | `sim_multistage.py::env.run(until=…)`, `met_sla` on NaN | 4, 5 | High |
| E4 | FIFO = one `simpy.Store` per stage; order is stage-entry order | IMPL | `sim_multistage.py:110-118` | 4 | High |
| E5 | Urgent-First enforces strict two-class priority | IMPL+EXP | `sim_multistage.py`; `verify_priority_store.py` TEST 2 | 4 | High |
| E6 | Urgent-First within-class order is heap-structural, **not** FIFO | IMPL+EXP | SimPy `PriorityItem.__lt__`; `verify_priority_store.py` TEST 1 (1,3,7,10,9,6,8,5,2,4) | 4, 6, LIM | High |
| E7 | `policies.py` is dead code, not the policy implementation | IMPL | grep: no importer | 4 (footnote) | High |
| E8 | RL-3 keeps two FIFO sub-queues per stage → FIFO within class | IMPL | `env_fullstage_rl.py:302-307` | 4 | High |
| E9 | RL-3 state = 16 features, exact ordering | IMPL | `env_fullstage_rl.py::_state` | 4, App C | High |
| E10 | RL-3 actions = 2 (0 urgent, 1 normal) | IMPL | `env_fullstage_rl.py::_decide` | 4, App C | High |
| E11 | RL-3 decides only when both queues non-empty; forced actions untrained | IMPL | `env_fullstage_rl.py::_decide` | 4 | High |
| E12 | Reward deferred to dispatch; all of an order's transitions share it; backlog → −p | IMPL | `env_fullstage_rl.py::_reward`, `:516-528` | 4, App C | High |
| E13 | RL-3 observes no future information | IMPL | 16 features enumerated; none forward-looking | 4, 6 | High |
| E14 | Training: 200 episodes, lr 1e-3, γ 0.99, batch 256, buffer 2e5, ε 1.0→0.05 | IMPL | `configs/rl3.yaml` | 4, App C | High |
| E15 | CRN: service times precomputed per (order_id, stage), shared by all policies | IMPL | `service_time_map.py` | 4 | High |
| E16 | Service time = max(min, (base + per_unit·units)·clip(N(1,0.12),0.8,1.25)) | IMPL | `service_times_multistage.py`, `sim_multistage.yaml` | 4 | High |
| E17 | Stage-differentiated workload multipliers (fragile 1.1 pick vs 1.8 pack) | IMPL | `planning_profile.yaml::workload_multipliers` | 3, 4, 5.1 | High |
| E18 | Dispatch has an urgency multiplier (urgent 1.3) — picking/packing do not | IMPL | `planning_profile.yaml::workload_multipliers.dispatch` | 4 | High |
| E19 | SLA: urgent 240 min, normal 1,440 min, on the operating clock | IMPL | `planning_profile.yaml::sla` | 4 | High |
| E20 | Feasible iff urgent ≥ 0.95 **and** normal ≥ 0.80 | IMPL | `sla_feasibility.py` | 4, 5 | High |
| E21 | Analytical estimate = ceil(Σ deterministic minutes / (H × 0.85)), min 1 | IMPL | `capacity_estimate.py` | 4, 5.2 | High |
| E22 | 16 dynamic candidates per month in a fixed structured order | IMPL | `candidate_generation.py`; `HIST` 16 regimes/month | 4, 5.2 | High |
| E23 | Stage Pressure = 0.40·util + 0.25·p95_norm + 0.20·late_share + 0.15·queue_norm | IMPL | `bottleneck.py`, `planning_profile.yaml::bottleneck_score` | 4, 5.4 | High |
| E24 | p95 and queue components are normalised **across stages**; utilisation is absolute | IMPL | `bottleneck.py::_normalise` | 4, 5.4, LIM | High |
| E25 | Historical economics: 20 / 5 / 15 €·h⁻¹ | EXP | `historical/run_manifest.json::cost_params` | 4, 5 | High |
| E26 | Future economics: 15 / 10 / 18 €·h⁻¹ — **different from Historical** | EXP | `future/run_manifest.json`, `FUT` columns | 4, 5.5, LIM | High |
| E27 | Data are synthetic: 240,000 orders, `scenario=seasonal_base`, 2026 | EXP | `data/uploads/orders_uploaded.csv` | 4, LIM | High |
| E28 | Monthly counts match `annual_share × 240,000` exactly (Jan 31,200; Dec 40,800) | EXP | ibid. vs `planning_profile.yaml::months` | 4 | High |
| E29 | Historical design: 12 months × 16 candidates × 3 policies = 576 rows, 1 seed/month | EXP | `HIST` | 4, 5 | High |
| E30 | Future design: December only, 16 × 3 = 48 rows, 3 replications, screening→validation | EXP | `FUT`, `future/run_manifest.json`, `future_screening.py` | 4, 5.5 | High |
| E31 | Lowest-total-cost policy split: RL-3 7/12, Urgent-First 5/12, FIFO 0/12 | EXP | `REC::best_total_policy` | 5.3, 6.2 | High |
| E31a | `best_total` is `idxmin(estimated_total_cost)` over **all** rows — no feasibility filter is applied by the code | IMPL | `export_rl3_monthly_recommendations.py:96` | 4, 5.2 | High |
| E31b | The unconstrained cheapest configuration nevertheless turned out to be feasible in **all 12** months — a property of these results, not a rule enforced by the selection | EXP+INTERP | `REC` vs `HIST::feasible` | 5.2, 5.3 | High |
| E32 | Feasibility counts: FIFO 84/192, UF 166/192, RL-3 167/192 | EXP | `HIST::feasible` by policy | 5.3, 6.2 | High |
| E33 | October `s753`: FIFO urgent 0.245 / UF 1.000 / RL-3 0.997; costs 90,460 / 49,635 / 45,400 | EXP | `HIST` month 10 | 5.3, 6.2 | High |
| E34 | October `s11_7_5`: all policies ≥ 0.9995 SLA, cost spread ≈ 30 | EXP | `HIST` month 10 | 5.3, 6.2 | High |
| E35 | Future December `s25_14_7` RL-3 urgent-action share 0.267 / 0.732 / 1.000 by stage | EXP | `FUT::p_urgent_pick/pack/dispatch` | 5.5, 6.2 | High |
| E36 | `min_feasible` = fewest workers among feasible; `best_total` = lowest total cost | IMPL | `export_rl3_monthly_recommendations.py:96,152` | 4, 5.2 | High |
| E37 | RL checkpoint `data/dqn_rl3_final.pt` absent from working tree; sha256 recorded | EXP | `checkpoint_provenance.json`; filesystem check | 4, App E, LIM | High |
| E38 | Aggregate SLA can mask class failure (Oct `s753` FIFO: total 0.81, urgent 0.245) | INTERP | derived from E33 | 4, 6 | High |
| E39 | Policy differences are capacity-regime dependent | INTERP | E33 + E34 | 6.2, 7 | Medium-High |
| E40 | Bounded candidate search ≠ global optimisation | LIM | `candidate_generation.py` | 4, 6, 7 | High |
| E41 | **FIFO has zero feasible configurations in January, February, November and December** — the four campaign months — across all 16 candidate workforces | EXP | `HIST::feasible` grouped by month × policy | 5.3, 6.2 | High |
| E42 | December `s22_11_5` (38 FTE): FIFO urgent 0.197 (infeasible); Urgent-First urgent 0.993 / normal **0.738** (infeasible); RL-3 urgent 0.992 / normal 0.854 (**feasible**) | EXP | `HIST` month 12; `bottleneck_analysis.json::policy_comparison` | 5.3, 6.2 | High |
| E43 | At that configuration RL-3 costs 115,040 vs Urgent-First 132,570 — RL-3 reaches feasibility where Urgent-First cannot, at 13% lower cost | EXP | ibid. | 5.3, 6.2 | High |
| E44 | Picking is the primary bottleneck in **all 12** months (pressure 0.91–1.00) | EXP | `bottleneck_analysis.json::bottleneck_ranking` | 5.4, 6.3 | High |
| E45 | December adaptive search tested `s23_11_5` (+1 picking), then **rejected** it: labour +2,400 vs late-penalty reduction −20 | EXP | `bottleneck_analysis.json::adaptive_search.trail` | 5.4, 6.3 | High |
| E46 | December break-even: one FTE costs 2,400/month and would need to prevent 120 urgent-late (or 480 normal-late, or 456 mixed) orders; the observed average penalty per late order is 5.27 | EXP | `bottleneck_analysis.json::break_even` | 5.4, 6.3 | High |
| E47 | A named bottleneck therefore does **not** imply that hiring is justified — the marginal FTE must be evaluated economically | INTERP | E45 + E46 | 5.4, 6.3, 7 | High |
| E48 | `bottleneck_analysis.json` per month also carries `policy_comparison`, `recommended_policy`, `capacity_level_diagnostics` and a natural-language `explanation` | EXP | ibid. | 4, 5.4, App G | High |
| E49 | **The analytical anchor over-estimates the workforce in all 12 months.** Simulated recommendation is always lower: Jan 31→25, Feb 27→25, Mar/Apr 15→13, May 9→8, Jun/Jul/Aug 8–9→7, Sep 16→14, Oct 19→17, Nov 36→30, **Dec 47→38** | EXP | `analytical_estimate_by_month` vs `REC::best_total_workers`; `fig_capacity_bottleneck.py` | 5.2, 6.1 | High |
| E50 | December's 9-FTE reduction is worth 9 × 2,400 = **21,600/month** under that run's economics — the direct value of simulating rather than sizing by formula | INTERP | E49 + E46 | 5.2, 6.1 | High |
| E51 | The adaptive search ran in **9 of 12** months (not Feb, Jul, Aug), always proposing +1 **picking** FTE, and **rejected it in all 9** | EXP | `bottleneck_analysis.json::adaptive_search.trail` | 5.4, 6.3 | High |
| E52 | Penalty reduction from that extra FTE ranged 20 (Jun, Dec) to 2,025 (Nov) against a constant labour cost of 2,400 — November is near break-even, the rest are not close | EXP | ibid. | 5.4, 6.3 | High |
| E53 | Annual mean workload units per order: picking 3.39, packing 2.46, dispatch 1.17 → ratio **1 : 0.72 : 0.34**, confirming stage-differentiated workload | EXP | `fig_demand.py` over `orders_uploaded.csv` | 5.1, 3 | High |
| E54 | Picking utilisation at the recommended configuration ranges 79.5% (Jul) to 98.8% (May); `late_wait_share` 83.7–100% | EXP | `t_bottleneck_ranking.csv` | 5.4 | High |

## Pending evidence (to be added during Phase E)

| # | Claim to establish | Planned source |
|---|---|---|
| P1 | Analytical anchor vs simulated recommendation divergence, per month | `analytical_estimate_by_month` vs `REC` |
| P2 | Stage Pressure decomposition per month (components, not just the score) | `bottleneck_analysis.json`; `HIST` pressure columns |
| P4 | Equal-workforce policy comparison isolated from workforce effects (§84) | `HIST` grouped by (month, regime) |
| P5 | Future replication spread / `prob_meets_sla_targets` | `FUT` |

*(P3 resolved: the adaptive-search trail and break-even block in `bottleneck_analysis.json`
provide the marginal-FTE evidence directly — see E45–E47. No reconstruction from neighbouring
grid candidates is needed.)*
