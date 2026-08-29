# FIGURE MANIFEST

Status: PLANNED · GENERATED · PLACED (rendered in the compiled PDF) · DROPPED.

Every quantitative figure is reproducible from the persisted run outputs by the named script
under `thesis_support/analysis/`. Conceptual diagrams are TikZ source inside the chapter files
and carry no external dependency.

`HIST` = `data/api_runs/latest/historical/rl3_monthly_capacity_cost_results.csv`
`FUT`  = `data/api_runs/latest/future/rl3_monthly_capacity_cost_results.csv`
`BN`   = `data/api_runs/latest/{mode}/bottleneck_analysis.json`

## Quantitative figures (matplotlib → PDF in `thesis/figures/`)

| ID | Fig. | RQ | Title | Source | Script | Main message | Status |
|---|---|---|---|---|---|---|---|
| F1  | 4.1 | RQ1 | Monthly order volume and urgency mix | orders CSV | `fig_demand.py` | Volume and urgent share peak together | PLACED |
| F2  | 4.2 | RQ1 | Mean workload units per order by stage | orders CSV | `fig_demand.py` | Stage workload is not 1:1:1 and shifts seasonally | PLACED |
| F3  | 5.1 | RQ1 | Analytical anchor vs simulated recommendation | summary JSON + REC | `fig_capacity_bottleneck.py` | The estimate over-sizes in all 12 months | PLACED |
| F4  | 5.2 | RQ2 | Feasible configurations by month and policy | HIST | `fig_policy.py` | FIFO feasible nowhere in the four campaign months | PLACED |
| F5  | 5.3 | RQ2 | Urgent vs normal SLA trade-off, all 576 runs | HIST | `fig_policy.py` | FIFO sacrifices the urgent class; the other two differ on normal | PLACED |
| F6  | 5.5 | RQ2 | December cost against total SLA | HIST | `fig_policy.py` | FIFO occupies a separate expensive low-service region | PLACED |
| F7  | 5.4 | RQ2 | Representative regimes: Oct `s753` vs `s11_7_5` | HIST | `fig_policy.py` | Policy matters under pressure, not under slack | PLACED |
| F8  | 5.7 | RQ3 | Stage Pressure by stage and month, decomposed | BN | `fig_capacity_bottleneck.py` | Picking dominates; components show why | PLACED |
| F9  | 5.8 | RQ3 | Marginal FTE: cost against benefit | BN | `fig_capacity_bottleneck.py` | Rejected in all nine months tested | PLACED |
| F11 | 5.9 | RQ4 | December forecast: cost vs feasibility reliability | FUT | `fig_capacity_bottleneck.py` | Forecast mode reports reliability, not a point estimate | PLACED |
| F12 | 5.6 | RQ2 | Equal-workforce comparison over all 192 matched configurations | HIST | `fig_equal_workforce.py` | The policy gap scales with pressure; RL-3's advantage holds only in the feasible region | PLACED |

## Conceptual diagrams (TikZ, inline in the chapter source)

| ID | Fig. | Title | Located in | Main message | Status |
|---|---|---|---|---|---|
| D1 | 3.1 | Structure of the modelled fulfilment process | `03_problem_framework.tex` | Three stages, own queues, decision at each | PLACED |
| D2 | 3.2 | Framework architecture | `03_problem_framework.tex` | Two entry points, one shared pipeline | PLACED |
| D3 | 4.3 | Operating-time compression | `04_methodology.tex` | Calendar month → 9,600 operating minutes | PLACED |
| D5 | 4.4 | RL-3 decision structure | `04_methodology.tex` | Shared network, class choice, deferred reward | PLACED |
| D6 | 4.5 | Adaptive capacity search | `04_methodology.tex` | Propose → simulate → accept/reject | PLACED |
| D4 | 4.6 | Common random numbers design | `04_methodology.tex` | Same service times across all policies | PLACED |

`F10` (Historical vs Future workflow diagram) was **DROPPED**: its content is fully covered by
D2, which already shows the two entry points converging into the shared pipeline. A second
diagram would have repeated it.

## Tables generated from data

| File | Used as | Script |
|---|---|---|
| `t_demand_profile.csv` | Table 5.1 | `fig_demand.py` |
| `t_policy_feasibility.csv` | Tables 5.4, F.1 | `fig_policy.py` |
| `t_representative_cases.csv` | Table 5.5 | `fig_policy.py` |
| `t_monthly_recommendations.csv` | Table 5.2 | `fig_capacity_bottleneck.py` |
| `t_bottleneck_ranking.csv` | Table 5.9 | `fig_capacity_bottleneck.py` |
| `t_equal_workforce_pairs.csv` | Table 5.6 | `fig_equal_workforce.py` |
| `t_pressure_bands.csv` | Table 5.8 | `fig_equal_workforce.py` |
| `t_adaptive_search.csv` | Table 5.10 | `fig_capacity_bottleneck.py` |
| Appendix D longtables (576 + 48 rows) | Tables D.1, D.2 | `gen_appendix_d.py` |

## Interface screenshots

| ID | Title | Status |
|---|---|---|
| S1 | Recommendation view | **NOT CAPTURED** |
| S2 | Capacity & Bottlenecks view | **NOT CAPTURED** |

The architecture planned up to four selective frontend screenshots as an Appendix G. These were
not captured: doing so requires running the web application, and the reinforcement learning
checkpoint needed to produce a live recommendation is absent from the working tree (see
`THESIS_STATE.md` §6.12). The thesis therefore describes the framework's outputs from the
persisted run artefacts rather than from the interface. This is a deliberate scope reduction and
is recorded in the completion report rather than silently dropped.
