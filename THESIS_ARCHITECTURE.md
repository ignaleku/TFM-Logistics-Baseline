# THESIS ARCHITECTURE

Locked plan for the IMIM Master's Thesis. Companion to `thesis_support/THESIS_STATE.md`,
which holds the verified implementation facts. This document holds the *argument*.

Source commit: `c42c8e4` · branch `thesis-branch`.

---

## 1. Title

**Simulation-Based Decision Support for Warehouse Workforce Planning and Order Sequencing
under Differentiated Service-Level Constraints**

## 2. Research problem

Warehouse fulfilment operations must decide, month by month, how much labour capacity to
allocate to each processing stage, and in which order to release the orders that queue at
those stages. The two decisions are usually taken separately and with different instruments:
capacity from aggregate workload ratios, sequencing from a fixed operating rule. Neither
instrument on its own reveals what a manager actually needs to know — whether a proposed
workforce will meet *class-specific* service commitments once queueing, arrival burstiness and
heterogeneous order workload are accounted for, where the resulting pressure concentrates, and
whether buying one more full-time equivalent is economically worthwhile.

The problem this thesis addresses is therefore: **how to support monthly warehouse
workforce-capacity planning and order sequencing jointly, under heterogeneous demand and
differentiated service-level requirements, in a way that is quantitatively grounded and
explainable to the manager who has to act on it.**

## 3. Objectives

See `THESIS_STATE.md` §5 (general + five specific objectives).

## 4. Research questions and how each is answered

| RQ | Method | Evidence (current outputs) | Figure / Table | Expected result | Chapter |
|---|---|---|---|---|---|
| **RQ1** — DES for monthly capacity planning; what it adds over an analytical estimate | Analytical anchor (`capacity_estimate.py`) vs 16 simulated candidates per month | `historical_analysis_summary.json::analytical_estimate_by_month` vs `rl3_monthly_capacity_cost_results.csv` (576 rows) | F1 demand profile · F2 stage workload · F3 analytical centre vs simulated recommendation · T-rec monthly recommendations | The anchor is a good centre but not the answer; the simulated recommendation departs from it, and feasibility is decided by queueing/SLA effects the formula cannot see | 5.1, 6.1 |
| **RQ2** — sequencing policies vs differentiated SLA and cost | 3 policies × 16 candidates × 12 months under CRN; both *equal-workforce* and *own-feasible-workforce* comparisons (§84) | 576-row grid; feasibility 84/166/167; cheapest-feasible 0/5/7 (FIFO/UF/RL-3) | F4 SLA by policy & month · F5 urgent-vs-normal trade-off · F6 cost vs SLA frontier · F7 October `s753` case | FIFO collapses on urgent class under pressure; UF protects urgent by sacrificing normal; RL-3 attains a better joint balance in 7/12 months — but differences vanish when capacity is ample | 5.2, 6.2 |
| **RQ3** — bottleneck diagnostics and adaptive capacity for explainability | Stage Pressure decomposition; adaptive search accept/reject; +1 FTE economics | per-stage pressure columns in both CSVs; `bottleneck_analysis.json`; `adaptive` rows | F8 pressure decomposition by stage · F9 +1 FTE cost-vs-benefit · T-bottleneck | Picking dominates pressure in nearly all months; a named bottleneck does **not** imply hiring — the marginal FTE must pay for itself | 5.3, 6.3 |
| **RQ4** — one framework, retrospective and forecast-based | Compare the two workflows' inputs, uncertainty structures and outputs | `latest/historical` (12 months, 1 seed) vs `latest/future` (December, 3 replications, screening+validation) | F10 workflow diagram · F11 replication spread / `prob_meets_sla_targets` · T-modes | Same engine and same decision logic serve both; they differ in uncertainty structure and economics, so their outputs answer different managerial questions and must not be pooled | 5.4, 6.4 |

## 5. Contribution statement

This thesis contributes an **integrated, explainable, simulation-based decision-support
framework** for monthly warehouse capacity and sequencing planning. Specifically:

*Methodological.* (i) An operating-time abstraction that forces physical simulated capacity and
economically paid capacity to be the same quantity, closing a gap that otherwise grants
workers unpaid processing time; (ii) a finite-horizon monthly design in which unfinished work
is retained as explicit backlog rather than absorbed by unlimited post-period processing;
(iii) a class-specific SLA feasibility criterion combined with an explicit economic objective;
(iv) a CRN-supported three-way policy comparison in which a learned policy is evaluated on
exactly the same stochastic realisations as two transparent heuristics.

*Practical.* A workflow that converts either an order-level history or an aggregate forecast
into a defensible monthly recommendation — workforce per stage, sequencing policy, expected
cost, class-level service attainment, and a named pressure location with its decomposition.

*Empirical.* Evidence that sequencing policy choice is **capacity-regime dependent**: it is
close to irrelevant when capacity is ample and decisive when capacity is tight, which
reframes "which policy is best?" as "under which regime does policy choice pay?".

Explicitly **not** claimed: a new RL algorithm, a global optimum, or industrial validation.

## 6. Research design

**Applied simulation-based experimental research.** Justification: the study builds a
computational model of a system, defines controlled experimental factors (workforce regime,
sequencing policy, month/scenario), holds stochastic realisations constant across treatments
via common random numbers, and measures response variables (class SLA, cost, utilisation,
waiting). This is experimentation on a simulated system, not design-science artefact
construction with evaluation cycles.

## 7. Chapter structure and page budget

| # | Chapter | Pages | Core content |
|---|---|---|---|
| — | Front matter | 8 | Cover, acknowledgements, conflict of interest, extended abstract (~1,000 w), keywords, UNESCO codes, TOC, LoF, LoT, abbreviations |
| 1 | Introduction | 8 | Background, motivation (incl. Baobab framing), problem, objectives, RQs, scope, contributions, structure |
| 2 | Theoretical foundations & literature review | 16 | Warehouse fulfilment & order picking; workforce/capacity planning; queueing & congestion; DES in warehousing; sequencing & priority rules; differentiated service; RL/DQN for operational control; simulation experimentation & CRN; decision support & explainability; **research gap** |
| 3 | Problem formulation & decision-support framework | 9 | Formal problem, notation, system architecture, the two workflows, design principles |
| 4 | Methodology | 20 | Data & heterogeneity; operating-time model; DES engine; the three policies **as implemented**; RL-3 spec; capacity estimate & candidate generation; feasibility & economics; Stage Pressure; adaptive search; CRN & replication; verification & validation; limitations of method |
| 5 | Results | 19 | 5.1 demand & workload · 5.2 capacity (RQ1) · 5.3 policies (RQ2) · 5.4 bottlenecks & adaptive capacity (RQ3) · 5.5 future planning (RQ4) |
| 6 | Discussion | 11 | Interpretation per RQ, comparison with literature, managerial implications, external validity, limitations |
| 7 | Conclusions, contributions, implications | 5 | RQ answers, contributions, implications, future work |
| — | References | 4 | 50–70 verified entries |
| — | Appendices A–G | +25 | Data schema · simulation detail · RL spec · full results · verification & reproducibility · additional analyses · interface |

Main body target ≈ **80 pages** (96 incl. front matter and references), appendices additional.

## 8. Figure plan

Full detail in `thesis_support/FIGURE_MANIFEST.md`. Categories:
- **Conceptual diagrams (TikZ/SVG, original):** warehouse flow; decision-support architecture;
  operating-time compression; Historical vs Future workflows; CRN design; RL-3 decision
  structure; adaptive capacity loop.
- **Quantitative figures (matplotlib, from persisted outputs):** monthly demand & urgency mix;
  stage workload units; analytical vs simulated capacity; SLA by policy/month; urgent–normal
  trade-off; cost-vs-SLA frontier; October `s753` representative case; Stage Pressure
  decomposition; RL decision share by stage; marginal-FTE economics; Future replication spread.
- **Interface screenshots (selective, ≤4):** Recommendation, Capacity & Bottlenecks, Policy
  Comparison, Future Planning.

## 9. Table plan

RQ→evidence matrix · order schema · workload multipliers · simulation parameters · policy
definitions · RL state features (16 rows) · RL hyperparameters · SLA & economic parameters ·
monthly recommendations (12 rows) · representative cases · bottleneck ranking · feasibility
counts by policy · Future screening/validation · limitations register.

## 10. Strongest existing evidence

1. **Policy feasibility asymmetry** — FIFO feasible in 84/192 configurations vs 166 (UF) and
   167 (RL-3). Large, unambiguous, directly answers RQ2.
2. **October `s753` under-capacity case** — FIFO urgent SLA 0.245 vs UF 1.000 vs RL-3 0.997,
   with RL-3 simultaneously best on normal class (0.894) *and* cheapest (45,400). A single
   configuration that demonstrates the entire argument.
3. **Regime dependence** — at October `s11_7_5` all three policies exceed 0.9995 SLA within a
   cost spread of 30. Prevents overclaiming and is itself a managerial finding.
4. **Cheapest-feasible split 7/5/0** across months — honest, non-uniform, defensible.
5. **Stage-differentiated RL behaviour** — urgent-action share 0.27 / 0.73 / 1.00 across
   picking / packing / dispatch in Future December.
6. **Analytical anchor vs simulated recommendation divergence** — motivates DES over formulas.

## 11. Limitations to state explicitly

Synthetic data provenance · no external industrial calibration · operating-time abstraction is
not wall-clock · monthly FTE, not shift rostering · SLA measured on the operating clock ·
minimal warehouse physics (no travel, layout, congestion, batching, blocking) · single seed per
month in Historical · Future covers one month · **RL checkpoint absent → RL results not
regenerable here** · Urgent-First's within-class ordering is unspecified by construction ·
different within-class mechanics between baseline and RL engines · heuristic, uncalibrated
Stage Pressure weights · cross-mode economics differ · bounded candidate search, not global
optimisation.

## 12. Claim risks and mitigations

| Risk | Mitigation |
|---|---|
| Reading "RL wins" into a 7/5 split | Report the split with its regime dependence; state where RL does *not* help |
| Treating Urgent-First as FIFO-within-class | Documented and empirically demonstrated (`verify_priority_store.py`) |
| Comparing Historical and Future costs | Rule recorded in `THESIS_STATE.md` §6.7; never compare absolute money across modes |
| Implying real client/Baobab data | Provenance verified and stated as synthetic throughout |
| Implying 22 workers on one shift | FTE semantics defined early and repeated in Results |
| Stage Pressure read causally | Phrased as "primary pressure location"; weights labelled heuristic |
| Calling the search an optimum | Always "bounded structured candidate set" |
| Overclaiming novelty | Integration/application framing; no "no previous study" claims |

## 13. Execution status

Phase A (understand) ✅ · Phase B (literature) → next · Phase C (architecture) ✅ ·
Phases D–M pending.
