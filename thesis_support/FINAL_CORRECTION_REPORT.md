# FINAL CORRECTION REPORT

Final thesis correction pass — **text and analysis presentation only**.

No file under `src/`, `configs/`, `data/` or `webapp/` was modified. No simulation was rerun,
no model retrained, no persisted result altered. Every factual change below was made by reading
the current implementation and correcting the thesis to describe it.

One new file was added under `thesis_support/analysis/` (`verify_pressure_bands.py`), a
read-only derivation script over persisted outputs. It is thesis-support tooling, not part of
the production system.

Source at start of pass: branch `thesis-branch`, commit `c42c8e4`.

---

## Summary

| Group | Items | Status |
|---|---|---|
| A — objective corrections | A1–A4 | all applied |
| B — claim discipline | B1–B5 | all applied |
| C — forecast / RQ4 statistics | C1–C3 | all applied |
| D — RL methodology / reproducibility | D1–D4 | all applied |
| E — theory / literature | E1–E2 | applied (2 verified references added) |
| F — terminology / editorial | F1–F4 | all applied |
| H — global consistency pass | — | complete |
| I — quality gate | — | complete |

**Compile:** 0 errors, 0 LaTeX warnings, 0 undefined references, 0 undefined citations,
2 overfull boxes (2.0 pt and 3.1 pt, both pre-existing), 114 pages, 17 figures, 34 tables,
37 references all cited.

---

## A. Mandatory objective corrections

### A1 — The December 19 % calculation

**Requested:** `9 / 47 = 19.15 %`, `9 / 38 = 23.68 %`. Stop calling nine FTE "19 % of the
recommended workforce".

**Verified:** analytical estimate 47 FTE, simulated recommendation 38 FTE
(`chapters/05_results.tex` Table 5.2, matching `data/api_runs/latest/historical/
rl3_monthly_recommendations_summary.csv`). The old wording was wrong in five places.

**Resolution.** Canonical formulation adopted throughout: *nine FTE — the recommendation lies
approximately 19 % **below the analytical estimate***. The results chapter now states both
ratios explicitly and names which one the thesis uses.

| File | Change |
|---|---|
| `chapters/05_results.tex` §5.2.2 | Replaced "roughly 19 % of the recommended workforce" with both ratios computed (`9/47 ≈ 19 %`, `9/38 ≈ 24 %`) and a statement that the first is used throughout |
| `frontmatter/extended_abstract.tex` | "roughly nineteen per cent of the recommended workforce" → "the simulated recommendation of 38 FTE lay approximately nineteen per cent below the analytical estimate of 47" |
| `chapters/06_discussion.tex` §6.1 | "roughly 19 % of the recommended workforce" → "roughly 19 % of the analytical estimate of 47" |
| `chapters/06_discussion.tex` §6.5 | "over-provides capacity by up to 19 %" → "by up to 19 % **of the estimate**" |
| `chapters/07_conclusions.tex` RQ1 | "roughly 19 % of the recommended workforce" → "the recommended 38 FTE lay roughly 19 % below the estimate of 47" |
| `chapters/07_conclusions.tex` closing | "over-provided by up to nineteen per cent" → base made explicit |

Note: the earlier grep in the source missed `\SI{19}{\percent}`; the siunitx form was the more
common one and is now covered.

### A2 — Forecast screening / validation simulation count

**Requested:** 48 + 24 = 72 actual, 144 exhaustive, 50 % reduction. Do not double-count
replication 1 for the finalists.

**Verified in code** (`src/analysis/future_screening.py` docstring; `src/api/runners.py::
run_future_planning`). Stage A evaluates every candidate × 3 policies on replication 0
(48 runs). Stage B loops `for r in range(1, n_reps)` over finalists only (4 × 3 × 2 = 24 runs),
and `per_rep_dfs_finalists` is *seeded with the finalists' existing screening rows* — replication
1 is provably not re-run. The persisted CSV confirms the shape: 36 rows tagged `screening`,
12 tagged `validated`.

**Resolution.** The thesis said **60 runs, ~58 % reduction**. Corrected to **72 runs against
144, exactly 50 %**, with the arithmetic shown and the non-repetition of replication 1 stated.

| File | Change |
|---|---|
| `chapters/05_results.tex` §5.5.1 | Full rewrite of the paragraph with the arithmetic broken out |
| `chapters/04_methodology.tex` §4.12.2 | Two-stage description rewritten with the same counts |
| `chapters/07_conclusions.tex` RQ4 | Counts added ("72 simulation runs against the 144…") |

### A3 — Data provenance for the two workflows

**Requested:** distinguish the retrospective 240,000-order dataset from the forecast's
generated streams; keep "no real data" prominent.

**Verified.** The forecast entry point never touches `data/uploads/orders_uploaded.csv`; it
calls `generate_future_scenario_orders` per replication. Direct corroboration from the persisted
output: the three December replications average **41,687** orders against the annual dataset's
**40,800** for December, because volume is drawn lognormally around the forecast.

**Resolution.** §4.1 restructured into *retrospective* and *forecast* paragraphs, with the
41,687 vs 40,800 figure cited as evidence. Corresponding changes in the Extended Abstract, §3.5.2
and Appendix A's provenance statement. Appendix E's environment table now lists the two data
sources separately. "No real, proprietary or client data were used" retained everywhere.

### A4 — Reconciling the formal problem with the actual recommendation rule

**Requested:** determine the current decision hierarchy from code and rewrite the formulation to
match; do not hide the infeasible fallback.

**Verified in code.** `src/analysis/bottleneck_report.py::_select_recommendation` is
**lexicographic**, not an unconstrained cost minimisation:

1. **Eligibility** — in forecast mode only `evaluation_stage == "validated"` rows are eligible.
2. **Feasibility first** — if any eligible row meets both floors, take the cheapest of those.
3. **Fallback** — otherwise, minimum `sla_violation`, ties broken by cost; reported infeasible.

Separately, `src/reporting/export_rl3_monthly_recommendations.py` publishes several
**descriptive** per-month columns, including a `best_total_*` "cheapest configuration" computed
with **no** feasibility filter. Table 5.2 was tabulated from that descriptive column
(`thesis_support/analysis/fig_capacity_bottleneck.py:182–188`).

**This is a correction to the supporting documentation as well as to the thesis.**
`THESIS_STATE.md §7` describes the selection rule as "unconstrained `idxmin(total_cost)`" — that
describes the exporter column, not the framework's recommendation. The thesis text at
`05_results.tex:87` and Appendix F.4 had inherited that mischaracterisation.

Row-by-row check performed: the two rules coincide in **all twelve months** of this run
(identical regime, policy and cost in every month; every recommendation feasible). Asserted in
the thesis as a verified property of this run, explicitly not as a property of the rule.

**Resolution.**

| File | Change |
|---|---|
| `chapters/03_problem_framework.tex` §3.2.2 | Eq. 3.5/3.6 relabelled the *planning criterion*. New **Eq. 3.7** states the implemented lexicographic selection over `W_cand × Π` with the least-violation fallback written out. Optimality disclaimer retained and now points at the equation. |
| `chapters/04_methodology.tex` | New **§4.8.1 "The recommendation rule as implemented"** — the three-step hierarchy, the fallback stated openly, and the distinction from the descriptive export columns |
| `chapters/05_results.tex` §5.2.1 | Now states the lexicographic rule, that the fallback was never reached, and that Table 5.2 comes from the descriptive column which coincided with it |
| `appendices/appendix_F_additional_analysis.tex` §F.4 | Table F.3 restructured: recommendation rule vs descriptive statistics, with the coincidence stated and its limits noted. Missing `\label{sec:app-rules}` added. |
| `chapters/03_problem_framework.tex` §3.2.2 opener | "The decision is to choose…" → "The planning criterion is to choose…" |

Chapter 1 objectives, §4.11 (adaptive search) and the Conclusions were checked: §4.11 already
described its own three-branch lexicographic acceptance rule correctly, and now reads
consistently with §4.8.1. No optimality claim exists anywhere.

---

## B. Claim-discipline corrections

### B1 — RL-3 vs Urgent-First: no over-attribution

**Resolution.** The numerical results are unchanged. Attribution language added at each of the
four places the comparison carries weight:

- `chapters/05_results.tex` §5.3.3 (the December headline) — new paragraph: the two
  implementations differ on *two* dimensions (state-dependent class selection **and**
  FIFO-vs-heap within-class ordering), the experiment cannot decompose them, and the result is a
  difference between the policies as implemented, not a causal estimate of the value of learning.
- `chapters/07_conclusions.tex` RQ2 — same qualification.
- `frontmatter/extended_abstract.tex` — same qualification, one sentence.
- `chapters/06_discussion.tex` — the existing within-class-ordering subsection (already
  well-hedged, and already recommending the FIFO-within-class variant as future work) was given
  the label `sec:disc-uf-ordering` so the results chapter can point at it. Its content was not
  weakened.

The Future Work item proposing an Urgent-First FIFO-within-class variant is retained unchanged.

### B2 — §5.3.5 renamed and reframed

**Requested:** the statistic is an absolute difference, which measures divergence, not advantage.
Rename; acknowledge volume/mix confounds; avoid causal language for utilisation.

**Robustness check performed** (read-only, over the persisted 576-row retrospective CSV; no
simulation input touched, nothing rerun). Recorded in
`thesis_support/analysis/verify_pressure_bands.py`:

| Band | n | abs. mean (EUR) | per 1,000 orders | signed, feasible exists | signed, neither feasible |
|---|---|---|---|---|---|
| below 70 % | 33 | 537 | 55 | −1 (31) | −8,608 (2) |
| 70–80 % | 47 | 1,484 | 96 | +858 (45) | −14,925 (2) |
| 80–90 % | 69 | 3,004 | 106 | +2,988 (69) | — |
| 90–95 % | 11 | 5,563 | 283 | +4,611 (8) | −7,987 (3) |
| above 95 % | 32 | 6,320 | 545 | +3,630 (14) | −8,388 (18) |

Signed = `C_UF − C_RL3`; positive means RL-3 cheaper.

Two findings, both now in the thesis:

1. The per-1,000-order normalisation is **also monotone** (55 → 545), so the divergence is not a
   volume artefact.
2. The **signed** difference is *not* monotone and reverses where neither policy is feasible.
   Above 95 % utilisation, 18 of 32 configurations are in that reversed group, so the growth in
   the absolute gap in the top band is partly reversal, not advantage.

**Resolution.**

| File | Change |
|---|---|
| `chapters/05_results.tex` §5.3.5 | **Retitled** "Policy divergence increases with capacity pressure". Table 5.8 extended with per-1,000-order and signed-by-feasibility columns; caption rewritten. Interpretation rewritten with both qualifications. Causal language replaced by "coincide with", "observed across the bands", "observed grouping variable, not an established cause". |
| `chapters/05_results.tex` fig. 5.6 caption | "the cost difference" → "the absolute cost difference … across the observed bands" |
| `chapters/06_discussion.tex` §6.6 | Same reframing; divergence vs advantage separated; causal caution added |
| `chapters/06_discussion.tex` §6.7 | Pressure-band screening-test paragraph reworded to "how far apart the two policies land"; association stated as observed, not causal |

The existing right-panel figure caption ("advantage holds wherever a feasible plan exists and
reverses where none does") was already correct and is now supported by the tabulated signed
split.

### B3 — Removed the absolute "cannot be repaired by capacity" claim

All three instances qualified to the evaluated range, with an explicit acknowledgement that a
sufficiently large workforce would eventually satisfy the floors:

| File | Old | New |
|---|---|---|
| `frontmatter/extended_abstract.tex` | "a discipline that ignores differentiated deadlines cannot be repaired by adding capacity" | "Within the workforce range evaluated … a sufficiently large workforce would eventually do so, but no candidate examined reached it" |
| `chapters/07_conclusions.tex` RQ2 | same absolute claim | "a statement about the candidate set rather than a law" |
| `chapters/06_discussion.tex` §6.3 | "cannot be repaired by resourcing" | "is expensive to repair by resourcing alone"; subsection retitled "…why capacity did not rescue it in the evaluated range"; a bounding paragraph added |
| `chapters/06_discussion.tex` §6.6 | "which no realistic capacity increase would have achieved" | "none of the capacity increases within that candidate range achieved" |

### B4 — Campaign-month statement corrected

**Verified from the persisted CSV:** FIFO has **108** infeasible configurations of 192, occurring
in **every one of the twelve months** (October's 2 is the smallest count). **64** are in the four
campaign months; **44** are spread across the eight non-campaign months. So "failures concentrate
entirely in the four campaign months" is provably wrong. What is confined to those months is
**complete** monthly infeasibility (0 of 16 feasible), against at least 8 of 16 feasible in every
other month (May's 8 is the minimum).

| File | Change |
|---|---|
| `chapters/07_conclusions.tex` RQ2 | "FIFO's failures concentrate entirely in the four campaign months" replaced with the 108 / 64 / 44 breakdown across all twelve months and the "complete monthly infeasibility" framing |
| `chapters/05_results.tex` §5.3.1 | Same correction, with the "at least eight of sixteen elsewhere" contrast |
| `chapters/05_results.tex` §5.6 summary item | Reworded to the same distinction |
| `frontmatter/extended_abstract.tex` | "More decisively, in the four campaign months…" → "its **complete** monthly infeasibility is confined to the four campaign months" |
| `chapters/06_discussion.tex` §6.3 | Partial-failure months acknowledged before the categorical claim |

### B5 — Little's law / throughput claims softened

**Requested:** do not claim Little's law guarantees identical finite-horizon completion counts.

**Supporting evidence already in the thesis:** January backlog shares are 8.59 / 8.97 / 8.95 %
and December's 10.42 / 11.09 / 11.10 % — i.e. differences of up to 0.38 and 0.68 percentage
points. Sequencing *does* move a few completions across the horizon; the correct claim is
near-invariance.

| File | Change |
|---|---|
| `chapters/02_theoretical_foundations.tex` §2.2.1 | New paragraph marking the boundary: the law constrains long-run averages in a stable system and does not imply identical finite-horizon counts under heterogeneous service times |
| `chapters/02_theoretical_foundations.tex` §2.3.2 | "should be essentially independent" → "close to independent", with the work-vs-count distinction stated; "the prediction to test is near-invariance rather than identity" |
| `chapters/05_results.tex` §5.3.8 | "Total throughput is therefore a property of the workforce, not of the policy" replaced by the empirical claim, with the 0.38 / 0.68 pp residuals quoted |
| `chapters/06_discussion.tex` §6.6 | Rewritten as: capacity fixes worker-minutes; the law does not guarantee identical completion counts; the tested policies produced almost identical utilisation and backlog, so their operative effect was the allocation of lateness between classes |

---

## C. Forecast / RQ4 statistical precision

RQ4 and the Future Planning workflow are retained in full. No output, interface label or
persisted value was changed.

### C1 — p90 with three replications

**Verified:** `src/analysis/replication_aggregation.py` computes
`p90_total_cost = grp["estimated_total_cost"].quantile(0.9)` over three values — pandas' linear
interpolation, so the result sits 80 % of the way from the second-largest to the largest and is
dominated by the worst replication.

**Resolution.** A new paragraph in `chapters/05_results.tex` §5.5.1, *"How the three-replication
statistics should be read"*, states this explicitly: it is a sample quantile of three
observations, descriptive of spread, not an estimate of a distributional tail, and it is not
used as one anywhere. The word **"pessimistic cost"** was removed from every occurrence
(§3.5.2, §4.12.2, §5.5.1, §7 RQ4) in favour of "the spread across the three replications" or
"the cost observed in the least favourable replication". Table 5.14's caption now labels the
column as a sample quantile.

### C2 — "Probability of meeting SLA"

**Verified:** `prob_meets_sla_targets = mean(feasible)` over three replications — values in
{0, ⅓, ⅔, 1}. Also verified: the aggregated `feasible` flag is a **majority** rule
(`prob >= 0.5`), *not* "all three".

**Resolution.** The same new paragraph explains that the interface label is
"probability of meeting the SLA targets" but the quantity is a **replication success rate over
three trials**, admitting only four values, with no confidence statement attached; and that the
framework's feasibility flag is a majority rule. Verified from the persisted output that in this
run every validated row scored exactly 0.0 or 1.0 (four FIFO rows at 0/3, eight urgency-aware
rows at 3/3), so the existing statement "met both floors in all three replications" is accurate
as observed data — but the system's *rule* is no longer described as requiring 3/3.
"Proportion / probability of replications" was replaced by "share of replications" in Chapters 3,
4, 5, 6 and 7 and in the Figure 5.7 caption, which now also notes the four admissible values.

### C3 — Forecast uncertainty method documented

**Resolution.** New **§4.12.3 "Forecast scenario generation and uncertainty"**, written only
from verified code and configuration:

- **Inputs** — planning month, expected annual volume (or monthly override), named uncertainty
  level; monthly volume = annual × the month's configured share (December 0.170).
- **What the uncertainty level is** — a table of the three configured presets
  (`low` 0.05/0.15, `standard` 0.10/0.30, `high` 0.20/0.50), with the explicit statement that
  these are **scenario assumptions supplied to the model, not forecast errors calibrated against
  any observed forecasting record** — exactly the disclosure the task asked for.
- **Seeding** — replication *r* of month *m* uses `42 + 10m + 1000r`.
- **Volume draw** — lognormal with mean equal to the expected volume and CV equal to the
  level's demand CV.
- **Arrival pattern** — configured calendar weights including the December campaign burst,
  multiplied by unit-mean lognormal noise at the level's arrival CV; intraday profile untouched.
- **Composition** — urgency, family, complexity, item count and stage workload units from the
  same generator as the annual dataset.
- **Common vs varying** — within a replication, all 16 configurations × 3 policies share one
  order stream and one CRN service-time map, so within-replication comparisons are matched;
  across replications the volume, arrival pattern and individual orders all change together.
- **Finalist selection** — the exact `_regime_rank_key` rule: configurations at which any policy
  is feasible rank first, ordered by their cheapest feasible policy's cost; infeasible ones
  after, by violation then cost; top four retained; ranking is over configurations, so a finalist
  carries all three policies forward.

---

## D. RL methodology / reproducibility hardening

### D1 — Production evaluation is greedy

**Verified:** `evaluate_rl3_monthly_capacity_cost.py:128`, `evaluate_rl3.py:68`,
`evaluate_rl3_multiseed.py:56` and `rl_audit.py:179` all call `run_episode(..., greedy=True)`;
`env_fullstage_rl.py:353` sets `should_record = not greedy`; `dqn_agent.py:67` takes `argmax`
with no exploration when greedy.

**Resolution.** New paragraph *"Evaluation is greedy"* in `chapters/04_methodology.tex` §4.5.3
and a new **§C.7 "Evaluation mode"** in Appendix C. Both state that argmax selection is used with
exploration disabled, that no transitions are recorded, that ε-greedy applies to training only,
and that no reported RL-3 metric contains exploratory randomness.

### D2 — Training pool and holdout documented

**Verified:** `main_train_rl3.py::build_training_pool` samples from three representative months
(June, October, December), generating 9 candidates each, of which the last 2 are held out — so
**21 training pairs and 6 holdout pairs**, persisted at `data/rl3_train_pool.json`.

**A documentation error was found and corrected.** The thesis (Appendix C) stated the
generalisation split as "twelve configurations for training, four held out (`s112`, `s231`,
`s322`, `s432`)". That is `planning_profile.yaml::rl_generalisation`, which is used **only** by
`evaluate_rl3_generalisation.py`. `rl_audit.py:271–276` documents it as *"a stale, month-agnostic
set of small regimes left over from before per-month dynamic regime generation existed"* that
"does not correspond to what any given month was actually trained on". `THESIS_STATE.md §6.11`
carried the same error.

**Resolution.** New **§C.6 "Training pool and holdout"** with **Table C.5** reproducing the
persisted manifest exactly (months, roles, order counts, all 21 training and 6 holdout
configurations, analytical centres). Three consequences stated:

1. Nine of the twelve retrospective months were **never trained on** — only June, October and
   December are in the pool, so every other month is an out-of-distribution evaluation.
2. The headline configurations **`s22_11_5` (December) and `s753` (October) are holdouts** —
   independently confirmed by `rl3_audit_report.json`, which records
   `seen_during_training: false` for them. This strengthens the finding rather than qualifying
   it, and is stated as such.
3. The unrelated 12/4 list is recorded as belonging to a separate script, explicitly to prevent
   the two being confused.

Seed logic (`base_seed` 123 for the agent and resources, per-episode seed from a seeded NumPy
generator) is stated in the same section. The methodology chapter's training paragraph was
updated to match.

### D3 — State-saturation limitation

**Verified:** `env_fullstage_rl.py:201–203` divides worker counts by
`capacity_feature_scale = 20` and clips to [0, 1]. The finding is stronger than the task
anticipated:

- **All sixteen December candidates staff picking at 22 FTE or more** (`s22_11_5` through
  `s30_17_9`), so `cap_pick` is identically **1.0 across the entire December candidate set** —
  that feature carries zero information in the peak month.
- Packing (11–17) and dispatch (5–9) stay below the ceiling and continue to vary, so states are
  not identical — "other state features continue to vary" is literally true here.
- Queue features saturate too: the normal-class divisor is 500, while December's maximum picking
  queue exceeds **3,122** orders and the mean picking queue at `s22_11_5` is above 1,000. The
  persisted `rl3_audit_report.json` already flags `state_saturation_signal: true` at that
  configuration.

**Resolution.** New **§C.8 "A representation limitation: feature saturation"** stating all of the
above, plus a short entry in `chapters/04_methodology.tex` §4.14 (methodological limitations) and
in `chapters/06_discussion.tex` §6.8 (limitations). Framed as a representation limitation, not as
invalidating the policy; notes that the agent nevertheless produced feasible lower-cost outcomes
at those configurations, and that the reported behaviour should not be extrapolated to capacities
well beyond the normalising constants.

### D4 — Checkpoint reproducibility

`data/dqn_rl3_final.pt` remains absent. Nothing was altered, recreated or faked.

**Resolution.** The wording was tightened in four places to the precise formulation requested,
and to close a gap the task specifically warned about — implying that a hash enables
regeneration:

> The persisted evaluation outputs are available, but exact regeneration of the RL-3 results
> requires the trained checkpoint, which was not available in the environment used for final
> thesis preparation.

Added everywhere it appears: the digest and file size are retained as **provenance evidence**
identifying which artefact produced the results, and explicitly *"a digest verifies a file one
already holds, and cannot produce one"*. Also added: retraining from the recorded configuration
would produce *a* policy, not *this* policy, since episodes are sampled stochastically. And an
explicit statement that **no part of the thesis claims end-to-end reproducibility of the RL-3
results from the distributed artefacts**.

Files: `appendices/appendix_C_rl.tex` §C.9, `appendices/appendix_E_validation.tex` §E.5,
`chapters/06_discussion.tex` §6.8, `chapters/04_methodology.tex` §4.14.

---

## E. Theory / literature hardening

### E1 — Support for the decision-support / explainability claims

§2.7 previously carried **zero citations**. Two references were found and verified against
independent bibliographic records before insertion:

| Key | Reference | Verified |
|---|---|---|
| `little1970` | Little, J. D. C. (1970). *Models and Managers: The Concept of a Decision Calculus.* Management Science 16(8), B466–B485. DOI `10.1287/mnsc.16.8.B466` | Title, author, journal, volume, issue, year, pages and DOI confirmed via EconPapers/RePEc record `inm:ormnsc:v:16:y:1970:i:8:p:b466-b485` |
| `rudin2019` | Rudin, C. (2019). *Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead.* Nature Machine Intelligence 1(5), 206–215. DOI `10.1038/s42256-019-0048-x` | Title, author, journal, volume, issue, year, pages and DOI confirmed |

Both are peer-reviewed and from recognised publishers (INFORMS; Nature Portfolio). No metadata
was fabricated. `little1970` also pairs naturally with `little1961`, already in the bibliography.

**How they are used.** `little1970` supports the claim that interpretability is a *functional
requirement* of a managerial model rather than presentation — the decision-calculus criteria
(understandable, robust, easy to control, complete on important issues) are exactly the position
§2.7 takes. `rudin2019` supports the claim that opacity carries a real cost in high-stakes
decisions and that an interpretable model should be preferred unless the opaque alternative
demonstrably performs better — which is precisely the standard §5.3 then applies to RL-3. Both
citations genuinely support the sentences that carry them.

### E2 — Bibliography discipline

Bibliography grew from **35 to 37** entries. Both additions are cited in text, both support
specific claims, neither was added for appearance. BibTeX reports 37 entries used, 0 warnings;
since BibTeX emits only cited entries, all 37 are cited.

---

## F. Terminology / editorial

| Item | Resolution |
|---|---|
| **F1** normal vs standard | Two instances of "standard orders" (`frontmatter/extended_abstract.tex:14`, `chapters/01_introduction.tex:55`) changed to "normal orders". Verified: `standard` is not a code-level class label — the classes are `urgent` / `normal`. Sweep confirms zero remaining instances. (Note: `standard` remains as the name of the *uncertainty level* preset, which **is** a code-level label and is correctly typeset as `\code{standard}`.) |
| **F2** σ_c terminology | Renamed from "service-time threshold" to **"system-time SLA threshold for class c"** in the Chapter 3 notation table, with a new clarifying sentence distinguishing σ_c (end-to-end time in system, waiting included) from t_{i,s} (per-stage processing duration), and noting both are on the operating clock. Propagated to §4.5.3, Appendix A (`sla_minutes` row), Appendix B (both parameter rows) and Appendix C (slack equation). Sweep confirms zero remaining instances of the old term. |
| **F3** chapter title hyphenation | Chapter 3's opening no longer splits "Frame-work". Fixed by a layout-only change: `\chapter[short]{Problem Formulation and\\Decision-Support \mbox{Framework}}` — the short form keeps the TOC and running heads unchanged, and the substantive title is identical. Verified visually on the rendered page. |
| **F4** overfull boxes | The new Table 5.8 initially overran by 13.9 pt. Fixed by shortening two column heads onto a second header row and reducing `\tabcolsep` to 4.5 pt — layout only, no numbers or content changed. Final state: **2 overfull boxes, 3.13 pt and 2.01 pt, both pre-existing** and unchanged by this pass. Per instruction, no substantive text was rewritten to chase them. |

---

## G. Confirmed unchanged

Verified untouched: `src/`, `configs/`, `data/`, `webapp/`; the Picking → Packing → Dispatch
flow; the implemented Urgent-First semantics (strict class precedence, heap-ordered within
class); RL-3; every historical and forecast result; December `s22_11_5` retrospective and
`s25_14_7` forecast; the Stage Pressure formula; the adaptive search; candidate generation; the
economic assumptions; SLA thresholds; operating-time semantics; synthetic data generation; the
CRN implementation. RQ4 and Future Planning are retained in full. No corrected policy variant was
created; no FIFO-within-class Urgent-First was implemented.

---

## H. Global consistency pass

Every term in the checklist was swept across all chapters, front matter and appendices after
editing:

| Term | Result |
|---|---|
| 19 % / nineteen per cent | 6 instances, all now referenced to the analytical estimate |
| 47 vs 38 FTE | consistent; both ratios stated once, in §5.2.2 |
| 72 vs 144 simulations, 50 % reduction | 3 locations, all consistent; no residual "60" or "58 %" |
| 240,000 retrospective orders | 11 instances; every one now scoped to the retrospective dataset |
| forecast-generated scenarios | consistent in Extended Abstract, §3.5.2, §4.1, §4.12.3, App. A, App. E |
| feasibility / recommendation / constrained problem | consistent between §3.2.2, §4.8.1, §4.11, §5.2.1, §5.5.1, App. F.4 |
| optimal / optimisation | 14 instances, all disclaimers; no optimality claim anywhere |
| RL advantage / causal attribution | qualified in §5.3.3, §5.3.5, §6.2, §6.6, §7 RQ2, Extended Abstract |
| Urgent-First / within-class ordering | consistent; §6.2 is now cross-referenced from §5.3.3 |
| FIFO capacity claims | 4 absolute claims found, all bounded to the evaluated range |
| campaign months | 17 instances; the "complete infeasibility" distinction holds throughout. Verified against Table 5.4: FIFO has infeasible configurations in all twelve months, so no "eleven months" or "seven further months" phrasing survives |
| p90 / probability / replication success rate | consistent; "pessimistic cost" eliminated |
| Little's law / throughput | 3 locations softened to near-invariance |
| checkpoint reproducibility | 4 locations, identical precise formulation |
| normal vs standard | 0 remaining instances of "standard orders" |

---

## I. Final quality gate

| Check | Result |
|---|---|
| Compile (`latexmk -pdf`) | **exit 0** |
| LaTeX errors | **0** |
| LaTeX warnings | **0** |
| Undefined references | **0** |
| Undefined citations | **0** |
| BibTeX warnings | **0** |
| Bibliography entries used | **37** (all cited) |
| Overfull boxes | **2** — 3.13 pt, 2.01 pt (both pre-existing) |
| Underfull boxes | 8 (all in tabularx/longtable cells; cosmetic, pre-existing) |
| Page count | **114** (was 105 before this pass; +9 from the new §4.8.1, §4.12.3, §C.6–C.8 and the expanded Table 5.8) |
| Figures | 17 |
| Tables | 34 (was 33; Table C.5 added) |
| Figures/tables render | verified — Ch. 3 opening, Table 5.8, Table C.5 inspected on the rendered PDF |
| Word-count minima | **all satisfied** — Theory 4,385 (min 3,000); Methodology 6,736 (min 3,000); Results 6,311 (min 4,000); Discussion 5,302 (min 3,000); Conclusions 2,220 (min 1,000). Grand total 34,476 words |

---

## Requested corrections that did not apply as stated

- **A4's premise that the selector "may select by total cost without imposing feasibility"** was
  half right. The *framework's recommendation selector* does impose feasibility, lexicographically
  — it was the thesis text (and `THESIS_STATE.md`) that mischaracterised it, having conflated it
  with a descriptive export column. The correction therefore went the other way from what the
  brief anticipated: the constrained formulation was **kept and made exact**, rather than
  replaced by a purely algorithmic description, because the code supports it. The fallback branch
  is documented openly as instructed.
- **D2's 12/4 generalisation split** turned out to be a stale configuration artefact rather than
  the training design. Documented as such rather than reproduced.
- **B2's optional per-1,000-order normalisation** was cheap enough to compute from the persisted
  CSV, so it was done and included; it strengthens rather than complicates the story.

## Points where the implementation forced different wording

- **C2** — the framework's aggregated feasibility flag is a **majority** rule (`prob ≥ 0.5`), not
  "all three replications". The thesis now describes the rule accurately while still reporting,
  correctly, that every validated configuration in this run scored 0/3 or 3/3.
- **C1** — the p90 is pandas' interpolated sample quantile — between the median and the largest of the three, four-fifths of the way towards the largest — not the maximum. The task offered
  "maximum observed cost" as a cleaner alternative, but that would misstate the persisted number,
  so the reported value was kept and described precisely as a sample quantile of three
  observations.
- **C3** — the "standard uncertainty" setting is a **configured preset**, not calibrated forecast
  error. Stated explicitly, as instructed.

---

## Remaining TODOs

Two placeholders, both confined to `thesis/frontmatter/titlepage.tex`, both requiring information
only the author holds. Neither is a correctness issue and neither was introduced by this pass:

1. `TODO_TUTOR_FULL_NAME`
2. `TODO_SUBMISSION_MONTH_YEAR`

The title page should additionally be replaced with the official IMIM cover sheet, per
`THESIS_STATE.md §10`.

Optional and not done (out of scope for a correction pass): frontend screenshots as an
Appendix G.

---

## Verdict

**SUBMISSION-READY**, conditional on the author filling the two title-page placeholders.

Final PDF: `C:\Users\ignal\Desktop\TFM-Logistics-Baseline\thesis\main.pdf` — 114 pages.
