"""
Extra-worker economics (spec §8), SLA feasibility (§10) and bottleneck-directed adaptive
capacity search (§9).

Reuses the exact same per-regime evaluation helpers as the base 16-regime grid
(src/rl/evaluate_rl3_monthly_capacity_cost.py::_run_baseline / _run_rl3 / _compute_costs) —
there is exactly one definition of "run a regime under a policy" and "compute its cost".
The adaptive search never resamples service times: it's always called with the same
service_time_map (and therefore the same scenario seed) as the parent 16-regime comparison,
so added-worker decisions are evaluated under identical stochastic conditions.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.analysis.bottleneck import STAGES, score_bottlenecks
from src.analysis.regime_naming import format_regime, parse_regime
from src.analysis.sla_feasibility import check_feasibility
from src.data.planning_profile import load_planning_profile
from src.rl.evaluate_rl3_monthly_capacity_cost import _compute_costs, _run_baseline, _run_rl3

_STAGE_IDX = {"picking": 0, "packing": 1, "dispatch": 2}


def worker_monthly_cost(worker_cost_per_hour: float, hours_per_worker_month: float) -> float:
    return float(worker_cost_per_hour) * float(hours_per_worker_month)


def break_even_metrics(
    worker_cost_per_hour: float,
    hours_per_worker_month: float,
    cost_late_urgent: float,
    cost_late_normal: float,
    urgent_late_orders: float = 0.0,
    normal_late_orders: float = 0.0,
) -> Dict[str, Any]:
    """Theoretical break-even quantity of late orders an extra worker's monthly cost would
    need to prevent to pay for itself, under three lenses: urgent-only, normal-only, and the
    current observed urgent/normal late-order mix."""
    wc = worker_monthly_cost(worker_cost_per_hour, hours_per_worker_month)
    urgent_be = (wc / cost_late_urgent) if cost_late_urgent > 0 else None
    normal_be = (wc / cost_late_normal) if cost_late_normal > 0 else None

    total_late = float(urgent_late_orders) + float(normal_late_orders)
    if total_late > 0:
        avg_penalty = (
            float(urgent_late_orders) * cost_late_urgent + float(normal_late_orders) * cost_late_normal
        ) / total_late
    else:
        avg_penalty = (cost_late_urgent + cost_late_normal) / 2.0
    mixed_be = (wc / avg_penalty) if avg_penalty > 0 else None

    return {
        "worker_monthly_cost": round(wc, 2),
        "urgent_only_break_even_orders": round(urgent_be, 1) if urgent_be is not None else None,
        "normal_only_break_even_orders": round(normal_be, 1) if normal_be is not None else None,
        "mixed_break_even_orders": round(mixed_be, 1) if mixed_be is not None else None,
        "current_avg_penalty_per_late_order": round(avg_penalty, 2),
    }


def _regime_label(pick: int, pack: int, disp: int) -> str:
    return format_regime(pick, pack, disp)


def _parse_regime(name: str) -> Tuple[int, int, int]:
    return parse_regime(name)


def evaluate_regime_all_policies(
    orders: pd.DataFrame,
    workers: Tuple[int, int, int],
    base_resources_cfg: Dict,
    sim_cfg: Dict,
    service_cfg: Dict,
    service_time_map: Dict,
    rl_agent,
    reward_cfg: Dict,
    seed: int,
    cost_params: Dict[str, float],
    sla_targets: Dict[str, float],
) -> Dict[str, Dict[str, Any]]:
    """Evaluate one workforce regime under fifo / urgent_first / rl3_dqn, all sharing
    `service_time_map`. Returns per-policy: raw metrics (incl. stage_metrics), costs,
    feasibility, and SLA violation score."""
    resources_cfg = {
        **base_resources_cfg,
        "picking_workers": workers[0], "packing_workers": workers[1], "dispatch_workers": workers[2],
    }
    total_workers = sum(workers)
    urgent_cnt = int((orders["order_type"] == "urgent").sum())
    normal_cnt = int((orders["order_type"] == "normal").sum())

    cu = cost_params["cost_late_urgent"]
    cn = cost_params["cost_late_normal"]
    wc = cost_params["worker_cost_per_hour"]
    hpm = cost_params["hours_per_worker_month"]
    u_target = sla_targets["urgent_target"]
    n_target = sla_targets["normal_target"]

    results: Dict[str, Dict[str, Any]] = {}
    for policy in ("fifo", "urgent_first", "rl3_dqn"):
        if policy == "rl3_dqn":
            m = _run_rl3(orders, rl_agent, resources_cfg, sim_cfg, service_cfg, reward_cfg, seed, service_time_map)
        else:
            m = _run_baseline(orders, policy, resources_cfg, sim_cfg, service_cfg, service_time_map)

        late, labour, ul, nl = _compute_costs(
            urgent_cnt, normal_cnt, m["sla_urgent"], m["sla_normal"], total_workers, cu, cn, wc, hpm
        )
        feasible, violation = check_feasibility(m["sla_urgent"], m["sla_normal"], u_target, n_target)

        results[policy] = {
            "metrics": m,
            "workers": workers,
            "total_workers": total_workers,
            "late_cost": late,
            "labour_cost": labour,
            "total_cost": late + labour,
            "urgent_late_orders": ul,
            "normal_late_orders": nl,
            "feasible": feasible,
            "sla_violation": violation,
        }
    return results


def pick_best_policy(results_by_policy: Dict[str, Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    """Feasible candidates win on lowest total cost; if none are feasible, pick the lowest
    SLA violation, tie-broken by lowest total cost (spec §10)."""
    feasible = [(p, r) for p, r in results_by_policy.items() if r["feasible"]]
    if feasible:
        return min(feasible, key=lambda kv: kv[1]["total_cost"])
    return min(results_by_policy.items(), key=lambda kv: (kv[1]["sla_violation"], kv[1]["total_cost"]))


def _decide_accept(current_best: Dict[str, Any], candidate_best: Dict[str, Any]) -> Tuple[bool, str]:
    """Shared adaptive-search accept/reject decision (spec §9.2), used identically by the
    single-seed (historical) and multi-seed screen-then-validate (future planning, §8)
    search variants — and, for future planning, also as the screening "is this candidate
    promising enough to validate" check (§8.B), so a candidate is never accepted using
    weaker criteria than the final decision uses."""
    total_cost_diff = candidate_best["total_cost"] - current_best["total_cost"]

    if not current_best["feasible"] and candidate_best["feasible"]:
        return True, "reaches SLA feasibility targets"
    if candidate_best["feasible"] and current_best["feasible"] and total_cost_diff < 0:
        return True, "lower total cost while remaining feasible"
    if (
        not current_best["feasible"] and not candidate_best["feasible"]
        and candidate_best["sla_violation"] < current_best["sla_violation"]
    ):
        return True, "reduces SLA violation (still below target)"
    reason = (
        "added labour cost not offset by penalty reduction"
        if total_cost_diff >= 0
        else "no feasibility or cost improvement over parent"
    )
    return False, reason


def _avg_stage_metrics(stage_metrics_list: List[Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """Average per-stage bottleneck-input metrics across replications (spec §8.C) — used to
    re-rank bottlenecks from a multi-seed aggregated result, the same way score_bottlenecks
    already consumes a single replication's stage_metrics."""
    n = len(stage_metrics_list)
    avg: Dict[str, Dict[str, Any]] = {}
    for stage in STAGES:
        avg[stage] = {
            "utilisation": sum(m[stage]["utilisation"] for m in stage_metrics_list) / n,
            "avg_wait_min": sum(m[stage]["avg_wait_min"] for m in stage_metrics_list) / n,
            "p95_wait_min": sum(m[stage]["p95_wait_min"] for m in stage_metrics_list) / n,
            "avg_queue_len": sum(m[stage]["avg_queue_len"] for m in stage_metrics_list) / n,
            "max_queue_len": int(round(sum(m[stage]["max_queue_len"] for m in stage_metrics_list) / n)),
            "late_wait_share": sum(m[stage]["late_wait_share"] for m in stage_metrics_list) / n,
        }
    return avg


def evaluate_regime_all_policies_multiseed(
    orders_by_replication: List[pd.DataFrame],
    service_time_maps: List[Dict],
    seeds: List[int],
    workers: Tuple[int, int, int],
    base_resources_cfg: Dict,
    sim_cfg: Dict,
    service_cfg: Dict,
    rl_agent,
    reward_cfg: Dict,
    cost_params: Dict[str, float],
    sla_targets: Dict[str, float],
) -> Dict[str, Dict[str, Any]]:
    """Evaluate one workforce regime under all three policies across multiple scenario
    replications (seeds), then aggregate (mean) per policy — the adaptive-search analogue of
    src/analysis/replication_aggregation.py::aggregate_replications, at single-regime
    granularity. Each replication reuses evaluate_regime_all_policies with its own
    (orders, service_time_map, seed) triple, so common random numbers hold within each
    replication (spec §11) exactly as they do for the base regime grid."""
    n = len(orders_by_replication)
    per_rep = [
        evaluate_regime_all_policies(
            orders_by_replication[i], workers, base_resources_cfg, sim_cfg, service_cfg,
            service_time_maps[i], rl_agent, reward_cfg, seeds[i], cost_params, sla_targets,
        )
        for i in range(n)
    ]

    u_target = sla_targets["urgent_target"]
    n_target = sla_targets["normal_target"]

    aggregated: Dict[str, Dict[str, Any]] = {}
    for policy in ("fifo", "urgent_first", "rl3_dqn"):
        rows = [rep[policy] for rep in per_rep]
        sla_urgent = sum(r["metrics"]["sla_urgent"] for r in rows) / n
        sla_normal = sum(r["metrics"]["sla_normal"] for r in rows) / n
        sla_rate = sum(r["metrics"]["sla_rate"] for r in rows) / n
        feasible, violation = check_feasibility(sla_urgent, sla_normal, u_target, n_target)
        aggregated[policy] = {
            "metrics": {
                "sla_urgent": sla_urgent, "sla_normal": sla_normal, "sla_rate": sla_rate,
                "stage_metrics": _avg_stage_metrics([r["metrics"]["stage_metrics"] for r in rows]),
            },
            "workers": workers,
            "total_workers": sum(workers),
            "late_cost": sum(r["late_cost"] for r in rows) / n,
            "labour_cost": sum(r["labour_cost"] for r in rows) / n,
            "total_cost": sum(r["total_cost"] for r in rows) / n,
            "urgent_late_orders": sum(r["urgent_late_orders"] for r in rows) / n,
            "normal_late_orders": sum(r["normal_late_orders"] for r in rows) / n,
            "feasible": feasible,
            "sla_violation": violation,
            "replication_count": n,
        }
    return aggregated


def _neighbour_candidates(
    current_workers: Tuple[int, int, int],
    bottleneck_rows: List[Dict[str, Any]],
    max_workers_by_stage: Dict[str, int],
    close_threshold: float,
    tested_labels: set,
) -> List[Tuple[str, Tuple[int, int, int], str]]:
    stages_to_try = [bottleneck_rows[0]["stage"]]
    if (
        len(bottleneck_rows) > 1
        and (bottleneck_rows[0]["pressure_score"] - bottleneck_rows[1]["pressure_score"]) <= close_threshold
    ):
        stages_to_try.append(bottleneck_rows[1]["stage"])

    candidates = []
    for stage in stages_to_try:
        new_workers = list(current_workers)
        idx = _STAGE_IDX[stage]
        if new_workers[idx] + 1 > max_workers_by_stage[stage]:
            continue
        new_workers[idx] += 1
        label = _regime_label(*new_workers)
        if label in tested_labels:
            continue
        candidates.append((stage, tuple(new_workers), label))
    return candidates


def default_max_workers_by_stage(
    analytical_workers: Dict[str, int],
    max_extra_workers_per_stage: int,
) -> Dict[str, int]:
    """Per-stage adaptive-search ceiling (spec §18): analytical estimate for that stage plus a
    configurable extra allowance — relative to expected workload, not a single global cap sized
    for the small static regimes."""
    return {stage: int(analytical_workers[stage]) + int(max_extra_workers_per_stage) for stage in STAGES}


def run_adaptive_capacity_search(
    orders: pd.DataFrame,
    base_resources_cfg: Dict,
    sim_cfg: Dict,
    service_cfg: Dict,
    service_time_map: Dict,
    rl_agent,
    reward_cfg: Dict,
    seed: int,
    parent_regime: str,
    parent_results: Dict[str, Dict[str, Any]],
    cost_params: Dict[str, float],
    sla_targets: Dict[str, float],
    profile: Optional[Dict] = None,
    max_workers_by_stage: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Bottleneck-directed search for extra capacity, starting from a parent workforce
    comparison result. Returns the final regime/policy plus the full decision trail
    (spec §9.2/§9.3) — every tested candidate, accepted or rejected, with a reason.

    `max_workers_by_stage` (spec §18): per-stage ceiling, normally
    `default_max_workers_by_stage(analytical_estimate, max_extra_workers_per_stage)` computed
    by the caller from that month's analytical capacity estimate. Falls back to the parent
    regime's own workers + the configured extra allowance if not given."""
    profile = profile or load_planning_profile()
    ad = profile["adaptive_search"]
    max_extra = int(ad["max_extra_workers_per_stage"])
    max_iterations = int(ad["max_extra_iterations"])
    max_candidates = int(ad["max_candidates_per_iteration"])
    close_threshold = float(ad["close_bottleneck_threshold"])

    tested = {parent_regime}
    current_regime = parent_regime
    current_workers = _parse_regime(parent_regime)
    current_policy, current_best = pick_best_policy(parent_results)
    current_all_results = parent_results

    if max_workers_by_stage is None:
        max_workers_by_stage = {
            stage: current_workers[_STAGE_IDX[stage]] + max_extra for stage in STAGES
        }

    trail: List[Dict[str, Any]] = []
    stop_reason = "max adaptive iterations reached"

    for iteration in range(1, max_iterations + 1):
        stage_rows = score_bottlenecks(current_best["metrics"]["stage_metrics"])
        candidates = _neighbour_candidates(
            current_workers, stage_rows, max_workers_by_stage, close_threshold, tested
        )[:max_candidates]

        if not candidates:
            stop_reason = "no untested valid neighbour remains (or max workers per stage reached)"
            break

        improved = False
        for stage, workers_tuple, label in candidates:
            tested.add(label)
            results = evaluate_regime_all_policies(
                orders, workers_tuple, base_resources_cfg, sim_cfg, service_cfg, service_time_map,
                rl_agent, reward_cfg, seed, cost_params, sla_targets,
            )
            policy, best = pick_best_policy(results)

            labour_increase = best["labour_cost"] - current_best["labour_cost"]
            penalty_reduction = current_best["late_cost"] - best["late_cost"]
            total_cost_diff = best["total_cost"] - current_best["total_cost"]

            accept, reason = _decide_accept(current_best, best)

            trail.append({
                "iteration": iteration,
                "parent_regime": current_regime,
                "candidate_regime": label,
                "added_stage": stage,
                "policy": policy,
                "labour_cost_increase": round(labour_increase, 2),
                "late_penalty_reduction": round(penalty_reduction, 2),
                "total_cost_diff": round(total_cost_diff, 2),
                "urgent_sla_before": round(current_best["metrics"]["sla_urgent"], 4),
                "urgent_sla_after": round(best["metrics"]["sla_urgent"], 4),
                "normal_sla_before": round(current_best["metrics"]["sla_normal"], 4),
                "normal_sla_after": round(best["metrics"]["sla_normal"], 4),
                "overall_sla_before": round(current_best["metrics"]["sla_rate"], 4),
                "overall_sla_after": round(best["metrics"]["sla_rate"], 4),
                "bottleneck_before": stage_rows[0]["stage"],
                "accepted": accept,
                "reason": reason,
            })

            if accept:
                current_regime, current_workers = label, workers_tuple
                current_policy, current_best = policy, best
                current_all_results = results
                improved = True
                break  # re-rank bottlenecks from the new best before generating more neighbours

        if not improved:
            stop_reason = "no tested neighbour improved the objective"
            break
        if all(current_workers[_STAGE_IDX[s]] >= max_workers_by_stage[s] for s in STAGES):
            stop_reason = "maximum workers per stage reached"
            break

    return {
        "final_regime": current_regime,
        "final_policy": current_policy,
        "final_result": current_best,
        "final_results_by_policy": current_all_results,
        "stop_reason": stop_reason,
        "trail": trail,
        "iterations_run": len({t["iteration"] for t in trail}),
        "simulations_executed": len(trail) * 3,
    }


def run_adaptive_capacity_search_validated(
    orders_by_replication: List[pd.DataFrame],
    service_time_maps: List[Dict],
    seeds: List[int],
    base_resources_cfg: Dict,
    sim_cfg: Dict,
    service_cfg: Dict,
    rl_agent,
    reward_cfg: Dict,
    parent_regime: str,
    parent_results: Dict[str, Dict[str, Any]],
    cost_params: Dict[str, float],
    sla_targets: Dict[str, float],
    profile: Optional[Dict] = None,
    max_workers_by_stage: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Future-planning adaptive capacity search (spec §8): every candidate is first screened
    on replication #1 alone against the current (already 3-replication-validated) parent; if
    not competitive it is rejected without spending the extra replications. Only a promising
    candidate is validated on the remaining replications and re-aggregated before the
    accept/reject decision is finalised — so the search is never comparing a 3-replication
    parent against a 1-replication candidate as if they were equivalent (spec §7.5/§11).

    `parent_results` must already be the 3-replication aggregated per-policy result for
    `parent_regime` (see evaluate_regime_all_policies_multiseed) — the same aggregation
    granularity every accepted candidate is judged against and eventually replaces.

    `max_workers_by_stage` (spec §18): see run_adaptive_capacity_search — same per-stage
    ceiling convention, normally derived from that month's analytical capacity estimate.
    """
    profile = profile or load_planning_profile()
    ad = profile["adaptive_search"]
    max_extra = int(ad["max_extra_workers_per_stage"])
    max_iterations = int(ad["max_extra_iterations"])
    max_candidates = int(ad["max_candidates_per_iteration"])
    close_threshold = float(ad["close_bottleneck_threshold"])

    tested = {parent_regime}
    current_regime = parent_regime
    current_workers = _parse_regime(parent_regime)
    current_policy, current_best = pick_best_policy(parent_results)
    current_all_results = parent_results

    if max_workers_by_stage is None:
        max_workers_by_stage = {
            stage: current_workers[_STAGE_IDX[stage]] + max_extra for stage in STAGES
        }

    trail: List[Dict[str, Any]] = []
    stop_reason = "max adaptive iterations reached"
    simulations_executed = 0

    for iteration in range(1, max_iterations + 1):
        stage_rows = score_bottlenecks(current_best["metrics"]["stage_metrics"])
        candidates = _neighbour_candidates(
            current_workers, stage_rows, max_workers_by_stage, close_threshold, tested
        )[:max_candidates]

        if not candidates:
            stop_reason = "no untested valid neighbour remains (or max workers per stage reached)"
            break

        improved = False
        for stage, workers_tuple, label in candidates:
            tested.add(label)

            # Stage A — screen on replication #1 only, against the 3-replication parent.
            screen_results = evaluate_regime_all_policies(
                orders_by_replication[0], workers_tuple, base_resources_cfg, sim_cfg, service_cfg,
                service_time_maps[0], rl_agent, reward_cfg, seeds[0], cost_params, sla_targets,
            )
            simulations_executed += 3
            screen_policy, screen_best = pick_best_policy(screen_results)
            promising, screen_reason = _decide_accept(current_best, screen_best)

            if not promising:
                trail.append({
                    "iteration": iteration, "parent_regime": current_regime, "candidate_regime": label,
                    "added_stage": stage, "policy": screen_policy,
                    "evaluation_stage": "screening_rejected",
                    "labour_cost_increase": round(screen_best["labour_cost"] - current_best["labour_cost"], 2),
                    "late_penalty_reduction": round(current_best["late_cost"] - screen_best["late_cost"], 2),
                    "total_cost_diff": round(screen_best["total_cost"] - current_best["total_cost"], 2),
                    "urgent_sla_before": round(current_best["metrics"]["sla_urgent"], 4),
                    "urgent_sla_after": round(screen_best["metrics"]["sla_urgent"], 4),
                    "normal_sla_before": round(current_best["metrics"]["sla_normal"], 4),
                    "normal_sla_after": round(screen_best["metrics"]["sla_normal"], 4),
                    "overall_sla_before": round(current_best["metrics"]["sla_rate"], 4),
                    "overall_sla_after": round(screen_best["metrics"]["sla_rate"], 4),
                    "bottleneck_before": stage_rows[0]["stage"],
                    "accepted": False,
                    "reason": "Rejected during screening.",
                })
                continue

            # Stage B — promising: validate on the remaining replications and re-aggregate.
            aggregated = evaluate_regime_all_policies_multiseed(
                orders_by_replication, service_time_maps, seeds, workers_tuple,
                base_resources_cfg, sim_cfg, service_cfg, rl_agent, reward_cfg, cost_params, sla_targets,
            )
            simulations_executed += 3 * (len(orders_by_replication) - 1)
            policy, best = pick_best_policy(aggregated)
            accept, reason = _decide_accept(current_best, best)

            trail.append({
                "iteration": iteration, "parent_regime": current_regime, "candidate_regime": label,
                "added_stage": stage, "policy": policy,
                "evaluation_stage": "validated",
                "labour_cost_increase": round(best["labour_cost"] - current_best["labour_cost"], 2),
                "late_penalty_reduction": round(current_best["late_cost"] - best["late_cost"], 2),
                "total_cost_diff": round(best["total_cost"] - current_best["total_cost"], 2),
                "urgent_sla_before": round(current_best["metrics"]["sla_urgent"], 4),
                "urgent_sla_after": round(best["metrics"]["sla_urgent"], 4),
                "normal_sla_before": round(current_best["metrics"]["sla_normal"], 4),
                "normal_sla_after": round(best["metrics"]["sla_normal"], 4),
                "overall_sla_before": round(current_best["metrics"]["sla_rate"], 4),
                "overall_sla_after": round(best["metrics"]["sla_rate"], 4),
                "bottleneck_before": stage_rows[0]["stage"],
                "accepted": accept,
                "reason": reason,
            })

            if accept:
                current_regime, current_workers = label, workers_tuple
                current_policy, current_best = policy, best
                current_all_results = aggregated
                improved = True
                break  # re-rank bottlenecks from the new (validated) best before continuing

        if not improved:
            stop_reason = "no tested neighbour improved the objective"
            break
        if all(current_workers[_STAGE_IDX[s]] >= max_workers_by_stage[s] for s in STAGES):
            stop_reason = "maximum workers per stage reached"
            break

    return {
        "final_regime": current_regime,
        "final_policy": current_policy,
        "final_result": current_best,
        "final_results_by_policy": current_all_results,
        "stop_reason": stop_reason,
        "trail": trail,
        "iterations_run": len({t["iteration"] for t in trail}),
        "simulations_executed": simulations_executed,
    }
