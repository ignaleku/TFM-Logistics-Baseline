# src/rl/evaluate_rl3_monthly_capacity_cost.py
"""
Monthly workforce-capacity cost optimisation for RL-3.

Evaluates all months × all worker regimes × all policies to find the
configuration that minimises total estimated operating cost.

Total cost = SLA penalty cost + monthly labour cost.

Economic assumptions are configurable; all values used are stored in the output
rows for full reproducibility. The core evaluation loop is exposed as an importable
function (evaluate_monthly_capacity_cost) so both the historical CLI/API flow and the
future-planning API flow share exactly one implementation — see src/api/runners.py.

Usage:
    python -m src.rl.evaluate_rl3_monthly_capacity_cost
    python -m src.rl.evaluate_rl3_monthly_capacity_cost \\
        --cost-late-urgent 20 --cost-late-normal 5 \\
        --worker-cost-per-hour 15 --hours-per-worker-month 160
"""
from __future__ import annotations

import calendar
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import argparse

import numpy as np
import pandas as pd
import torch
import yaml

from src.analysis.bottleneck import STAGES, score_bottlenecks
from src.analysis.sla_feasibility import check_feasibility
from src.data.planning_profile import load_planning_profile
from src.rl.dqn_agent import QNetwork
from src.rl.env_fullstage_rl import FullStageRLRunner
from src.rl.replay_buffer import ReplayBuffer
from src.simulation.multistage.sim_multistage import run_simulation_multistage
from src.simulation.multistage.service_time_map import build_service_time_map
from src.simulation.multistage.operating_time import slice_month_operating_time, with_operating_horizon

# ---------------------------------------------------------------------------
# Economic assumptions — defaults; override at runtime via argparse / function args
# ---------------------------------------------------------------------------
COST_LATE_URGENT           = 20.0
COST_LATE_NORMAL           = 5.0
WORKER_COST_PER_HOUR       = 15.0
HOURS_PER_WORKER_PER_MONTH = 160.0

NAN = float("nan")

# (label, picking_workers, packing_workers, dispatch_workers) — single source of truth is
# configs/planning_profile.yaml::regimes.
REGIMES = [
    (label, workers[0], workers[1], workers[2])
    for label, workers in load_planning_profile()["regimes"].items()
]
REGIME_LOOKUP = {label: entry for entry in REGIMES for label in (entry[0],)}

_STAGE_METRIC_COLS = [
    "utilisation", "avg_wait_min", "p95_wait_min",
    "avg_queue_len", "max_queue_len", "late_wait_share", "pressure_score",
]

CSV_COLS = [
    "month", "month_name", "regime", "policy",
    "picking_workers", "packing_workers", "dispatch_workers", "total_workers",
    "total_orders", "urgent_orders", "normal_orders", "urgent_share",
    "total_sla", "urgent_sla", "normal_sla",
    "mean_system_time_min", "p90_system_time_min",
    "urgent_late_orders", "normal_late_orders",
    "completed_orders", "unfinished_orders", "unfinished_urgent_orders",
    "unfinished_normal_orders", "backlog_share",
    "estimated_late_cost", "estimated_worker_cost", "estimated_total_cost",
    "savings_total_cost_vs_fifo_same_month_regime",
    "savings_total_cost_vs_urgent_first_same_month_regime",
    "p_urgent_overall", "p_urgent_pick", "p_urgent_pack", "p_urgent_dispatch",
    "decisions_total", "decisions_pick", "decisions_pack", "decisions_dispatch",
    "cost_late_urgent", "cost_late_normal",
    "worker_cost_per_hour", "hours_per_worker_month",
    "feasible", "urgent_sla_target", "normal_sla_target", "sla_violation",
    "scenario_seed",
] + [f"{stage}_{col}" for stage in STAGES for col in _STAGE_METRIC_COLS]


# ── Agent ─────────────────────────────────────────────────────────────────────

class _GreedyAgent:
    def __init__(self, q_net: QNetwork, device: str) -> None:
        self.q = q_net
        self.device = device

    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.q(s).squeeze(0).argmax().item())


def load_rl3_agent(checkpoint_path: Path, rl_cfg: Dict, device: str = "cpu") -> _GreedyAgent:
    input_dim  = int(rl_cfg["network"].get("input_dim", 13))
    hidden_dim = int(rl_cfg["network"]["hidden_dim"])
    q_net = QNetwork(input_dim, hidden_dim, output_dim=2)
    q_net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    q_net.eval()
    return _GreedyAgent(q_net, device)


# ── Run helpers ───────────────────────────────────────────────────────────────

def _run_baseline(orders, policy, resources_cfg, sim_cfg, service_cfg, service_time_map):
    cfg = dict(sim_cfg)
    cfg["policy"] = policy
    _, summary = run_simulation_multistage(orders, cfg, resources_cfg, service_cfg, service_time_map=service_time_map)
    return summary


def _run_rl3(orders, agent, resources_cfg, sim_cfg, service_cfg, reward_cfg, seed, service_time_map):
    runner = FullStageRLRunner(
        sim_cfg=sim_cfg,
        resources_cfg=resources_cfg,
        service_cfg=service_cfg,
        seed=seed,
        reward_cfg=reward_cfg,
    )
    buf = ReplayBuffer(capacity=1)
    return runner.run_episode(
        orders=orders, agent=agent, buffer=buf, episode_seed=seed, greedy=True,
        service_time_map=service_time_map,
    )


def _compute_costs(urgent_cnt, normal_cnt, urgent_sla, normal_sla,
                   total_workers, cu, cn, wc, hpm):
    ul     = urgent_cnt * (1 - urgent_sla)
    nl     = normal_cnt * (1 - normal_sla)
    late   = ul * cu + nl * cn
    labour = total_workers * wc * hpm
    return late, labour, ul, nl


# ── Month parsing ─────────────────────────────────────────────────────────────

def parse_months(months_str: str) -> List[int]:
    """Return sorted list of month numbers (1-12) from a comma-separated string.

    Accepts integers (1-12), short names (Jan), and full names (January).
    Case-insensitive.
    """
    abbr_to_num = {v.lower(): k for k, v in enumerate(calendar.month_abbr) if k}
    name_to_num = {v.lower(): k for k, v in enumerate(calendar.month_name) if k}

    result: List[int] = []
    for token in months_str.split(","):
        token = token.strip()
        if not token:
            continue
        if token.isdigit():
            n = int(token)
            if not 1 <= n <= 12:
                raise ValueError(f"Month number out of range: {token!r}")
            result.append(n)
        elif token.lower() in abbr_to_num:
            result.append(abbr_to_num[token.lower()])
        elif token.lower() in name_to_num:
            result.append(name_to_num[token.lower()])
        else:
            raise ValueError(
                f"Unrecognised month: {token!r}. "
                "Use a number (1-12), short name (Jan), or full name (January)."
            )
    return sorted(set(result))


_parse_months = parse_months  # backward-compatible alias


# ── Core evaluation (importable — shared by CLI, historical API, future-planning API) ──

def evaluate_monthly_capacity_cost(
    orders_all: pd.DataFrame,
    checkpoint_path: Path,
    cost_late_urgent: float = COST_LATE_URGENT,
    cost_late_normal: float = COST_LATE_NORMAL,
    worker_cost_per_hour: float = WORKER_COST_PER_HOUR,
    hours_per_worker_month: float = HOURS_PER_WORKER_PER_MONTH,
    months: Optional[List[int]] = None,
    regime_names: Optional[List[str]] = None,
    regimes_by_month: Optional[Dict[int, Dict[str, Tuple[int, int, int]]]] = None,
    root: Optional[Path] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    seed_offset: int = 123,
) -> pd.DataFrame:
    """Evaluate month(s) × regime(s) × {fifo, urgent_first, rl3_dqn} and return the results
    DataFrame (same schema written to rl3_monthly_capacity_cost_results.csv).

    Regime resolution, in priority order:
      1. `regimes_by_month[month_num]` (label -> (picking, packing, dispatch)) if given — the
         business-planning path (spec §14/§17): dynamic candidates generated around that
         month's analytical capacity estimate, different per month.
      2. `regime_names` (static labels from configs/planning_profile.yaml::regimes) if given —
         kept for RL research / benchmark / generalisation diagnostics (spec §12).
      3. All 16 static base regimes (legacy default).

    `months` defaults to every month present in `orders_all`. Each month's orders are sliced
    and compressed onto that month's finite operating horizon (operating_time.py) before
    simulation — worker resources are only ever "on the clock" for hours_per_worker_month.
    `progress_cb(done, total)` is called after each simulation, if given.
    """
    root = root or Path(__file__).resolve().parents[2]

    with open(root / "configs" / "sim_multistage.yaml", encoding="utf-8") as f:
        sim_cfg_full = yaml.safe_load(f)
    with open(root / "configs" / "rl3.yaml", encoding="utf-8") as f:
        rl_cfg = yaml.safe_load(f)

    sim_cfg        = with_operating_horizon(sim_cfg_full["simulation"], hours_per_worker_month)
    base_resources = sim_cfg_full["resources"]
    service_cfg    = sim_cfg_full["service_time"]
    reward_cfg     = rl_cfg.get("reward", {})

    planning_profile   = load_planning_profile()
    sla_targets        = planning_profile["sla"]
    bottleneck_weights = planning_profile["bottleneck_score"]

    orders_all = orders_all.sort_values("arrival_time").reset_index(drop=True)
    rl3_agent = load_rl3_agent(Path(checkpoint_path), rl_cfg)

    all_months_in_data = sorted(orders_all["month"].unique())
    months = [m for m in all_months_in_data if m in set(months)] if months else all_months_in_data
    if not months:
        raise ValueError("No matching months found in the orders data.")

    def _regimes_for_month(month_num: int) -> List[Tuple[str, int, int, int]]:
        if regimes_by_month is not None:
            month_regimes = regimes_by_month.get(month_num, {})
            return [(label, w[0], w[1], w[2]) for label, w in month_regimes.items()]
        if regime_names:
            unknown = [r for r in regime_names if r not in REGIME_LOOKUP]
            if unknown:
                raise ValueError(f"Unknown regime(s): {unknown}. Available: {[r[0] for r in REGIMES]}")
            return [REGIME_LOOKUP[r] for r in regime_names]
        return REGIMES

    total_runs = sum(len(_regimes_for_month(m)) for m in months) * 3
    rows: List[Dict] = []
    done = 0

    cu, cn, wc, hpm = cost_late_urgent, cost_late_normal, worker_cost_per_hour, hours_per_worker_month
    horizon_minutes = sim_cfg["operating_horizon_minutes"]

    for month_num in months:
        active_regimes = _regimes_for_month(month_num)
        month_orders = slice_month_operating_time(orders_all, month_num, horizon_minutes)
        month_name = calendar.month_name[month_num]
        total_cnt = len(month_orders)
        urgent_cnt = int((month_orders["order_type"] == "urgent").sum())
        normal_cnt = int((month_orders["order_type"] == "normal").sum())
        urgent_share = urgent_cnt / total_cnt if total_cnt > 0 else 0.0
        seed = seed_offset + month_num

        # Common random numbers: one service-time map per month, shared across every regime
        # AND every policy (fifo / urgent_first / rl3_dqn) evaluated for that month — service
        # time doesn't depend on workforce size, only on the order's workload units, so this
        # is valid to reuse across the whole regime grid.
        service_time_map = build_service_time_map(month_orders, service_cfg, seed)

        for regime_name, n_pick, n_pack, n_disp in active_regimes:
            total_workers = n_pick + n_pack + n_disp
            worker_cost = total_workers * wc * hpm
            resources_cfg = {
                **base_resources,
                "picking_workers": n_pick, "packing_workers": n_pack, "dispatch_workers": n_disp,
            }

            policy_metrics: Dict = {}
            for policy in ("fifo", "urgent_first", "rl3_dqn"):
                try:
                    if policy == "rl3_dqn":
                        policy_metrics[policy] = _run_rl3(
                            month_orders, rl3_agent, resources_cfg, sim_cfg, service_cfg, reward_cfg, seed,
                            service_time_map,
                        )
                    else:
                        policy_metrics[policy] = _run_baseline(
                            month_orders, policy, resources_cfg, sim_cfg, service_cfg, service_time_map
                        )
                except Exception as e:
                    raise RuntimeError(
                        f"Simulation failed for month={month_name} regime={regime_name} "
                        f"policy={policy} workers=({n_pick},{n_pack},{n_disp}) orders={total_cnt}"
                    ) from e
                done += 1
                if progress_cb:
                    progress_cb(done, total_runs)

            total_costs: Dict = {}
            for policy, m in policy_metrics.items():
                late, _, ul, nl = _compute_costs(
                    urgent_cnt, normal_cnt, m["sla_urgent"], m["sla_normal"], total_workers, cu, cn, wc, hpm
                )
                total_costs[policy] = (late, worker_cost, late + worker_cost, ul, nl)

            fifo_total = total_costs["fifo"][2]
            uf_total = total_costs["urgent_first"][2]

            for policy, m in policy_metrics.items():
                is_rl = policy == "rl3_dqn"
                late, w_cost, total, ul, nl = total_costs[policy]

                feasible, sla_violation = check_feasibility(
                    m["sla_urgent"], m["sla_normal"], sla_targets["urgent_target"], sla_targets["normal_target"]
                )
                bottleneck_by_stage = {r["stage"]: r for r in score_bottlenecks(m["stage_metrics"], bottleneck_weights)}

                row = {
                    "month": month_num, "month_name": month_name, "regime": regime_name, "policy": policy,
                    "picking_workers": n_pick, "packing_workers": n_pack, "dispatch_workers": n_disp,
                    "total_workers": total_workers,
                    "total_orders": total_cnt, "urgent_orders": urgent_cnt, "normal_orders": normal_cnt,
                    "urgent_share": round(urgent_share, 4),
                    "total_sla": m["sla_rate"], "urgent_sla": m["sla_urgent"], "normal_sla": m["sla_normal"],
                    "mean_system_time_min": m.get("mean_system_min", NAN) if is_rl else m["mean_system_min"],
                    "p90_system_time_min":  m.get("p90_system_min",  NAN) if is_rl else m["p90_system_min"],
                    "urgent_late_orders": round(ul, 2), "normal_late_orders": round(nl, 2),
                    "completed_orders": int(m.get("completed_orders", total_cnt)),
                    "unfinished_orders": int(m.get("unfinished_orders", 0)),
                    "unfinished_urgent_orders": int(m.get("unfinished_urgent_orders", 0)),
                    "unfinished_normal_orders": int(m.get("unfinished_normal_orders", 0)),
                    "backlog_share": round(float(m.get("backlog_share", 0.0)), 4),
                    "estimated_late_cost": round(late, 2), "estimated_worker_cost": round(w_cost, 2),
                    "estimated_total_cost": round(total, 2),
                    "savings_total_cost_vs_fifo_same_month_regime": round(fifo_total - total, 2),
                    "savings_total_cost_vs_urgent_first_same_month_regime": round(uf_total - total, 2),
                    "p_urgent_overall":  m.get("p_urgent_decisions", NAN) if is_rl else NAN,
                    "p_urgent_pick":     m.get("pick_pct_urgent",    NAN) if is_rl else NAN,
                    "p_urgent_pack":     m.get("pack_pct_urgent",    NAN) if is_rl else NAN,
                    "p_urgent_dispatch": m.get("disp_pct_urgent",    NAN) if is_rl else NAN,
                    "decisions_total":    int(m.get("total_decisions", 0)) if is_rl else NAN,
                    "decisions_pick":     int(m.get("pick_dec_pts", 0))    if is_rl else NAN,
                    "decisions_pack":     int(m.get("pack_dec_pts", 0))    if is_rl else NAN,
                    "decisions_dispatch": int(m.get("disp_dec_pts", 0))    if is_rl else NAN,
                    "cost_late_urgent": cu, "cost_late_normal": cn,
                    "worker_cost_per_hour": wc, "hours_per_worker_month": hpm,
                    "feasible": feasible,
                    "urgent_sla_target": sla_targets["urgent_target"],
                    "normal_sla_target": sla_targets["normal_target"],
                    "sla_violation": sla_violation,
                    "scenario_seed": seed,
                }
                for stage in STAGES:
                    br = bottleneck_by_stage[stage]
                    row[f"{stage}_utilisation"] = br["utilisation"]
                    row[f"{stage}_avg_wait_min"] = br["avg_wait_min"]
                    row[f"{stage}_p95_wait_min"] = br["p95_wait_min"]
                    row[f"{stage}_avg_queue_len"] = br["avg_queue_len"]
                    row[f"{stage}_max_queue_len"] = br["max_queue_len"]
                    row[f"{stage}_late_wait_share"] = br["late_wait_share"]
                    row[f"{stage}_pressure_score"] = br["pressure_score"]
                rows.append(row)

    return pd.DataFrame(rows)[CSV_COLS]


# ── Interpretation (CLI display only) ───────────────────────────────────────────

def _fmt_row(row) -> str:
    return (
        f"  {calendar.month_abbr[int(row['month'])]:<4} "
        f"{row['regime']:<8} {row['policy']:<14} "
        f"workers={int(row['total_workers'])} "
        f"late=${row['estimated_late_cost']:>8,.0f}  "
        f"labour=${row['estimated_worker_cost']:>8,.0f}  "
        f"total=${row['estimated_total_cost']:>8,.0f}  "
        f"SLA={row['total_sla']:.4f}  U={row['urgent_sla']:.4f}"
    )


def _print_interpretation(df: pd.DataFrame) -> None:
    months = sorted(df["month"].unique())

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    print("\n  1. Best configuration by total estimated cost (per month):\n")
    best_total = df.loc[df.groupby("month")["estimated_total_cost"].idxmin()]
    rl3_best_months = []
    for _, row in best_total.sort_values("month").iterrows():
        if row["policy"] == "rl3_dqn":
            rl3_best_months.append(calendar.month_abbr[int(row["month"])])
        print(_fmt_row(row))

    print("\n  2. Minimum-worker config reaching urgent_sla >= 0.95 (per month):\n")
    for m in months:
        sub = df[(df["month"] == m) & (df["urgent_sla"] >= 0.95)]
        if sub.empty:
            print(f"  {calendar.month_abbr[m]:<4}  none reach urgent_sla >= 0.95")
        else:
            row = sub.loc[sub["total_workers"].idxmin()]
            print(_fmt_row(row))

    print("\n  3. Minimum-worker config reaching total_sla >= 0.80 (per month):\n")
    for m in months:
        sub = df[(df["month"] == m) & (df["total_sla"] >= 0.80)]
        if sub.empty:
            print(f"  {calendar.month_abbr[m]:<4}  none reach total_sla >= 0.80")
        else:
            row = sub.loc[sub["total_workers"].idxmin()]
            print(_fmt_row(row))

    print(f"\n  4. RL-3 selected as best total-cost policy:")
    if rl3_best_months:
        print(f"     YES — in months: {', '.join(rl3_best_months)}")
    else:
        print(f"     NO — RL-3 never has the lowest total cost across all month × regime combinations")

    rl3_best_policy = []
    for (m, reg), grp in df.groupby(["month", "regime"]):
        best_policy = grp.loc[grp["estimated_total_cost"].idxmin(), "policy"]
        if best_policy == "rl3_dqn":
            rl3_best_policy.append(f"{calendar.month_abbr[m]}/{reg}")
    if rl3_best_policy:
        print(f"     RL-3 is cheapest policy within its regime in: {', '.join(rl3_best_policy)}")

    print()


# ── CLI entry point ──────────────────────────────────────────────────────────

def main() -> None:
    root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description="Monthly workforce-capacity cost optimisation for RL-3"
    )
    parser.add_argument("--orders", default="data/orders_base.csv")
    parser.add_argument("--checkpoint", default="data/dqn_rl3_final.pt")
    parser.add_argument("--output", default="data/rl3_monthly_capacity_cost_results.csv")
    parser.add_argument("--cost-late-urgent",       type=float, default=COST_LATE_URGENT)
    parser.add_argument("--cost-late-normal",        type=float, default=COST_LATE_NORMAL)
    parser.add_argument("--worker-cost-per-hour",   type=float, default=WORKER_COST_PER_HOUR)
    parser.add_argument("--hours-per-worker-month", type=float, default=HOURS_PER_WORKER_PER_MONTH)
    parser.add_argument(
        "--months", default=None,
        help="Comma-separated months to evaluate (e.g. Jan,Feb or 1,2,12 or January). "
             "Omit to evaluate all months present in the orders file.",
    )
    parser.add_argument(
        "--regimes", default=None,
        help="Comma-separated regime names to evaluate (e.g. s321 or s222,s321). "
             "Omit to evaluate all regimes.",
    )
    args = parser.parse_args()

    orders_all = pd.read_csv(root / args.orders, parse_dates=["arrival_time"])

    months = parse_months(args.months) if args.months else None
    regime_names = [r.strip() for r in args.regimes.split(",") if r.strip()] if args.regimes else None

    if months:
        available = set(orders_all["month"].unique())
        missing = set(months) - available
        if missing:
            print(f"Warning: requested months not found in data: {[calendar.month_name[m] for m in sorted(missing)]}")

    print(f"Economic assumptions:")
    print(f"  --cost-late-urgent        = {args.cost_late_urgent}")
    print(f"  --cost-late-normal        = {args.cost_late_normal}")
    print(f"  --worker-cost-per-hour    = {args.worker_cost_per_hour}")
    print(f"  --hours-per-worker-month  = {args.hours_per_worker_month}")
    print(f"Checkpoint : {root / args.checkpoint}")

    def _progress(done: int, total: int) -> None:
        if done % 10 == 0 or done == total:
            print(f"[{done:>4}/{total}] simulations complete")

    df = evaluate_monthly_capacity_cost(
        orders_all,
        checkpoint_path=root / args.checkpoint,
        cost_late_urgent=args.cost_late_urgent,
        cost_late_normal=args.cost_late_normal,
        worker_cost_per_hour=args.worker_cost_per_hour,
        hours_per_worker_month=args.hours_per_worker_month,
        months=months,
        regime_names=regime_names,
        root=root,
        progress_cb=_progress,
    )

    out_path = root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  ({len(df)} rows)")

    _print_interpretation(df)


if __name__ == "__main__":
    main()
