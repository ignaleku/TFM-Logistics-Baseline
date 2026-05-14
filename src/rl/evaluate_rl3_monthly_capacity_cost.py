# src/rl/evaluate_rl3_monthly_capacity_cost.py
"""
Monthly workforce-capacity cost optimisation for RL-3.

Evaluates all months × all worker regimes × all policies to find the
configuration that minimises total estimated operating cost.

Total cost = SLA penalty cost + monthly labour cost.

Economic assumptions are configurable via argparse; all values used are
stored in the output CSV for full reproducibility.

Usage:
    python -m src.rl.evaluate_rl3_monthly_capacity_cost
    python -m src.rl.evaluate_rl3_monthly_capacity_cost \\
        --cost-late-urgent 20 --cost-late-normal 5 \\
        --worker-cost-per-hour 15 --hours-per-worker-month 160
"""
from __future__ import annotations

import calendar
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import torch
import yaml

from src.rl.dqn_agent import QNetwork
from src.rl.env_fullstage_rl import FullStageRLRunner
from src.rl.replay_buffer import ReplayBuffer
from src.simulation.multistage.sim_multistage import run_simulation_multistage

# ---------------------------------------------------------------------------
# Economic assumptions — defaults; override at runtime via argparse
# ---------------------------------------------------------------------------
COST_LATE_URGENT           = 20.0
COST_LATE_NORMAL           = 5.0
WORKER_COST_PER_HOUR       = 15.0
HOURS_PER_WORKER_PER_MONTH = 160.0

NAN = float("nan")

# (label, picking_workers, packing_workers, dispatch_workers)
REGIMES = [
    ("s111", 1, 1, 1),   # minimal
    ("s211", 2, 1, 1),   # picking-focused low capacity
    ("s121", 1, 2, 1),   # packing-focused low capacity
    ("s112", 1, 1, 2),   # dispatch-focused low capacity
    ("s221", 2, 2, 1),   # picking + packing
    ("s212", 2, 1, 2),   # picking + dispatch
    ("s122", 1, 2, 2),   # packing + dispatch
    ("s311", 3, 1, 1),   # strong picking, limited downstream
    ("s231", 2, 3, 1),   # strong packing
    ("s312", 3, 1, 2),   # strong picking + dispatch
    ("s222", 2, 2, 2),   # balanced medium capacity
    # Peak-capacity candidates
    ("s321", 3, 2, 1),   # high picking + packing, limited dispatch
    ("s322", 3, 2, 2),   # high picking, balanced downstream
    ("s331", 3, 3, 1),   # strong picking + packing, dispatch limited
    ("s332", 3, 3, 2),   # strong all-round, heavy packing
    ("s432", 4, 3, 2),   # maximum throughput
]

CSV_COLS = [
    "month", "month_name", "regime", "policy",
    "picking_workers", "packing_workers", "dispatch_workers", "total_workers",
    "total_orders", "urgent_orders", "normal_orders", "urgent_share",
    "total_sla", "urgent_sla", "normal_sla",
    "mean_system_time_min", "p90_system_time_min",
    "urgent_late_orders", "normal_late_orders",
    "estimated_late_cost", "estimated_worker_cost", "estimated_total_cost",
    "savings_total_cost_vs_fifo_same_month_regime",
    "savings_total_cost_vs_urgent_first_same_month_regime",
    "p_urgent_overall", "p_urgent_pick", "p_urgent_pack", "p_urgent_dispatch",
    "decisions_total", "decisions_pick", "decisions_pack", "decisions_dispatch",
    "cost_late_urgent", "cost_late_normal",
    "worker_cost_per_hour", "hours_per_worker_month",
]


# ── Agent ─────────────────────────────────────────────────────────────────────

class _GreedyAgent:
    def __init__(self, q_net: QNetwork, device: str) -> None:
        self.q = q_net
        self.device = device

    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.q(s).squeeze(0).argmax().item())


# ── Run helpers ───────────────────────────────────────────────────────────────

def _run_baseline(orders, policy, resources_cfg, sim_cfg, service_cfg):
    cfg = dict(sim_cfg)
    cfg["policy"] = policy
    _, summary = run_simulation_multistage(orders, cfg, resources_cfg, service_cfg)
    return summary


def _run_rl3(orders, agent, resources_cfg, sim_cfg, service_cfg, reward_cfg, seed):
    runner = FullStageRLRunner(
        sim_cfg=sim_cfg,
        resources_cfg=resources_cfg,
        service_cfg=service_cfg,
        seed=seed,
        reward_cfg=reward_cfg,
    )
    buf = ReplayBuffer(capacity=1)
    return runner.run_episode(
        orders=orders, agent=agent, buffer=buf, episode_seed=seed, greedy=True
    )


def _compute_costs(urgent_cnt, normal_cnt, urgent_sla, normal_sla,
                   total_workers, cu, cn, wc, hpm):
    ul     = urgent_cnt * (1 - urgent_sla)
    nl     = normal_cnt * (1 - normal_sla)
    late   = ul * cu + nl * cn
    labour = total_workers * wc * hpm
    return late, labour, ul, nl


# ── Interpretation ────────────────────────────────────────────────────────────

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


# ── Month parsing ─────────────────────────────────────────────────────────────

def _parse_months(months_str: str) -> list[int]:
    """Return sorted list of month numbers (1-12) from a comma-separated string.

    Accepts integers (1-12), short names (Jan), and full names (January).
    Case-insensitive.
    """
    abbr_to_num = {v.lower(): k for k, v in enumerate(calendar.month_abbr) if k}
    name_to_num = {v.lower(): k for k, v in enumerate(calendar.month_name) if k}

    result: list[int] = []
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


# ── Main ──────────────────────────────────────────────────────────────────────

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

    cu  = args.cost_late_urgent
    cn  = args.cost_late_normal
    wc  = args.worker_cost_per_hour
    hpm = args.hours_per_worker_month

    with open(root / "configs" / "sim_multistage.yaml", encoding="utf-8") as f:
        sim_cfg_full = yaml.safe_load(f)
    with open(root / "configs" / "rl3.yaml", encoding="utf-8") as f:
        rl_cfg = yaml.safe_load(f)

    sim_cfg        = sim_cfg_full["simulation"]
    base_resources = sim_cfg_full["resources"]
    service_cfg    = sim_cfg_full["service_time"]
    reward_cfg     = rl_cfg.get("reward", {})

    orders_all = (
        pd.read_csv(root / args.orders, parse_dates=["arrival_time"])
        .sort_values("arrival_time")
        .reset_index(drop=True)
    )

    ckpt_path  = root / args.checkpoint
    device     = "cpu"
    input_dim  = int(rl_cfg["network"].get("input_dim", 13))
    hidden_dim = int(rl_cfg["network"]["hidden_dim"])
    q_net      = QNetwork(input_dim, hidden_dim, output_dim=2)
    q_net.load_state_dict(torch.load(ckpt_path, map_location=device))
    q_net.eval()
    rl3_agent = _GreedyAgent(q_net, device)

    all_months_in_data = sorted(orders_all["month"].unique())

    if args.months:
        requested = set(_parse_months(args.months))
        months = [m for m in all_months_in_data if m in requested]
        missing = requested - set(months)
        if missing:
            missing_names = [calendar.month_name[m] for m in sorted(missing)]
            print(f"Warning: requested months not found in data: {missing_names}")
        if not months:
            raise SystemExit("No matching months found in the orders data. Aborting.")
    else:
        months = all_months_in_data

    regime_lookup = {label: entry for entry in REGIMES for label in (entry[0],)}
    if args.regimes:
        requested_regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
        unknown = [r for r in requested_regimes if r not in regime_lookup]
        if unknown:
            available = [label for label, *_ in REGIMES]
            raise ValueError(
                f"Unknown regime(s): {unknown}. "
                f"Available regimes: {available}"
            )
        active_regimes = [regime_lookup[r] for r in requested_regimes]
    else:
        active_regimes = REGIMES

    total_runs = len(months) * len(active_regimes) * 3

    print(f"Economic assumptions:")
    print(f"  --cost-late-urgent        = {cu}")
    print(f"  --cost-late-normal        = {cn}")
    print(f"  --worker-cost-per-hour    = {wc}")
    print(f"  --hours-per-worker-month  = {hpm}")
    print(f"Checkpoint : {ckpt_path}")
    if args.months:
        print(f"Months     : {[calendar.month_abbr[m] for m in months]}  (filtered from --months {args.months!r})")
    else:
        print(f"Months     : {[calendar.month_abbr[m] for m in months]}  (all months in data)")
    if args.regimes:
        print(f"Regimes    : {[r[0] for r in active_regimes]}  (filtered from --regimes {args.regimes!r})")
    else:
        print(f"Regimes    : {[r[0] for r in active_regimes]}  (all regimes)")
    print(f"Total runs : {total_runs}  ({len(months)} months × {len(active_regimes)} regimes × 3 policies)\n")

    rows = []
    done = 0
    sim_num = 0

    for month_num in months:
        month_orders = (
            orders_all[orders_all["month"] == month_num]
            .sort_values("arrival_time")
            .reset_index(drop=True)
        )
        month_name   = calendar.month_name[month_num]
        month_abbr   = calendar.month_abbr[month_num]
        total_cnt    = len(month_orders)
        urgent_cnt   = int((month_orders["order_type"] == "urgent").sum())
        normal_cnt   = int((month_orders["order_type"] == "normal").sum())
        urgent_share = urgent_cnt / total_cnt if total_cnt > 0 else 0.0
        seed         = 123 + month_num

        for regime_name, n_pick, n_pack, n_disp in active_regimes:
            total_workers = n_pick + n_pack + n_disp
            worker_cost   = total_workers * wc * hpm
            workers_tuple = (n_pick, n_pack, n_disp)

            resources_cfg = {
                **base_resources,
                "picking_workers":  n_pick,
                "packing_workers":  n_pack,
                "dispatch_workers": n_disp,
            }

            policy_metrics: dict = {}
            for policy in ("fifo", "urgent_first", "rl3_dqn"):
                sim_num += 1
                print(
                    f"[RUN {sim_num}/{total_runs}] month={month_abbr} regime={regime_name} "
                    f"policy={policy} workers={workers_tuple} orders={total_cnt}"
                )
                try:
                    if policy == "rl3_dqn":
                        policy_metrics[policy] = _run_rl3(
                            month_orders, rl3_agent, resources_cfg, sim_cfg, service_cfg, reward_cfg, seed
                        )
                    else:
                        policy_metrics[policy] = _run_baseline(
                            month_orders, policy, resources_cfg, sim_cfg, service_cfg
                        )
                except Exception as e:
                    raise RuntimeError(
                        f"Simulation failed for month={month_abbr} regime={regime_name} "
                        f"policy={policy} workers={workers_tuple} orders={total_cnt}"
                    ) from e

            total_costs: dict = {}
            for policy, m in policy_metrics.items():
                late, _, ul, nl = _compute_costs(
                    urgent_cnt, normal_cnt, m["sla_urgent"], m["sla_normal"],
                    total_workers, cu, cn, wc, hpm
                )
                total_costs[policy] = (late, worker_cost, late + worker_cost, ul, nl)

            fifo_total = total_costs["fifo"][2]
            uf_total   = total_costs["urgent_first"][2]

            for policy, m in policy_metrics.items():
                is_rl = policy == "rl3_dqn"
                late, w_cost, total, ul, nl = total_costs[policy]

                row = {
                    "month":        month_num,
                    "month_name":   month_name,
                    "regime":       regime_name,
                    "policy":       policy,
                    "picking_workers":  n_pick,
                    "packing_workers":  n_pack,
                    "dispatch_workers": n_disp,
                    "total_workers": total_workers,
                    "total_orders":  total_cnt,
                    "urgent_orders": urgent_cnt,
                    "normal_orders": normal_cnt,
                    "urgent_share":  round(urgent_share, 4),
                    "total_sla":     m["sla_rate"],
                    "urgent_sla":    m["sla_urgent"],
                    "normal_sla":    m["sla_normal"],
                    "mean_system_time_min": m.get("mean_system_min", NAN) if is_rl else m["mean_system_min"],
                    "p90_system_time_min":  m.get("p90_system_min",  NAN) if is_rl else m["p90_system_min"],
                    "urgent_late_orders":      round(ul, 2),
                    "normal_late_orders":      round(nl, 2),
                    "estimated_late_cost":     round(late, 2),
                    "estimated_worker_cost":   round(w_cost, 2),
                    "estimated_total_cost":    round(total, 2),
                    "savings_total_cost_vs_fifo_same_month_regime":
                        round(fifo_total - total, 2),
                    "savings_total_cost_vs_urgent_first_same_month_regime":
                        round(uf_total - total, 2),
                    "p_urgent_overall":  m.get("p_urgent_decisions", NAN) if is_rl else NAN,
                    "p_urgent_pick":     m.get("pick_pct_urgent",    NAN) if is_rl else NAN,
                    "p_urgent_pack":     m.get("pack_pct_urgent",    NAN) if is_rl else NAN,
                    "p_urgent_dispatch": m.get("disp_pct_urgent",    NAN) if is_rl else NAN,
                    "decisions_total":   int(m.get("total_decisions", 0))  if is_rl else NAN,
                    "decisions_pick":    int(m.get("pick_dec_pts",   0))   if is_rl else NAN,
                    "decisions_pack":    int(m.get("pack_dec_pts",   0))   if is_rl else NAN,
                    "decisions_dispatch":int(m.get("disp_dec_pts",   0))   if is_rl else NAN,
                    "cost_late_urgent":       cu,
                    "cost_late_normal":       cn,
                    "worker_cost_per_hour":   wc,
                    "hours_per_worker_month": hpm,
                }
                rows.append(row)
                done += 1
                print(
                    f"[{done:>3}/{total_runs}] {month_abbr:<4} "
                    f"{regime_name}  {policy:<14}  "
                    f"W={total_workers}  SLA={m['sla_rate']:.4f}  "
                    f"U={m['sla_urgent']:.4f}  late=${late:,.0f}  total=${total:,.0f}"
                )

    df = pd.DataFrame(rows)[CSV_COLS]
    out_path = root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  ({len(df)} rows)")

    _print_interpretation(df)


if __name__ == "__main__":
    main()
