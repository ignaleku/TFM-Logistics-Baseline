# src/rl/evaluate_rl5_monthly_capacity_cost.py
"""
Monthly workforce-capacity cost optimisation for RL-5.

Evaluates all months × all worker regimes × all policies (432 simulations)
to find the configuration that minimises total estimated operating cost.

Total cost = SLA penalty cost + monthly labour cost.

Usage:
    python -m src.rl.evaluate_rl5_monthly_capacity_cost
    python -m src.rl.evaluate_rl5_monthly_capacity_cost --checkpoint data/dqn_rl5_v2_final.pt
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
from src.rl.env_5stage_rl import FiveStageRLRunner
from src.rl.replay_buffer import ReplayBuffer
from src.simulation.multistage.sim_5stage import run_simulation_5stage

# ---------------------------------------------------------------------------
# Economic assumptions — edit these to recalculate
# ---------------------------------------------------------------------------
COST_LATE_URGENT         = 20.0
COST_LATE_NORMAL         = 5.0
WORKER_COST_PER_HOUR     = 15.0
HOURS_PER_WORKER_PER_MONTH = 160.0

NAN = float("nan")

# (name, pick, qc, pack, lab, disp)
REGIMES = [
    ("s11111", 1, 1, 1, 1, 1),
    ("s21111", 2, 1, 1, 1, 1),
    ("s31111", 3, 1, 1, 1, 1),
    ("s32111", 3, 2, 1, 1, 1),
    ("s32121", 3, 2, 1, 2, 1),   # reinforced labelling
    ("s32112", 3, 2, 1, 1, 2),   # reinforced dispatch
    ("s32211", 3, 2, 2, 1, 1),
    ("s32212", 3, 2, 2, 1, 2),   # reinforced dispatch after packing
    ("s32221", 3, 2, 2, 2, 1),
    ("s33211", 3, 3, 2, 1, 1),   # reinforced QC
    ("s42211", 4, 2, 2, 1, 1),   # extra picking
    ("s33322", 3, 3, 3, 2, 2),
]

CSV_COLS = [
    "month", "month_name", "regime", "policy",
    "picking_workers", "quality_check_workers", "packing_workers",
    "labelling_workers", "dispatch_workers", "total_workers",
    "total_orders", "urgent_orders", "normal_orders", "urgent_share",
    "total_sla", "urgent_sla", "normal_sla",
    "mean_system_time_min", "p90_system_time_min",
    "urgent_late_orders", "normal_late_orders",
    "estimated_late_cost", "estimated_worker_cost", "estimated_total_cost",
    "savings_total_cost_vs_fifo_same_month_regime",
    "savings_total_cost_vs_urgent_first_same_month_regime",
    "p_urgent_overall", "p_urgent_pick", "p_urgent_quality_check",
    "p_urgent_pack", "p_urgent_labelling", "p_urgent_dispatch",
    "decisions_total", "decisions_pick", "decisions_quality_check",
    "decisions_pack", "decisions_labelling", "decisions_dispatch",
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
    _, summary = run_simulation_5stage(orders, cfg, resources_cfg, service_cfg)
    return summary


def _run_rl5(orders, agent, resources_cfg, sim_cfg, service_cfg, reward_cfg, seed):
    runner = FiveStageRLRunner(
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


def _compute_costs(urgent_cnt, normal_cnt, urgent_sla, normal_sla, total_workers):
    ul      = urgent_cnt * (1 - urgent_sla)
    nl      = normal_cnt * (1 - normal_sla)
    late    = ul * COST_LATE_URGENT + nl * COST_LATE_NORMAL
    labour  = total_workers * WORKER_COST_PER_HOUR * HOURS_PER_WORKER_PER_MONTH
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

    # ── 1. Best by total cost per month ───────────────────────────────────────
    print("\n  1. Best configuration by total estimated cost (per month):\n")
    best_total = df.loc[df.groupby("month")["estimated_total_cost"].idxmin()]
    rl5_best_months_total = []
    for _, row in best_total.sort_values("month").iterrows():
        if row["policy"] == "rl5_dqn":
            rl5_best_months_total.append(calendar.month_abbr[int(row["month"])])
        print(_fmt_row(row))

    # ── 2. Best by SLA penalty cost only (ignoring labour) ───────────────────
    print("\n  2. Best configuration by SLA penalty cost only (labour ignored):\n")
    best_late = df.loc[df.groupby("month")["estimated_late_cost"].idxmin()]
    for _, row in best_late.sort_values("month").iterrows():
        print(_fmt_row(row))

    # ── 3. Min workers with urgent_sla >= 0.95 ───────────────────────────────
    print("\n  3. Minimum-worker config reaching urgent_sla >= 0.95 (per month):\n")
    for m in months:
        sub = df[(df["month"] == m) & (df["urgent_sla"] >= 0.95)]
        if sub.empty:
            print(f"  {calendar.month_abbr[m]:<4}  none reach urgent_sla >= 0.95")
        else:
            row = sub.loc[sub["total_workers"].idxmin()]
            print(_fmt_row(row))

    # ── 4. Min workers with total_sla >= 0.80 ────────────────────────────────
    print("\n  4. Minimum-worker config reaching total_sla >= 0.80 (per month):\n")
    for m in months:
        sub = df[(df["month"] == m) & (df["total_sla"] >= 0.80)]
        if sub.empty:
            print(f"  {calendar.month_abbr[m]:<4}  none reach total_sla >= 0.80")
        else:
            row = sub.loc[sub["total_workers"].idxmin()]
            print(_fmt_row(row))

    # ── 5. Months where hiring extra workers is economically justified ────────
    print("\n  5. Months where hiring extra workers reduces total cost:\n")
    LOW_WORKERS  = set(r for r, *w in REGIMES if sum(w) <= 7)   # s11111, s21111, s31111
    HIGH_WORKERS = set(r for r, *w in REGIMES if sum(w) > 7)    # s32111 and above
    for m in months:
        sub = df[df["month"] == m]
        low_best  = sub[sub["regime"].isin(LOW_WORKERS)]["estimated_total_cost"].min()
        high_best = sub[sub["regime"].isin(HIGH_WORKERS)]["estimated_total_cost"].min()
        justified = high_best < low_best
        saving    = low_best - high_best
        verdict   = f"YES  (saves ${saving:,.0f})" if justified else f"no   (extra cost ${-saving:,.0f})"
        print(f"  {calendar.month_abbr[m]:<4}  low-worker best=${low_best:,.0f}  "
              f"high-worker best=${high_best:,.0f}  → {verdict}")

    # ── 6. RL-5 selected as best total-cost policy ────────────────────────────
    print(f"\n  6. RL-5 selected as best total-cost policy:")
    if rl5_best_months_total:
        print(f"     YES — in months: {', '.join(rl5_best_months_total)}")
    else:
        print(f"     NO — RL-5 never has the lowest total cost across all month × regime combinations")

    # Check if rl5 is best *policy* within any (month, regime) group
    rl5_best_policy = []
    for (m, reg), grp in df.groupby(["month", "regime"]):
        best_policy = grp.loc[grp["estimated_total_cost"].idxmin(), "policy"]
        if best_policy == "rl5_dqn":
            rl5_best_policy.append(f"{calendar.month_abbr[m]}/{reg}")
    if rl5_best_policy:
        print(f"     RL-5 is cheapest policy within its regime in: {', '.join(rl5_best_policy)}")

    # ── 7. urgent_first dominance ─────────────────────────────────────────────
    print(f"\n  7. urgent_first vs RL-5 — when urgent penalties dominate:")
    uf_beats_rl5 = []
    rl5_beats_uf = []
    for (m, reg), grp in df.groupby(["month", "regime"]):
        uf_row  = grp[grp["policy"] == "urgent_first"]
        rl5_row = grp[grp["policy"] == "rl5_dqn"]
        if uf_row.empty or rl5_row.empty:
            continue
        uf_cost  = uf_row["estimated_total_cost"].iloc[0]
        rl5_cost = rl5_row["estimated_total_cost"].iloc[0]
        if uf_cost < rl5_cost:
            uf_beats_rl5.append(f"{calendar.month_abbr[m]}/{reg}")
        else:
            rl5_beats_uf.append(f"{calendar.month_abbr[m]}/{reg}")
    print(f"     urgent_first cheaper: {len(uf_beats_rl5)}/{len(uf_beats_rl5)+len(rl5_beats_uf)} regimes")
    print(f"     RL-5 cheaper or equal: {len(rl5_beats_uf)}/{len(uf_beats_rl5)+len(rl5_beats_uf)} regimes")

    # ── 8. Short interpretation ───────────────────────────────────────────────
    print("\n  8. Capacity planning interpretation:")
    print()
    print("     • Labour cost is fixed per regime regardless of demand volume.")
    print("       In low-demand months, adding workers may cost more than the SLA")
    print("       penalties they avoid — fewer workers can be optimal.")
    print()
    print("     • In high-demand months, SLA penalties accumulate faster.")
    print("       Extra workers reduce late orders and can lower total cost,")
    print("       making larger regimes economically justified.")
    print()
    print("     • Policy and workforce planning interact: RL-5 may close the gap")
    print("       between a smaller regime and a larger one by prioritising smarter,")
    print("       potentially deferring the need for additional headcount.")
    print()
    uf_dom = len(uf_beats_rl5) > len(rl5_beats_uf)
    if uf_dom:
        print("     • urgent_first tends to outperform RL-5 on total cost in this dataset.")
        print("       With COST_LATE_URGENT=20 and a minority urgent share, the greedy")
        print("       priority rule captures most of the urgent penalty savings.")
    else:
        print("     • RL-5 tends to match or beat urgent_first on total cost,")
        print("       suggesting the learned policy adds value beyond simple priority rules.")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    root = Path(__file__).resolve().parents[2]
    v2_ckpt = root / "data" / "dqn_rl5_v2_final.pt"
    default_ckpt = "data/dqn_rl5_v2_final.pt" if v2_ckpt.exists() else "data/dqn_rl5_final.pt"

    parser = argparse.ArgumentParser(
        description="Monthly workforce-capacity cost optimisation for RL-5"
    )
    parser.add_argument("--checkpoint", default=default_ckpt)
    parser.add_argument("--output", default="data/rl5_monthly_capacity_cost_results.csv")
    args = parser.parse_args()

    # Load configs
    with open(root / "configs" / "sim_5stage.yaml", encoding="utf-8") as f:
        sim_cfg_full = yaml.safe_load(f)

    rl_cfg_path = root / "configs" / "rl5_v2.yaml"
    if not rl_cfg_path.exists():
        rl_cfg_path = root / "configs" / "rl5.yaml"
    with open(rl_cfg_path, encoding="utf-8") as f:
        rl_cfg = yaml.safe_load(f)

    sim_cfg        = sim_cfg_full["simulation"]
    base_resources = sim_cfg_full["resources"]
    service_cfg    = sim_cfg_full["service_time"]
    reward_cfg     = rl_cfg.get("reward", {})

    # Load orders
    orders_all = (
        pd.read_csv(root / "data" / "orders_base.csv", parse_dates=["arrival_time"])
        .sort_values("arrival_time")
        .reset_index(drop=True)
    )

    # Load RL-5 agent
    ckpt_path  = root / args.checkpoint
    device     = "cpu"
    input_dim  = int(rl_cfg["network"].get("input_dim", 19))
    hidden_dim = int(rl_cfg["network"]["hidden_dim"])
    q_net      = QNetwork(input_dim, hidden_dim, output_dim=2)
    q_net.load_state_dict(torch.load(ckpt_path, map_location=device))
    q_net.eval()
    rl5_agent = _GreedyAgent(q_net, device)

    months     = sorted(orders_all["month"].unique())
    total_runs = len(months) * len(REGIMES) * 3  # 432

    print(f"Economic assumptions:")
    print(f"  COST_LATE_URGENT          = {COST_LATE_URGENT}")
    print(f"  COST_LATE_NORMAL          = {COST_LATE_NORMAL}")
    print(f"  WORKER_COST_PER_HOUR      = {WORKER_COST_PER_HOUR}")
    print(f"  HOURS_PER_WORKER_PER_MONTH= {HOURS_PER_WORKER_PER_MONTH}")
    print(f"Checkpoint : {ckpt_path}")
    print(f"Config     : {rl_cfg_path.name}")
    print(f"Months     : {[calendar.month_abbr[m] for m in months]}")
    print(f"Total runs : {total_runs}  ({len(months)} months × {len(REGIMES)} regimes × 3 policies)\n")

    rows = []
    done = 0

    for month_num in months:
        month_orders = (
            orders_all[orders_all["month"] == month_num]
            .sort_values("arrival_time")
            .reset_index(drop=True)
        )
        month_name   = calendar.month_name[month_num]
        total_cnt    = len(month_orders)
        urgent_cnt   = int((month_orders["order_type"] == "urgent").sum())
        normal_cnt   = int((month_orders["order_type"] == "normal").sum())
        urgent_share = urgent_cnt / total_cnt if total_cnt > 0 else 0.0
        seed         = 123 + month_num

        for regime_name, n_pick, n_qc, n_pack, n_lab, n_disp in REGIMES:
            total_workers = n_pick + n_qc + n_pack + n_lab + n_disp
            worker_cost   = total_workers * WORKER_COST_PER_HOUR * HOURS_PER_WORKER_PER_MONTH

            resources_cfg = {
                **base_resources,
                "picking_workers":       n_pick,
                "quality_check_workers": n_qc,
                "packing_workers":       n_pack,
                "labelling_workers":     n_lab,
                "dispatch_workers":      n_disp,
            }

            # Run all three policies
            policy_metrics: dict = {}
            for policy in ("fifo", "urgent_first"):
                policy_metrics[policy] = _run_baseline(
                    month_orders, policy, resources_cfg, sim_cfg, service_cfg
                )
            policy_metrics["rl5_dqn"] = _run_rl5(
                month_orders, rl5_agent, resources_cfg, sim_cfg, service_cfg, reward_cfg, seed
            )

            # Compute costs per policy
            total_costs: dict = {}
            for policy, m in policy_metrics.items():
                late, _, ul, nl = _compute_costs(
                    urgent_cnt, normal_cnt, m["sla_urgent"], m["sla_normal"], total_workers
                )
                total_costs[policy] = (late, worker_cost, late + worker_cost, ul, nl)

            fifo_total = total_costs["fifo"][2]
            uf_total   = total_costs["urgent_first"][2]

            for policy, m in policy_metrics.items():
                is_rl = policy == "rl5_dqn"
                late, w_cost, total, ul, nl = total_costs[policy]

                row = {
                    "month":        month_num,
                    "month_name":   month_name,
                    "regime":       regime_name,
                    "policy":       policy,
                    "picking_workers":       n_pick,
                    "quality_check_workers": n_qc,
                    "packing_workers":       n_pack,
                    "labelling_workers":     n_lab,
                    "dispatch_workers":      n_disp,
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
                    "p_urgent_overall":        m.get("p_urgent_decisions", NAN) if is_rl else NAN,
                    "p_urgent_pick":           m.get("pick_pct_urgent",    NAN) if is_rl else NAN,
                    "p_urgent_quality_check":  m.get("qc_pct_urgent",      NAN) if is_rl else NAN,
                    "p_urgent_pack":           m.get("pack_pct_urgent",     NAN) if is_rl else NAN,
                    "p_urgent_labelling":      m.get("lab_pct_urgent",      NAN) if is_rl else NAN,
                    "p_urgent_dispatch":       m.get("disp_pct_urgent",     NAN) if is_rl else NAN,
                    "decisions_total":          int(m.get("total_decisions", 0)) if is_rl else NAN,
                    "decisions_pick":           int(m.get("pick_dec_pts",   0))  if is_rl else NAN,
                    "decisions_quality_check":  int(m.get("qc_dec_pts",    0))  if is_rl else NAN,
                    "decisions_pack":           int(m.get("pack_dec_pts",   0))  if is_rl else NAN,
                    "decisions_labelling":      int(m.get("lab_dec_pts",    0))  if is_rl else NAN,
                    "decisions_dispatch":       int(m.get("disp_dec_pts",   0))  if is_rl else NAN,
                }
                rows.append(row)
                done += 1
                print(
                    f"[{done:>3}/{total_runs}] {calendar.month_abbr[month_num]:<4} "
                    f"{regime_name}  {policy:<14}  "
                    f"W={total_workers}  SLA={m['sla_rate']:.4f}  "
                    f"U={m['sla_urgent']:.4f}  late=${late:,.0f}  total=${total:,.0f}"
                )

    df = pd.DataFrame(rows)[CSV_COLS]
    out_path = root / args.output
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  ({len(df)} rows)")

    _print_interpretation(df)


if __name__ == "__main__":
    main()
