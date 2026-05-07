# src/rl/evaluate_rl5_monthly.py
"""
Monthly evaluation of RL-5 DQN vs FIFO and urgent_first.

Groups orders_base.csv by calendar month and evaluates each independently
to analyse seasonality and policy value under varying demand.

Usage:
    python -m src.rl.evaluate_rl5_monthly
    python -m src.rl.evaluate_rl5_monthly --checkpoint data/dqn_rl5_v2_final.pt --regime s32111 --output data/rl5_monthly_eval_results.csv
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

COST_LATE_URGENT = 20.0
COST_LATE_NORMAL = 5.0

NAN = float("nan")

KNOWN_REGIMES = {
    "s11111": (1, 1, 1, 1, 1),
    "s21111": (2, 1, 1, 1, 1),
    "s31111": (3, 1, 1, 1, 1),
    "s32111": (3, 2, 1, 1, 1),
    "s32211": (3, 2, 2, 1, 1),
    "s32221": (3, 2, 2, 2, 1),
    "s33322": (3, 3, 3, 2, 2),
}

CSV_COLS = [
    "month", "month_name", "regime", "policy",
    "total_orders", "urgent_orders", "normal_orders", "urgent_share",
    "total_sla", "urgent_sla", "normal_sla",
    "mean_system_time_min", "p90_system_time_min",
    "p_urgent_overall", "p_urgent_pick", "p_urgent_quality_check",
    "p_urgent_pack", "p_urgent_labelling", "p_urgent_dispatch",
    "decisions_total", "decisions_pick", "decisions_quality_check",
    "decisions_pack", "decisions_labelling", "decisions_dispatch",
    "urgent_late_orders", "normal_late_orders",
    "estimated_late_cost", "savings_vs_fifo", "savings_vs_urgent_first",
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


def _late_cost(urgent_cnt, normal_cnt, urgent_sla, normal_sla):
    ul = urgent_cnt * (1 - urgent_sla)
    nl = normal_cnt * (1 - normal_sla)
    return ul * COST_LATE_URGENT + nl * COST_LATE_NORMAL, ul, nl


# ── Interpretation ────────────────────────────────────────────────────────────

def _print_interpretation(df: pd.DataFrame) -> None:
    rl_df   = df[df["policy"] == "rl5_dqn"].set_index("month")
    fifo_df = df[df["policy"] == "fifo"].set_index("month")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    # 1. Month with highest order volume
    peak = fifo_df["total_orders"].idxmax()
    print(f"\n  1. Highest order volume: {calendar.month_name[peak]} "
          f"({int(fifo_df.loc[peak, 'total_orders']):,} orders)")

    # 2. Month with worst FIFO SLA
    worst = fifo_df["total_sla"].idxmin()
    print(f"\n  2. Worst FIFO total_sla: {calendar.month_name[worst]} "
          f"(SLA={fifo_df.loc[worst, 'total_sla']:.4f})")

    # 3. Month where RL-5 saves most vs FIFO
    best_vs_fifo = rl_df["savings_vs_fifo"].idxmax()
    print(f"\n  3. RL-5 max savings vs FIFO: {calendar.month_name[best_vs_fifo]} "
          f"(${rl_df.loc[best_vs_fifo, 'savings_vs_fifo']:,.0f})")

    # 4. Month where RL-5 saves most vs urgent_first
    best_vs_uf = rl_df["savings_vs_urgent_first"].idxmax()
    print(f"\n  4. RL-5 max savings vs urgent_first: {calendar.month_name[best_vs_uf]} "
          f"(${rl_df.loc[best_vs_uf, 'savings_vs_urgent_first']:,.0f})")

    # 5 & 6. Cheapest policy per month
    cost_pivot = df.pivot_table(index="month", columns="policy", values="estimated_late_cost")
    rl5_cheapest = [
        calendar.month_abbr[m]
        for m in cost_pivot.index
        if "rl5_dqn" in cost_pivot.columns
        and cost_pivot.loc[m, "rl5_dqn"] == cost_pivot.loc[m].min()
    ]
    uf_cheapest = [
        calendar.month_abbr[m]
        for m in cost_pivot.index
        if "urgent_first" in cost_pivot.columns
        and cost_pivot.loc[m, "urgent_first"] == cost_pivot.loc[m].min()
    ]
    print(f"\n  5. Months where RL-5 has lowest estimated cost: "
          f"{', '.join(rl5_cheapest) if rl5_cheapest else 'none'}")
    print(f"\n  6. Months where urgent_first has lowest estimated cost: "
          f"{', '.join(uf_cheapest) if uf_cheapest else 'none'}")

    # 7. RL-5 more valuable in high-demand months?
    corr_data = rl_df[["total_orders", "savings_vs_fifo"]].dropna()
    if len(corr_data) >= 3:
        corr = corr_data["total_orders"].corr(corr_data["savings_vs_fifo"])
        if corr > 0.4:
            verdict = f"YES — positive correlation ({corr:.2f}) with order volume"
        elif corr < -0.2:
            verdict = f"NO — negative correlation ({corr:.2f})"
        else:
            verdict = f"UNCLEAR — weak correlation ({corr:.2f})"
    else:
        verdict = "insufficient months to determine"
    print(f"\n  7. RL-5 more valuable in high-demand months? {verdict}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    root = Path(__file__).resolve().parents[2]
    v2_ckpt = root / "data" / "dqn_rl5_v2_final.pt"
    default_ckpt = "data/dqn_rl5_v2_final.pt" if v2_ckpt.exists() else "data/dqn_rl5_final.pt"

    parser = argparse.ArgumentParser(
        description="Monthly evaluation of RL-5 DQN vs FIFO and urgent_first"
    )
    parser.add_argument("--checkpoint", default=default_ckpt)
    parser.add_argument("--regime",     default="s32111",
                        choices=list(KNOWN_REGIMES))
    parser.add_argument("--output",     default="data/rl5_monthly_eval_results.csv")
    args = parser.parse_args()

    n_pick, n_qc, n_pack, n_lab, n_disp = KNOWN_REGIMES[args.regime]

    # Load sim config
    with open(root / "configs" / "sim_5stage.yaml", encoding="utf-8") as f:
        sim_cfg_full = yaml.safe_load(f)

    # Prefer rl5_v2.yaml for network params
    rl_cfg_path = root / "configs" / "rl5_v2.yaml"
    if not rl_cfg_path.exists():
        rl_cfg_path = root / "configs" / "rl5.yaml"
    with open(rl_cfg_path, encoding="utf-8") as f:
        rl_cfg = yaml.safe_load(f)

    sim_cfg        = sim_cfg_full["simulation"]
    base_resources = sim_cfg_full["resources"]
    service_cfg    = sim_cfg_full["service_time"]
    reward_cfg     = rl_cfg.get("reward", {})

    resources_cfg = {
        **base_resources,
        "picking_workers":       n_pick,
        "quality_check_workers": n_qc,
        "packing_workers":       n_pack,
        "labelling_workers":     n_lab,
        "dispatch_workers":      n_disp,
    }

    orders_all = (
        pd.read_csv(root / "data" / "orders_base.csv", parse_dates=["arrival_time"])
        .sort_values("arrival_time")
        .reset_index(drop=True)
    )

    ckpt_path  = root / args.checkpoint
    device     = "cpu"
    input_dim  = int(rl_cfg["network"].get("input_dim", 19))
    hidden_dim = int(rl_cfg["network"]["hidden_dim"])
    q_net      = QNetwork(input_dim, hidden_dim, output_dim=2)
    q_net.load_state_dict(torch.load(ckpt_path, map_location=device))
    q_net.eval()
    rl5_agent = _GreedyAgent(q_net, device)

    months = sorted(orders_all["month"].unique())
    print(f"Regime    : {args.regime}  (pick={n_pick} qc={n_qc} pack={n_pack} lab={n_lab} disp={n_disp})")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Config    : {rl_cfg_path.name}")
    print(f"Months    : {[calendar.month_abbr[m] for m in months]}")
    print(f"Total runs: {len(months) * 3}\n")

    hdr = (
        f"{'Month':<10} {'Policy':<14} {'Orders':>7} {'Urgent%':>8} "
        f"{'SLA':>6} {'SLA-U':>6} {'SLA-N':>6} "
        f"{'Cost':>10} {'vs FIFO':>10} {'vs UF':>10}"
    )
    print(hdr)
    print("-" * len(hdr))

    rows = []

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

        # Run all three policies
        policy_metrics: dict = {}
        for policy in ("fifo", "urgent_first"):
            policy_metrics[policy] = _run_baseline(
                month_orders, policy, resources_cfg, sim_cfg, service_cfg
            )
        policy_metrics["rl5_dqn"] = _run_rl5(
            month_orders, rl5_agent, resources_cfg, sim_cfg, service_cfg, reward_cfg, seed
        )

        # Costs using actual monthly order counts
        costs: dict = {}
        late_orders: dict = {}
        for policy, m in policy_metrics.items():
            cost, ul, nl = _late_cost(urgent_cnt, normal_cnt, m["sla_urgent"], m["sla_normal"])
            costs[policy]       = cost
            late_orders[policy] = (ul, nl)

        fifo_cost = costs["fifo"]
        uf_cost   = costs["urgent_first"]

        for policy, m in policy_metrics.items():
            is_rl     = policy == "rl5_dqn"
            cost      = costs[policy]
            ul, nl    = late_orders[policy]

            row = {
                "month":        month_num,
                "month_name":   month_name,
                "regime":       args.regime,
                "policy":       policy,
                "total_orders": total_cnt,
                "urgent_orders": urgent_cnt,
                "normal_orders": normal_cnt,
                "urgent_share": round(urgent_share, 4),
                "total_sla":   m["sla_rate"],
                "urgent_sla":  m["sla_urgent"],
                "normal_sla":  m["sla_normal"],
                "mean_system_time_min": m.get("mean_system_min", NAN) if is_rl else m["mean_system_min"],
                "p90_system_time_min":  m.get("p90_system_min",  NAN) if is_rl else m["p90_system_min"],
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
                "urgent_late_orders":      round(ul, 2),
                "normal_late_orders":      round(nl, 2),
                "estimated_late_cost":     round(cost, 2),
                "savings_vs_fifo":         round(fifo_cost - cost, 2),
                "savings_vs_urgent_first": round(uf_cost   - cost, 2),
            }
            rows.append(row)
            print(
                f"{month_name:<10} {policy:<14} {total_cnt:>7} {urgent_share:>8.1%} "
                f"{m['sla_rate']:>6.4f} {m['sla_urgent']:>6.4f} {m['sla_normal']:>6.4f} "
                f"{cost:>10,.0f} {fifo_cost - cost:>10,.0f} {uf_cost - cost:>10,.0f}"
            )

    df = pd.DataFrame(rows)[CSV_COLS]
    out_path = root / args.output
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  ({len(df)} rows)")

    _print_interpretation(df)


if __name__ == "__main__":
    main()
