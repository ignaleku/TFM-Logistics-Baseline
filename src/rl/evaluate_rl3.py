# src/rl/evaluate_rl3.py
from __future__ import annotations

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

EPISODE_ORDERS = 10_000
EVAL_SEED = 123

# (label, picking_workers, packing_workers, dispatch_workers)
REGIMES = [
    ("s111", 1, 1, 1),
    ("s211", 2, 1, 1),
    ("s221", 2, 2, 1),
    ("s311", 3, 1, 1),
    ("s321", 3, 2, 1),
    ("s222", 2, 2, 2),
    ("s332", 3, 3, 2),
]

NAN = float("nan")


class _GreedyAgent:
    def __init__(self, q_net: QNetwork, device: str):
        self.q = q_net
        self.device = device

    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.q(s).squeeze(0).argmax().item())


def _run_baseline(orders, policy, resources_cfg, sim_cfg, service_cfg):
    cfg = dict(sim_cfg)
    cfg["policy"] = policy
    _, summary = run_simulation_multistage(orders, cfg, resources_cfg, service_cfg)
    return summary


def _run_rl3(orders, agent, resources_cfg, sim_cfg, service_cfg, reward_cfg):
    runner = FullStageRLRunner(
        sim_cfg=sim_cfg,
        resources_cfg=resources_cfg,
        service_cfg=service_cfg,
        seed=EVAL_SEED,
        reward_cfg=reward_cfg,
    )
    buf = ReplayBuffer(capacity=1)
    return runner.run_episode(
        orders=orders, agent=agent, buffer=buf, episode_seed=EVAL_SEED, greedy=True
    )


def _fmt_pct(v) -> str:
    if isinstance(v, float) and np.isnan(v):
        return "—"
    return f"{v:.2%}"


def _print_interpretation(df: pd.DataFrame) -> None:
    rl = df[df["policy"] == "rl3_dqn"].set_index("regime")
    fifo = df[df["policy"] == "fifo"].set_index("regime")
    uf = df[df["policy"] == "urgent_first"].set_index("regime")

    best_total = rl["total_sla"].idxmax()
    best_urgent = rl["urgent_sla"].idxmax()

    beats_fifo = [r for r in rl.index if rl.loc[r, "total_sla"] > fifo.loc[r, "total_sla"]]
    matches_uf = [
        r for r in rl.index
        if rl.loc[r, "total_sla"] >= uf.loc[r, "total_sla"] - 0.001
    ]

    print("\n" + "=" * 65)
    print("INTERPRETATION")
    print("=" * 65)
    print(f"  Best total SLA  (RL-3): {best_total}  ({rl.loc[best_total, 'total_sla']:.4f})")
    print(f"  Best urgent SLA (RL-3): {best_urgent}  ({rl.loc[best_urgent, 'urgent_sla']:.4f})")
    print(f"  RL-3 > FIFO (total SLA):       {', '.join(beats_fifo) if beats_fifo else 'none'}")
    print(f"  RL-3 >= urgent_first (total):  {', '.join(matches_uf) if matches_uf else 'none'}")

    print()
    print(f"  {'Regime':<8}  {'Most active stage':<20}  {'pick dec':>8}  {'pack dec':>8}  {'disp dec':>8}")
    print(f"  {'-'*8}  {'-'*20}  {'-'*8}  {'-'*8}  {'-'*8}")
    for r in rl.index:
        dp = int(rl.loc[r, "decisions_pick"])
        dk = int(rl.loc[r, "decisions_pack"])
        dd = int(rl.loc[r, "decisions_dispatch"])
        top = max(("pick", dp), ("pack", dk), ("dispatch", dd), key=lambda x: x[1])[0]
        print(f"  {r:<8}  {top:<20}  {dp:>8}  {dk:>8}  {dd:>8}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RL-3 DQN against FIFO and urgent_first")
    parser.add_argument("--checkpoint", default="data/dqn_rl3_final.pt",
                        help="Path to RL-3 model checkpoint (relative to repo root or absolute)")
    parser.add_argument("--output", default="data/rl3_eval_results.csv")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]

    with open(root / "configs" / "sim_multistage.yaml", encoding="utf-8") as f:
        sim_cfg_full = yaml.safe_load(f)
    with open(root / "configs" / "rl3.yaml", encoding="utf-8") as f:
        rl_cfg = yaml.safe_load(f)

    sim_cfg = sim_cfg_full["simulation"]
    base_resources = sim_cfg_full["resources"]
    service_cfg = sim_cfg_full["service_time"]
    reward_cfg = rl_cfg.get("reward", {})

    orders = (
        pd.read_csv(root / "data" / "orders_base.csv", parse_dates=["arrival_time"])
        .sort_values("arrival_time")
        .reset_index(drop=True)
        .iloc[:EPISODE_ORDERS]
        .copy()
    )

    ckpt_path = root / args.checkpoint
    device = "cpu"
    input_dim = int(rl_cfg["network"].get("input_dim", 13))
    hidden_dim = int(rl_cfg["network"]["hidden_dim"])
    q_net = QNetwork(input_dim, hidden_dim, output_dim=2)
    q_net.load_state_dict(torch.load(ckpt_path, map_location=device))
    q_net.eval()
    rl3_agent = _GreedyAgent(q_net, device)

    print(f"Checkpoint : {ckpt_path}")
    print(f"Orders     : {len(orders):,} | Seed: {EVAL_SEED}")
    print(f"Input dim  : {input_dim} | Regimes: {len(REGIMES)}\n")

    hdr = (
        f"{'Regime':<8} {'Policy':<14} {'SLA':>6} {'SLA-U':>6} {'SLA-N':>6} "
        f"{'mean(min)':>10} {'p90(min)':>9} "
        f"{'%U-tot':>7} {'%U-pick':>8} {'%U-pack':>8} {'%U-disp':>8}"
    )
    print(hdr)
    print("-" * len(hdr))

    rows = []

    for regime_name, n_pick, n_pack, n_disp in REGIMES:
        resources_cfg = {
            **base_resources,
            "picking_workers": n_pick,
            "packing_workers": n_pack,
            "dispatch_workers": n_disp,
        }

        for policy in ("fifo", "urgent_first"):
            m = _run_baseline(orders, policy, resources_cfg, sim_cfg, service_cfg)
            row = {
                "regime": regime_name,
                "policy": policy,
                "total_sla": m["sla_rate"],
                "urgent_sla": m["sla_urgent"],
                "normal_sla": m["sla_normal"],
                "mean_system_time_min": m["mean_system_min"],
                "p90_system_time_min": m["p90_system_min"],
                "p_urgent_overall": NAN,
                "p_urgent_pick": NAN,
                "p_urgent_pack": NAN,
                "p_urgent_dispatch": NAN,
                "decisions_total": NAN,
                "decisions_pick": NAN,
                "decisions_pack": NAN,
                "decisions_dispatch": NAN,
            }
            rows.append(row)
            print(
                f"{regime_name:<8} {policy:<14} "
                f"{row['total_sla']:6.4f} {row['urgent_sla']:6.4f} {row['normal_sla']:6.4f} "
                f"{row['mean_system_time_min']:10.1f} {row['p90_system_time_min']:9.1f} "
                f"{'—':>7} {'—':>8} {'—':>8} {'—':>8}"
            )

        m = _run_rl3(orders, rl3_agent, resources_cfg, sim_cfg, service_cfg, reward_cfg)
        pu_tot = m.get("p_urgent_decisions", NAN)
        pu_pick = m.get("pick_pct_urgent", NAN)
        pu_pack = m.get("pack_pct_urgent", NAN)
        pu_disp = m.get("disp_pct_urgent", NAN)
        dec_total = int(m.get("total_decisions", 0))
        dec_pick = int(m.get("pick_dec_pts", 0))
        dec_pack = int(m.get("pack_dec_pts", 0))
        dec_disp = int(m.get("disp_dec_pts", 0))

        row = {
            "regime": regime_name,
            "policy": "rl3_dqn",
            "total_sla": m["sla_rate"],
            "urgent_sla": m["sla_urgent"],
            "normal_sla": m["sla_normal"],
            "mean_system_time_min": m.get("mean_system_min", NAN),
            "p90_system_time_min": m.get("p90_system_min", NAN),
            "p_urgent_overall": pu_tot,
            "p_urgent_pick": pu_pick,
            "p_urgent_pack": pu_pack,
            "p_urgent_dispatch": pu_disp,
            "decisions_total": dec_total,
            "decisions_pick": dec_pick,
            "decisions_pack": dec_pack,
            "decisions_dispatch": dec_disp,
        }
        rows.append(row)
        print(
            f"{regime_name:<8} {'rl3_dqn':<14} "
            f"{row['total_sla']:6.4f} {row['urgent_sla']:6.4f} {row['normal_sla']:6.4f} "
            f"{row['mean_system_time_min']:10.1f} {row['p90_system_time_min']:9.1f} "
            f"{_fmt_pct(pu_tot):>7} {_fmt_pct(pu_pick):>8} {_fmt_pct(pu_pack):>8} {_fmt_pct(pu_disp):>8}"
        )
        print(
            f"         decisions: pick={dec_pick}  pack={dec_pack}  disp={dec_disp}  "
            f"total={dec_total}"
        )
        print()

    df_out = pd.DataFrame(rows)
    out_path = root / args.output
    df_out.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")

    _print_interpretation(df_out)


if __name__ == "__main__":
    main()