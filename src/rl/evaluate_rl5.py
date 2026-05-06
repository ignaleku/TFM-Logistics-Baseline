# src/rl/evaluate_rl5.py
"""
Evaluate RL-5 DQN against FIFO and urgent_first baselines
across seven capacity regimes.

Usage:
    python -m src.rl.evaluate_rl5
    python -m src.rl.evaluate_rl5 --checkpoint data/dqn_rl5_final.pt
"""
from __future__ import annotations

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

EPISODE_ORDERS = 10_000
EVAL_SEED      = 123

# (label, pick, qc, pack, lab, disp)
REGIMES = [
    ("s11111", 1, 1, 1, 1, 1),
    ("s21111", 2, 1, 1, 1, 1),
    ("s31111", 3, 1, 1, 1, 1),
    ("s32111", 3, 2, 1, 1, 1),
    ("s32211", 3, 2, 2, 1, 1),
    ("s32221", 3, 2, 2, 2, 1),
    ("s33322", 3, 3, 3, 2, 2),
]

NAN = float("nan")


# ── Agent wrapper ─────────────────────────────────────────────────────────────

class _GreedyAgent:
    def __init__(self, q_net: QNetwork, device: str) -> None:
        self.q      = q_net
        self.device = device

    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.q(s).squeeze(0).argmax().item())


# ── Run helpers ───────────────────────────────────────────────────────────────

def _run_baseline(orders, policy, resources_cfg, sim_cfg, service_cfg):
    cfg           = dict(sim_cfg)
    cfg["policy"] = policy
    _, summary    = run_simulation_5stage(orders, cfg, resources_cfg, service_cfg)
    return summary


def _run_rl5(orders, agent, resources_cfg, sim_cfg, service_cfg, reward_cfg):
    runner = FiveStageRLRunner(
        sim_cfg       = sim_cfg,
        resources_cfg = resources_cfg,
        service_cfg   = service_cfg,
        seed          = EVAL_SEED,
        reward_cfg    = reward_cfg,
    )
    buf = ReplayBuffer(capacity=1)
    return runner.run_episode(
        orders       = orders,
        agent        = agent,
        buffer       = buf,
        episode_seed = EVAL_SEED,
        greedy       = True,
    )


# ── Formatting ────────────────────────────────────────────────────────────────

def _fmt_pct(v) -> str:
    if isinstance(v, float) and np.isnan(v):
        return "—"
    return f"{v:.2%}"


# ── Interpretation ────────────────────────────────────────────────────────────

def _print_interpretation(df: pd.DataFrame) -> None:
    rl   = df[df["policy"] == "rl5_dqn"].set_index("regime")
    fifo = df[df["policy"] == "fifo"].set_index("regime")
    uf   = df[df["policy"] == "urgent_first"].set_index("regime")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    # 1. Best policy per regime
    print("\n  1. Best policy per regime (total_sla):")
    print(f"  {'Regime':<8} {'Winner':<16} {'SLA':>6}  {'RL5 vs FIFO':>12}  {'RL5 vs UF':>10}")
    print(f"  {'-'*8} {'-'*16} {'-'*6}  {'-'*12}  {'-'*10}")
    for regime_name, *_ in REGIMES:
        sub = df[df["regime"] == regime_name]
        if sub.empty:
            continue
        best    = sub.loc[sub["total_sla"].idxmax()]
        rl_sla  = rl.loc[regime_name,   "total_sla"] if regime_name in rl.index   else NAN
        fi_sla  = fifo.loc[regime_name, "total_sla"] if regime_name in fifo.index else NAN
        uf_sla  = uf.loc[regime_name,   "total_sla"] if regime_name in uf.index   else NAN
        d_fi    = f"{rl_sla - fi_sla:+.4f}" if not np.isnan(rl_sla) and not np.isnan(fi_sla) else "—"
        d_uf    = f"{rl_sla - uf_sla:+.4f}" if not np.isnan(rl_sla) and not np.isnan(uf_sla) else "—"
        print(
            f"  {regime_name:<8} {best['policy']:<16} {best['total_sla']:6.4f}  "
            f"{d_fi:>12}  {d_uf:>10}"
        )

    # 2. Regimes where RL-5 beats FIFO
    beats_fifo = [
        r for r in rl.index
        if r in fifo.index and rl.loc[r, "total_sla"] > fifo.loc[r, "total_sla"]
    ]
    print(
        f"\n  2. RL-5 beats FIFO (total_sla): "
        f"{', '.join(beats_fifo) if beats_fifo else 'none'}"
    )

    # 3. Regimes where RL-5 matches or beats urgent_first
    matches_uf = [
        r for r in rl.index
        if r in uf.index and rl.loc[r, "total_sla"] >= uf.loc[r, "total_sla"] - 0.001
    ]
    print(
        f"\n  3. RL-5 >= urgent_first − 0.001: "
        f"{', '.join(matches_uf) if matches_uf else 'none'}"
    )

    # 4. Most active decision stage per regime
    stage_dec_cols = {
        "pick":     "decisions_pick",
        "qc":       "decisions_quality_check",
        "pack":     "decisions_pack",
        "lab":      "decisions_labelling",
        "dispatch": "decisions_dispatch",
    }
    print(f"\n  4. Most active decision stage per regime (RL-5):")
    print(
        f"  {'Regime':<8} {'Top stage':<12} {'Decisions':>10}  "
        f"{'2nd stage':<12} {'Decisions':>10}"
    )
    print(f"  {'-'*8} {'-'*12} {'-'*10}  {'-'*12} {'-'*10}")
    for r in rl.index:
        counts = {}
        for stage, col in stage_dec_cols.items():
            v = rl.loc[r, col]
            if not np.isnan(v):
                counts[stage] = int(v)
        if not counts:
            continue
        sorted_stages = sorted(counts, key=counts.get, reverse=True)
        top1 = sorted_stages[0]
        top2 = sorted_stages[1] if len(sorted_stages) > 1 else "—"
        cnt2 = counts.get(top2, 0) if top2 != "—" else 0
        print(
            f"  {r:<8} {top1:<12} {counts[top1]:>10}  "
            f"{top2:<12} {cnt2:>10}"
        )

    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate RL-5 DQN against FIFO and urgent_first"
    )
    parser.add_argument("--checkpoint", default="data/dqn_rl5_final.pt",
                        help="Path to RL-5 checkpoint (relative to repo root or absolute)")
    parser.add_argument("--output", default="data/rl5_eval_results.csv")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]

    with open(root / "configs" / "sim_5stage.yaml", encoding="utf-8") as f:
        sim_cfg_full = yaml.safe_load(f)
    with open(root / "configs" / "rl5.yaml", encoding="utf-8") as f:
        rl_cfg = yaml.safe_load(f)

    sim_cfg        = sim_cfg_full["simulation"]
    base_resources = sim_cfg_full["resources"]
    service_cfg    = sim_cfg_full["service_time"]
    reward_cfg     = rl_cfg.get("reward", {})

    orders = (
        pd.read_csv(root / "data" / "orders_base.csv", parse_dates=["arrival_time"])
        .sort_values("arrival_time")
        .reset_index(drop=True)
        .iloc[:EPISODE_ORDERS]
        .copy()
    )

    ckpt_path  = root / args.checkpoint
    device     = "cpu"
    input_dim  = int(rl_cfg["network"].get("input_dim", 19))
    hidden_dim = int(rl_cfg["network"]["hidden_dim"])
    q_net      = QNetwork(input_dim, hidden_dim, output_dim=2)
    q_net.load_state_dict(torch.load(ckpt_path, map_location=device))
    q_net.eval()
    rl5_agent = _GreedyAgent(q_net, device)

    print(f"Checkpoint : {ckpt_path}")
    print(f"Orders     : {len(orders):,} | Seed: {EVAL_SEED}")
    print(f"Input dim  : {input_dim} | Regimes: {len(REGIMES)}\n")

    hdr = (
        f"{'Regime':<8} {'Policy':<14} {'SLA':>6} {'SLA-U':>6} {'SLA-N':>6} "
        f"{'mean(min)':>10} {'p90(min)':>9} "
        f"{'%U-tot':>7} {'%U-pick':>8} {'%U-qc':>7} "
        f"{'%U-pack':>8} {'%U-lab':>7} {'%U-disp':>8}"
    )
    print(hdr)
    print("-" * len(hdr))

    rows = []

    for regime_name, n_pick, n_qc, n_pack, n_lab, n_disp in REGIMES:
        resources_cfg = {
            **base_resources,
            "picking_workers":       n_pick,
            "quality_check_workers": n_qc,
            "packing_workers":       n_pack,
            "labelling_workers":     n_lab,
            "dispatch_workers":      n_disp,
        }

        # ── FIFO and urgent_first baselines ───────────────────────────────────

        for policy in ("fifo", "urgent_first"):
            m   = _run_baseline(orders, policy, resources_cfg, sim_cfg, service_cfg)
            row = {
                "regime":                   regime_name,
                "policy":                   policy,
                "total_sla":                m["sla_rate"],
                "urgent_sla":               m["sla_urgent"],
                "normal_sla":               m["sla_normal"],
                "mean_system_time_min":     m["mean_system_min"],
                "p90_system_time_min":      m["p90_system_min"],
                "p_urgent_overall":         NAN,
                "p_urgent_pick":            NAN,
                "p_urgent_quality_check":   NAN,
                "p_urgent_pack":            NAN,
                "p_urgent_labelling":       NAN,
                "p_urgent_dispatch":        NAN,
                "decisions_total":          NAN,
                "decisions_pick":           NAN,
                "decisions_quality_check":  NAN,
                "decisions_pack":           NAN,
                "decisions_labelling":      NAN,
                "decisions_dispatch":       NAN,
            }
            rows.append(row)
            print(
                f"{regime_name:<8} {policy:<14} "
                f"{row['total_sla']:6.4f} {row['urgent_sla']:6.4f} {row['normal_sla']:6.4f} "
                f"{row['mean_system_time_min']:10.1f} {row['p90_system_time_min']:9.1f} "
                f"{'—':>7} {'—':>8} {'—':>7} {'—':>8} {'—':>7} {'—':>8}"
            )

        # ── RL-5 ──────────────────────────────────────────────────────────────

        m = _run_rl5(orders, rl5_agent, resources_cfg, sim_cfg, service_cfg, reward_cfg)

        pu_tot  = m.get("p_urgent_decisions", NAN)
        pu_pick = m.get("pick_pct_urgent",    NAN)
        pu_qc   = m.get("qc_pct_urgent",      NAN)
        pu_pack = m.get("pack_pct_urgent",     NAN)
        pu_lab  = m.get("lab_pct_urgent",      NAN)
        pu_disp = m.get("disp_pct_urgent",     NAN)

        dec_pick = int(m.get("pick_dec_pts",  0))
        dec_qc   = int(m.get("qc_dec_pts",    0))
        dec_pack = int(m.get("pack_dec_pts",  0))
        dec_lab  = int(m.get("lab_dec_pts",   0))
        dec_disp = int(m.get("disp_dec_pts",  0))

        row = {
            "regime":                   regime_name,
            "policy":                   "rl5_dqn",
            "total_sla":                m["sla_rate"],
            "urgent_sla":               m["sla_urgent"],
            "normal_sla":               m["sla_normal"],
            "mean_system_time_min":     m.get("mean_system_min", NAN),
            "p90_system_time_min":      m.get("p90_system_min",  NAN),
            "p_urgent_overall":         pu_tot,
            "p_urgent_pick":            pu_pick,
            "p_urgent_quality_check":   pu_qc,
            "p_urgent_pack":            pu_pack,
            "p_urgent_labelling":       pu_lab,
            "p_urgent_dispatch":        pu_disp,
            "decisions_total":          int(m.get("total_decisions", 0)),
            "decisions_pick":           dec_pick,
            "decisions_quality_check":  dec_qc,
            "decisions_pack":           dec_pack,
            "decisions_labelling":      dec_lab,
            "decisions_dispatch":       dec_disp,
        }
        rows.append(row)
        print(
            f"{regime_name:<8} {'rl5_dqn':<14} "
            f"{row['total_sla']:6.4f} {row['urgent_sla']:6.4f} {row['normal_sla']:6.4f} "
            f"{row['mean_system_time_min']:10.1f} {row['p90_system_time_min']:9.1f} "
            f"{_fmt_pct(pu_tot):>7} {_fmt_pct(pu_pick):>8} {_fmt_pct(pu_qc):>7} "
            f"{_fmt_pct(pu_pack):>8} {_fmt_pct(pu_lab):>7} {_fmt_pct(pu_disp):>8}"
        )
        print(
            f"         decisions: pick={dec_pick}  qc={dec_qc}  pack={dec_pack}  "
            f"lab={dec_lab}  disp={dec_disp}  "
            f"total={row['decisions_total']}"
        )
        print()

    df_out   = pd.DataFrame(rows)
    out_path = root / args.output
    df_out.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")

    _print_interpretation(df_out)


if __name__ == "__main__":
    main()
