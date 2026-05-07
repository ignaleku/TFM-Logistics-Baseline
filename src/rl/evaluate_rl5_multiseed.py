# src/rl/evaluate_rl5_multiseed.py
"""
Multi-window evaluation of RL-5 DQN vs FIFO and urgent_first.

5 windows × 7 regimes × 3 policies = 105 runs.

Usage:
    python -m src.rl.evaluate_rl5_multiseed
    python -m src.rl.evaluate_rl5_multiseed --checkpoint data/dqn_rl5_v2_final.pt
    python -m src.rl.evaluate_rl5_multiseed --checkpoint data/dqn_rl5_v2_final.pt --output data/rl5_v2_eval_multiseed_results.csv
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
N_WINDOWS = 5
SEEDS = [42, 43, 44, 45, 46]

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

CSV_COLS = [
    "window_id", "start_idx", "seed", "regime", "policy",
    "total_sla", "urgent_sla", "normal_sla",
    "mean_system_time_min", "p90_system_time_min",
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


def _window_starts(n_orders: int) -> list:
    max_start = n_orders - EPISODE_ORDERS
    if max_start <= 0:
        return [0] * N_WINDOWS
    step = max_start // (N_WINDOWS - 1)
    return [min(i * step, max_start) for i in range(N_WINDOWS)]


# ── Summary printing ──────────────────────────────────────────────────────────

def _mean_std(series: pd.Series) -> str:
    v = series.dropna()
    if len(v) == 0:
        return f"{'n/a':>17}"
    return f"{v.mean():>7.4f} ± {v.std(ddof=0):>6.4f}"


def _print_summary(df: pd.DataFrame) -> None:
    sla_cols = [
        "total_sla", "urgent_sla", "normal_sla",
        "mean_system_time_min", "p90_system_time_min",
    ]
    pct_cols = [
        "p_urgent_overall", "p_urgent_pick", "p_urgent_quality_check",
        "p_urgent_pack", "p_urgent_labelling", "p_urgent_dispatch",
    ]

    print("\n=== Aggregated results — mean ± std across windows ===\n")

    hdr1 = (
        f"{'Regime':<8} {'Policy':<14} "
        f"{'total_sla':>17}  {'urgent_sla':>17}  {'normal_sla':>17}  "
        f"{'mean_sys(min)':>17}  {'p90_sys(min)':>17}"
    )
    print(hdr1)
    print("-" * len(hdr1))
    for (regime, policy), grp in df.groupby(["regime", "policy"]):
        vals = "  ".join(_mean_std(grp[c]) for c in sla_cols)
        print(f"{regime:<8} {policy:<14} {vals}")

    print()

    hdr2 = (
        f"{'Regime':<8} {'Policy':<14} "
        f"{'%U_all':>17}  {'%U_pick':>17}  {'%U_qc':>17}  "
        f"{'%U_pack':>17}  {'%U_lab':>17}  {'%U_disp':>17}"
    )
    print(hdr2)
    print("-" * len(hdr2))
    for (regime, policy), grp in df.groupby(["regime", "policy"]):
        vals = "  ".join(_mean_std(grp[c]) for c in pct_cols)
        print(f"{regime:<8} {policy:<14} {vals}")
    print()


# ── Interpretation ────────────────────────────────────────────────────────────

def _print_interpretation(df: pd.DataFrame) -> None:
    agg = df.groupby(["regime", "policy"]).mean(numeric_only=True)

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    # 1. Regimes where RL-5 beats FIFO in mean total_sla
    beats_fifo = []
    for regime_name, *_ in REGIMES:
        try:
            rl_sla   = agg.loc[(regime_name, "rl5_dqn"), "total_sla"]
            fifo_sla = agg.loc[(regime_name, "fifo"),    "total_sla"]
            if rl_sla > fifo_sla:
                beats_fifo.append(f"{regime_name} (Δ={rl_sla - fifo_sla:+.4f})")
        except KeyError:
            pass
    print(f"\n  1. RL-5 beats FIFO in mean total_sla:")
    print(f"     {', '.join(beats_fifo) if beats_fifo else 'none'}")

    # 2. Regimes where RL-5 matches or beats urgent_first
    matches_uf = []
    for regime_name, *_ in REGIMES:
        try:
            rl_sla = agg.loc[(regime_name, "rl5_dqn"),      "total_sla"]
            uf_sla = agg.loc[(regime_name, "urgent_first"),  "total_sla"]
            if rl_sla >= uf_sla - 0.001:
                matches_uf.append(f"{regime_name} (Δ={rl_sla - uf_sla:+.4f})")
        except KeyError:
            pass
    print(f"\n  2. RL-5 matches or beats urgent_first in mean total_sla (tol=0.001):")
    print(f"     {', '.join(matches_uf) if matches_uf else 'none'}")

    # 3. Regimes with high std for RL-5
    HIGH_STD = 0.01
    std_ser = (
        df[df["policy"] == "rl5_dqn"]
        .groupby("regime")["total_sla"]
        .std(ddof=0)
        .sort_values(ascending=False)
    )
    high = [(r, v) for r, v in std_ser.items() if v >= HIGH_STD]
    print(f"\n  3. Regimes with high std (≥ {HIGH_STD}) for RL-5 total_sla:")
    if high:
        for r, v in high:
            print(f"     {r}: std={v:.4f}")
    else:
        print(f"     none (all < {HIGH_STD})")

    # 4. Most active RL decision stage per regime
    stage_cols = {
        "pick":          "decisions_pick",
        "quality_check": "decisions_quality_check",
        "pack":          "decisions_pack",
        "labelling":     "decisions_labelling",
        "dispatch":      "decisions_dispatch",
    }
    print(f"\n  4. Most active decision stage per regime (RL-5, mean decisions):")
    print(f"  {'Regime':<8} {'Top stage':<16} {'Mean dec':>9}  {'2nd stage':<16} {'Mean dec':>9}")
    print(f"  {'-'*8} {'-'*16} {'-'*9}  {'-'*16} {'-'*9}")
    rl_df = df[df["policy"] == "rl5_dqn"]
    for regime_name, *_ in REGIMES:
        sub = rl_df[rl_df["regime"] == regime_name]
        if sub.empty:
            continue
        counts = {stage: sub[col].mean() for stage, col in stage_cols.items()}
        sorted_s = sorted(counts, key=counts.get, reverse=True)
        top1 = sorted_s[0]
        if len(sorted_s) >= 2:
            top2, cnt2 = sorted_s[1], counts[sorted_s[1]]
        else:
            top2, cnt2 = "—", 0
        print(
            f"  {regime_name:<8} {top1:<16} {counts[top1]:>9.0f}  "
            f"{top2:<16} {cnt2:>9.0f}"
        )
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    root = Path(__file__).resolve().parents[2]
    v2_ckpt = root / "data" / "dqn_rl5_v2_final.pt"
    default_ckpt = "data/dqn_rl5_v2_final.pt" if v2_ckpt.exists() else "data/dqn_rl5_final.pt"

    parser = argparse.ArgumentParser(
        description="Multi-window evaluation of RL-5 DQN vs FIFO and urgent_first"
    )
    parser.add_argument("--checkpoint", default=default_ckpt)
    parser.add_argument("--output", default="data/rl5_eval_multiseed_results.csv")
    args = parser.parse_args()

    with open(root / "configs" / "sim_5stage.yaml", encoding="utf-8") as f:
        sim_cfg_full = yaml.safe_load(f)
    with open(root / "configs" / "rl5.yaml", encoding="utf-8") as f:
        rl_cfg = yaml.safe_load(f)

    sim_cfg        = sim_cfg_full["simulation"]
    base_resources = sim_cfg_full["resources"]
    service_cfg    = sim_cfg_full["service_time"]
    reward_cfg     = rl_cfg.get("reward", {})

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

    n_orders   = len(orders_all)
    starts     = _window_starts(n_orders)
    total_runs = N_WINDOWS * len(REGIMES) * 3  # 105

    print(f"Orders available : {n_orders:,}")
    print(f"Orders per window: {EPISODE_ORDERS:,}")
    print(f"Windows ({N_WINDOWS})       : start indices {starts}")
    print(f"Seeds            : {SEEDS}")
    print(f"Regimes          : {[r for r, *_ in REGIMES]}")
    print(f"Checkpoint       : {ckpt_path}")
    print(f"Total runs       : {total_runs}\n")

    rows = []
    done = 0

    for regime_name, n_pick, n_qc, n_pack, n_lab, n_disp in REGIMES:
        resources_cfg = {
            **base_resources,
            "picking_workers":       n_pick,
            "quality_check_workers": n_qc,
            "packing_workers":       n_pack,
            "labelling_workers":     n_lab,
            "dispatch_workers":      n_disp,
        }

        for w_id, (start_idx, seed) in enumerate(zip(starts, SEEDS)):
            orders = orders_all.iloc[start_idx : start_idx + EPISODE_ORDERS].copy()

            # Baselines
            for policy in ("fifo", "urgent_first"):
                m = _run_baseline(orders, policy, resources_cfg, sim_cfg, service_cfg)
                rows.append({
                    "window_id":               w_id,
                    "start_idx":               start_idx,
                    "seed":                    seed,
                    "regime":                  regime_name,
                    "policy":                  policy,
                    "total_sla":               m["sla_rate"],
                    "urgent_sla":              m["sla_urgent"],
                    "normal_sla":              m["sla_normal"],
                    "mean_system_time_min":    m["mean_system_min"],
                    "p90_system_time_min":     m["p90_system_min"],
                    "p_urgent_overall":        NAN,
                    "p_urgent_pick":           NAN,
                    "p_urgent_quality_check":  NAN,
                    "p_urgent_pack":           NAN,
                    "p_urgent_labelling":      NAN,
                    "p_urgent_dispatch":       NAN,
                    "decisions_total":         NAN,
                    "decisions_pick":          NAN,
                    "decisions_quality_check": NAN,
                    "decisions_pack":          NAN,
                    "decisions_labelling":     NAN,
                    "decisions_dispatch":      NAN,
                })
                done += 1
                print(
                    f"[{done:>3}/{total_runs}] {regime_name}  {policy:<14}  w{w_id}  seed={seed}  "
                    f"SLA={m['sla_rate']:.4f}  U={m['sla_urgent']:.4f}  N={m['sla_normal']:.4f}"
                )

            # RL-5 DQN
            m = _run_rl5(
                orders, rl5_agent, resources_cfg, sim_cfg, service_cfg, reward_cfg, seed
            )
            pu = m.get("p_urgent_decisions", NAN)
            rows.append({
                "window_id":               w_id,
                "start_idx":               start_idx,
                "seed":                    seed,
                "regime":                  regime_name,
                "policy":                  "rl5_dqn",
                "total_sla":               m["sla_rate"],
                "urgent_sla":              m["sla_urgent"],
                "normal_sla":              m["sla_normal"],
                "mean_system_time_min":    m.get("mean_system_min", NAN),
                "p90_system_time_min":     m.get("p90_system_min",  NAN),
                "p_urgent_overall":        pu,
                "p_urgent_pick":           m.get("pick_pct_urgent", NAN),
                "p_urgent_quality_check":  m.get("qc_pct_urgent",   NAN),
                "p_urgent_pack":           m.get("pack_pct_urgent",  NAN),
                "p_urgent_labelling":      m.get("lab_pct_urgent",   NAN),
                "p_urgent_dispatch":       m.get("disp_pct_urgent",  NAN),
                "decisions_total":         int(m.get("total_decisions", 0)),
                "decisions_pick":          int(m.get("pick_dec_pts",   0)),
                "decisions_quality_check": int(m.get("qc_dec_pts",    0)),
                "decisions_pack":          int(m.get("pack_dec_pts",   0)),
                "decisions_labelling":     int(m.get("lab_dec_pts",    0)),
                "decisions_dispatch":      int(m.get("disp_dec_pts",   0)),
            })
            done += 1
            print(
                f"[{done:>3}/{total_runs}] {regime_name}  {'rl5_dqn':<14}  w{w_id}  seed={seed}  "
                f"SLA={m['sla_rate']:.4f}  U={m['sla_urgent']:.4f}  N={m['sla_normal']:.4f}  "
                f"%U={pu:.4f}"
            )

    df = pd.DataFrame(rows)[CSV_COLS]
    out_path = root / args.output
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  ({len(df)} rows)")

    _print_summary(df)
    _print_interpretation(df)


if __name__ == "__main__":
    main()
