# src/rl/evaluate_rl3_sensitivity.py
"""
Bottleneck sensitivity evaluation for RL-3.

Evaluates the trained RL-3 policy under five service-time scenarios that
shift the bottleneck across picking, packing, and dispatch stages.
Compares against FIFO and urgent_first baselines.

Usage:
    python -m src.rl.evaluate_rl3_sensitivity
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.rl.dqn_agent import QNetwork
from src.rl.evaluate_rl3 import (
    EPISODE_ORDERS,
    EVAL_SEED,
    NAN,
    REGIMES,
    _GreedyAgent,
    _run_baseline,
    _run_rl3,
)

OUTPUT_PATH = "data/rl3_bottleneck_sensitivity_results.csv"


# ---------------------------------------------------------------------------
# Core helper
# ---------------------------------------------------------------------------

def apply_stage_multipliers(service_cfg: dict, multipliers: dict) -> dict:
    """Return a deepcopy of service_cfg with per-stage multipliers applied.

    Only base_minutes and minutes_per_item are scaled.
    sigma and class_multiplier are left untouched.
    """
    cfg = copy.deepcopy(service_cfg)
    for stage, factor in multipliers.items():
        if stage in cfg:
            cfg[stage]["base_minutes"] = float(cfg[stage]["base_minutes"]) * float(factor)
            cfg[stage]["minutes_per_item"] = float(cfg[stage]["minutes_per_item"]) * float(factor)
    return cfg


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------

def _print_interpretation(df: pd.DataFrame) -> None:
    rl   = df[df["policy"] == "rl3_dqn"].copy()
    fifo = df[df["policy"] == "fifo"].copy()
    uf   = df[df["policy"] == "urgent_first"].copy()

    scenarios     = list(df["scenario"].unique())
    base_scenarios = [s for s in scenarios if s != "base"]

    print("\n" + "=" * 72)
    print("SENSITIVITY INTERPRETATION")
    print("=" * 72)

    # 1. Best policy by scenario × regime
    print("\n1. Best policy (total_sla) by scenario × regime:")
    print(f"  {'Scenario':<16} {'Regime':<8} {'Winner':<16} {'SLA':>6}  {'vs FIFO':>8}  {'vs UF':>8}")
    print(f"  {'-'*16} {'-'*8} {'-'*16} {'-'*6}  {'-'*8}  {'-'*8}")
    for scen in scenarios:
        for regime_name, *_ in REGIMES:
            sub  = df[(df["scenario"] == scen) & (df["regime"] == regime_name)]
            if sub.empty:
                continue
            best = sub.loc[sub["total_sla"].idxmax()]
            rl_sla   = rl[(rl["scenario"] == scen) & (rl["regime"] == regime_name)]["total_sla"]
            fi_sla   = fifo[(fifo["scenario"] == scen) & (fifo["regime"] == regime_name)]["total_sla"]
            uf_sla   = uf[(uf["scenario"] == scen) & (uf["regime"] == regime_name)]["total_sla"]
            delta_fi = (rl_sla.values[0] - fi_sla.values[0]) if (len(rl_sla) and len(fi_sla)) else NAN
            delta_uf = (rl_sla.values[0] - uf_sla.values[0]) if (len(rl_sla) and len(uf_sla)) else NAN
            fi_str   = f"{delta_fi:+.4f}" if not np.isnan(delta_fi) else "—"
            uf_str   = f"{delta_uf:+.4f}" if not np.isnan(delta_uf) else "—"
            print(f"  {scen:<16} {regime_name:<8} {best['policy']:<16} {best['total_sla']:6.4f}  {fi_str:>8}  {uf_str:>8}")

    # 2. Cases where RL-3 beats FIFO
    print("\n2. Cases where RL-3 strictly beats FIFO (total_sla):")
    beats_fifo = []
    for scen in scenarios:
        for regime_name, *_ in REGIMES:
            rl_v  = rl[(rl["scenario"] == scen) & (rl["regime"] == regime_name)]["total_sla"]
            fi_v  = fifo[(fifo["scenario"] == scen) & (fifo["regime"] == regime_name)]["total_sla"]
            if len(rl_v) and len(fi_v) and rl_v.values[0] > fi_v.values[0]:
                beats_fifo.append(f"{scen}/{regime_name} (+{rl_v.values[0] - fi_v.values[0]:.4f})")
    print(f"  {', '.join(beats_fifo) if beats_fifo else 'none'}")

    # 3. Cases where RL-3 matches or beats urgent_first
    print("\n3. Cases where RL-3 >= urgent_first − 0.001 (total_sla):")
    matches_uf = []
    for scen in scenarios:
        for regime_name, *_ in REGIMES:
            rl_v = rl[(rl["scenario"] == scen) & (rl["regime"] == regime_name)]["total_sla"]
            uf_v = uf[(uf["scenario"] == scen) & (uf["regime"] == regime_name)]["total_sla"]
            if len(rl_v) and len(uf_v) and rl_v.values[0] >= uf_v.values[0] - 0.001:
                matches_uf.append(f"{scen}/{regime_name}")
    print(f"  {', '.join(matches_uf) if matches_uf else 'none'}")

    # 4. RL-3 SLA drop vs base scenario (mean across regimes)
    print("\n4. RL-3 mean total_sla drop vs base (across all regimes):")
    base_mean = rl[rl["scenario"] == "base"]["total_sla"].mean()
    drops: dict[str, float] = {}
    for scen in base_scenarios:
        scen_mean = rl[rl["scenario"] == scen]["total_sla"].mean()
        drops[scen] = base_mean - scen_mean
        direction = "drop" if drops[scen] >= 0 else "gain"
        print(f"  {scen:<16}: {direction} = {drops[scen]:+.4f}  (base={base_mean:.4f}, scen={scen_mean:.4f})")

    # 5. Hardest bottleneck scenario for RL-3
    print("\n5. Hardest bottleneck scenario for RL-3 (largest mean SLA drop):")
    hardest = max(drops, key=lambda s: drops[s])
    print(f"  {hardest}  (mean drop vs base: {drops[hardest]:+.4f})")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    root = Path(__file__).resolve().parents[2]

    with open(root / "configs" / "sim_multistage.yaml", encoding="utf-8") as f:
        sim_cfg_full = yaml.safe_load(f)
    with open(root / "configs" / "rl3.yaml", encoding="utf-8") as f:
        rl_cfg = yaml.safe_load(f)
    with open(root / "configs" / "sensitivity_scenarios.yaml", encoding="utf-8") as f:
        sensitivity_cfg = yaml.safe_load(f)

    sim_cfg          = sim_cfg_full["simulation"]
    base_resources   = sim_cfg_full["resources"]
    base_service_cfg = sim_cfg_full["service_time"]
    reward_cfg       = rl_cfg.get("reward", {})
    scenarios        = sensitivity_cfg["scenarios"]

    orders = (
        pd.read_csv(root / "data" / "orders_base.csv", parse_dates=["arrival_time"])
        .sort_values("arrival_time")
        .reset_index(drop=True)
        .iloc[:EPISODE_ORDERS]
        .copy()
    )

    ckpt_path  = root / "data" / "dqn_rl3_final.pt"
    device     = "cpu"
    input_dim  = int(rl_cfg["network"].get("input_dim", 13))
    hidden_dim = int(rl_cfg["network"]["hidden_dim"])
    q_net      = QNetwork(input_dim, hidden_dim, output_dim=2)
    q_net.load_state_dict(torch.load(ckpt_path, map_location=device))
    q_net.eval()
    rl3_agent = _GreedyAgent(q_net, device)

    print(f"Checkpoint : {ckpt_path}")
    print(f"Orders     : {len(orders):,} | Seed: {EVAL_SEED}")
    print(f"Scenarios  : {list(scenarios.keys())}")
    print(f"Regimes    : {[r[0] for r in REGIMES]}")
    print()

    hdr = (
        f"{'Scenario':<16} {'Regime':<8} {'Policy':<14} "
        f"{'SLA':>6} {'SLA-U':>6} {'SLA-N':>6} "
        f"{'mean(min)':>10} {'p90(min)':>9}"
    )
    print(hdr)
    print("-" * len(hdr))

    rows: list[dict] = []

    for scen_name, multipliers in scenarios.items():
        service_cfg = apply_stage_multipliers(base_service_cfg, multipliers)

        pick_mult = float(multipliers.get("picking",  1.0))
        pack_mult = float(multipliers.get("packing",  1.0))
        disp_mult = float(multipliers.get("dispatch", 1.0))

        for regime_name, n_pick, n_pack, n_disp in REGIMES:
            resources_cfg = {
                **base_resources,
                "picking_workers":  n_pick,
                "packing_workers":  n_pack,
                "dispatch_workers": n_disp,
            }

            # FIFO and urgent_first baselines
            for policy in ("fifo", "urgent_first"):
                m = _run_baseline(orders, policy, resources_cfg, sim_cfg, service_cfg)
                row = {
                    "scenario":              scen_name,
                    "regime":                regime_name,
                    "policy":                policy,
                    "picking_workers":       n_pick,
                    "packing_workers":       n_pack,
                    "dispatch_workers":      n_disp,
                    "picking_multiplier":    pick_mult,
                    "packing_multiplier":    pack_mult,
                    "dispatch_multiplier":   disp_mult,
                    "total_sla":             m["sla_rate"],
                    "urgent_sla":            m["sla_urgent"],
                    "normal_sla":            m["sla_normal"],
                    "mean_system_time_min":  m["mean_system_min"],
                    "p90_system_time_min":   m["p90_system_min"],
                    "p_urgent_overall":      NAN,
                    "p_urgent_pick":         NAN,
                    "p_urgent_pack":         NAN,
                    "p_urgent_dispatch":     NAN,
                    "decisions_total":       NAN,
                    "decisions_pick":        NAN,
                    "decisions_pack":        NAN,
                    "decisions_dispatch":    NAN,
                }
                rows.append(row)
                print(
                    f"{scen_name:<16} {regime_name:<8} {policy:<14} "
                    f"{row['total_sla']:6.4f} {row['urgent_sla']:6.4f} {row['normal_sla']:6.4f} "
                    f"{row['mean_system_time_min']:10.1f} {row['p90_system_time_min']:9.1f}"
                )

            # RL-3
            m = _run_rl3(orders, rl3_agent, resources_cfg, sim_cfg, service_cfg, reward_cfg)
            row = {
                "scenario":              scen_name,
                "regime":                regime_name,
                "policy":                "rl3_dqn",
                "picking_workers":       n_pick,
                "packing_workers":       n_pack,
                "dispatch_workers":      n_disp,
                "picking_multiplier":    pick_mult,
                "packing_multiplier":    pack_mult,
                "dispatch_multiplier":   disp_mult,
                "total_sla":             m["sla_rate"],
                "urgent_sla":            m["sla_urgent"],
                "normal_sla":            m["sla_normal"],
                "mean_system_time_min":  m.get("mean_system_min", NAN),
                "p90_system_time_min":   m.get("p90_system_min", NAN),
                "p_urgent_overall":      m.get("p_urgent_decisions", NAN),
                "p_urgent_pick":         m.get("pick_pct_urgent", NAN),
                "p_urgent_pack":         m.get("pack_pct_urgent", NAN),
                "p_urgent_dispatch":     m.get("disp_pct_urgent", NAN),
                "decisions_total":       int(m.get("total_decisions", 0)),
                "decisions_pick":        int(m.get("pick_dec_pts", 0)),
                "decisions_pack":        int(m.get("pack_dec_pts", 0)),
                "decisions_dispatch":    int(m.get("disp_dec_pts", 0)),
            }
            rows.append(row)
            print(
                f"{scen_name:<16} {regime_name:<8} {'rl3_dqn':<14} "
                f"{row['total_sla']:6.4f} {row['urgent_sla']:6.4f} {row['normal_sla']:6.4f} "
                f"{row['mean_system_time_min']:10.1f} {row['p90_system_time_min']:9.1f}"
            )

        print()  # blank line between scenarios

    df_out   = pd.DataFrame(rows)
    out_path = root / OUTPUT_PATH
    df_out.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")

    _print_interpretation(df_out)


if __name__ == "__main__":
    main()