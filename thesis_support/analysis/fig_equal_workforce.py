"""
THESIS-ONLY. Figure F12 and its tables: the systematic equal-workforce policy comparison
(evidence-ledger item P4, RQ2).

Every (month, workforce) pair in the retrospective grid was evaluated under all three policies
on identical stochastic realisations, which makes the grid a set of 192 matched triples. This
script pairs them and reports the distribution of policy differences, rather than relying on
the individual configurations highlighted in the main text.

Two views are produced:
  1. how the RL-3 vs Urgent-First cost gap scales with capacity pressure;
  2. how the comparison splits between the feasible and infeasible regions.

Reads : data/api_runs/latest/historical/rl3_monthly_capacity_cost_results.csv
Writes: thesis/figures/f12_equal_workforce.pdf
        thesis/tables/t_equal_workforce_pairs.csv
        thesis/tables/t_pressure_bands.csv
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import POLICY_COLOUR, POLICY_LABEL, apply_style, load_results, save, save_table

# Capacity-pressure bands, defined on picking utilisation (the dominant stage throughout).
BAND_EDGES = [0.0, 0.70, 0.80, 0.90, 0.95, 1.01]
BAND_LABELS = ["<70%", "70-80%", "80-90%", "90-95%", ">95%"]


def build_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (month, workforce), with each policy's outcome in its own columns."""
    p = df.pivot_table(
        index=["month", "regime"],
        columns="policy",
        values=["urgent_sla", "normal_sla", "total_sla", "estimated_total_cost",
                "feasible", "picking_utilisation", "total_workers"],
    )
    out = pd.DataFrame(index=p.index)
    out["picking_utilisation"] = p[("picking_utilisation", "fifo")]
    out["total_workers"] = p[("total_workers", "fifo")]
    for pol in ("fifo", "urgent_first", "rl3_dqn"):
        out[f"{pol}_cost"] = p[("estimated_total_cost", pol)]
        out[f"{pol}_urgent"] = p[("urgent_sla", pol)]
        out[f"{pol}_normal"] = p[("normal_sla", pol)]
        out[f"{pol}_feasible"] = p[("feasible", pol)].astype(bool)

    out["d_cost_rl_uf"] = out["rl3_dqn_cost"] - out["urgent_first_cost"]
    out["d_normal_rl_uf"] = out["rl3_dqn_normal"] - out["urgent_first_normal"]
    out["d_urgent_rl_uf"] = out["rl3_dqn_urgent"] - out["urgent_first_urgent"]
    out["any_feasible"] = out["rl3_dqn_feasible"] | out["urgent_first_feasible"]
    out["band"] = pd.cut(out["picking_utilisation"], BAND_EDGES, labels=BAND_LABELS)
    return out.reset_index()


def fig_equal_workforce(pairs: pd.DataFrame):
    # wspace keeps the right panel's y-axis label clear of the left panel's bar annotations.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.9, 3.3),
                                   gridspec_kw={"wspace": 0.42})

    # ── Left: absolute RL-3 vs Urgent-First cost gap by capacity-pressure band ────────────
    g = (pairs.assign(gap=pairs["d_cost_rl_uf"].abs())
              .groupby("band", observed=True)["gap"].agg(["mean", "count"]))
    g = g.reindex(BAND_LABELS).dropna()
    bars = ax1.bar(range(len(g)), g["mean"], color="#3E5C76", width=0.62)
    for i, (m, n) in enumerate(zip(g["mean"], g["count"])):
        ax1.text(i, m + 130, f"{m:,.0f}", ha="center", va="bottom", fontsize=7)
        ax1.text(i, -430, f"n={int(n)}", ha="center", fontsize=6.5, color="#666666")
    ax1.set_xticks(range(len(g)))
    ax1.set_xticklabels(g.index, fontsize=7.5)
    ax1.set_xlabel("Picking utilisation band", fontsize=8)
    ax1.set_ylabel("Mean |cost difference| (EUR)")
    ax1.set_ylim(-700, g["mean"].max() * 1.28)
    ax1.set_title("RL-3 vs Urgent-First at equal workforce", fontsize=8.5)

    # ── Right: normal-class SLA difference, split by feasible region ──────────────────────
    feas = pairs.loc[pairs["any_feasible"], "d_normal_rl_uf"] * 100
    infeas = pairs.loc[~pairs["any_feasible"], "d_normal_rl_uf"] * 100
    parts = ax2.boxplot([feas.values, infeas.values], widths=0.5, patch_artist=True,
                        medianprops=dict(color="black", linewidth=1.2),
                        flierprops=dict(marker="o", markersize=2.5, alpha=0.5))
    for patch, colour in zip(parts["boxes"], ["#2E7D5B", "#B3452F"]):
        patch.set_facecolor(colour)
        patch.set_alpha(0.55)
    ax2.axhline(0, color="#666666", linewidth=0.8, linestyle="--")
    ax2.set_xticklabels([f"at least one\nfeasible\n(n={len(feas)})",
                         f"neither\nfeasible\n(n={len(infeas)})"], fontsize=7.5)
    ax2.set_ylabel("Normal-class SLA difference\n(RL-3 minus Urgent-First, pts)")
    ax2.set_title("Where RL-3's advantage holds", fontsize=8.5)

    fig.suptitle("Systematic comparison over all 192 matched configurations", y=1.02)
    return fig


def main() -> None:
    apply_style()
    df = load_results("historical")
    pairs = build_pairs(df)
    n = len(pairs)
    print(f"  matched (month, workforce) configurations: {n}")

    # Pairwise summary across all pairs.
    rows = []
    for a, b in [("rl3_dqn", "urgent_first"), ("rl3_dqn", "fifo"), ("urgent_first", "fifo")]:
        dc = pairs[f"{a}_cost"] - pairs[f"{b}_cost"]
        du = pairs[f"{a}_urgent"] - pairs[f"{b}_urgent"]
        dn = pairs[f"{a}_normal"] - pairs[f"{b}_normal"]
        rows.append({
            "comparison": f"{POLICY_LABEL[a]} vs {POLICY_LABEL[b]}",
            "cheaper_in": int((dc < 0).sum()),
            "mean_cost_diff": round(float(dc.mean())),
            "urgent_better_in": int((du > 0).sum()),
            "mean_urgent_diff_pts": round(float(du.mean() * 100), 2),
            "normal_better_in": int((dn > 0).sum()),
            "mean_normal_diff_pts": round(float(dn.mean() * 100), 2),
        })
    summary = pd.DataFrame(rows)
    save_table(summary, "t_equal_workforce_pairs")
    print(summary.to_string(index=False))

    # Pressure bands.
    g = (pairs.assign(gap=pairs["d_cost_rl_uf"].abs())
              .groupby("band", observed=True)
              .agg(configs=("gap", "size"), mean_gap=("gap", "mean"), max_gap=("gap", "max"))
              .reindex(BAND_LABELS).dropna().reset_index())
    g["mean_gap"] = g["mean_gap"].round(0).astype(int)
    g["max_gap"] = g["max_gap"].round(0).astype(int)
    g["configs"] = g["configs"].astype(int)
    save_table(g, "t_pressure_bands")
    print()
    print(g.to_string(index=False))

    # Feasible / infeasible split.
    print()
    for label, mask in [("at least one feasible", pairs["any_feasible"]),
                        ("neither feasible", ~pairs["any_feasible"])]:
        s = pairs.loc[mask, "d_normal_rl_uf"]
        c = pairs.loc[mask, "d_cost_rl_uf"]
        print(f"  {label:<24} n={mask.sum():3d}  "
              f"normal delta mean {s.mean()*100:+.2f} pts (min {s.min()*100:+.2f})  "
              f"cost mean {c.mean():+,.0f}  RL-3 cheaper {int((c < 0).sum())}")

    save(fig_equal_workforce(pairs), "f12_equal_workforce")


if __name__ == "__main__":
    main()
