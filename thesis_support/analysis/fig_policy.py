"""
THESIS-ONLY. Figures F4-F7 and the policy tables (RQ2).

Comparisons are reported at EQUAL WORKFORCE throughout (each policy evaluated on the same
(month, regime) under common random numbers), which isolates the sequencing effect from the
capacity effect. The separate "each policy at its own cheapest configuration" view is reported
as a table, never mixed into the equal-workforce figures (see THESIS_STATE.md 6 / master
prompt section 84).

Reads : data/api_runs/latest/historical/rl3_monthly_capacity_cost_results.csv (576 rows)
Writes: f4_feasibility_by_month.pdf, f5_class_sla_tradeoff.pdf,
        f6_cost_vs_sla.pdf, f7_representative_cases.pdf
        t_policy_feasibility.csv, t_representative_cases.csv
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (MONTH_ABBR, NORMAL_TARGET, POLICIES, POLICY_COLOUR, POLICY_LABEL,
                    POLICY_MARKER, URGENT_TARGET, apply_style, load_results, save, save_table)


def fig_feasibility(df: pd.DataFrame):
    """F4 — how many of the 16 candidate workforces are feasible, per month per policy."""
    counts = (df[df["feasible"]].groupby(["month", "policy"]).size()
              .unstack(fill_value=0).reindex(range(1, 13), fill_value=0))
    for p in POLICIES:
        if p not in counts:
            counts[p] = 0

    fig, ax = plt.subplots(figsize=(6.3, 3.1))
    x = np.arange(12)
    w = 0.27
    for i, p in enumerate(POLICIES):
        ax.bar(x + (i - 1) * w, counts[p].values, width=w,
               color=POLICY_COLOUR[p], label=POLICY_LABEL[p])
    ax.set_xticks(x)
    ax.set_xticklabels(MONTH_ABBR)
    ax.set_ylabel("Feasible configurations\n(out of 16 candidates)")
    ax.set_ylim(0, 18)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.16))
    ax.set_title("Configurations meeting both SLA floors, by month and policy", pad=22)

    # Annotate the campaign months where FIFO is feasible nowhere.
    for m in (1, 2, 11, 12):
        ax.annotate("0", xy=(m - 1 - w, 0.25), ha="center", va="bottom",
                    fontsize=7, color="#B3452F", fontweight="bold")
    return fig, counts


def fig_class_tradeoff(df: pd.DataFrame):
    """F5 — urgent vs normal SLA at equal workforce; the feasible region is boxed."""
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    for p in POLICIES:
        d = df[df["policy"] == p]
        ax.scatter(d["normal_sla"] * 100, d["urgent_sla"] * 100, s=14, alpha=0.62,
                   color=POLICY_COLOUR[p], marker=POLICY_MARKER[p],
                   label=POLICY_LABEL[p], linewidths=0)

    ax.axvline(NORMAL_TARGET * 100, color="#666666", linestyle="--", linewidth=0.9)
    ax.axhline(URGENT_TARGET * 100, color="#666666", linestyle="--", linewidth=0.9)
    ax.add_patch(plt.Rectangle((NORMAL_TARGET * 100, URGENT_TARGET * 100), 100, 100,
                               facecolor="#2E7D5B", alpha=0.07, zorder=0))
    ax.text(99.4, 95.6, "feasible region", ha="right", va="bottom",
            fontsize=7.5, color="#2E7D5B", style="italic")
    ax.set_xlabel("Normal-class SLA attainment (%)")
    ax.set_ylabel("Urgent-class SLA attainment (%)")
    ax.set_xlim(60, 101)
    ax.set_ylim(0, 103)
    ax.legend(loc="lower left")
    ax.set_title("Class-level service trade-off\n(all 576 month x workforce x policy runs)")
    return fig


def fig_cost_vs_sla(df: pd.DataFrame):
    """F6 — cost against total SLA, December only (the binding month)."""
    d12 = df[df["month"] == 12]
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for p in POLICIES:
        d = d12[d12["policy"] == p]
        ax.scatter(d["total_sla"] * 100, d["estimated_total_cost"] / 1000.0,
                   s=32, color=POLICY_COLOUR[p], marker=POLICY_MARKER[p],
                   label=POLICY_LABEL[p], alpha=0.8, linewidths=0)
    # Mark the recommended configuration.
    rec = d12[(d12["regime"] == "s22_11_5") & (d12["policy"] == "rl3_dqn")].iloc[0]
    # Plain underscores: this label is rendered by matplotlib, not LaTeX.
    ax.annotate("s22_11_5 (recommended)",
                xy=(rec["total_sla"] * 100, rec["estimated_total_cost"] / 1000.0),
                xytext=(-6, 26), textcoords="offset points", fontsize=7.5, ha="right",
                arrowprops=dict(arrowstyle="->", lw=0.7, color="#333333"))
    ax.set_xlabel("Total SLA attainment (%)")
    ax.set_ylabel("Estimated total cost (thousand EUR)")
    ax.legend(loc="upper right")
    ax.set_title("December: cost against service, all 16 candidate workforces")
    return fig


def fig_cases(df: pd.DataFrame):
    """F7 — the two October regimes that show policy choice is regime-dependent.

    Note: these labels are rendered by matplotlib, not by LaTeX, so regime names must contain
    plain underscores. Escaping them as `s11\\_7\\_5` would print the backslash literally.
    """
    cases = [("s753", "Under-capacity: October s753 (15 FTE)"),
             ("s11_7_5", "Well-resourced: October s11_7_5 (23 FTE)")]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.3), sharey=True)
    x = np.arange(2)
    w = 0.26
    handles = None
    for ax, (regime, title) in zip(axes, cases):
        d = df[(df["month"] == 10) & (df["regime"] == regime)]
        for i, p in enumerate(POLICIES):
            r = d[d["policy"] == p].iloc[0]
            vals = [r["urgent_sla"] * 100, r["normal_sla"] * 100]
            bars = ax.bar(x + (i - 1) * w, vals, width=w, color=POLICY_COLOUR[p],
                          label=POLICY_LABEL[p])
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{v:.1f}",
                        ha="center", va="bottom", fontsize=6.5)
        ax.axhline(URGENT_TARGET * 100, color="#666666", linestyle="--", linewidth=0.8)
        ax.axhline(NORMAL_TARGET * 100, color="#999999", linestyle=":", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(["Urgent class", "Normal class"])
        ax.set_ylim(0, 125)
        ax.set_title(title, fontsize=8.5)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()

    axes[0].set_ylabel("SLA attainment (%)")
    # Legend below the panels: the only region guaranteed free of bars, value labels and
    # subplot titles. The two floor lines are identified in the caption instead of by
    # in-axes annotations, which collided with the bars at every position tried.
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.07),
               frameon=False, fontsize=8)
    fig.suptitle("Policy choice matters under capacity pressure, not under slack", y=1.01)
    return fig


def main() -> None:
    apply_style()
    df = load_results("historical")

    # ── Feasibility summary ────────────────────────────────────────────────────────────────
    fig, counts = fig_feasibility(df)
    save(fig, "f4_feasibility_by_month")
    tot = df.groupby("policy")["feasible"].agg(["sum", "count"])
    print("  feasibility by policy (of 192 runs each):")
    for p in POLICIES:
        print(f"    {POLICY_LABEL[p]:<13} {int(tot.loc[p,'sum']):3d}/{int(tot.loc[p,'count'])}")
    zero = [MONTH_ABBR[m - 1] for m in range(1, 13) if counts.loc[m, "fifo"] == 0]
    print(f"  months with ZERO feasible FIFO configurations: {zero}")

    ct = counts.reset_index().rename(columns={"month": "month_num"})
    ct.insert(1, "month", [MONTH_ABBR[i - 1] for i in ct["month_num"]])
    save_table(ct[["month"] + POLICIES], "t_policy_feasibility")

    save(fig_class_tradeoff(df), "f5_class_sla_tradeoff")
    save(fig_cost_vs_sla(df), "f6_cost_vs_sla")
    save(fig_cases(df), "f7_representative_cases")

    # ── Representative case table ──────────────────────────────────────────────────────────
    rows = []
    for month, regime in [(10, "s753"), (10, "s11_7_5"), (12, "s22_11_5")]:
        for p in POLICIES:
            r = df[(df["month"] == month) & (df["regime"] == regime)
                   & (df["policy"] == p)].iloc[0]
            rows.append({
                "month": MONTH_ABBR[month - 1],
                "regime": regime,
                "fte": int(r["total_workers"]),
                "policy": POLICY_LABEL[p],
                "urgent_sla": round(r["urgent_sla"] * 100, 1),
                "normal_sla": round(r["normal_sla"] * 100, 1),
                "total_sla": round(r["total_sla"] * 100, 1),
                "feasible": "yes" if r["feasible"] else "no",
                "total_cost": int(r["estimated_total_cost"]),
            })
    cases = pd.DataFrame(rows)
    save_table(cases, "t_representative_cases")
    print("\n  representative cases:")
    print(cases.to_string(index=False))


if __name__ == "__main__":
    main()
