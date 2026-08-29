"""
THESIS-ONLY. Figures F3, F8, F9 and F11 plus supporting tables (RQ1, RQ3, RQ4).

F3   Analytical capacity anchor vs the workforce actually recommended after simulation.
F8   Stage Pressure decomposition by month (historical).
F9   The marginal-FTE decision: what the adaptive search tested and why it was rejected.
F11  Future planning (December): candidate cost vs feasibility probability across replications.

Reads : data/api_runs/latest/{historical,future}/*
Writes: f3_analytical_vs_recommended.pdf, f8_stage_pressure.pdf,
        f9_marginal_fte.pdf, f11_future_planning.pdf
        t_monthly_recommendations.csv, t_bottleneck_ranking.csv, t_adaptive_search.csv
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (MONTH_ABBR, POLICY_COLOUR, POLICY_LABEL, STAGES, STAGE_COLOUR,
                    STAGE_LABEL, apply_style, load_bottleneck, load_recommendations,
                    load_results, save, save_table)
import json
from common import RUNS


def fig_analytical_vs_recommended(rec: pd.DataFrame):
    """F3 — the analytical estimate centres the search; simulation decides the answer."""
    with open(RUNS / "historical" / "historical_analysis_summary.json", encoding="utf-8") as fh:
        summary = json.load(fh)
    est = summary["analytical_estimate_by_month"]

    an, si = [], []
    for i, name in enumerate(["January", "February", "March", "April", "May", "June", "July",
                              "August", "September", "October", "November", "December"]):
        w = est[name]["workers"]
        an.append(w["picking"] + w["packing"] + w["dispatch"])
        si.append(int(rec.iloc[i]["best_total_workers"]))

    fig, ax = plt.subplots(figsize=(6.3, 3.1))
    x = np.arange(12)
    ax.bar(x - 0.19, an, width=0.38, color="#9BB1C4", label="Analytical estimate (anchor)")
    ax.bar(x + 0.19, si, width=0.38, color="#3E5C76", label="Recommended after simulation")
    for i, (a, s) in enumerate(zip(an, si)):
        d = s - a
        if d != 0:
            ax.text(i + 0.19, s + 0.7, f"{d:+d}", ha="center", va="bottom",
                    fontsize=6.5, color="#B3452F")
    ax.set_xticks(x)
    ax.set_xticklabels(MONTH_ABBR)
    ax.set_ylabel("Total workforce (monthly FTE)")
    ax.set_ylim(0, max(max(an), max(si)) * 1.22)
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.16))
    ax.set_title("Analytical capacity anchor vs simulated recommendation", pad=22)
    print("  analytical vs recommended (total FTE):")
    print("    " + ", ".join(f"{MONTH_ABBR[i]} {an[i]}->{si[i]}" for i in range(12)))
    return fig, an, si


def fig_stage_pressure(bn: dict):
    """F8 — stacked Stage Pressure components for the primary stage each month, plus the
    ranking of all three stages."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.3, 5.0), sharex=True,
                                   gridspec_kw={"height_ratios": [1.15, 1]})
    x = np.arange(12)

    # Top: pressure score of each stage, per month.
    for s in STAGES:
        vals = []
        for m in bn["months"]:
            row = next(r for r in m["bottleneck_ranking"] if r["stage"] == s)
            vals.append(row["pressure_score"])
        ax1.plot(x, vals, marker="o", markersize=3.6, linewidth=1.5,
                 color=STAGE_COLOUR[s], label=STAGE_LABEL[s])
    ax1.set_ylabel("Stage Pressure score")
    ax1.set_ylim(0, 1.12)
    ax1.legend(loc="lower right", ncol=3)
    ax1.set_title("Stage Pressure by stage and month, and its decomposition")

    # Bottom: decomposition of the PRIMARY stage's score.
    comps = ["utilisation_component", "wait_component", "late_wait_component", "queue_component"]
    labels = ["Utilisation (0.40)", "p95 wait (0.25)", "Late-order wait (0.20)", "Queue (0.15)"]
    cols = ["#3E5C76", "#7A9E9F", "#C9A227", "#B3452F"]
    bottom = np.zeros(12)
    for c, lab, col in zip(comps, labels, cols):
        vals = np.array([m["bottleneck_ranking"][0][c] for m in bn["months"]])
        ax2.bar(x, vals, bottom=bottom, width=0.62, color=col, label=lab)
        bottom += vals
    ax2.set_xticks(x)
    ax2.set_xticklabels(MONTH_ABBR)
    ax2.set_ylabel("Primary-stage score\ncontribution")
    ax2.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.16), fontsize=7)
    return fig


def fig_marginal_fte(bn: dict):
    """F9 — the +1 FTE decision at the identified bottleneck, per month."""
    rows = []
    for m in bn["months"]:
        ad = m.get("adaptive_search") or {}
        for t in (ad.get("trail") or []):
            rows.append({
                "month": MONTH_ABBR[m["month"] - 1],
                "parent": t["parent_regime"], "candidate": t["candidate_regime"],
                "stage": t["added_stage"], "labour": t["labour_cost_increase"],
                "penalty": t["late_penalty_reduction"], "net": t["total_cost_diff"],
                "accepted": t["accepted"], "reason": t.get("reason", ""),
            })
    tr = pd.DataFrame(rows)
    if tr.empty:
        return None, tr

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    x = np.arange(len(tr))
    ax.bar(x - 0.2, tr["labour"], width=0.4, color="#B3452F", label="Added labour cost")
    ax.bar(x + 0.2, tr["penalty"], width=0.4, color="#2E7D5B",
           label="Late-penalty reduction")
    for i, r in tr.iterrows():
        ax.text(i + 0.2, r["penalty"] + 60, f"{r['penalty']:.0f}", ha="center",
                va="bottom", fontsize=7)
        ax.text(i - 0.2, r["labour"] + 60, f"{r['labour']:.0f}", ha="center",
                va="bottom", fontsize=7)
        ax.text(i, -230, "rejected" if not r["accepted"] else "accepted", ha="center",
                fontsize=7, color="#B3452F" if not r["accepted"] else "#2E7D5B",
                style="italic")
    ax.set_xticks(x)
    ax.set_xticklabels(tr["month"], fontsize=8)
    ax.set_xlabel("Every tested candidate adds one picking FTE to that month's "
                  "recommended workforce", fontsize=7.5, labelpad=8)
    ax.set_ylabel("Monthly cost effect (EUR)")
    ax.set_ylim(-400, max(tr["labour"].max(), tr["penalty"].max()) * 1.3)
    ax.legend(loc="upper right")
    ax.set_title("Adding one FTE at the identified bottleneck: cost against benefit")
    return fig, tr


def fig_future(fut: pd.DataFrame):
    """F11 — December future planning: cost vs probability of meeting both SLA floors."""
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    val = fut[fut["evaluation_stage"] == "validated"]
    scr = fut[fut["evaluation_stage"] != "validated"]
    for p in ["fifo", "urgent_first", "rl3_dqn"]:
        d = val[val["policy"] == p]
        ax.scatter(d["estimated_total_cost"] / 1000.0, d["prob_meets_sla_targets"] * 100,
                   s=46, color=POLICY_COLOUR[p], label=f"{POLICY_LABEL[p]} (validated)",
                   marker="o", alpha=0.9, linewidths=0)
        d2 = scr[scr["policy"] == p]
        ax.scatter(d2["estimated_total_cost"] / 1000.0, d2["prob_meets_sla_targets"] * 100,
                   s=16, color=POLICY_COLOUR[p], marker="x", alpha=0.45, linewidths=0.8)
    ax.set_xlabel("Expected total cost (thousand EUR)")
    ax.set_ylabel("Replications meeting both SLA floors (%)")
    ax.set_ylim(-6, 108)
    ax.legend(loc="center right", fontsize=7)
    ax.set_title("Future planning, December: cost against feasibility reliability\n"
                 "(circles = validated on 3 replications, crosses = screened on 1)",
                 fontsize=9)
    return fig


def main() -> None:
    apply_style()
    rec = load_recommendations()
    bn = load_bottleneck("historical")

    fig, an, si = fig_analytical_vs_recommended(rec)
    save(fig, "f3_analytical_vs_recommended")

    save(fig_stage_pressure(bn), "f8_stage_pressure")

    fig9, trail = fig_marginal_fte(bn)
    if fig9 is not None:
        save(fig9, "f9_marginal_fte")
        save_table(trail, "t_adaptive_search")
        print("\n  adaptive-search trail:")
        print(trail.to_string(index=False))

    # Monthly recommendation table.
    t = pd.DataFrame({
        "month": MONTH_ABBR,
        "orders": rec["total_orders"].astype(int),
        "analytical_fte": an,
        "regime": rec["best_total_regime"],
        "policy": rec["best_total_policy"].map(POLICY_LABEL),
        "fte": rec["best_total_workers"].astype(int),
        "urgent_sla": (rec["best_total_urgent_sla"] * 100).round(1),
        "normal_sla": (rec["best_total_normal_sla"] * 100).round(1),
        "total_cost": rec["best_total_cost"].astype(int),
    })
    save_table(t, "t_monthly_recommendations")
    print("\n  monthly recommendations:")
    print(t.to_string(index=False))

    # Bottleneck ranking table.
    rows = []
    for m in bn["months"]:
        top = m["bottleneck_ranking"][0]
        rows.append({"month": MONTH_ABBR[m["month"] - 1], "primary": STAGE_LABEL[top["stage"]],
                     "pressure": top["pressure_score"],
                     "utilisation": round(top["utilisation"] * 100, 1),
                     "p95_wait_min": top["p95_wait_min"],
                     "late_wait_share": round(top["late_wait_share"] * 100, 1)})
    bt = pd.DataFrame(rows)
    save_table(bt, "t_bottleneck_ranking")
    print("\n  bottleneck ranking:")
    print(bt.to_string(index=False))

    save(fig_future(load_results("future")), "f11_future_planning")


if __name__ == "__main__":
    main()
