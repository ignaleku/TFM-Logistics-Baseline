"""
THESIS-ONLY. Figures F1-F2: demand profile and stage-differentiated workload (RQ1).

F1  Monthly order volume with the urgent share overlaid.
F2  Mean workload units per order by stage and month, showing that one order does NOT impose
    equal workload at the three stages.

Reads : data/uploads/orders_uploaded.csv (the synthetic order-level dataset AS RUN)
Writes: thesis/figures/f1_demand_profile.pdf, thesis/figures/f2_stage_workload.pdf
        thesis/tables/t_demand_profile.csv
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from common import (MONTH_ABBR, STAGES, STAGE_COLOUR, STAGE_LABEL, apply_style,
                    load_orders, save, save_table)


def build_monthly(orders: pd.DataFrame) -> pd.DataFrame:
    g = orders.groupby("month")
    df = pd.DataFrame({
        "month": range(1, 13),
        "orders": g.size().reindex(range(1, 13)).values,
        "urgent_share": g.apply(
            lambda d: (d["order_type"] == "urgent").mean(), include_groups=False
        ).reindex(range(1, 13)).values,
        "mean_items": g["num_items"].mean().reindex(range(1, 13)).values,
    })
    for s in STAGES:
        df[f"{s}_units"] = g[f"{s}_units"].mean().reindex(range(1, 13)).values
    return df


def fig_demand(m: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    bars = ax.bar(MONTH_ABBR, m["orders"] / 1000.0, color="#3E5C76", width=0.62,
                  label="Order volume")
    ax.set_ylabel("Orders (thousands)")
    # No per-bar value labels: the urgent-share line crosses the bar tops, so labels there
    # collide with it. Exact values are given in the accompanying table.
    ax.set_ylim(0, m["orders"].max() / 1000.0 * 1.18)

    ax2 = ax.twinx()
    ax2.plot(MONTH_ABBR, m["urgent_share"] * 100, color="#B3452F", marker="o",
             markersize=4, linewidth=1.6, label="Urgent share")
    ax2.set_ylabel("Urgent orders (%)", color="#B3452F")
    ax2.tick_params(axis="y", colors="#B3452F")
    ax2.set_ylim(0, 30)
    ax2.grid(False)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.16))
    ax.set_title("Monthly order volume and urgency mix", pad=22)
    return fig


def fig_workload(m: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    for s in STAGES:
        ax.plot(MONTH_ABBR, m[f"{s}_units"], marker="o", markersize=4, linewidth=1.6,
                color=STAGE_COLOUR[s], label=STAGE_LABEL[s])
    ax.set_ylabel("Mean workload units per order")
    ax.set_ylim(0, m[[f"{s}_units" for s in STAGES]].to_numpy().max() * 1.18)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.15))
    ax.set_title("Mean workload units per order, by stage", pad=20)
    return fig


def main() -> None:
    apply_style()
    orders = load_orders()
    m = build_monthly(orders)

    print(f"  orders loaded: {len(orders):,}  scenario={orders['scenario'].unique().tolist()}")
    ratio = m[[f"{s}_units" for s in STAGES]].mean()
    print("  annual mean units/order by stage: "
          + ", ".join(f"{STAGE_LABEL[s]}={ratio[f'{s}_units']:.2f}" for s in STAGES))
    print(f"  picking:packing:dispatch ratio = "
          f"1 : {ratio['packing_units']/ratio['picking_units']:.2f} : "
          f"{ratio['dispatch_units']/ratio['picking_units']:.2f}")

    save(fig_demand(m), "f1_demand_profile")
    save(fig_workload(m), "f2_stage_workload")

    out = m.copy()
    out["month_name"] = [MONTH_ABBR[i] for i in range(12)]
    out["urgent_share"] = (out["urgent_share"] * 100).round(1)
    for s in STAGES:
        out[f"{s}_units"] = out[f"{s}_units"].round(2)
    out["mean_items"] = out["mean_items"].round(2)
    save_table(out[["month_name", "orders", "urgent_share", "mean_items",
                    "picking_units", "packing_units", "dispatch_units"]],
               "t_demand_profile")


if __name__ == "__main__":
    main()
