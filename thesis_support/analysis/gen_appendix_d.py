"""
THESIS-ONLY. Generates appendix D (complete results) as LaTeX directly from the persisted run
outputs, so the appendix cannot drift from the data through manual transcription.

Writes: thesis/appendices/appendix_D_results.tex
"""
from __future__ import annotations

import pandas as pd

from common import MONTH_ABBR, POLICIES, POLICY_LABEL, ROOT, load_results

OUT = ROOT / "thesis" / "appendices" / "appendix_D_results.tex"

HEADER = r"""\chapter{Complete Results}
\label{app:D}

This appendix reports the full experimental grid summarised in \Cref{ch:results}. All values are
taken directly from the persisted run outputs; the tables are generated programmatically from
those files rather than transcribed.

\section{Retrospective run: all configurations}

\Cref{tab:app-full} lists every month, workforce configuration and policy evaluated --- 576 rows
in total. Costs are in euros under the retrospective run's economic assumptions
(\Cref{tab:sla-econ}) and are not comparable with the forecast run.

\begin{footnotesize}
\begin{longtable}{llrrrrrlr}
\caption{Complete retrospective results: 12 months $\times$ 16 configurations $\times$ 3
policies.}\label{tab:app-full}\\
\toprule
\textbf{Month} & \textbf{Config.} & \textbf{FTE} & \textbf{Policy} & \textbf{Urg.\ \%} &
\textbf{Nor.\ \%} & \textbf{Tot.\ \%} & \textbf{Feas.} & \textbf{Cost} \\
\midrule
\endfirsthead
\multicolumn{9}{l}{\emph{\Cref{tab:app-full} continued from previous page}}\\
\toprule
\textbf{Month} & \textbf{Config.} & \textbf{FTE} & \textbf{Policy} & \textbf{Urg.\ \%} &
\textbf{Nor.\ \%} & \textbf{Tot.\ \%} & \textbf{Feas.} & \textbf{Cost} \\
\midrule
\endhead
\midrule \multicolumn{9}{r}{\emph{continued on next page}}\\
\endfoot
\bottomrule
\endlastfoot
"""

FOOTER_HIST = r"""\end{longtable}
\end{footnotesize}
"""


def esc(label: str) -> str:
    return label.replace("_", r"\_")


def policy_abbr(p: str) -> str:
    return {"fifo": "FIFO", "urgent_first": "UF", "rl3_dqn": "RL-3"}[p]


def main() -> None:
    hist = load_results("historical")
    lines = [HEADER]

    for month in range(1, 13):
        md = hist[hist["month"] == month]
        regimes = list(dict.fromkeys(md["regime"]))
        for regime in regimes:
            for j, pol in enumerate(POLICIES):
                r = md[(md["regime"] == regime) & (md["policy"] == pol)]
                if r.empty:
                    continue
                r = r.iloc[0]
                month_cell = MONTH_ABBR[month - 1] if (regime == regimes[0] and j == 0) else ""
                regime_cell = f"\\rgm{{{esc(regime)}}}" if j == 0 else ""
                fte_cell = f"{int(r['total_workers'])}" if j == 0 else ""
                lines.append(
                    f"{month_cell} & {regime_cell} & {fte_cell} & {policy_abbr(pol)} & "
                    f"{r['urgent_sla']*100:.1f} & {r['normal_sla']*100:.1f} & "
                    f"{r['total_sla']*100:.1f} & "
                    f"{'yes' if r['feasible'] else 'no'} & "
                    f"{int(round(r['estimated_total_cost'])):,} \\\\".replace(",", "\\,")
                )
        lines.append(r"\midrule")

    lines.append(FOOTER_HIST)

    # ── Forecast run ───────────────────────────────────────────────────────────────────────
    fut = load_results("future")
    lines.append(r"""
\section{Forecast run: December candidates}

\Cref{tab:app-future} lists every candidate evaluated by the forecast workflow for December.
Values are means across replications for validated candidates and single-replication values for
screened candidates. Costs use the forecast run's economic assumptions and are not comparable
with the retrospective tables above.

\begin{footnotesize}
\begin{longtable}{llrrrrrlr}
\caption{Complete forecast results for December.}\label{tab:app-future}\\
\toprule
\textbf{Config.} & \textbf{Policy} & \textbf{FTE} & \textbf{Urg.\ \%} & \textbf{Nor.\ \%} &
\textbf{Tot.\ \%} & \textbf{Feas.\ prob.} & \textbf{Stage} & \textbf{Cost} \\
\midrule
\endfirsthead
\toprule
\textbf{Config.} & \textbf{Policy} & \textbf{FTE} & \textbf{Urg.\ \%} & \textbf{Nor.\ \%} &
\textbf{Tot.\ \%} & \textbf{Feas.\ prob.} & \textbf{Stage} & \textbf{Cost} \\
\midrule
\endhead
\bottomrule
\endlastfoot
""")

    for regime in dict.fromkeys(fut["regime"]):
        for j, pol in enumerate(POLICIES):
            r = fut[(fut["regime"] == regime) & (fut["policy"] == pol)]
            if r.empty:
                continue
            r = r.iloc[0]
            regime_cell = f"\\rgm{{{esc(regime)}}}" if j == 0 else ""
            fte_cell = f"{int(r['total_workers'])}" if j == 0 else ""
            stage = str(r.get("evaluation_stage", ""))
            lines.append(
                f"{regime_cell} & {policy_abbr(pol)} & {fte_cell} & "
                f"{r['urgent_sla']*100:.1f} & {r['normal_sla']*100:.1f} & "
                f"{r['total_sla']*100:.1f} & {r['prob_meets_sla_targets']*100:.0f}\\% & "
                f"{stage} & {int(round(r['estimated_total_cost'])):,} \\\\".replace(",", "\\,")
            )
        lines.append(r"\midrule")

    lines.append(r"""\end{longtable}
\end{footnotesize}
""")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote {OUT.relative_to(ROOT)}  ({len(hist)} historical + {len(fut)} forecast rows)")


if __name__ == "__main__":
    main()
