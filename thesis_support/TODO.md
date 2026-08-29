# THESIS TODO

## Outstanding (author action)
- [ ] Replace `thesis/frontmatter/titlepage.tex` with the official IMIM cover page.
      The file is self-contained; nothing else in the project depends on it.
      It currently holds the only two placeholders: TODO_TUTOR_FULL_NAME and
      TODO_SUBMISSION_MONTH_YEAR.

## Optional
- [ ] Frontend screenshots as an Appendix G. Not captured: producing a live recommendation
      needs the RL checkpoint, which is absent from the working tree.
- [ ] Further references beyond the current 35. Any addition must be CrossRef-verified first
      (see LITERATURE_LEDGER for the rule and the six rejections it caught).

## Completed
- [x] Phase A - system reconstruction and fact verification
- [x] Phase B - literature discovery and CrossRef verification (35 entries)
- [x] Phase C - architecture lock (THESIS_ARCHITECTURE.md)
- [x] Phase D - LaTeX project, compiles clean
- [x] Phase E - 11 quantitative figures + 6 TikZ diagrams + 33 tables, all reproducible
- [x] Phases F-K - all chapters and front matter written
- [x] Phase L - hostile review (4 advisor consultations)
- [x] Phase M - final polish; all 17 figures visually inspected in the rendered PDF
- [x] Manual compliance gate - all word-count minimums exceeded
- [x] UNESCO codes verified against the official nomenclature (two were wrong, now corrected)

## Environment notes
- Use `py -3.12` (has pandas/numpy/simpy/matplotlib). Default `python` is 3.14 and has none.
- torch is NOT installed and the RL checkpoint is absent, so RL results cannot be regenerated.
  Documented as a limitation in Appendix E and Chapter 6.
- Set PYTHONIOENCODING=utf-8 when printing non-ASCII from scripts.
- Write LaTeX with the Write/Edit tools, NOT bash heredocs: heredocs collapse a double
  backslash to a single one and silently break tables and line breaks.
- Regenerate all figures/tables:
      cd thesis_support/analysis
      py -3.12 fig_demand.py && py -3.12 fig_policy.py && py -3.12 fig_capacity_bottleneck.py
      py -3.12 fig_equal_workforce.py && py -3.12 gen_appendix_d.py
  Then rebuild: cd thesis && latexmk -pdf main.tex
