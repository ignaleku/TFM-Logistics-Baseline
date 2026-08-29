"""
THESIS-ONLY. Approximate word counts per chapter, for checking compliance with the
IMIM Master's Thesis Manual minimums. Strips LaTeX commands, tables, figures and equations,
so the count reflects prose only and is deliberately conservative.
"""
from __future__ import annotations

import re
from pathlib import Path

THESIS = Path(__file__).resolve().parents[2] / "thesis"

MINIMUMS = {
    "02_theoretical_foundations": ("Theoretical basis", 3000),
    "04_methodology":             ("Methodology", 3000),
    "05_results":                 ("Findings / Results", 4000),
    "06_discussion":              ("Discussion", 3000),
    "07_conclusions":             ("Conclusions", 1000),
}

FLOAT_ENVS = ("tabular", "tabularx", "longtable", "table", "figure",
              "equation", "center", "footnotesize")


def count_words(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"(?<!\\)%.*", "", text)
    for env in FLOAT_ENVS:
        text = re.sub(rf"\\begin\{{{env}\*?\}}.*?\\end\{{{env}\*?\}}", " ", text, flags=re.S)
    text = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", text)
    text = re.sub(r"[{}$&\\_^~#]", " ", text)
    return len([w for w in text.split() if re.search(r"[A-Za-z]", w)])


def main() -> None:
    print(f"{'File':<34}{'Words':>7}   Manual requirement")
    print("-" * 74)

    total = 0
    failures = []
    for path in sorted((THESIS / "chapters").glob("*.tex")):
        key = path.stem
        n = count_words(path)
        total += n
        note = ""
        if key in MINIMUMS:
            label, minimum = MINIMUMS[key]
            if n >= minimum:
                note = f"   OK  {label} (min {minimum:,})"
            else:
                note = f"   BELOW MINIMUM: {label} needs {minimum:,}"
                failures.append((label, n, minimum))
        print(f"{key:<34}{n:>7}{note}")

    print("-" * 74)
    print(f"{'CHAPTERS TOTAL':<34}{total:>7}")

    extra = 0
    for folder in ("appendices", "frontmatter"):
        sub = sum(count_words(p) for p in (THESIS / folder).glob("*.tex"))
        extra += sub
        print(f"{folder:<34}{sub:>7}")
    print(f"{'GRAND TOTAL':<34}{total + extra:>7}")

    print()
    if failures:
        print("FAILS MANUAL MINIMUMS:")
        for label, n, minimum in failures:
            print(f"  - {label}: {n:,} words, needs {minimum:,} ({minimum - n:,} short)")
    else:
        print("All manual word-count minimums satisfied.")


if __name__ == "__main__":
    main()
