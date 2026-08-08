"""
Workforce regime label formatting/parsing — the single implementation used everywhere a
(picking, packing, dispatch) worker tuple needs a short label or vice versa.

Static base regimes (configs/planning_profile.yaml::regimes) all have single-digit worker
counts, so the historical `sNNN` format (e.g. "s432" = 4 picking / 3 packing / 2 dispatch) reads
unambiguously. Under the corrected operating-time capacity model, analytically-estimated /
dynamically-generated candidates and adaptive-search results can legitimately need >=10 workers
in a stage at peak demand — "s1063" would be ambiguous (10/6/3? 1/06/3? 1/0/63?). Any regime
with a worker count >= 10 therefore uses an explicit underscore-delimited format instead:
"s10_6_3". The compact form is kept for the common case so existing base-regime labels
(s111 … s432) are unchanged.
"""
from __future__ import annotations

import re
from typing import Tuple

_COMPACT_RE = re.compile(r"^s(\d)(\d)(\d)$")
_EXPANDED_RE = re.compile(r"^s(\d+)_(\d+)_(\d+)$")


def format_regime(picking: int, packing: int, dispatch: int) -> str:
    """s432 when every stage is a single digit, else the unambiguous s10_6_3 form."""
    p, k, d = int(picking), int(packing), int(dispatch)
    if p < 10 and k < 10 and d < 10:
        return f"s{p}{k}{d}"
    return f"s{p}_{k}_{d}"


def parse_regime(label: str) -> Tuple[int, int, int]:
    """Inverse of format_regime. Accepts both the compact (s432) and expanded (s10_6_3) forms."""
    if not isinstance(label, str):
        raise ValueError(f"Regime label must be a string, got {label!r}")

    m = _EXPANDED_RE.match(label)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))

    m = _COMPACT_RE.match(label)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))

    raise ValueError(
        f"Unrecognised regime label: {label!r}. Expected 'sPKD' (single-digit workers, e.g. "
        "'s432') or 's P_K_D' (any worker count, e.g. 's10_6_3')."
    )


def total_workers(label: str) -> int:
    return sum(parse_regime(label))
