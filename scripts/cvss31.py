"""CVSS v3.1 base-score calculator (FIRST spec, section 7), no external deps."""

from __future__ import annotations

import math
import re

VECTOR_RE = re.compile(
    r"CVSS:3\.[01]/AV:[NALP]/AC:[LH]/PR:[NLH]/UI:[NR]/S:[UC]/C:[NLH]/I:[NLH]/A:[NLH]"
)

_AV = {"N": 0.85, "A": 0.62, "L": 0.55, "P": 0.2}
_AC = {"L": 0.77, "H": 0.44}
_PR_UNCHANGED = {"N": 0.85, "L": 0.62, "H": 0.27}
_PR_CHANGED = {"N": 0.85, "L": 0.68, "H": 0.5}
_UI = {"N": 0.85, "R": 0.62}
_CIA = {"H": 0.56, "L": 0.22, "N": 0.0}


def _roundup(x: float) -> float:
    i = round(x * 100_000)
    if i % 10_000 == 0:
        return i / 100_000.0
    return (math.floor(i / 10_000) + 1) / 10.0


def base_score(vector: str) -> float | None:
    """Return the 0.0-10.0 base score, or None if the vector is malformed."""
    m = VECTOR_RE.search(vector or "")
    if not m:
        return None
    parts = dict(p.split(":", 1) for p in m.group(0).split("/")[1:])

    changed = parts["S"] == "C"
    pr = (_PR_CHANGED if changed else _PR_UNCHANGED)[parts["PR"]]
    iss = 1 - (
        (1 - _CIA[parts["C"]]) * (1 - _CIA[parts["I"]]) * (1 - _CIA[parts["A"]])
    )
    if changed:
        impact = 7.52 * (iss - 0.029) - 3.25 * (iss - 0.02) ** 15
    else:
        impact = 6.42 * iss
    exploitability = 8.22 * _AV[parts["AV"]] * _AC[parts["AC"]] * pr * _UI[parts["UI"]]

    if impact <= 0:
        return 0.0
    if changed:
        return _roundup(min(1.08 * (impact + exploitability), 10))
    return _roundup(min(impact + exploitability, 10))


def extract_last_vector(text: str) -> str | None:
    matches = VECTOR_RE.findall(text or "")
    return matches[-1] if matches else None


if __name__ == "__main__":
    for v, want in [
        ("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H", 9.8),
        ("CVSS:3.1/AV:L/AC:L/PR:L/UI:N/S:U/C:N/I:N/A:H", 5.5),
        ("CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:C/C:L/I:L/A:N", 6.1),
        ("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:N", 0.0),
    ]:
        got = base_score(v)
        print(f"{'OK ' if got == want else 'BAD'} {v} -> {got} (expected {want})")
