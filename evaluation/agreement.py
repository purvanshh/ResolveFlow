"""Agreement statistics for ordinal ratings."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def exact_agreement(a: Iterable[float], b: Iterable[float]) -> float:
    aa, bb = list(a), list(b)
    if not aa:
        return 0.0
    return float(np.mean([x == y for x, y in zip(aa, bb, strict=False)]))


def within_one_agreement(a: Iterable[float], b: Iterable[float]) -> float:
    aa, bb = list(a), list(b)
    if not aa:
        return 0.0
    return float(np.mean([abs(x - y) <= 1 for x, y in zip(aa, bb, strict=False)]))


def spearman(a: Iterable[float], b: Iterable[float]) -> float:
    aa = np.asarray(list(a), dtype=float)
    bb = np.asarray(list(b), dtype=float)
    if len(aa) < 2 or np.std(aa) == 0 or np.std(bb) == 0:
        return 0.0
    ra = aa.argsort().argsort().astype(float)
    rb = bb.argsort().argsort().astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def weighted_cohen_kappa(a: Iterable[float], b: Iterable[float], grades: int = 5) -> float:
    """Quadratic weighted kappa for ordinal scores 1..grades."""
    aa = [int(x) for x in a]
    bb = [int(x) for x in b]
    n = len(aa)
    if n == 0:
        return 0.0
    O = np.zeros((grades, grades), dtype=float)
    for x, y in zip(aa, bb, strict=False):
        i = max(0, min(grades - 1, x - 1))
        j = max(0, min(grades - 1, y - 1))
        O[i, j] += 1
    O /= O.sum()
    row = O.sum(axis=1)
    col = O.sum(axis=0)
    E = np.outer(row, col)
    W = np.zeros((grades, grades), dtype=float)
    for i in range(grades):
        for j in range(grades):
            W[i, j] = ((i - j) ** 2) / ((grades - 1) ** 2)
    den = (W * E).sum()
    if den == 0:
        return 1.0
    return float(1.0 - (W * O).sum() / den)


def dimension_agreement(
    human_rows: list[dict], judge_rows: list[dict], dimensions: list[str]
) -> dict:
    out = {}
    # align by example_id
    jmap = {r["example_id"]: r for r in judge_rows}
    for dim in dimensions:
        hvals, jvals = [], []
        for h in human_rows:
            j = jmap.get(h["example_id"])
            if not j or dim not in h or dim not in j:
                continue
            hvals.append(float(h[dim]))
            jvals.append(float(j[dim]))
        out[dim] = {
            "n": len(hvals),
            "exact_agreement": exact_agreement(hvals, jvals),
            "within_one_agreement": within_one_agreement(hvals, jvals),
            "spearman": spearman(hvals, jvals),
            "weighted_kappa": weighted_cohen_kappa(hvals, jvals),
        }
    return out
