"""
A1 (CONFIRM) — co-occurrence-preserving nulls for the headline DoM cosine (SPEC §3-A1).

Null B (exact): partition segments into the four joint cells {neither, i-only, j-only, both} and
REASSIGN which segments occupy each cell while keeping cell SIZES fixed. Marginals and co-occurrence
are preserved exactly; only the link between labels and activations is broken. Null A is a softer
robustness check that permutes labels within (chain, position-tercile) bins.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .metrics import cosine


def _dom_cos(X, mi, mj) -> float:
    di = X[mi].mean(0) - X[~mi].mean(0)
    dj = X[mj].mean(0) - X[~mj].mean(0)
    return cosine(di, dj)


def _summary(observed: float, null: np.ndarray) -> dict:
    return {
        "observed": float(observed),
        "null_mean": float(null.mean()),
        "pct_2_5": float(np.percentile(null, 2.5)),
        "p_value": float((np.sum(null <= observed) + 1) / (len(null) + 1)),
    }


def null_b(X, mi, mj, n: int = 1000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    mi = np.asarray(mi, dtype=bool)
    mj = np.asarray(mj, dtype=bool)
    code = mi.astype(int) * 2 + mj.astype(int)        # 0 neither, 1 j-only, 2 i-only, 3 both
    observed = _dom_cos(X, mi, mj)
    null = np.empty(n)
    for t in range(n):
        perm = rng.permutation(code)                  # fixed cell sizes, segments reassigned
        null[t] = _dom_cos(X, perm >= 2, perm % 2 == 1)
    return _summary(observed, null)


def null_a(X, mi, mj, chain_ids, positions, n: int = 1000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    mi = np.asarray(mi, dtype=bool)
    mj = np.asarray(mj, dtype=bool)
    code = mi.astype(int) * 2 + mj.astype(int)
    bins = _tercile_bins(chain_ids, positions)
    observed = _dom_cos(X, mi, mj)
    null = np.empty(n)
    for t in range(n):
        perm = code.copy()
        for ids in bins:
            perm[ids] = rng.permutation(code[ids])
        null[t] = _dom_cos(X, perm >= 2, perm % 2 == 1)
    return _summary(observed, null)


def _tercile_bins(chain_ids, positions) -> list:
    by_chain: dict = defaultdict(list)
    for i, (c, p) in enumerate(zip(chain_ids, positions)):
        by_chain[c].append((p, i))
    bins = []
    for lst in by_chain.values():
        lst.sort()
        m = len(lst)
        terc: dict = defaultdict(list)
        for rank, (_, i) in enumerate(lst):
            terc[min(2, rank * 3 // m) if m else 0].append(i)
        bins.extend(np.array(v) for v in terc.values())
    return bins
