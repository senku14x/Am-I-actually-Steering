"""Label co-occurrence: presence matrix + phi coefficient (the A2 x-axis; SPEC §3-A2)."""
from __future__ import annotations

import numpy as np


def presence_matrix(seg_label_lists, labels) -> np.ndarray:
    """[N_segments, n_labels] binary matrix from per-segment label lists."""
    idx = {lab: i for i, lab in enumerate(labels)}
    p = np.zeros((len(seg_label_lists), len(labels)), dtype=bool)
    for r, labs in enumerate(seg_label_lists):
        for lab in labs:
            if lab in idx:
                p[r, idx[lab]] = True
    return p


def phi_coefficient(a, b) -> float:
    """Phi (mean-square-contingency) for two binary arrays via the 2x2 table. 0.0 if degenerate."""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    n11 = float(np.sum(a & b))
    n10 = float(np.sum(a & ~b))
    n01 = float(np.sum(~a & b))
    n00 = float(np.sum(~a & ~b))
    den = np.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    if den == 0:
        return 0.0
    return float((n11 * n00 - n10 * n01) / den)


def all_pairs_phi(presence: np.ndarray, labels, min_count: int) -> dict:
    """phi for every label pair where both labels occur >= min_count times. Keys are (i, j), i<j."""
    counts = presence.sum(0)
    out: dict[tuple[int, int], float] = {}
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if counts[i] >= min_count and counts[j] >= min_count:
                out[(i, j)] = phi_coefficient(presence[:, i], presence[:, j])
    return out
