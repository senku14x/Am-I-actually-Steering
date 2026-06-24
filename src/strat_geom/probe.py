"""
Probe-vs-DoM orthogonality (SPEC §6) — an INDEPENDENT result, not artifact evidence.

A linear probe finds the model's actual discriminant direction for a label; the DoM is the
mean-difference direction. If they are far apart (low |cos|, while Fisher/AUC show the label IS
linearly decodable), the DoM "behavioral direction" is not the model's feature axis.
"""
from __future__ import annotations

import numpy as np

from .dom import dom_vector
from .metrics import cosine, fisher_ratio


def train_probe(X, y):
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

    clf = LogisticRegression(max_iter=1000).fit(X, y)
    w = clf.coef_[0]
    nw = np.linalg.norm(w)
    return (w / nw if nw > 0 else w), clf


def probe_vs_dom(X, mask, min_count: int):
    from sklearn.metrics import roc_auc_score  # noqa: PLC0415

    mask = np.asarray(mask, dtype=bool)
    if mask.sum() < min_count or (~mask).sum() < min_count:
        return None
    w, clf = train_probe(X, mask.astype(int))
    dom = dom_vector(X, mask, min_count)
    dn = np.linalg.norm(dom)
    domu = dom / dn if dn > 0 else dom
    return {
        "cos_probe_dom": abs(float(cosine(w, domu))),       # probe-weight sign is arbitrary
        "fisher_ratio": float(fisher_ratio(X[~mask], X[mask])),
        "probe_auc": float(roc_auc_score(mask.astype(int), clf.decision_function(X))),
    }
