"""A3 (SUPPORT) — within-stratum control: recompute the headline cosine inside payoff strata.

Rules out a composition artifact (SPEC §3-A3): if opp/ded anti-alignment is just payoff_analysis
mixing, it collapses inside payoff_analysis=1 (and =0). Report per-class counts — deduction thins
within payoff=0, so small strata are descriptive only.
"""
from __future__ import annotations

from .dom import dom_cosine


def within_stratum(X, presence, labels, stratum_label, value, headline_pair, min_count) -> dict:
    idx = {lab: i for i, lab in enumerate(labels)}
    sel = presence[:, idx[stratum_label]]
    if not value:
        sel = ~sel
    Xs, Ps = X[sel], presence[sel]
    cos = dom_cosine(Xs, Ps[:, idx[headline_pair[0]]], Ps[:, idx[headline_pair[1]]], min_count)
    return {"stratum": f"{stratum_label}={int(bool(value))}", "cosine": cos, "n": int(sel.sum())}
