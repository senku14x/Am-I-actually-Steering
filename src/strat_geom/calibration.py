"""
A2 (LEAD) — complementary-label-pair calibration (SPEC §3-A2).

Regress DoM cosine on label-complementarity (phi) across all label pairs. If opp/ded sits on the
complementarity line (high R2, headline inside the band) the headline geometry is a labelling
artifact (branch B); if it is a negative outlier beyond the band it is special (branch A).
"""
from __future__ import annotations

import numpy as np

from .metrics import robust_regression


def calibrate(cosines: dict, phis: dict, headline_pair, labels) -> dict:
    idx = {lab: i for i, lab in enumerate(labels)}
    hkey = tuple(sorted((idx[headline_pair[0]], idx[headline_pair[1]])))
    pairs = sorted(k for k in cosines if k in phis)
    if len(pairs) < 3:
        return {"n_pairs": len(pairs), "error": "too few pairs for calibration"}

    x = np.array([phis[k] for k in pairs])
    y = np.array([cosines[k] for k in pairs])
    fit = robust_regression(x, y)
    out: dict = {
        "n_pairs": len(pairs), "r2": fit.r2, "slope": fit.slope, "intercept": fit.intercept,
        "headline_pair": list(headline_pair),
    }
    if hkey in pairs:
        hi = pairs.index(hkey)
        hw = float(fit.pred_interval_halfwidth(np.array([x[hi]]))[0])
        hresid = float(y[hi] - float(fit.predict(np.array([x[hi]]))[0]))
        out.update({
            "headline_cosine": float(y[hi]), "headline_phi": float(x[hi]),
            "headline_studentized_resid": float(fit.studentized[hi]),
            "headline_residual": hresid, "headline_pred_band_halfwidth": hw,
            "headline_below_band": bool(hresid < -hw),
            "headline_in_95_band": bool(abs(hresid) <= hw),
        })
    else:
        out["headline_present"] = False

    # built-in positive control: the most-complementary pair (min phi) should be most anti-aligned
    minphi = min(phis, key=lambda k: phis[k])
    out["most_complementary_pair"] = [labels[minphi[0]], labels[minphi[1]]]
    out["most_complementary_phi"] = float(phis[minphi])
    out["most_complementary_cosine"] = float(cosines.get(minphi, float("nan")))

    # branch hint (SPEC §4): outlier-below+|t|>2 -> A; on-line high-R2 -> B; else C
    stud, below = out.get("headline_studentized_resid"), out.get("headline_below_band")
    if below and stud is not None and abs(stud) > 2:
        out["branch_hint"] = "A"
    elif fit.r2 > 0.5 and out.get("headline_in_95_band"):
        out["branch_hint"] = "B"
    else:
        out["branch_hint"] = "C"
    return out
