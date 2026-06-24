"""Shared statistics for Stage-A geometry (pure numpy/scipy/sklearn; CPU)."""
from __future__ import annotations

import numpy as np


def cosine(a, b) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class RobustFit:
    """Theil-Sen robust line y~x plus residual diagnostics (SPEC §3-A2).

    Slope/intercept are the robust Theil-Sen estimates; r2, residuals, internally-studentized
    residuals and the prediction band are derived from that fit with OLS-style leverage on x.
    """

    def __init__(self, x, y):
        from sklearn.linear_model import TheilSenRegressor

        self.x = np.asarray(x, dtype=float).ravel()
        self.y = np.asarray(y, dtype=float).ravel()
        self.n = self.x.size
        ts = TheilSenRegressor(random_state=0).fit(self.x.reshape(-1, 1), self.y)
        self.slope = float(ts.coef_[0])
        self.intercept = float(ts.intercept_)
        self.resid = self.y - self.predict(self.x)
        ss_res = float(np.sum(self.resid ** 2))
        ss_tot = float(np.sum((self.y - self.y.mean()) ** 2))
        self.r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        xbar = self.x.mean()
        self._sxx = float(np.sum((self.x - xbar) ** 2))
        self.leverage = (1.0 / self.n + (self.x - xbar) ** 2 / self._sxx
                         if self._sxx > 0 else np.full(self.n, 1.0 / self.n))
        self.sigma = float(np.sqrt(ss_res / max(self.n - 2, 1)))
        denom = self.sigma * np.sqrt(np.clip(1.0 - self.leverage, 1e-12, None))
        self.studentized = self.resid / np.where(denom > 0, denom, 1e-12)

    def predict(self, xq):
        return self.slope * np.asarray(xq, dtype=float) + self.intercept

    def pred_interval_halfwidth(self, xq, level: float = 0.95):
        from scipy.stats import t

        xq = np.asarray(xq, dtype=float)
        xbar = self.x.mean()
        if self._sxx > 0:
            se = self.sigma * np.sqrt(1.0 + 1.0 / self.n + (xq - xbar) ** 2 / self._sxx)
        else:
            se = self.sigma * np.sqrt(1.0 + 1.0 / self.n) * np.ones_like(xq)
        tcrit = float(t.ppf(0.5 + level / 2.0, max(self.n - 2, 1)))
        return tcrit * se


def robust_regression(x, y) -> RobustFit:
    return RobustFit(x, y)


def fisher_ratio(v0, v1) -> float:
    """(between-class / within-class) variance along the class-mean-difference direction."""
    v0 = np.asarray(v0, dtype=float)
    v1 = np.asarray(v1, dtype=float)
    w = v1.mean(0) - v0.mean(0)
    nw = np.linalg.norm(w)
    if nw == 0:
        return 0.0
    w = w / nw
    p0, p1 = v0 @ w, v1 @ w
    within = float(p0.var() + p1.var())
    between = float((p1.mean() - p0.mean()) ** 2)
    return between / within if within > 0 else float("inf")
