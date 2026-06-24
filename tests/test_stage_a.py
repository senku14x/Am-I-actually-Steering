"""CPU tests for the Stage-A statistics on SYNTHETIC arrays (no real data, no GPU)."""
import numpy as np

from strat_geom.calibration import calibrate
from strat_geom.cooccur import all_pairs_phi, phi_coefficient, presence_matrix
from strat_geom.dom import cosine_matrix, dom_cosine, dom_vector
from strat_geom.metrics import cosine, fisher_ratio, robust_regression
from strat_geom.nulls import null_a, null_b
from strat_geom.probe import probe_vs_dom
from strat_geom.strata import within_stratum


def test_cosine_identities():
    v = np.array([1.0, 2.0, 3.0])
    assert abs(cosine(v, v) - 1.0) < 1e-9
    assert abs(cosine(v, -v) + 1.0) < 1e-9
    assert cosine(v, np.zeros(3)) == 0.0


def test_fisher_ratio_separable_is_large():
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 0.1, (50, 3))
    b = rng.normal(5.0, 0.1, (50, 3))
    assert fisher_ratio(a, b) > 50


def test_robust_regression_recovers_line():
    x = np.linspace(-1, 1, 30)
    y = 0.5 * x - 0.2
    fit = robust_regression(x, y)
    assert abs(fit.slope - 0.5) < 1e-6 and abs(fit.intercept + 0.2) < 1e-6 and fit.r2 > 0.999


def test_phi_coefficient_known():
    a = np.array([1, 1, 0, 0], dtype=bool)
    assert abs(phi_coefficient(a, a) - 1.0) < 1e-9
    assert abs(phi_coefficient(a, ~a) + 1.0) < 1e-9
    assert phi_coefficient(np.ones(4, bool), np.array([1, 0, 1, 0], bool)) == 0.0


def test_presence_and_all_pairs_phi():
    labels = ["a", "b", "c"]
    P = presence_matrix([["a", "b"], ["a"], ["b"], ["a", "b"]], labels)
    assert P.shape == (4, 3) and P[:, 0].sum() == 3 and P[:, 2].sum() == 0
    phis = all_pairs_phi(P, labels, min_count=1)
    assert (0, 1) in phis and (0, 2) not in phis           # c never occurs -> dropped


def test_dom_vector_meandiff_and_min_count():
    X = np.array([[2.0, 0.0], [2.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    mask = np.array([True, True, False, False])
    assert np.allclose(dom_vector(X, mask, 2), [2.0, 0.0])
    assert dom_vector(X, mask, 3) is None
    assert dom_cosine(X, mask, ~mask, 2) is not None


def test_cosine_matrix_keys():
    X = np.random.default_rng(0).normal(0, 1, (40, 6))
    P = np.random.default_rng(1).random((40, 3)) < 0.5
    cm = cosine_matrix(X, P, ["a", "b", "c"], min_count=5)
    assert all(i < j for (i, j) in cm)


def test_calibrate_recovers_linear_and_branches():
    labels = [f"l{i}" for i in range(10)]
    rng = np.random.default_rng(0)
    cosines, phis = {}, {}
    for i in range(10):
        for j in range(i + 1, 10):
            phi = float(rng.uniform(-1, 1))
            phis[(i, j)] = phi
            cosines[(i, j)] = -0.7 * phi + 0.05            # exact line -> r2 ~ 1
    cal = calibrate(cosines, phis, ("l0", "l1"), labels)
    assert cal["r2"] > 0.99 and cal["branch_hint"] in ("A", "B", "C")
    assert cal["headline_in_95_band"] is True


def test_null_b_breaks_label_activation_link():
    rng = np.random.default_rng(0)
    n = 200
    mi = np.zeros(n, bool)
    mj = np.zeros(n, bool)
    mi[:100] = True            # both(0:50) + i_only(50:100)
    mj[:50] = True
    mj[100:150] = True         # both(0:50) + j_only(100:150); neither = 150:200
    X = np.zeros((n, 4))
    X[:, 0] = mi.astype(float) - mj.astype(float)          # structure tied to TRUE membership
    X += rng.normal(0, 0.05, (n, 4))
    res = null_b(X, mi, mj, n=300, seed=0)
    assert res["observed"] < -0.5                          # genuinely anti-aligned
    assert res["null_mean"] > res["observed"] + 0.3        # null pulled toward 0


def test_null_a_shape():
    rng = np.random.default_rng(0)
    n = 60
    mi = rng.random(n) < 0.5
    mj = rng.random(n) < 0.5
    X = rng.normal(0, 1, (n, 4))
    chain_ids = [f"c{i // 10}" for i in range(n)]
    res = null_a(X, mi, mj, chain_ids, list(range(n)), n=50, seed=0)
    assert set(res) == {"observed", "null_mean", "pct_2_5", "p_value"}


def test_within_stratum_counts_subset():
    labels = ["a", "b", "payoff_analysis"]
    n = 20
    P = np.zeros((n, 3), bool)
    P[:10, 2] = True
    P[:, 0] = True
    P[:5, 1] = True
    X = np.random.default_rng(0).normal(0, 1, (n, 8))
    r = within_stratum(X, P, labels, "payoff_analysis", 1, ("a", "b"), min_count=2)
    assert r["n"] == 10


def test_probe_vs_dom_separable():
    rng = np.random.default_rng(0)
    n = 100
    mask = np.zeros(n, bool)
    mask[:50] = True
    X = np.zeros((n, 5))
    X[mask] = rng.normal(2.0, 0.5, (50, 5))
    X[~mask] = rng.normal(-2.0, 0.5, (50, 5))
    r = probe_vs_dom(X, mask, min_count=10)
    assert r["probe_auc"] > 0.95 and r["fisher_ratio"] > 1.0
