"""
Stage-A driver (SPEC §3) — observational geometry; no GPU, no interventions.

Per-model: A2 calibration + A1 nulls + A3 strata + probe-vs-DoM at the headline layer ->
artifacts/<model>/stage_a.json (+ dom_vectors.npz). Cross-model: the A2 cross-model leg (does
cosine-agreement across models exceed phi-agreement?) -> artifacts/cross_model.json.

  python scripts/run_stage1.py --model configs/model_llama8b.yaml     # one model (after pipeline)
  python scripts/run_stage1.py --cross                                # after all models are done
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from strat_geom.calibration import calibrate  # noqa: E402
from strat_geom.config import load_config  # noqa: E402
from strat_geom.cooccur import all_pairs_phi  # noqa: E402
from strat_geom.dom import cosine_matrix, dom_vector, load_model_segments  # noqa: E402
from strat_geom.io import artifact_path, provenance, write_json  # noqa: E402
from strat_geom.nulls import null_a, null_b  # noqa: E402
from strat_geom.probe import probe_vs_dom  # noqa: E402
from strat_geom.strata import within_stratum  # noqa: E402


def run_model(cfg, judge) -> dict:
    labels, layer = cfg.labels, cfg.headline_layer
    X, P, meta = load_model_segments(cfg, layer, judge)
    if len(X) == 0:
        raise SystemExit("no segments — run generate/segment/activations/annotate for this model first")
    idx = {lab: i for i, lab in enumerate(labels)}
    hi, hj = idx[cfg.headline_pair[0]], idx[cfg.headline_pair[1]]

    cosines = cosine_matrix(X, P, labels, cfg.min_count)
    phis = all_pairs_phi(P, labels, cfg.min_count)
    cal = calibrate(cosines, phis, cfg.headline_pair, labels)

    nulls = {}
    if P[:, hi].sum() >= cfg.min_count and P[:, hj].sum() >= cfg.min_count:
        chain_ids = [m["chain_id"] for m in meta]
        positions = [m["seg_idx"] for m in meta]
        nulls = {
            "null_b": null_b(X, P[:, hi], P[:, hj], n=cfg.n_permutation, seed=cfg.seed),
            "null_a": null_a(X, P[:, hi], P[:, hj], chain_ids, positions,
                             n=cfg.n_permutation, seed=cfg.seed),
        }

    strata = []
    if "payoff_analysis" in idx:
        for v in (1, 0):
            strata.append(within_stratum(X, P, labels, "payoff_analysis", v,
                                         cfg.headline_pair, cfg.min_count))

    probe = {lab: probe_vs_dom(X, P[:, idx[lab]], cfg.min_count) for lab in cfg.headline_pair}

    pairs = [{"a": labels[i], "b": labels[j], "cosine": cosines[(i, j)], "phi": phis.get((i, j))}
             for (i, j) in sorted(cosines)]
    result = {
        "provenance": provenance(cfg, judge=judge, headline_layer=layer, n_segments=len(X)),
        "headline_layer": layer, "n_segments": int(len(X)), "min_count": cfg.min_count,
        "headline_pair": list(cfg.headline_pair),
        "pairs": pairs, "calibration": cal, "nulls": nulls, "strata": strata, "probe": probe,
    }
    write_json(artifact_path(cfg, "stage_a.json"), result)
    doms = {lab: dom_vector(X, P[:, idx[lab]], cfg.min_count) for lab in labels}
    doms = {lab: v.astype(np.float32) for lab, v in doms.items() if v is not None}
    if doms:
        np.savez(artifact_path(cfg, "dom_vectors.npz"), **doms)
    print(f"wrote {artifact_path(cfg, 'stage_a.json')}")
    print(f"  calibration: r2={cal.get('r2')!r} headline_resid={cal.get('headline_residual')!r} "
          f"branch_hint={cal.get('branch_hint')!r}")
    if nulls:
        print(f"  null_b: {nulls['null_b']}")
    return result


def run_cross() -> dict:
    rows = [(p.parent.name, json.loads(p.read_text()))
            for p in sorted((ROOT / "artifacts").glob("*/stage_a.json"))]
    if len(rows) < 2:
        raise SystemExit("need >=2 models' stage_a.json for the cross-model leg")
    maps = {name: {f"{d['a']}|{d['b']}": d for d in res["pairs"]} for name, res in rows}
    shared = sorted(set.intersection(*(set(m) for m in maps.values())))
    if len(shared) < 2:
        raise SystemExit("models share <2 label pairs — cannot correlate")
    names = [name for name, _ in rows]
    cos = {n: np.array([maps[n][k]["cosine"] for k in shared]) for n in names}
    phi = {n: np.array([(maps[n][k]["phi"] or 0.0) for k in shared]) for n in names}
    cos_corr, phi_corr = {}, {}
    for a, b in itertools.combinations(names, 2):
        cos_corr[f"{a}~{b}"] = float(np.corrcoef(cos[a], cos[b])[0, 1])
        phi_corr[f"{a}~{b}"] = float(np.corrcoef(phi[a], phi[b])[0, 1])
    out = {
        "models": names, "n_shared_pairs": len(shared),
        "cosine_agreement": cos_corr, "phi_agreement": phi_corr,
        "note": "excess cosine-agreement beyond phi-agreement = shared model structure (SPEC §3-A2)",
    }
    write_json(ROOT / "artifacts" / "cross_model.json", out)
    print(f"wrote {ROOT / 'artifacts' / 'cross_model.json'}")
    print(f"  cosine_agreement: {cos_corr}")
    print(f"  phi_agreement:    {phi_corr}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage-A observational geometry (SPEC §3).")
    ap.add_argument("--base", default="configs/base.yaml")
    ap.add_argument("--model", help="path to a configs/model_*.yaml (per-model run)")
    ap.add_argument("--judge", default=None, help="judge id for the annotations file (default: cfg judge)")
    ap.add_argument("--cross", action="store_true", help="cross-model combiner over existing artifacts")
    args = ap.parse_args()
    if args.cross:
        run_cross()
        return
    if not args.model:
        ap.error("provide --model <cfg> or --cross")
    cfg = load_config(args.base, args.model)
    judge = args.judge or (cfg.extra.get("judge", {}) or {}).get("model")
    run_model(cfg, judge)


if __name__ == "__main__":
    main()
