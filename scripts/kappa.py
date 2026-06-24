"""
Inter-judge agreement — Cohen's kappa per label between two (or more) judges' annotations (SPEC
§5-B6). Quantifies how much the taxonomy labelling depends on the annotator, which bears directly on
the artifact thesis: a label with low kappa is one whose geometry is judge-dependent (not robust).

  python scripts/kappa.py --model configs/model_llama8b.yaml \
      --judges z-ai/glm-5,deepseek/deepseek-v3.2
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from strat_geom.config import load_config  # noqa: E402
from strat_geom.io import (  # noqa: E402
    annotations_path, artifact_path, provenance, read_jsonl, write_json,
)


def _load(cfg, judge: str) -> dict:
    return {(r["chain_id"], r["seg_idx"]): set(r["labels"])
            for r in read_jsonl(annotations_path(cfg, judge))}


def main() -> None:
    ap = argparse.ArgumentParser(description="Inter-judge Cohen's kappa per label (SPEC §5-B6).")
    ap.add_argument("--base", default="configs/base.yaml")
    ap.add_argument("--model", required=True)
    ap.add_argument("--judges", required=True, help="comma-separated judge ids (>=2)")
    args = ap.parse_args()

    from sklearn.metrics import cohen_kappa_score  # noqa: PLC0415

    cfg = load_config(args.base, args.model)
    judges = [j.strip() for j in args.judges.split(",") if j.strip()]
    if len(judges) < 2:
        ap.error("need >=2 judges")
    anns = {j: _load(cfg, j) for j in judges}
    shared = sorted(set.intersection(*(set(a) for a in anns.values())))
    if not shared:
        raise SystemExit("no shared (chain_id, seg_idx) across the judges — check the judge ids")

    per_label = {}
    for lab in cfg.labels:
        ks = []
        for a, b in itertools.combinations(judges, 2):
            va = [lab in anns[a][k] for k in shared]
            vb = [lab in anns[b][k] for k in shared]
            # kappa is undefined if a rater is constant; skip those degenerate label/pair cases
            if (any(va) or any(vb)) and not (all(va) and all(vb)):
                ks.append(float(cohen_kappa_score(va, vb)))
        support = sum(1 for k in shared if any(lab in anns[j][k] for j in judges))
        per_label[lab] = {"kappa": (sum(ks) / len(ks) if ks else None), "support": support}

    defined = [v["kappa"] for v in per_label.values() if v["kappa"] is not None]
    macro = sum(defined) / len(defined) if defined else None
    out = {
        "provenance": provenance(cfg, judges=judges, n_shared=len(shared)),
        "judges": judges, "n_shared_segments": len(shared),
        "macro_kappa": macro, "per_label": per_label,
    }
    write_json(artifact_path(cfg, "kappa.json"), out)
    print(f"wrote {artifact_path(cfg, 'kappa.json')}")
    print(f"  macro kappa: {macro}")
    for lab, v in sorted(per_label.items(), key=lambda x: (x[1]["kappa"] is None, x[1]["kappa"] or 0)):
        print(f"  {lab:28s} kappa={v['kappa']!r}  support={v['support']}")


if __name__ == "__main__":
    main()
