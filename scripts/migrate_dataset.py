"""
Migrate v1 dataset -> v2 unified schema (SPEC §0: D1-D5).

Reads v1 `final_dataset.json` (train) and `ood.json` (heldout), emits:
  data/final_dataset_v2.json   (split="train")
  data/heldout_v2.json         (split="heldout")

Adds: answer_type, accepted_answers, solution_concept, length stats, and review flags.
Prints a report including a demonstration that the v1 substring metric produced false positives
that the v2 typed scorer removes.

Usage:
  python scripts/migrate_dataset.py --in-dir <v1 data dir> --out-dir data
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from strat_geom.dataset import (  # noqa: E402
    TaskRecord, classify_answer_type, build_accepted_answers, tag_solution_concept,
    normalize, score_answer,
)


def _coerce_v1(t: dict, split: str) -> dict:
    """Map a v1 record (train: id/task ; heldout: task_id/prompt) to common raw fields."""
    return {
        "id": t.get("id") or t.get("task_id") or "",
        "task": t.get("task") or t.get("prompt") or "",
        "category": t.get("category", ""),
        "subcategory": t.get("subcategory", ""),
        "difficulty": t.get("difficulty", "unknown"),
        "ground_truth": str(t.get("ground_truth", "")),
        "optimal_action": str(t.get("optimal_action", "")),
        "ground_truth_explanation": str(t.get("ground_truth_explanation", "")),
        "reasoning_required": t.get("reasoning_required", []) or [],
        "split": split,
    }


def build_record(raw: dict) -> TaskRecord:
    atype, conf, reasons = classify_answer_type(raw["ground_truth"], raw["optimal_action"])
    accepted = build_accepted_answers(raw["ground_truth"], raw["optimal_action"], atype)
    concept, c_conf = tag_solution_concept(raw["subcategory"], raw["category"])

    review_reasons = list(reasons)
    needs_review = False
    if conf == "low":
        needs_review = True
        review_reasons.append(f"answer_type_confidence={conf}")
    if atype != "freeform" and not accepted:
        needs_review = True
        review_reasons.append("no accepted_answers derived")
    if c_conf == "low":
        needs_review = True            # heuristic solution_concept -> flag for D5 verification
        review_reasons.append(f"solution_concept={concept} (low confidence)")
    # GT verbatim in prompt -> prompt-echo leakage risk for any substring-style check
    if normalize(raw["ground_truth"]) and normalize(raw["ground_truth"]) in normalize(raw["task"]):
        review_reasons.append("ground_truth verbatim in prompt (echo-leakage)")

    return TaskRecord(
        id=raw["id"], split=raw["split"], category=raw["category"],
        subcategory=raw["subcategory"], difficulty=raw["difficulty"], task=raw["task"],
        answer_type=atype, ground_truth=raw["ground_truth"], accepted_answers=accepted,
        ground_truth_explanation=raw["ground_truth_explanation"],
        solution_concept=concept, reasoning_required=raw["reasoning_required"],
        answer_type_confidence=conf, needs_review=needs_review,
        review_reasons=review_reasons, n_task_chars=len(raw["task"]),
    )


def migrate(tasks: list[dict], split: str) -> list[TaskRecord]:
    return [build_record(_coerce_v1(t, split)) for t in tasks]


def report(records: list[TaskRecord], name: str) -> None:
    n = len(records)
    print(f"\n=== {name}: {n} records ===")
    for fld in ("answer_type", "answer_type_confidence", "solution_concept", "category"):
        c = collections.Counter(getattr(r, fld) for r in records)
        print(f"  {fld}: {dict(c.most_common())}")
    nrev = sum(1 for r in records if r.needs_review)
    print(f"  needs_review: {nrev}/{n} ({100*nrev/n:.0f}%)")
    no_acc = [r.id for r in records if r.answer_type != "freeform" and not r.accepted_answers]
    print(f"  non-freeform with no accepted_answers: {len(no_acc)} {no_acc[:5]}")


def demo_d1_fix(records: list[TaskRecord]) -> None:
    """Show v1 substring metric false-positives that the v2 scorer removes."""
    print("\n=== D1 demonstration (substring false-positives removed) ===")
    # a deliberately WRONG answer that nonetheless contains short GTs as substrings
    wrong = "After analysis, I am not sure. Information increases. ANSWER: Cooperate"
    fp_v1 = fp_fixed = checked = 0
    examples = []
    for r in records:
        gt = r.ground_truth.strip().lower()
        if not gt or len(gt) > 8:
            continue
        checked += 1
        v1_correct = gt in wrong.lower()                      # old metric
        v2 = score_answer(wrong, r.to_dict())                  # new metric
        v2_correct = v2["correct"] is True
        if v1_correct and not v2_correct:
            fp_v1 += 1
            if len(examples) < 8:
                examples.append((r.id, repr(r.ground_truth)))
        if v2_correct:
            fp_fixed += 1
    print(f"  short-GT tasks checked: {checked}")
    print(f"  v1 substring marked CORRECT (false positive): {fp_v1}")
    print(f"  v2 typed scorer marked correct on the same wrong answer: {fp_fixed}")
    print(f"  examples flipped FP->correctly-not-credited: {examples}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="v1 data dir with final_dataset.json + ood.json")
    ap.add_argument("--out-dir", default=str(ROOT / "data"))
    ap.add_argument("--train-file", default="final_dataset.json")
    ap.add_argument("--heldout-file", default="ood.json")
    args = ap.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_raw = json.load(open(in_dir / args.train_file))["tasks"]
    heldout_raw = json.load(open(in_dir / args.heldout_file))["tasks"]

    train = migrate(train_raw, "train")
    heldout = migrate(heldout_raw, "heldout")

    for recs, fname, meta_split in ((train, "final_dataset_v2.json", "train"),
                                    (heldout, "heldout_v2.json", "heldout")):
        out = {
            "metadata": {
                "schema_version": "2.0", "split": meta_split, "n": len(recs),
                "source": str(in_dir),
                "notes": "D1 typed scoring; D2 unified schema; D3 'ood'->'heldout'; "
                         "D5 solution_concept (heuristic, needs_review flags).",
            },
            "tasks": [r.to_dict() for r in recs],
        }
        json.dump(out, open(out_dir / fname, "w"), indent=2)
        print(f"wrote {out_dir / fname}")

    report(train, "train")
    report(heldout, "heldout")
    demo_d1_fix(train)


if __name__ == "__main__":
    main()
