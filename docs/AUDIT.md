# v1 dataset audit (record)

Audit of the v1 dataset (`final_dataset.json` 300, `ood.json` 50) that motivated the D1–D6 fixes.

## Clean (verified)
- No duplicate IDs, no exact/near-duplicate task texts, **zero train↔heldout text leakage**
  (exact or >0.85 prefix similarity).
- All 9 fields present on all 300 train tasks; `ground_truth_explanation` present (median 454 chars).
- Non-strategic controls are genuinely non-strategic (0/50 contain strategic vocabulary).
- Ground truths spot-check as game-theoretically correct across all 6 strategic categories.
- Difficulty balanced across categories (easy/med/hard ≈ 61/148/91).

## Problems → fixes

| # | Problem | Fix |
|---|---|---|
| D1 | `ground_truth_substring_hit` broken both ways: 14 train GTs ≤8 chars (`"A"`,`"Up"`,`"In"`,`"0"`) → false positives; median GT 90 chars → false negatives; **heldout 23/50 GTs verbatim in prompt** | typed `answer_type` + structured `ANSWER:` extraction + type-aware scoring |
| D2 | schema mismatch: train `id/task`(+difficulty/explanation/reasoning_required) vs heldout `task_id/prompt` (missing those) | one unified record schema |
| D3 | "OOD" is same-distribution held-out (`val_batch_*` ids, category subset) — not a shift | rename to `split="heldout"`; no transfer claim without a real OOD split |
| D4 | controls ~½ the length of strategic tasks (median 227 vs 510 chars) | record `n_task_chars`; report length-matched comparisons |
| D5 | GTs mix solution concepts; some contestable | `solution_concept` tags + `needs_review` flags; second-model/human verify a stratified sample |
| D6 | (analysis discipline) taxonomy must stay fixed | freeze the 10-label taxonomy for the geometry analysis |

Minor: `subcategory` too granular (114 values, 77 with ≤2 tasks) — analyze at `category`.
`reasoning_required` is a useful a-priori label and an upstream source of the segment-label
co-occurrence structure (a covariate for the Stage-A co-occurrence null).

## Migration result (current)
- train: answer_type {freeform 158, categorical 135, numeric 7}; needs_review 86/300 (29%).
- heldout: answer_type {categorical 42, freeform 6, numeric 2}; needs_review 50/50 (keyword-inferred
  solution concepts flagged for verification).
- D1 check: 3 v1 short-GT substring false-positives → 0 under the typed scorer.
