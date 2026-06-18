# Data dictionary

Two splits, one schema (v2). Built by `scripts/migrate_dataset.py` from the v1 dataset.

- `final_dataset_v2.json` — 300 tasks, `split="train"` (250 strategic + 50 non-strategic control)
- `heldout_v2.json` — 50 tasks, `split="heldout"` (held-out tasks from the **same** distribution;
  see D3 — this is *not* a distribution-shift "OOD" set)

Large regenerable artifacts (reasoning chains, segments, annotations, activations) are **not**
committed; they are produced on a GPU host from these task files (see `../SPEC.md` §2).

## Record schema

| field | type | notes |
|---|---|---|
| `id` | str | unique task id (e.g. `matrix_01`) |
| `split` | str | `train` \| `heldout` |
| `category` | str | one of 7 categories (incl. `non_strategic_control`); the analysis unit |
| `subcategory` | str | fine-grained; **too granular for analysis** (114 values, 77 with ≤2 tasks) |
| `difficulty` | str | `easy` \| `medium` \| `hard` \| `unknown` (heldout) |
| `task` | str | the prompt shown to the model |
| `answer_type` | str | `categorical` \| `numeric` \| `freeform` — drives scoring (D1) |
| `ground_truth` | str | canonical answer (may include explanation) |
| `accepted_answers` | list[str] | normalized acceptable answers (categorical/numeric); empty for freeform |
| `ground_truth_explanation` | str | rationale; usable by an LLM judge |
| `solution_concept` | str | e.g. `dominant_strategy`, `nash`, `backward_induction`, `bayesian`, `folk_theorem`, `computation`, `heuristic` |
| `reasoning_required` | list[str] | a-priori task-level reasoning tags (also a covariate for the co-occurrence null) |
| `answer_type_confidence` | str | `high` \| `medium` \| `low` |
| `needs_review` | bool | flagged for the D5 second-model/human verification pass |
| `review_reasons` | list[str] | why it was flagged |
| `n_task_chars` | int | prompt length (D4 length-control covariate) |

## Scoring (D1 — replaces v1's broken substring metric)

Use `strat_geom.dataset.score_answer(model_text, record)`:
1. **extract** the model's final answer (prefers an `ANSWER: <x>` line, then `\boxed{}`, then a
   flagged weak fallback);
2. **score by type** — categorical: normalized whole-token set-match vs `accepted_answers`
   (kills the `"A"`/`"In"`/`"Up"` substring false positives); numeric: value parse + tolerance
   (handles `$`, `%`, `K/M/B`); freeform: `correct=None` → defer to an LLM judge.

**Never** use bare `ground_truth in answer`. v1's metric also leaked on the heldout split (46% of
GTs appear verbatim in their own prompt) — irrelevant now because scoring reads the extracted
answer, not the prompt.

## Known first-pass limitations (flagged via `needs_review`, for the D5 pass)

- Some non-strategic control answers are numeric but currently typed categorical/freeform.
- `solution_concept` is heuristic (subcategory map + keyword fallback); verify a stratified
  ~40-task sample with a second model + spot human check, and flag contestable GTs.
- A few strategic GTs mix solution concepts (equilibrium vs best-response-to-a-stated-opponent).
