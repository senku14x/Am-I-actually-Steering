# SPEC — "Geometry of Strategic Reasoning" v2 (workshop-paper redo)

Detailed protocol: every experiment has hypothesis / procedure / expected-if-real vs -if-artifact
/ control / decision rule / sanity checks / reuse / compute.

## Context

The v1 project reports `cos(u_opp, u_deduction) = -0.707` ("antagonistic geometry," cross-model)
and a "DoM paradox" (ablate `u_opp` and steer `+u_opp` both raise opp-modeling-labeled output).
Audit shows the headline geometry is plausibly an **annotation-complementarity artifact** (their
own matrix: payoff/none_other = -0.719, *more* anti-aligned than opp/ded; the existing permutation
null cannot test this), and the "paradox" is confounded (ablate=all-positions vs steer=last-token;
per-segment rates vary with generation length; same GPT judge defines and scores the vector).
Goal: a rigorous workshop paper, rebuilt in a **standalone GitHub repo**, **regenerating all
artifacts** (chains, segments, annotations, activations). Dedicated GPU(s) available; annotation
spend is cost-sensitive but the decisive Stage-1 needs **zero new annotation**.

**Decisions locked:** standalone GitHub repo; **three source-model families** —
DeepSeek-R1-Distill-Qwen-14B + DeepSeek-R1-Distill-Llama-8B + one newer open-weight reasoning model
from a distinct family (confirm at implementation by availability + VRAM; pick the most
architecturally distinct). Geometry on all three (cheap); interventions Qwen-first.

---

## 0. Dataset audit (done) and REQUIRED fixes before any GPU time

Audited v1 `final_dataset.json` (300) + `ood.json` (50).
**Clean:** no dup IDs / near-dup tasks / train↔heldout leakage; all train fields present; controls
genuinely non-strategic; GTs spot-check game-theoretically correct; difficulty balanced.

**Fixes (v2 dataset):**
- **D1 — Replace the substring correctness metric.** 14 train GTs ≤8 chars → false positives;
  median GT 90 chars → false negatives; heldout 23/50 GTs verbatim in prompt. Add
  `answer_type ∈ {categorical, numeric, freeform}`; require a fixed answer format (`ANSWER: <x>`);
  score: categorical→normalized set-match vs `accepted_answers`; numeric→value parse + tolerance;
  freeform→LLM judge validated vs GT. Never bare substring.
- **D2 — Unify schema** train+heldout: `{id, split, category, subcategory, difficulty, task,
  answer_type, ground_truth, accepted_answers, ground_truth_explanation, solution_concept,
  reasoning_required}`.
- **D3 — Reframe "OOD"** as a held-out split of the SAME distribution (`split="heldout"`); no
  distribution-shift transfer claim unless a genuinely-OOD split is later built.
- **D4 — Length-control the contrast** (controls ~½ length of strategic tasks): record prompt +
  chain length; report length-matched comparisons when controls are a class.
- **D5 — Tag `solution_concept`** per task; flag contestable GTs; verify a stratified ~40-task GT
  sample with a second model + spot human check.
- **D6 — Keep the 10-label taxonomy FIXED** (the artifact thesis is about this taxonomy; changing
  it confounds the test). Taxonomy revisions are a separate ablation.
- Analyze at `category` level (subcategory too granular: 114 subcats, 77 with ≤2 tasks).

---

## 1. Repo / project layout

**Standalone GitHub repo.** Config-driven, no Colab paths, small-model-first for CI. (Interim:
staged in the monorepo dev branch for safekeeping; lift to the standalone repo once created.)

```
strategic-geometry-v2/
  SPEC.md  README.md  requirements.txt  .gitignore
  configs/   base.yaml; model_{qwen0_5b,qwen14b,llama8b,...}.yaml; exp_{stage1,intervention}.yaml
  data/      final_dataset_v2.json; heldout_v2.json   (large artifacts gitignored)
  src/strat_geom/
    config.py dataset.py io.py generate.py segment.py annotate.py activations.py
    dom.py nulls.py calibration.py strata.py probe.py cooccur.py metrics.py
    intervene/{hooks.py, conditions.py, routing.py}
  scripts/   migrate_dataset.py run_pipeline.py run_stage1.py run_interventions.py run_routing.py
  tests/     test_dataset.py test_dom.py test_nulls.py test_smoke.py
  results/   gitignored; outputs stamped with git SHA + config hash + measured norm
```

---

## 2. Generation / annotation / extraction pipeline (REGENERATE everything)

1. **Chains:** fixed decoding (`do_sample=False`; `max_new_tokens` set from the measured length
   distribution so <10% truncation — v1 had 30–46%). Save full_output + parsed think/answer +
   token counts + truncation flag. Reuse `run_batched`/`parse_output` (fix `n_tokens` KeyError;
   record absolute counts).
2. **Segments:** paragraph split + 1200-char hard-wrap; persist char offsets; strict re-alignment
   test (v1 `full_output.find` is fragile to duplicate substrings — flag chains with >5% align
   failures; fall back to offset-tracked splitting).
3. **Annotations:** chain-mode, fixed taxonomy (D6), temperature 0; record annotator model +
   prompt hash. Recreate gold-calibration scripts; fresh 150-segment gold; report per-label F1.
4. **Activations:** all layers, mean-pooled per segment, fp16. **Measure** `mean_act_norm` per
   layer (kills hardcoded 158.6).
5. **De-risk first** on Qwen-0.5B-Instruct, 20-task fixture (CI only, not evidence).

---

## 3. STAGE A — Gating validation (DO FIRST; ~1 day; no GPU; all models). Lead on A2.

### A2 (LEAD) — Complementary-label-pair calibration
- **Hypothesis (artifact):** opp/ded anti-alignment is predicted by label complementarity; not a
  special pair. **(real):** opp/ded is a negative outlier beyond the complementarity trend.
- **Procedure:** all 45 label pairs (both ≥ min_count, start 20): x=phi coefficient (2×2
  contingency), y=with-vs-without DoM cosine at headline layer. Robust-regress y~x; 95% prediction
  band; studentized residual of opp/ded; locate payoff/none_other (built-in positive control).
- **Expected if real:** opp/ded below band, |studentized resid|>2. **if artifact:** opp/ded on the
  line, R²>0.5, payoff/none_other at bottom-left endpoint.
- **Decision:** in-band → B; clear outlier both models → A; mixed → C. Headline = R².
- **Cross-model:** correlate the 45-cosine vector across models and the 45-phi vector across
  models; excess cosine-agreement beyond phi-agreement = shared model structure.
- **Sanity:** payoff/none_other must be most-complementary & most-anti-aligned; `cos(v,v)=1`,
  `cos(v,-v)=-1`. **Reuse:** `compute_dom_vectors`, `cosine_sim`, co-occurrence counts. CPU/min.

### A1 (CONFIRM) — Co-occurrence-preserving null
- **Null B (primary, exact):** partition into {neither, opp-only, ded-only, both}; reassign which
  segments occupy each cell keeping cell SIZES fixed → marginals + co-occurrence exact, link
  broken; recompute cosine; n=1000.
- **Null A (robustness):** surrogate labels within (chain, position-tercile) bins (preserves
  position structure; uses reasoning_required + position).
- **Expected if real:** observed below 2.5th pct of both nulls. **if artifact:** observed inside
  null (null mean ≈ -0.65..-0.72).
- **Sanity:** OLD independent-draw null ≈ -0.005; NEW null mean < 0 (else bug).
- **Caveat:** nulls don't preserve higher-order co-occurrence → under-preserve → "survives"
  conservative, "dies" strong. **Reuse:** wvw block + paired resampling. CPU.

### A3 (SUPPORT) — Within-stratum control (rule out payoff composition)
- **Procedure:** restrict to payoff_analysis=1 (and =0); recompute opp/ded cosine.
- **Expected if real:** stays < -0.4. **if composition artifact:** collapses to 0.
- **Guardrail:** report per-class counts; deduction thins within payoff=0 (min_count 50; <100/class
  descriptive only). **Reuse:** `compute_dom_vectors` + extra mask.

**Stage-A output:** persisted `verdict.json` (decision inputs + branch + SHA + config hash).

---

## 4. Branch logic (2-of-3; A1 & A2 primary, A3 supporting; both-model consistency)

- **A — positive geometry+causal:** beats null AND outlier, both models. → §5.
- **B — deflationary methods (default prior):** in-null OR on-line (high R²). Legs: A2 (R² +
  payoff/none_other control), A1 (std permutation passes / new null fails), INDEPENDENT
  probe-orthogonality (DoM ≠ discriminant; Fisher 2.5–3.3×). ≈no new generation.
- **C — partial (most likely):** decompose complementarity vs residual; minimal causal test
  (norm-matched random + operator-matched 2×2 + routing).

**Framing guards:** (i) probe-orthogonality is NOT artifact evidence — independent result;
(ii) complementarity is built-in by construction but the −0.707 *magnitude* is empirical;
(iii) "cross-model = annotator consistency" is a hypothesis tested by A2's cross-model leg.

---

## 5. STAGE B — Intervention redesign (required for A; minimal subset for B/C)

One `InterventionHook(mode∈{ablate,add}, positions∈{all,last}, direction, alpha)` + config-driven
condition matrix + reused `run_batched`. Qwen first.
- **B1 Norm-matched random steering** (fixes "any large last-token vector"): random unit dir at
  `α·measured_norm`, ≥5 seeds; effect must exceed 95% random band.
- **B2 Operator-matched 2×2** (fixes ablate-all vs steer-last): {ablate,add}×{all,last}; claim a
  "paradox" only within matched position.
- **B3 Length/truncation matching:** absolute opp_mod counts/task + length-matched/non-truncated
  subset alongside rates; effect must survive both.
- **B4 Chain-position analysis:** opp_mod rate per position-decile; dip-then-recover under ablation
  = routing-around evidence.
- **B5 Routing-around measurement (highest-value positive result):** under L24 ablation, read the
  residual's `u_opp` component at L+1…L_end; specific recovery of `u_opp` (not random) = routing.
- **B6 Break annotator circularity:** 2nd judge (require replication, report κ); validated lexical
  proxies; calibrate on intervened text. Decide 2nd-judge spend only if Stage 1 → A.

**Minimal causal set (B/C):** baseline, ablate_opp{all,last}, ablate_random(all)×5,
steer_+α(last), steer_random(last)×5 @ matched norm, routing readout.

---

## 6. Geometry / probe analyses (reused, support all branches)

Layer-sweep cosine + SVD-depth (free, all layers); probe-vs-DoM + reg sweep +
within-class-variance/Fisher. Single headline layer for causal claims; sweep for geometry figures.

## 7. Scope discipline

Geometry/calibration on all three families (free); 3-way cross-model A2 agreement is the strongest
calibration + annotator-consistency test. Interventions Qwen-first; extend only if A clears.
De-risk on Qwen-0.5B (CI). ~10 intervention conditions max. 50 strategic tasks; heldout/transfer →
appendix.

## 8. Verification & CI

- **Stage 0 (Qwen-0.5B):** activation shapes/no-NaNs; char↔token round-trip + align-failure flag;
  DoM identities; D1 scorer unit tests.
- **Stage A:** old null ≈ -0.005; new null mean < 0; payoff/none_other most complementary +
  anti-aligned; stratifier reproduces pooled value; branch decision stable across ≥3 seeds.
- **Stage B:** ablate-then-add-back reconstructs hidden state; `add` α=0 no-op; assert steering
  norm = α·measured_norm; random 5-seed band is the pre-registered null; refuse rate comparisons
  across wildly different segment counts; report baseline judge–judge κ first.
- **Global:** every stage persists `verdict.json` (inputs + branch + git SHA + config hash).

## 9. Sequencing

0. Coordinate standalone-repo creation/access; confirm third model family.
1. Repo skeleton + `config.py`; migrate/clean dataset (D1–D6); CI smoke on Qwen-0.5B.
2. Regenerate chains→segments→annotations→activations for all three; gold calibration; per-layer
   norms.
3. Stage A (A2→A1→A3), 3-way cross-model → `verdict.json` → branch.
4. Branch B/C: calibration + null + probe legs; minimal confirmatory intervention on Qwen.
5. Branch A only: full intervention suite + 2nd judge + replication on other families.
