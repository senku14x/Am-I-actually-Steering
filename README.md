# Steering Strategic Reasoning

Falsification-first rebuild of *"The Internal Geometry of Strategic Reasoning in Thinking Language
Models."* The v1 headline (`cos(u_opp, u_deduction) = -0.707`, "antagonistic geometry") is most
plausibly an **annotation-complementarity artifact**; this project tests that *before* building any
interpretation on it.

**Core idea.** A difference-of-means (DoM) contrast between two near-complementary annotator labels
is anti-aligned largely by construction. A cheap, no-GPU **Stage-A gate** decides — on existing
activations — whether the opponent-modeling vs deduction anti-alignment exceeds a complementarity
baseline. The outcome selects the paper:

- **A** positive geometry + causal paper (effect survives),
- **B** deflationary methods paper ("DoM behavioral directions encode annotation co-occurrence
  structure, not model features"),
- **C** partial — decompose how much of the headline is artifact vs residual.

See **[`SPEC.md`](./SPEC.md)** for the full protocol (every experiment has hypothesis / procedure /
expected-if-real-vs-artifact / control / decision rule / sanity checks) and
**[`docs/AUDIT.md`](./docs/AUDIT.md)** for the dataset audit.

## Status

Bootstrapping. Built and CPU-tested: the unified v2 dataset + typed answer scorer (replaces v1's
broken substring metric, see [`data/README.md`](./data/README.md)); and the **generation pipeline**
(`generate` → `segment` → `activations`) — a config-driven think-region parser (no hardcoded
`<think>`), offset-exact segmentation, and char↔token activation mapping — driven by
[`scripts/run_pipeline.py`](./scripts/run_pipeline.py). GPU stages import torch lazily, so all the
pure logic is unit-tested without a GPU. Next: the Qwen-0.5B de-risk run on the GPU host, then
Stage-A analysis modules (`dom`, `nulls`, `calibration`).

## Install

```bash
pip install -e ".[dev]"        # CPU analysis + tests
# pip install -e ".[gpu]"      # + torch/transformers (GPU host)
# pip install -e ".[annotate]" # + openai/pydantic (LLM judge)
```

## Quickstart

```bash
# rebuild the unified dataset from v1 inputs
python scripts/migrate_dataset.py --in-dir <v1 data dir> --out-dir data
pytest
```

## Workflow (GPU host loop)

Code is authored and CPU-tested here; generation/extraction runs on a GPU host; small derived
results return via GitHub for analysis here:

1. **here:** edit + `pytest`, push.
2. **host:** `git pull` → `pip install -e ".[gpu,annotate]"` →
   `python scripts/run_pipeline.py --model configs/model_qwen0_5b.yaml --limit 20`
3. **host:** commit + push the tracked outputs — `artifacts/<model>/` (per-layer norms, reports,
   later DoM/figures) and `data/annotations/` (paid judge output). Raw chains/segments/activations
   stay on the host (gitignored, reproducible from a fixed-decode rerun).
4. **here:** `git pull` and analyse the artifacts.

The first run is the SPEC §2.5 de-risk on Qwen2.5-0.5B (CI plumbing, **not** evidence) — it
validates the whole loop and the Qwen chat-template/hook path before any 14B GPU spend.

## Layout

```
SPEC.md                 detailed experiment protocol
docs/AUDIT.md           dataset audit findings
configs/                base.yaml + per-model configs (qwen0_5b/qwen14b/llama8b/phi4_reasoning)
data/                   final_dataset_v2.json, heldout_v2.json; annotations tracked, chains/activations gitignored
src/strat_geom/         library: config, dataset, io, generate, segment, activations (+ analysis modules incoming)
scripts/                runnable entry points (migrate_dataset, run_pipeline, ...)
artifacts/              small derived results that return from the GPU host (tracked)
tests/                  CPU unit/smoke tests
```

## License

MIT — see [`LICENSE`](./LICENSE).
