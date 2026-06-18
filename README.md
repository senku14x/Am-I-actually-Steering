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

Bootstrapping. The CPU/data layer is built and tested: a unified v2 dataset with a typed answer
scorer that replaces v1's broken substring metric (see [`data/README.md`](./data/README.md)).
Next: Stage-A analysis modules (`dom`, `nulls`, `calibration`). GPU steps (regenerating
chains/activations across three model families, interventions) run on a GPU host.

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

## Layout

```
SPEC.md                 detailed experiment protocol
docs/AUDIT.md           dataset audit findings
configs/                base.yaml + per-model configs (env-interpolated paths)
data/                   final_dataset_v2.json, heldout_v2.json (+ data dictionary); large artifacts gitignored
src/strat_geom/         library: config, dataset (scorer), + analysis/intervention modules (incoming)
scripts/                runnable entry points (migrate_dataset, run_stage1, ...)
tests/                  CPU unit/smoke tests
```

## License

MIT — see [`LICENSE`](./LICENSE).
