# artifacts/

Small **derived** results that travel back from the GPU host to GitHub for analysis here. This
directory is **tracked** (the heavy inputs that produced it are not — see `.gitignore`).

Layout: `artifacts/<model_short>/<name>`, e.g. `artifacts/qwen0_5b/derisk_report.json`.

Each file is written via `strat_geom.io` and carries a `provenance` block (git SHA + config hash +
UTC timestamp), so any result is traceable to the exact code and config that produced it.

What belongs here (small, analysis-ready): de-risk reports, per-layer activation norms, DoM
vectors, the 45-pair cosine/phi tables, `verdict.json`, metrics, figures.

What does **not** (regenerable on a fixed-decode rerun, kept on the host): raw chains, segments,
and activation tensors. Paid annotations are the exception — they live under `data/annotations/`
and are tracked separately so an instance teardown never forces re-paying the judge.
