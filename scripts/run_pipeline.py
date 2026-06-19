"""
Generation pipeline driver — the entry point the GPU host runs after `git pull`.

Runs generate -> segment -> activations for one model config and writes a small, TRACKED
`artifacts/<model_short>/derisk_report.json` (provenance + counts + per-layer norms) that gets
pushed back to GitHub for analysis here. The heavy chains/segments/activations stay gitignored on
the host (deterministically reproducible from a fixed-decode rerun).

First use is the SPEC §2.5 de-risk: validate the whole loop + the Qwen chat-template/hook path on
Qwen2.5-0.5B for cents before spending 14B GPU hours.

  pip install -e ".[gpu,annotate]"
  python scripts/run_pipeline.py --model configs/model_qwen0_5b.yaml --limit 20

Stages are selectable (`--stages segment` re-runs just segmentation from existing chains on CPU).
"""
from __future__ import annotations

import argparse
import statistics as stats
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from strat_geom.activations import extract_activations  # noqa: E402
from strat_geom.annotate import (  # noqa: E402
    annotate_chain, annotation_prompt_hash, make_client, validate_taxonomy,
)
from strat_geom.config import load_config  # noqa: E402
from strat_geom.dataset import score_answer  # noqa: E402
from strat_geom.generate import generate_chains  # noqa: E402
from strat_geom.io import (  # noqa: E402
    annotations_path, artifact_path, chains_path, load_tasks, provenance, read_jsonl,
    segments_path, write_json, write_jsonl,
)
from strat_geom.segment import realign_failures, segment_chain  # noqa: E402


def _summarize_generation(chains: list[dict]) -> dict:
    toks = [c["n_gen_tokens"] for c in chains]
    n = len(chains)
    return {
        "n_chains": n,
        "truncation_rate": round(sum(c["truncated"] for c in chains) / n, 4) if n else None,
        "has_think_rate": round(sum(c["has_think"] for c in chains) / n, 4) if n else None,
        "gen_tokens": {
            "mean": round(stats.mean(toks), 1) if toks else None,
            "median": stats.median(toks) if toks else None,
            "max": max(toks) if toks else None,
        },
    }


def _summarize_segmentation(all_segs: list[dict], realign_fail: int) -> dict:
    lens = [s["char_end"] - s["char_start"] for s in all_segs]
    return {
        "n_segments": len(all_segs),
        "by_region": dict(Counter(s["region"] for s in all_segs)),
        "realign_failures": realign_fail,            # MUST be 0 (offset-exact invariant)
        "seg_chars": {
            "mean": round(stats.mean(lens), 1) if lens else None,
            "max": max(lens) if lens else None,
        },
    }


def _summarize_scoring(chains: list[dict], tasks_by_id: dict) -> dict:
    res = [score_answer(c["completion"], tasks_by_id[c["id"]]) for c in chains if c["id"] in tasks_by_id]
    auto = [r for r in res if r["correct"] is not None]
    return {
        "n_scored": len(res),
        "n_auto_scored": len(auto),
        "n_needs_judge": sum(r["needs_judge"] for r in res),
        "auto_accuracy": round(sum(r["correct"] for r in auto) / len(auto), 4) if auto else None,
        "methods": dict(Counter(r["method"] for r in res)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Generation pipeline driver (GPU host).")
    ap.add_argument("--base", default="configs/base.yaml")
    ap.add_argument("--model", required=True, help="path to a configs/model_*.yaml")
    ap.add_argument("--data", default="data/final_dataset_v2.json")
    ap.add_argument("--limit", type=int, default=20, help="first N tasks (de-risk fixture size)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--stages", default="generate,segment,activations")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    overrides = {"seed": args.seed} if args.seed is not None else {}
    cfg = load_config(args.base, args.model, **overrides)
    stages = {s.strip() for s in args.stages.split(",") if s.strip()}
    tasks = load_tasks(args.data)[:args.limit]
    tasks_by_id = {t["id"]: t for t in tasks}

    report: dict = {
        "provenance": provenance(cfg, stages=sorted(stages), n_tasks=len(tasks), data=args.data),
        "config": {
            "n_layers": cfg.n_layers, "hidden_dim": cfg.hidden_dim,
            "headline_layer": cfg.headline_layer, "max_new_tokens": cfg.max_new_tokens,
            "think_open": cfg.extra.get("think_open", "<think>"),
            "think_close": cfg.extra.get("think_close", "</think>"),
            "chat_template": cfg.extra.get("chat_template"),
        },
    }

    # --- generate ----------------------------------------------------------
    if "generate" in stages:
        chains = generate_chains(cfg, tasks, batch_size=args.batch_size)
        write_jsonl(chains_path(cfg), chains)
        report["generation"] = _summarize_generation(chains)
    else:
        chains = list(read_jsonl(chains_path(cfg)))

    # --- segment -----------------------------------------------------------
    segments_by_chain: dict[str, list[dict]] = defaultdict(list)
    if "segment" in stages:
        all_segs, fails = [], 0
        for ch in chains:
            segs = segment_chain(ch, cfg)
            fails += len(realign_failures(ch["completion"], segs))
            for s in segs:
                d = s.to_dict()
                segments_by_chain[ch["id"]].append(d)
                all_segs.append(d)
        write_jsonl(segments_path(cfg), all_segs)
        report["segmentation"] = _summarize_segmentation(all_segs, fails)
        if fails:
            print(f"WARNING: {fails} segment realignment failures (offset invariant broken)")
    else:
        for s in read_jsonl(segments_path(cfg)):
            segments_by_chain[s["chain_id"]].append(s)

    # --- activations -------------------------------------------------------
    if "activations" in stages:
        report["activations"] = extract_activations(cfg, chains, dict(segments_by_chain))

    # --- annotate (LLM judge; SPEC §2.3) ----------------------------------
    if "annotate" in stages:
        judge_cfg = cfg.extra.get("judge", {})
        validate_taxonomy(cfg.labels)
        client = make_client(judge_cfg)
        ann_records: list[dict] = []
        total_unknown = 0
        label_counter: Counter = Counter()
        for ch in chains:
            segs = segments_by_chain.get(ch["id"], [])
            if cfg.think_only:
                segs = [s for s in segs if s.get("region") == "think"]
            if not segs:
                continue
            recs, n_unknown = annotate_chain(client, judge_cfg, ch, segs, cfg.labels)
            ann_records.extend(recs)
            total_unknown += n_unknown
            for r in recs:
                label_counter.update(r["labels"])
        write_jsonl(annotations_path(cfg), ann_records)
        report["annotation"] = {
            "n_segments_annotated": len(ann_records),
            "unknown_labels_dropped": total_unknown,        # >0 => judge drifting off-taxonomy
            "judge_model": judge_cfg.get("model"),
            "prompt_hash": annotation_prompt_hash(cfg.labels),
            "label_distribution": dict(label_counter),
        }

    # --- scoring sanity (D1) ----------------------------------------------
    report["scoring"] = _summarize_scoring(chains, tasks_by_id)

    out = artifact_path(cfg, "derisk_report.json")
    write_json(out, report)
    print(f"\nwrote {out}")
    for k in ("generation", "segmentation", "activations", "annotation", "scoring"):
        if k in report:
            print(f"  {k}: {report[k]}")


if __name__ == "__main__":
    main()
