"""
Per-segment activation extraction + per-layer norm measurement (SPEC §2.4).

The fragile part — mapping a segment's character span to the tokens that cover it — is isolated in
`token_span_for_char_span`, a pure function unit-tested in CI (this is the "char<->token round-trip"
de-risk in SPEC §8). The GPU forward pass (`extract_activations`) wraps it and is run on the host.

We measure `mean_act_norm` per layer rather than trusting v1's hardcoded 158.6, so steering/ablation
magnitudes are calibrated to the actual representation scale of each model.
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import Config


def token_span_for_char_span(offset_mapping: list[tuple[int, int]], cs: int, ce: int):
    """Inclusive (lo, hi) token indices whose character offsets overlap [cs, ce).

    `offset_mapping` is the per-token (char_start, char_end) list for the completion. Zero-width
    entries (special tokens, often (0, 0)) are skipped. Returns (None, None) if nothing overlaps.
    """
    lo = hi = None
    for i, (tcs, tce) in enumerate(offset_mapping):
        if tce <= tcs:            # zero-width / special token
            continue
        if tce <= cs:             # token ends before the segment starts
            continue
        if tcs >= ce:             # token starts after the segment ends
            break
        if lo is None:
            lo = i
        hi = i
    return lo, hi


def extract_activations(cfg: Config, chains: list[dict], segments_by_chain: dict[str, list],
                        out_dir: str | Path | None = None, device: str | None = None) -> dict:
    """Mean-pool hidden states over each segment's tokens, all layers, fp16. GPU host only.

    Returns a summary dict (per-layer mean norms, shapes, mapping-failure count) suitable for the
    tracked de-risk report; the heavy per-chain arrays are written under out_dir (gitignored).
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    from .io import activations_dir  # noqa: PLC0415

    out_dir = Path(out_dir) if out_dir is not None else activations_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=torch.float16, device_map=device,
    )
    model.eval()

    norm_sum: dict[int, float] = {}
    n_pooled = 0
    map_failures = 0
    n_layers_seen = 0
    index_rows: list[dict] = []
    for chain in chains:
        segs = segments_by_chain.get(chain["id"], [])
        if not segs:
            continue
        prompt_ids = tok(chain["prompt"], return_tensors="pt",
                         add_special_tokens=False).input_ids.to(device)
        comp = tok(chain["completion"], return_offsets_mapping=True, add_special_tokens=False,
                   return_tensors="pt")
        offsets = [tuple(x) for x in comp["offset_mapping"][0].tolist()]
        comp_ids = comp["input_ids"].to(device)
        input_ids = torch.cat([prompt_ids, comp_ids], dim=1)
        comp_start = prompt_ids.shape[1]
        with torch.no_grad():
            hs = model(input_ids, output_hidden_states=True).hidden_states  # (L+1) x [1, T, H]
        n_layers_seen = len(hs)
        seg_vecs = []
        for row, seg in enumerate(segs):
            lo, hi = token_span_for_char_span(offsets, seg["char_start"], seg["char_end"])
            mapped = lo is not None
            if not mapped:
                map_failures += 1
                seg_vecs.append(np.zeros((len(hs), model.config.hidden_size), dtype=np.float16))
            else:
                a, b = comp_start + lo, comp_start + hi + 1
                vec = torch.stack([layer[0, a:b, :].mean(0) for layer in hs]).float().cpu()
                for li in range(len(hs)):
                    norm_sum[li] = norm_sum.get(li, 0.0) + float(vec[li].norm())
                n_pooled += 1
                seg_vecs.append(vec.to(torch.float16).numpy())
            # row i of {chain}.npy <-> this (chain_id, seg_idx); the Stage-A loader joins on it.
            index_rows.append({"chain_id": chain["id"], "seg_idx": seg["seg_idx"],
                               "region": seg.get("region"), "row": row, "mapped": mapped,
                               "file": f"{chain['id']}.npy"})
        np.save(out_dir / f"{chain['id']}.npy", np.stack(seg_vecs))

    (out_dir / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in index_rows))
    mean_norms = {li: (norm_sum[li] / n_pooled) for li in sorted(norm_sum)} if n_pooled else {}
    return {
        "n_segments_pooled": n_pooled,
        "map_failures": map_failures,
        "n_layers": n_layers_seen,
        "mean_act_norm_per_layer": mean_norms,
        "headline_layer_norm": mean_norms.get(cfg.headline_layer),
    }
