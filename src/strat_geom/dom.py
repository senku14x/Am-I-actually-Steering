"""
Difference-of-means (DoM) "behavioral directions" + the data loader that joins activations to labels.

DoM_k(layer) = mean(activations of segments WITH label k) - mean(without). The headline geometry
is cos(DoM_opp, DoM_ded). `load_model_segments` reconstructs the aligned (activation, labels) table
from the pipeline's outputs using the activation index.jsonl (SPEC §3/§6).
"""
from __future__ import annotations

import numpy as np

from .config import Config
from .metrics import cosine


def dom_vector(X: np.ndarray, mask, min_count: int):
    """mean(X[mask]) - mean(X[~mask]) at one layer, or None if either group < min_count."""
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() < min_count or (~mask).sum() < min_count:
        return None
    return X[mask].mean(0) - X[~mask].mean(0)


def dom_cosine(X: np.ndarray, mi, mj, min_count: int):
    di = dom_vector(X, mi, min_count)
    dj = dom_vector(X, mj, min_count)
    if di is None or dj is None:
        return None
    return cosine(di, dj)


def cosine_matrix(X: np.ndarray, presence: np.ndarray, labels, min_count: int) -> dict:
    """cos(DoM_i, DoM_j) for every label pair meeting min_count. Keys (i, j), i<j."""
    out: dict[tuple[int, int], float] = {}
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            c = dom_cosine(X, presence[:, i], presence[:, j], min_count)
            if c is not None:
                out[(i, j)] = c
    return out


def load_model_segments(cfg: Config, layer: int, judge: str | None = None):
    """Return (X[N, H] float32, presence[N, n_labels] bool, meta) for one model at one layer.

    Joins activation rows (index.jsonl + per-chain .npy) to annotation labels by (chain_id, seg_idx);
    drops unmapped rows and (when cfg.think_only) non-think segments.
    """
    from .io import activations_dir, annotations_path, read_jsonl

    adir = activations_dir(cfg)
    ann = {(r["chain_id"], r["seg_idx"]): r["labels"]
           for r in read_jsonl(annotations_path(cfg, judge))}
    labels = cfg.labels
    idx = {lab: i for i, lab in enumerate(labels)}
    npy_cache: dict[str, np.ndarray] = {}
    rows_X, rows_p, meta = [], [], []
    for row in read_jsonl(adir / "index.jsonl"):
        if not row.get("mapped", True):
            continue
        if cfg.think_only and row.get("region") != "think":
            continue
        key = (row["chain_id"], row["seg_idx"])
        if key not in ann:
            continue
        fname = row["file"]
        if fname not in npy_cache:
            npy_cache[fname] = np.load(adir / fname)
        rows_X.append(npy_cache[fname][row["row"], layer, :].astype(np.float32))
        p = np.zeros(len(labels), dtype=bool)
        for lab in ann[key]:
            if lab in idx:
                p[idx[lab]] = True
        rows_p.append(p)
        meta.append({"chain_id": row["chain_id"], "seg_idx": row["seg_idx"],
                     "region": row.get("region")})
    X = np.array(rows_X, dtype=np.float32) if rows_X else np.zeros((0, cfg.hidden_dim), np.float32)
    P = np.array(rows_p, dtype=bool) if rows_p else np.zeros((0, len(labels)), dtype=bool)
    return X, P, meta
