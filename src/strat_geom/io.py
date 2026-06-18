"""
Artifact I/O + provenance for the regenerate-everything pipeline.

Workflow this serves (the user's loop): author here -> push -> pull on the GPU host -> run -> push
small results back -> analyse here. Two consequences encoded in this module:

  1. Every persisted artifact is stamped with the git SHA + config hash that produced it (SPEC §1),
     so a result on GitHub is traceable to exact code + config.
  2. The heavy/cheap split is explicit and matches .gitignore:
       chains / segments / activations -> data/...      (gitignored: large, deterministically
                                                          reproducible from a fixed-decode rerun)
       annotations                     -> data/annotations/ (TRACKED: paid LLM-judge output; must
                                                          survive ephemeral-instance teardown)
       derived summaries               -> artifacts/...  (TRACKED: small; this is what gets analysed)
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from .config import Config


# --- repo / dataset --------------------------------------------------------

def repo_root() -> Path:
    """Repo root (dir containing pyproject.toml), so tracked artifacts land inside the repo
    regardless of env-configured DATA_ROOT/RESULTS_ROOT."""
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def load_tasks(path: str | Path) -> list[dict]:
    """Load the task list from a v2 dataset file ({metadata, tasks}) or a bare list."""
    obj = json.loads(Path(path).read_text())
    if isinstance(obj, dict) and "tasks" in obj:
        return obj["tasks"]
    if isinstance(obj, list):
        return obj
    raise ValueError(f"{path}: expected {{'tasks': [...]}} or a list, got {type(obj).__name__}")


# --- jsonl / json ----------------------------------------------------------

def write_jsonl(path: str | Path, records: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


# --- provenance ------------------------------------------------------------

def git_sha() -> str:
    """Short HEAD sha, suffixed '-dirty' if the tree has uncommitted changes; 'nogit' off-repo."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "nogit"


def config_hash(cfg: Config, length: int = 12) -> str:
    blob = json.dumps(asdict(cfg), sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:length]


def provenance(cfg: Config, **extra: Any) -> dict:
    return {
        "git_sha": git_sha(),
        "config_hash": config_hash(cfg),
        "model_short": cfg.model_short,
        "model_name": cfg.model_name,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **extra,
    }


# --- path layout (single source of truth for where artifacts live) ---------

def chains_path(cfg: Config) -> Path:
    return cfg.data_path / "chains" / f"{cfg.model_short}.jsonl"


def segments_path(cfg: Config) -> Path:
    return cfg.data_path / "segments" / f"{cfg.model_short}.jsonl"


def activations_dir(cfg: Config) -> Path:
    return cfg.data_path / "activations" / cfg.model_short


def annotations_path(cfg: Config) -> Path:
    return cfg.data_path / "annotations" / f"{cfg.model_short}.jsonl"


def artifact_path(cfg: Config, name: str) -> Path:
    """Tracked small-artifact path: <repo>/artifacts/<model_short>/<name>."""
    return repo_root() / "artifacts" / cfg.model_short / name
