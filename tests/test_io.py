"""Tests for io.py — dataset loading, jsonl round-trip, provenance, artifact paths."""
import json

from strat_geom.config import Config
from strat_geom.io import (
    artifact_path, config_hash, load_tasks, provenance, read_jsonl, repo_root, write_jsonl,
)


def test_load_tasks_envelope(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps({"metadata": {}, "tasks": [{"id": "a"}, {"id": "b"}]}))
    assert [t["id"] for t in load_tasks(p)] == ["a", "b"]


def test_load_tasks_bare_list(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps([{"id": "a"}]))
    assert load_tasks(p)[0]["id"] == "a"


def test_jsonl_roundtrip_unicode(tmp_path):
    p = tmp_path / "x.jsonl"
    recs = [{"id": i, "v": "ü"} for i in range(3)]
    assert write_jsonl(p, recs) == 3
    assert list(read_jsonl(p)) == recs


def test_provenance_keys_and_config_hash_deterministic():
    cfg = Config(model_short="m")
    prov = provenance(cfg, stage="x")
    assert {"git_sha", "config_hash", "model_short", "created_utc", "stage"} <= set(prov)
    assert config_hash(cfg) == config_hash(cfg)
    assert config_hash(cfg) != config_hash(Config(model_short="m", n_layers=1))


def test_artifact_path_is_tracked_repo_location():
    ap = artifact_path(Config(model_short="qwen0_5b"), "derisk_report.json")
    assert ap.parent == repo_root() / "artifacts" / "qwen0_5b"
    assert ap.name == "derisk_report.json"
