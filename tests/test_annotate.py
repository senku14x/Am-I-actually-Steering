"""Pure-helper tests for annotate.py (no openai / no network)."""
from pathlib import Path

import yaml

from strat_geom.annotate import (
    LABEL_DEFINITIONS,
    _extract_json,
    annotation_prompt_hash,
    build_annotation_prompt,
    parse_annotation_response,
    parse_grade_response,
    validate_taxonomy,
)

BASE_LABELS = yaml.safe_load(
    (Path(__file__).resolve().parents[1] / "configs" / "base.yaml").read_text()
)["labels"]


def test_every_configured_label_has_a_definition():
    validate_taxonomy(BASE_LABELS)                       # must not raise
    assert set(BASE_LABELS) <= set(LABEL_DEFINITIONS)


def test_build_prompt_lists_labels_and_segments():
    segs = [{"seg_idx": 0, "text": "compute the payoffs"}, {"seg_idx": 1, "text": "they will defect"}]
    prompt = build_annotation_prompt("c1", segs, BASE_LABELS)
    assert "payoff_analysis" in prompt and "opponent_modeling" in prompt
    assert "[0]" in prompt and "[1]" in prompt and "they will defect" in prompt


def test_extract_json_handles_code_fences_and_prose():
    assert _extract_json('```json\n{"0": ["deduction"]}\n```') == {"0": ["deduction"]}
    assert _extract_json('sure:\n{"1": ["none_other"]} done') == {"1": ["none_other"]}
    assert _extract_json("not json at all") == {}


def test_parse_validates_labels_and_defaults_empty_to_none_other():
    text = '{"0": ["payoff_analysis", "made_up_label"], "1": []}'
    per_seg, n_unknown = parse_annotation_response(text, 2, BASE_LABELS)
    assert per_seg[0] == ["payoff_analysis"]             # unknown dropped
    assert n_unknown == 1
    assert per_seg[1] == ["none_other"]                  # empty -> none_other


def test_parse_handles_string_and_missing_segments():
    per_seg, _ = parse_annotation_response('{"0": "deduction"}', 2, BASE_LABELS)
    assert per_seg[0] == ["deduction"]                   # str coerced to list
    assert per_seg[1] == ["none_other"]                  # missing index -> none_other


def test_annotation_prompt_hash_is_stable_and_order_invariant():
    assert annotation_prompt_hash(BASE_LABELS) == annotation_prompt_hash(BASE_LABELS)
    assert len(annotation_prompt_hash(BASE_LABELS)) == 12


def test_parse_grade_response():
    assert parse_grade_response('{"correct": true, "rationale": "ok"}')["correct"] is True
    assert parse_grade_response('{"correct": false}')["correct"] is False
    assert parse_grade_response("garbage")["correct"] is None
