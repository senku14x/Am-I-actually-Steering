"""Tests for the v2 dataset schema + typed answer scorer (the D1 fix)."""
import json
from pathlib import Path

import pytest

from strat_geom.dataset import (
    SCHEMA_FIELDS, parse_number, numbers_match, classify_answer_type,
    extract_final_answer, score_answer, _whole_token_present,
)

DATA = Path(__file__).resolve().parents[1] / "data"


# --- parse_number ---------------------------------------------------------

@pytest.mark.parametrize("s,expected", [
    ("$60", 60.0), ("80%", 80.0), ("0", 0.0), ("$449K", 449_000.0),
    ("$2,400", 2400.0), ("-5", -5.0), ("3.5", 3.5), ("no number", None),
])
def test_parse_number(s, expected):
    assert parse_number(s) == expected


def test_numbers_match_tolerance():
    assert numbers_match(80.0, 80.0)
    assert numbers_match(60.0, 60.4, abs_=0.5)
    assert not numbers_match(60.0, 65.0)


# --- answer-type classification ------------------------------------------

@pytest.mark.parametrize("gt,oa,expected", [
    ("Defect", "Choose Defect", "categorical"),
    ("$60", "Bid $60 up to your value", "numeric"),
    ("80%", "Update posterior to 80%", "numeric"),
    ("Mixed strategy: 50% Heads, 50% Tails", "Randomize 50-50", "categorical"),  # ratio action
    # clean short optimal_action -> categorical even if GT carries an explanation
    ("Commit to a price just below $450K because it captures surplus", "Commit to publishing the price", "categorical"),
    # genuinely long/multi-clause action -> freeform (needs a judge)
    ("Cooperate then defect on the last round to maximize total payoff",
     "Cooperate in rounds 1 through 4, then defect in round 5", "freeform"),
])
def test_classify_answer_type(gt, oa, expected):
    atype, conf, _ = classify_answer_type(gt, oa)
    assert atype == expected


# --- final-answer extraction ---------------------------------------------

def test_extract_prefers_answer_field():
    text = "lots of reasoning...\nSo I conclude.\nANSWER: Defect"
    ans, method = extract_final_answer(text)
    assert ans == "Defect" and method == "answer_field"


def test_extract_boxed_then_lastline():
    assert extract_final_answer(r"work \boxed{42} done")[0] == "42"
    assert extract_final_answer("a\nb\nfinal thought")[1] == "last_line_weak"


# --- the core D1 fix: no substring false positives ------------------------

def test_short_categorical_no_false_positive():
    """'A' must NOT match the article 'a' / 'In' must NOT match 'information'."""
    rec_A = {"answer_type": "categorical", "ground_truth": "A", "accepted_answers": ["a"]}
    rec_In = {"answer_type": "categorical", "ground_truth": "In", "accepted_answers": ["in"]}
    wrong = "After analysis, information increases. ANSWER: Cooperate"
    assert score_answer(wrong, rec_A)["correct"] is False
    assert score_answer(wrong, rec_In)["correct"] is False


def test_categorical_correct_match():
    rec = {"answer_type": "categorical", "ground_truth": "Defect",
           "accepted_answers": ["defect", "choose defect"]}
    assert score_answer("reasoning...\nANSWER: Defect", rec)["correct"] is True
    assert score_answer("reasoning...\nANSWER: I will choose Defect", rec)["correct"] is True


def test_whole_token_guard():
    assert _whole_token_present("up", "move up now")
    assert not _whole_token_present("up", "upon reflection")
    assert not _whole_token_present("a", "banana")


def test_numeric_scoring_with_tolerance_and_units():
    rec = {"answer_type": "numeric", "ground_truth": "80%", "accepted_answers": ["80%"]}
    assert score_answer("ANSWER: 80%", rec)["correct"] is True
    assert score_answer("ANSWER: 79.9%", rec)["correct"] is True   # within tolerance (rounding)
    assert score_answer("ANSWER: 70%", rec)["correct"] is False    # clearly outside tolerance
    rec2 = {"answer_type": "numeric", "ground_truth": "$60", "accepted_answers": ["60"]}
    assert score_answer("ANSWER: $60", rec2)["correct"] is True


def test_freeform_defers_to_judge():
    rec = {"answer_type": "freeform", "ground_truth": "Mixed 50/50", "accepted_answers": []}
    out = score_answer("ANSWER: randomize evenly", rec)
    assert out["correct"] is None and out["needs_judge"] is True


def test_no_answer_extracted_needs_judge():
    rec = {"answer_type": "categorical", "ground_truth": "Defect", "accepted_answers": ["defect"]}
    out = score_answer("", rec)
    assert out["correct"] is None and out["needs_judge"] is True


# --- migrated-output schema integrity (requires migration to have been run) -----

@pytest.mark.skipif(not (DATA / "final_dataset_v2.json").exists(),
                    reason="run scripts/migrate_dataset.py first")
def test_migrated_schema_complete():
    for fname, n_expected, split in (("final_dataset_v2.json", 300, "train"),
                                     ("heldout_v2.json", 50, "heldout")):
        d = json.load(open(DATA / fname))
        assert d["metadata"]["n"] == n_expected
        for r in d["tasks"]:
            assert set(SCHEMA_FIELDS).issubset(r.keys())
            assert r["answer_type"] in ("categorical", "numeric", "freeform")
            assert r["split"] == split
            if r["answer_type"] != "freeform":
                assert r["accepted_answers"], f"{r['id']} missing accepted_answers"
