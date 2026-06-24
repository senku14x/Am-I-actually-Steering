"""Tests for the pure char<->token mapping (SPEC §8 round-trip de-risk); no torch."""
from strat_geom.activations import token_span_for_char_span


def test_overlap_basic():
    om = [(0, 3), (3, 6), (6, 9), (9, 12)]
    assert token_span_for_char_span(om, 3, 9) == (1, 2)


def test_partial_overlap_tokens_are_included():
    om = [(0, 4), (4, 8), (8, 12)]
    assert token_span_for_char_span(om, 2, 6) == (0, 1)


def test_zero_width_special_tokens_skipped():
    om = [(0, 0), (0, 3), (3, 6), (0, 0)]
    assert token_span_for_char_span(om, 0, 6) == (1, 2)


def test_no_overlap_returns_none():
    om = [(0, 3), (3, 6)]
    assert token_span_for_char_span(om, 10, 20) == (None, None)
