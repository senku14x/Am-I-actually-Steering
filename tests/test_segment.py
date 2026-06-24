"""Tests for segment.py — the offset-exact invariant is the whole point (SPEC §2.2)."""
from strat_geom.config import Config
from strat_geom.segment import _paragraph_spans, realign_failures, segment_chain, segment_spans


def _cfg(**extra) -> Config:
    c = Config()
    c.extra = dict(extra)
    return c


def test_wrap_spans_contiguous_and_bounded():
    text = "word " * 100                       # 500 chars, one paragraph
    spans = list(segment_spans(text, 0, len(text), max_chars=120))
    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    assert all(e - s <= 120 for s, e in spans)
    for (_, e), (s2, _) in zip(spans, spans[1:]):
        assert e == s2                          # contiguous within a paragraph


def test_paragraph_split_excludes_separators():
    text = "para one\n\npara two\n\n\npara three"
    spans = _paragraph_spans(text, 0, len(text))
    assert [text[s:e] for s, e in spans] == ["para one", "para two", "para three"]


def test_segment_chain_offset_invariant_and_regions():
    completion = "<think>" + ("alpha " * 50) + "</think>\n\nANSWER: alpha"
    segs = segment_chain({"id": "t1", "completion": completion}, _cfg(), max_chars=80)
    assert segs
    assert realign_failures(completion, segs) == []        # the invariant
    assert {s.region for s in segs} == {"think", "answer"}
    close = completion.find("</think>")
    assert all(s.char_end <= close for s in segs if s.region == "think")


def test_segment_chain_hardcut_without_whitespace():
    completion = "x" * 300                       # no whitespace -> forced hard cuts
    segs = segment_chain({"id": "t", "completion": completion}, _cfg(), max_chars=100)
    assert realign_failures(completion, segs) == []
    assert max(s.char_end - s.char_start for s in segs) <= 100


def test_whitespace_only_segments_skipped():
    completion = "real text\n\n   \n\nmore text"
    segs = segment_chain({"id": "t", "completion": completion}, _cfg())
    assert segs and all(s.text.strip() for s in segs)
