"""
Chain -> segments with exact character offsets (SPEC §2.2).

v1 located segments with `full_output.find(segment_text)`, which silently mis-aligns whenever a
segment string occurs more than once in a chain. Here we never search for segment *content*: we
split on paragraph boundaries and hard-wrap long paragraphs while tracking (char_start, char_end),
so by construction `completion[seg.char_start:seg.char_end] == seg.text`. `realign_failures`
re-checks that identity and is asserted in CI — it should always be empty.

Offsets are into the model's `completion` string (the assistant turn only), the same coordinate
system the activation extractor maps tokens against, so a segment's text and its pooled activation
refer to the same characters.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .config import Config
from .generate import parse_completion

_PARA_SEP = re.compile(r"(?:[ \t]*\n){2,}")   # a run of >=2 line breaks = blank-line separator


@dataclass
class Segment:
    chain_id: str
    seg_idx: int
    region: str          # "think" | "answer"
    char_start: int
    char_end: int
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def _paragraph_spans(text: str, lo: int, hi: int) -> list[tuple[int, int]]:
    """Spans of paragraphs within text[lo:hi], blank-line separated (separators excluded)."""
    spans: list[tuple[int, int]] = []
    pos = lo
    for m in _PARA_SEP.finditer(text, lo, hi):
        if m.start() > pos:
            spans.append((pos, m.start()))
        pos = m.end()
    if pos < hi:
        spans.append((pos, hi))
    return spans


def _wrap_spans(text: str, lo: int, hi: int, max_chars: int):
    """Sub-divide text[lo:hi] into contiguous spans each <= max_chars, breaking at the last
    whitespace before the limit when possible (hard cut otherwise)."""
    i = lo
    while i < hi:
        if hi - i <= max_chars:
            yield (i, hi)
            return
        window_end = i + max_chars
        brk = text.rfind(" ", i, window_end)
        if brk <= i:
            brk = text.rfind("\n", i, window_end)
        if brk <= i:
            brk = window_end          # no breakpoint -> hard cut at the limit
        yield (i, brk)
        i = brk


def segment_spans(text: str, lo: int, hi: int, max_chars: int = 1200):
    """Yield (start, end) segment spans covering the paragraphs in text[lo:hi]."""
    for ps, pe in _paragraph_spans(text, lo, hi):
        yield from _wrap_spans(text, ps, pe, max_chars)


def segment_chain(chain: dict, cfg: Config, max_chars: int = 1200) -> list[Segment]:
    """Segment a chain record's completion into think/answer segments with exact offsets."""
    completion = chain.get("completion", "")
    cid = chain["id"]
    # Always re-parse from the completion so the CURRENT parser/config governs regions (e.g. the
    # `reasoning` flag), not spans frozen at generation time. Segments (char spans) are identical
    # either way — only the region label can change — so re-running `segment` alone is enough to
    # relabel without re-extracting activations.
    parsed = parse_completion(completion, cfg)
    think_span, answer_span = parsed["think_span"], parsed["answer_span"]

    out: list[Segment] = []
    idx = 0
    for region, span in (("think", think_span), ("answer", answer_span)):
        if not span:
            continue
        s, e = span
        for ws, we in segment_spans(completion, s, e, max_chars):
            text = completion[ws:we]
            if not text.strip():               # skip whitespace-only spans
                continue
            out.append(Segment(cid, idx, region, ws, we, text))
            idx += 1
    return out


def realign_failures(completion: str, segments: list[Segment]) -> list[int]:
    """seg_idx of any segment whose stored text != completion[start:end] (should be empty)."""
    return [s.seg_idx for s in segments if completion[s.char_start:s.char_end] != s.text]
