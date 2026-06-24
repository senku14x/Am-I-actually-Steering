"""Pure-helper tests for generate.py (no torch): think-region parsing + prompt building."""
from strat_geom.config import Config
from strat_geom.generate import DEFAULT_SYSTEM, build_prompt, parse_completion


def _cfg(**extra) -> Config:
    c = Config()
    c.extra = dict(extra)
    return c


def test_default_system_requires_answer_format():
    assert "ANSWER:" in DEFAULT_SYSTEM


def test_parse_open_and_close():
    cfg = _cfg()
    comp = "<think>reasoning here</think>\nANSWER: 42"
    p = parse_completion(comp, cfg)
    assert p["has_think"]
    assert comp[p["think_span"][0]:p["think_span"][1]] == "reasoning here"
    assert "ANSWER: 42" in comp[p["answer_span"][0]:p["answer_span"][1]]


def test_parse_close_only_when_template_ate_opener():
    cfg = _cfg()
    comp = "reasoning without opener</think>\nANSWER: x"
    p = parse_completion(comp, cfg)
    assert p["has_think"] and p["think_span"][0] == 0
    assert comp[p["think_span"][0]:p["think_span"][1]] == "reasoning without opener"


def test_parse_no_tags_is_all_answer():
    cfg = _cfg()
    comp = "just an answer\nANSWER: 7"
    p = parse_completion(comp, cfg)
    assert not p["has_think"] and p["think_span"] is None
    assert p["answer_span"] == [0, len(comp)]


def test_parse_reasoning_model_no_tags_is_all_think():
    # reasoning model (<think> in the prompt) truncated before </think> -> whole completion is think
    cfg = _cfg(reasoning=True)
    comp = "still reasoning, hit the token cap"
    p = parse_completion(comp, cfg)
    assert p["has_think"] and p["think_span"] == [0, len(comp)]
    assert p["answer_span"] == [len(comp), len(comp)]


def test_parse_truncated_midthink_has_empty_answer():
    cfg = _cfg()
    comp = "<think>still thinking, cut off"
    p = parse_completion(comp, cfg)
    assert p["has_think"] and p["answer_span"] == [len(comp), len(comp)]


def test_parse_custom_delimiters_from_config():
    cfg = _cfg(think_open="<reason>", think_close="</reason>")
    comp = "<reason>r</reason>A"
    p = parse_completion(comp, cfg)
    assert comp[p["think_span"][0]:p["think_span"][1]] == "r"
    assert comp[p["answer_span"][0]:p["answer_span"][1]] == "A"


class _FakeTok:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"


def test_build_prompt_uses_chat_template():
    out = build_prompt({"task": "2+2?"}, _cfg(), _FakeTok())
    assert "2+2?" in out and out.endswith("<assistant>") and DEFAULT_SYSTEM in out


def test_build_prompt_falls_back_without_template():
    out = build_prompt({"task": "hi"}, _cfg(), object())
    assert "hi" in out and DEFAULT_SYSTEM in out
