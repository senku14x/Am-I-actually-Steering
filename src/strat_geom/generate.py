"""
Chain generation (SPEC §2.1) + the config-driven think-region parser.

Two layers:
  * Pure helpers (`build_prompt`, `parse_completion`) — no torch, unit-tested in CI. The think
    delimiters and chat behaviour are read from config, NOT hardcoded to R1's `<think>`/`</think>`,
    so a native-recipe model (Phi-4-reasoning) is handled by swapping config, not code.
  * `generate_chains` — loads the model and runs fixed decoding on the GPU host. torch/transformers
    are imported lazily inside it so importing this module on a CPU/CI box stays cheap.

Decoding is deterministic (`do_sample=False`); `max_new_tokens` comes from config (set per model
from the measured length distribution, SPEC §2.1) and every chain records a `truncated` flag so the
v1 "30–46% silently truncated" failure is visible, not hidden.
"""
from __future__ import annotations

from typing import Any

from .config import Config

# The scorer (dataset.extract_final_answer) keys off an "ANSWER: <x>" line, so we must ask for it.
DEFAULT_SYSTEM = (
    "Solve the problem. Reason step by step, then end your reply with your final answer on its own "
    "line in exactly this format:\nANSWER: <answer>"
)


def _think_delims(cfg: Config) -> tuple[str, str]:
    return cfg.extra.get("think_open", "<think>"), cfg.extra.get("think_close", "</think>")


def build_prompt(task: dict, cfg: Config, tokenizer: Any) -> str:
    """Render the chat prompt for a task. Uses the tokenizer's chat template when available so each
    model gets its own template; falls back to a plain system+user concatenation."""
    system = cfg.extra.get("system_prompt", DEFAULT_SYSTEM)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": task["task"]}]
    apply = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply):
        try:
            return apply(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return f"{system}\n\n{task['task']}\n"


def parse_completion(completion: str, cfg: Config) -> dict:
    """Split a completion into think/answer character spans using the model's delimiters.

    Matching the literal delimiter (`</think>`) is safe — unlike v1's search for arbitrary segment
    *content*, a structural marker is unambiguous. Handles three shapes:
      * open + close present      -> think = between them
      * close only (template ate the opener, common for R1 distills) -> think = [0, close)
      * neither (plain instruct, e.g. the 0.5B CI model)             -> all answer
    Returns {think_span: [s,e]|None, answer_span: [s,e], has_think: bool} with offsets into
    `completion`.
    """
    open_, close = _think_delims(cfg)
    n = len(completion)
    o = completion.find(open_)
    c = completion.find(close)
    if c != -1:
        think_start = o + len(open_) if (o != -1 and o < c) else 0
        return {"think_span": [think_start, c], "answer_span": [c + len(close), n], "has_think": True}
    if o != -1:
        # opened but never closed -> likely truncated mid-think; no answer region.
        return {"think_span": [o + len(open_), n], "answer_span": [n, n], "has_think": True}
    return {"think_span": None, "answer_span": [0, n], "has_think": False}


def generate_chains(cfg: Config, tasks: list[dict], batch_size: int = 8,
                    device: str | None = None, dtype: str = "float16") -> list[dict]:
    """Generate one chain per task with fixed decoding. GPU host only (lazy torch import)."""
    import torch  # noqa: PLC0415  (host-only dependency)
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=getattr(torch, dtype), device_map=device,
    )
    model.eval()

    records: list[dict] = []
    for start in range(0, len(tasks), batch_size):
        batch = tasks[start:start + batch_size]
        prompts = [build_prompt(t, cfg, tok) for t in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=cfg.max_new_tokens, do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        gen = out[:, enc["input_ids"].shape[1]:]                  # completion tokens only
        for t, prompt, row in zip(batch, prompts, gen):
            row_list = row.tolist()
            eos = tok.eos_token_id
            truncated = eos not in row_list                       # no EOS => hit the token cap
            n_gen = len(row_list)
            completion = tok.decode(row, skip_special_tokens=True)
            parsed = parse_completion(completion, cfg)
            records.append({
                "id": t["id"], "model_short": cfg.model_short, "prompt": prompt,
                "completion": completion, "think_span": parsed["think_span"],
                "answer_span": parsed["answer_span"], "has_think": parsed["has_think"],
                "n_gen_tokens": n_gen, "truncated": truncated, "answer_type": t.get("answer_type"),
            })
    return records
