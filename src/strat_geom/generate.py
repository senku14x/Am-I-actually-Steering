"""
Chain generation (SPEC §2.1) + the config-driven think-region parser.

Backends (select with `backend=`):
  * "vllm" (default) — fast batched decoding for the real GPU-host runs.
  * "hf"             — transformers.generate fallback (small de-risk / hosts without vLLM).
Both share the torch-free pure helpers (build_messages / build_prompt / parse_completion), which are
unit-tested in CI. Think delimiters + chat behaviour are config-driven, NOT hardcoded to R1's
`<think>`/`</think>`, so a native-recipe model (Phi-4-reasoning) is handled by swapping config.

Decoding is deterministic (greedy; temperature 0 / `do_sample=False`); `max_new_tokens` comes from
config, and every chain records a `truncated` flag so the v1 "30–46% silently truncated" failure is
visible, not hidden. The stored `prompt` is the chat-templated string the activation stage re-tokens
to reconstruct context.
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


def build_messages(task: dict, cfg: Config) -> list[dict]:
    """Chat messages for a task — shared by both backends so prompts are identical."""
    system = cfg.extra.get("system_prompt", DEFAULT_SYSTEM)
    return [{"role": "system", "content": system}, {"role": "user", "content": task["task"]}]


def build_prompt(task: dict, cfg: Config, tokenizer: Any) -> str:
    """Render the chat prompt string. Uses the tokenizer's chat template when available so each model
    gets its own template; falls back to a plain system+user concatenation."""
    messages = build_messages(task, cfg)
    apply = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply):
        try:
            return apply(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return f"{messages[0]['content']}\n\n{task['task']}\n"


def parse_completion(completion: str, cfg: Config) -> dict:
    """Split a completion into think/answer character spans using the model's delimiters.

    Matching the literal delimiter (`</think>`) is safe — unlike v1's search for arbitrary segment
    *content*, a structural marker is unambiguous. Handles four shapes:
      * open + close present      -> think = between them
      * close only (template ate the opener, common for R1 distills) -> think = [0, close)
      * neither, reasoning model  -> the `<think>` opener lives in the prompt and the chain was
        truncated before `</think>`, so the whole completion is unfinished THINK (set
        `reasoning: true` in the model config). This recovers truncated chains instead of
        mislabelling them as answer.
      * neither, plain instruct (0.5B CI model) -> all answer
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
        # opened but never closed -> truncated mid-think; no answer region.
        return {"think_span": [o + len(open_), n], "answer_span": [n, n], "has_think": True}
    if cfg.extra.get("reasoning", False):
        # opener was in the prompt and the chain hit the token cap before `</think>` -> all think.
        return {"think_span": [0, n], "answer_span": [n, n], "has_think": True}
    return {"think_span": None, "answer_span": [0, n], "has_think": False}


def _record(task: dict, cfg: Config, prompt: str, completion: str, n_gen: int,
            truncated: bool) -> dict:
    parsed = parse_completion(completion, cfg)
    return {
        "id": task["id"], "model_short": cfg.model_short, "prompt": prompt,
        "completion": completion, "think_span": parsed["think_span"],
        "answer_span": parsed["answer_span"], "has_think": parsed["has_think"],
        "n_gen_tokens": n_gen, "truncated": truncated, "answer_type": task.get("answer_type"),
    }


def generate_chains(cfg: Config, tasks: list[dict], backend: str = "vllm", batch_size: int = 8,
                    device: str | None = None, dtype: str | None = None,
                    gpu_memory_utilization: float = 0.90) -> list[dict]:
    """Generate one chain per task with greedy decoding. GPU host only. `backend` ∈ {vllm, hf}."""
    if backend == "vllm":
        return _generate_vllm(cfg, tasks, dtype=dtype or "auto",
                              gpu_memory_utilization=gpu_memory_utilization)
    if backend == "hf":
        return _generate_hf(cfg, tasks, batch_size=batch_size, device=device,
                           dtype=dtype or "float16")
    raise ValueError(f"unknown backend {backend!r} (use 'vllm' or 'hf')")


def _generate_vllm(cfg: Config, tasks: list[dict], dtype: str = "auto",
                   gpu_memory_utilization: float = 0.90) -> list[dict]:
    from transformers import AutoTokenizer  # noqa: PLC0415
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    llm = LLM(model=cfg.model_name, dtype=dtype, gpu_memory_utilization=gpu_memory_utilization,
              max_model_len=cfg.extra.get("max_model_len"), trust_remote_code=True)
    convs = [build_messages(t, cfg) for t in tasks]
    sp = SamplingParams(temperature=0.0, max_tokens=cfg.max_new_tokens)
    outs = llm.chat(convs, sp, use_tqdm=True)                 # vLLM applies the model's chat template

    records: list[dict] = []
    for t, out in zip(tasks, outs):
        o = out.outputs[0]
        prompt = build_prompt(t, cfg, tok)                    # stored for the activation stage
        records.append(_record(t, cfg, prompt, o.text, len(o.token_ids),
                               o.finish_reason == "length"))
    return records


def _generate_hf(cfg: Config, tasks: list[dict], batch_size: int = 8, device: str | None = None,
                 dtype: str = "float16") -> list[dict]:
    import torch  # noqa: PLC0415
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
            truncated = tok.eos_token_id not in row_list          # no EOS => hit the token cap
            completion = tok.decode(row, skip_special_tokens=True)
            records.append(_record(t, cfg, prompt, completion, len(row_list), truncated))
    return records
