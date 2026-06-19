"""
LLM judge / annotator (SPEC §2.3, §5-B6) — the instrument that defines the labels whose DoM
geometry the whole project scrutinises. Two jobs, one configurable client:

  * annotate_chain  -> assign the FIXED 10-label taxonomy (D6) to each segment (feeds DoM/Stage-A)
  * grade_freeform  -> validate a freeform answer vs ground truth (the D1 needs_judge cases)

Provider-agnostic by design: the client is OpenAI-*compatible*, so the judge is whatever
`cfg.extra["judge"]` points at — OpenAI, or an open model (DeepSeek / Qwen / GLM / Llama) served
behind vLLM locally or via DeepInfra/Together/Fireworks. Switching judges is a config change.

Independence matters here more than raw capability. The audit's whole worry is annotator
circularity, so:
  * the PRIMARY judge should be a different lineage than the subject model — do NOT annotate
    DeepSeek-R1-Distill-Qwen chains with a DeepSeek/Qwen judge (shared lineage inflates apparent
    structure); use such a model as the INDEPENDENT second judge for the κ replication instead.
  * temperature 0, fixed prompt; the prompt hash + judge id are recorded for provenance.

`openai` is imported lazily inside `make_client`, so this module (and its pure helpers) import
without the `[annotate]` extra installed.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any

# Reconstructed label definitions. NOTE: reconcile against v1's annotation guide before treating
# the taxonomy as truly "fixed" (D6) — the label SET is locked in base.yaml; these glosses steer the
# judge and are the part most likely to drift from v1. Keep edits here, in one place.
LABEL_DEFINITIONS: dict[str, str] = {
    "opponent_modeling": "reasoning about the other player's beliefs, incentives, type, or likely "
                         "actions",
    "iterated_reasoning": "multi-level 'I think that they think...' / k-level / best-response "
                          "recursion",
    "equilibrium_identification": "invoking or locating an equilibrium concept (Nash, dominant "
                                  "strategy, subgame-perfect, ...)",
    "payoff_analysis": "computing, comparing, or enumerating payoffs / utilities / outcomes",
    "strategic_uncertainty": "explicit reasoning about uncertainty over the opponent's type, "
                             "strategy, or information",
    "cooperative_reasoning": "reasoning about cooperation, coordination, joint gains, trust, or "
                             "fairness",
    "initialization": "restating / setting up the problem, listing givens, defining notation",
    "deduction": "step-by-step logical or mathematical inference toward a conclusion",
    "backtracking": "revisiting, correcting, or abandoning an earlier line ('wait, actually...')",
    "none_other": "none of the above apply / filler / administrative text",
}

_PROMPT_VERSION = "annot-v1"

ANNOTATION_SYSTEM = (
    "You are a careful annotator of reasoning traces from strategic-game problems. You assign "
    "labels from a FIXED taxonomy to each numbered segment. A segment may take several labels or "
    "just 'none_other'. Output ONLY a JSON object mapping each segment index (as a string) to a "
    "list of labels from the taxonomy. Do not invent labels."
)

GRADE_SYSTEM = (
    "You grade whether a model's final answer to a problem is correct given the ground truth. "
    "Output ONLY a JSON object: {\"correct\": true|false, \"rationale\": \"<one sentence>\"}."
)


# --- taxonomy / provenance -------------------------------------------------

def validate_taxonomy(labels: list[str]) -> None:
    """Ensure every configured label has a definition (catches base.yaml / code drift)."""
    missing = [lab for lab in labels if lab not in LABEL_DEFINITIONS]
    if missing:
        raise ValueError(f"labels without definitions in annotate.LABEL_DEFINITIONS: {missing}")


def _taxonomy_block(labels: list[str]) -> str:
    return "\n".join(f"- {lab}: {LABEL_DEFINITIONS[lab]}" for lab in labels)


def annotation_prompt_hash(labels: list[str]) -> str:
    """Stable hash of the annotation instrument (system + taxonomy + version) for provenance."""
    blob = " ".join([_PROMPT_VERSION, ANNOTATION_SYSTEM, _taxonomy_block(labels)])
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


# --- prompt building (pure) ------------------------------------------------

def build_annotation_prompt(chain_id: str, segments: list[dict], labels: list[str],
                            max_seg_chars: int = 1200) -> str:
    lines = [
        "Taxonomy:", _taxonomy_block(labels), "",
        f"Segments of reasoning chain {chain_id} (label each by its index):", "",
    ]
    for i, s in enumerate(segments):
        text = s["text"].strip().replace("\n", " ")
        if len(text) > max_seg_chars:
            text = text[:max_seg_chars] + " …"
        lines.append(f"[{i}] {text}")
    lines += ["", 'Return JSON like {"0": ["payoff_analysis"], "1": ["opponent_modeling", '
              '"deduction"]}.']
    return "\n".join(lines)


def build_grade_prompt(record: dict, model_answer: str) -> str:
    return "\n".join([
        f"Problem:\n{record.get('task', '')}", "",
        f"Ground truth: {record.get('ground_truth', '')}",
        f"Explanation: {record.get('ground_truth_explanation', '')}", "",
        f"Model's final answer: {model_answer}", "",
        "Is the model's answer correct?",
    ])


# --- response parsing (pure) -----------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _extract_json(text: str) -> Any:
    """Best-effort JSON object from a model reply (handles code fences / surrounding prose)."""
    if not text:
        return {}
    m = _FENCE.search(text)
    candidate = m.group(1) if m else text
    try:
        return json.loads(candidate)
    except Exception:
        a, b = candidate.find("{"), candidate.rfind("}")
        if 0 <= a < b:
            try:
                return json.loads(candidate[a:b + 1])
            except Exception:
                return {}
        return {}


def parse_annotation_response(text: str, n_segments: int,
                              labels: list[str]) -> tuple[list[list[str]], int]:
    """Map the judge's JSON to per-segment label lists. Unknown labels are dropped and counted;
    an empty result defaults to ['none_other']. Returns (per_seg_labels, n_unknown_dropped)."""
    obj = _extract_json(text)
    labelset = set(labels)
    per_seg: list[list[str]] = []
    n_unknown = 0
    for i in range(n_segments):
        raw = obj.get(str(i), obj.get(i, [])) if isinstance(obj, dict) else []
        if isinstance(raw, str):
            raw = [raw]
        clean = []
        for lab in raw or []:
            lab = str(lab).strip()
            if lab in labelset:
                clean.append(lab)
            else:
                n_unknown += 1
        per_seg.append(sorted(set(clean)) or ["none_other"])
    return per_seg, n_unknown


def parse_grade_response(text: str) -> dict:
    obj = _extract_json(text)
    correct = obj.get("correct") if isinstance(obj, dict) else None
    return {"correct": bool(correct) if isinstance(correct, bool) else None,
            "rationale": (obj.get("rationale", "") if isinstance(obj, dict) else "")}


# --- client + calls (network; lazy openai) ---------------------------------

def make_client(judge_cfg: dict):
    """OpenAI-compatible client from a judge config block (OpenRouter by default).

    Key resolution: env named by `api_key_env` (OPENROUTER_API_KEY) -> JUDGE_API_KEY ->
    OPENAI_API_KEY -> "EMPTY" (keyless local servers). OpenRouter's optional attribution headers
    are sent when configured.
    """
    from openai import OpenAI  # noqa: PLC0415  ([annotate] extra)
    api_key = (os.environ.get(judge_cfg.get("api_key_env", "OPENROUTER_API_KEY"))
               or os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY")
    headers = {}
    if judge_cfg.get("http_referer"):
        headers["HTTP-Referer"] = judge_cfg["http_referer"]
    if judge_cfg.get("x_title"):
        headers["X-Title"] = judge_cfg["x_title"]
    kwargs: dict = {"api_key": api_key, "base_url": judge_cfg.get("base_url") or None}
    if headers:
        kwargs["default_headers"] = headers
    return OpenAI(**kwargs)


def _complete(client, judge_cfg: dict, system: str, user: str) -> str:
    resp = client.chat.completions.create(
        model=judge_cfg["model"],
        temperature=judge_cfg.get("temperature", 0),
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return resp.choices[0].message.content or ""


def annotate_chain(client, judge_cfg: dict, chain: dict, segments: list[dict],
                   labels: list[str]) -> tuple[list[dict], int]:
    """Label one chain's segments. Returns (annotation_records, n_unknown_dropped)."""
    prompt = build_annotation_prompt(chain["id"], segments, labels)
    text = _complete(client, judge_cfg, ANNOTATION_SYSTEM, prompt)
    per_seg, n_unknown = parse_annotation_response(text, len(segments), labels)
    records = [
        {"chain_id": chain["id"], "seg_idx": s["seg_idx"], "region": s.get("region"),
         "labels": per_seg[i]}
        for i, s in enumerate(segments)
    ]
    return records, n_unknown


def grade_freeform(client, judge_cfg: dict, record: dict, model_answer: str) -> dict:
    text = _complete(client, judge_cfg, GRADE_SYSTEM, build_grade_prompt(record, model_answer))
    return parse_grade_response(text)
