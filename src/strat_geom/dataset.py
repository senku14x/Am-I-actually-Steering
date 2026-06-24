"""
Dataset schema + answer scoring (fixes D1, D2 from SPEC §0).

Why this exists
---------------
v1 scored correctness with `ground_truth.lower() in answer.lower()` (bare substring). The audit
showed this is broken both ways:
  - 14 train GTs are <=8 chars ("A", "Up", "In", "Yes", "0") -> the substring is in almost any
    text -> false positives (e.g. gt "A" matches the article "a" everywhere).
  - median GT is ~90 chars and many are full sentences -> the model never echoes them verbatim
    -> false negatives.
  - on the heldout split 23/50 GTs appear verbatim in their own prompt -> prompt-echo leakage.

The fix here:
  1. typed answers: `answer_type in {categorical, numeric, freeform}`,
  2. structured extraction: pull the model's final answer from a required `ANSWER: <x>` line
     (with weak fallbacks that are flagged, never silently trusted),
  3. type-aware scoring: categorical -> normalized whole-token set-match vs `accepted_answers`;
     numeric -> value parse + tolerance; freeform -> defer to an LLM judge (not auto-scored).

This module is intentionally conservative: where a heuristic is uncertain it sets `needs_review`
so the later second-model/human pass (D5) can correct it, rather than pretending to be validated.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Canonical v2 record (D2: unified schema for train + heldout)
# ---------------------------------------------------------------------------

ANSWER_TYPES = ("categorical", "numeric", "freeform")

SCHEMA_FIELDS = (
    "id", "split", "category", "subcategory", "difficulty", "task",
    "answer_type", "ground_truth", "accepted_answers",
    "ground_truth_explanation", "solution_concept", "reasoning_required",
    # provenance / flags
    "answer_type_confidence", "needs_review", "review_reasons",
    "n_task_chars",
)


@dataclass
class TaskRecord:
    id: str
    split: str                       # "train" | "heldout"
    category: str
    subcategory: str
    difficulty: str                  # "easy" | "medium" | "hard" | "unknown"
    task: str
    answer_type: str                 # one of ANSWER_TYPES
    ground_truth: str
    accepted_answers: list[str]      # normalized canonical answers (categorical/numeric)
    ground_truth_explanation: str
    solution_concept: str
    reasoning_required: list[str]
    answer_type_confidence: str = "low"   # "high" | "medium" | "low"
    needs_review: bool = False
    review_reasons: list[str] = field(default_factory=list)
    n_task_chars: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

_WS = re.compile(r"\s+")
# strip surrounding markdown/quotes/punctuation but keep $ % - . for numerics
_EDGE = re.compile(r"^[\s\"'`*_(){}\[\]:.]+|[\s\"'`*_(){}\[\]:.]+$")


def normalize(s: Any) -> str:
    """Lowercase, collapse whitespace, strip edge punctuation. Keeps internal $/%/digits."""
    s = "" if s is None else str(s)
    s = _WS.sub(" ", s).strip().lower()
    s = _EDGE.sub("", s)
    return s.strip()


# ---------------------------------------------------------------------------
# Numeric parsing
# ---------------------------------------------------------------------------

_SUFFIX = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}
# a single numeric token: optional $, digits with optional ,/. , optional k/m/b, optional %
_NUM = re.compile(r"[-+]?\$?\s*\d[\d,]*\.?\d*\s*[kmb]?\s*%?", re.IGNORECASE)


def parse_number(s: str) -> Optional[float]:
    """Parse the first numeric token in `s` to a float. '$449K'->449000, '80%'->80, '0'->0."""
    if s is None:
        return None
    m = _NUM.search(str(s))
    if not m:
        return None
    tok = m.group(0).lower().replace(" ", "").replace(",", "").replace("$", "")
    pct = tok.endswith("%")
    if pct:
        tok = tok[:-1]
    mult = 1
    if tok and tok[-1] in _SUFFIX:
        mult = _SUFFIX[tok[-1]]
        tok = tok[:-1]
    try:
        val = float(tok) * mult
    except ValueError:
        return None
    return val


def numbers_match(a: float, b: float, rel: float = 0.01, abs_: float = 0.5) -> bool:
    return abs(a - b) <= max(abs_, rel * max(abs(a), abs(b)))


# ---------------------------------------------------------------------------
# Answer-type classification (heuristic first pass; flags uncertainty)
# ---------------------------------------------------------------------------

# A "ratio" expression like 50-50 / 50/50 / 60-40 — treat as categorical (not a single number).
_RATIO = re.compile(r"^\d{1,3}\s*[-/]\s*\d{1,3}$")
_SENTENCE = re.compile(r"[.;]\s+\S|\bbecause\b|\bsince\b|\btherefore\b|->|=>", re.IGNORECASE)


def _is_pure_number(cn: str) -> bool:
    """True if `cn` is a short, number-dominated token (no real words)."""
    if not cn or _RATIO.match(cn):
        return False
    return parse_number(cn) is not None and len(cn) <= 12 and not re.search(r"[a-z]{3,}", cn)


def classify_answer_type(ground_truth: str, optimal_action: str) -> tuple[str, str, list[str]]:
    """
    Returns (answer_type, confidence, reasons).

    Order of evidence:
      1. ground_truth's leading clause is a pure number ($60, 80%, 0)  -> numeric
      2. optimal_action canonical is a pure number                     -> numeric
      3. ratio expression (50-50)                                      -> categorical
      4. short action phrase (no sentence structure)                   -> categorical
      5. otherwise                                                     -> freeform (needs judge)
    """
    gt_clause = normalize(_leading_clause(ground_truth))
    oa = normalize(optimal_action)
    canon = oa or gt_clause

    if not canon:
        return "freeform", "low", ["empty canonical answer"]

    # 1 & 2: numeric if a clean short number appears as the GT's stated answer or the action
    if _is_pure_number(gt_clause):
        return "numeric", "high", ["ground_truth is a pure number"]
    if _is_pure_number(oa):
        return "numeric", "high", ["optimal_action is a pure number"]

    if _RATIO.match(canon):
        return "categorical", "high", ["ratio expression"]

    n_words = len(canon.split())
    if n_words <= 5 and not _SENTENCE.search(optimal_action or _leading_clause(ground_truth)):
        conf = "high" if n_words <= 3 else "medium"
        return "categorical", conf, ["short action phrase"]

    return "freeform", "medium", ["multi-clause / sentence answer"]


def _leading_clause(s: str) -> str:
    """First clause of a GT before a dash/colon/sentence break (e.g. 'Yes - value ...' -> 'Yes')."""
    s = (s or "").strip()
    for sep in (" - ", " — ", ": ", ". ", ", "):
        i = s.find(sep)
        if 0 < i <= 40:
            return s[:i].strip()
    return s


# ---------------------------------------------------------------------------
# Accepted-answer construction (categorical / numeric)
# ---------------------------------------------------------------------------

# action words to extract as canonical categorical answers
def build_accepted_answers(ground_truth: str, optimal_action: str, answer_type: str) -> list[str]:
    if answer_type == "freeform":
        return []
    if answer_type == "numeric":
        # keep just the clean number token(s); scoring parses values, not strings
        out = []
        for src in (ground_truth, optimal_action):
            n = parse_number(src)
            if n is not None:
                tok = normalize(_NUM.search(str(src)).group(0))
                if tok and tok not in out:
                    out.append(tok)
        return out
    cands: list[str] = []
    for src in (optimal_action, _leading_clause(ground_truth), ground_truth):
        c = normalize(src)
        if not c:
            continue
        cands.append(c)
        # also a trimmed action core: drop common lead-ins ("choose ", "play ", "bid ", ...)
        core = re.sub(r"^(choose|play|pick|select|bid|go|do|take|rank|set)\s+", "", c).strip()
        if core and core != c:
            cands.append(core)
    # dedupe, keep short distinctive ones first
    seen, out = set(), []
    for c in sorted(set(cands), key=len):
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


# ---------------------------------------------------------------------------
# Final-answer extraction from a model generation
# ---------------------------------------------------------------------------

_ANSWER_LINE = re.compile(r"answer\s*[:\-]\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_BOXED = re.compile(r"\\boxed\{([^}]*)\}")


def extract_final_answer(text: str) -> tuple[str, str]:
    """
    Return (answer_text, method). Prefers an explicit 'ANSWER: x' line, then \\boxed{}, then the
    last non-empty line as a *flagged* weak fallback.
    """
    if not text:
        return "", "empty"
    m = list(_ANSWER_LINE.finditer(text))
    if m:
        return m[-1].group(1).strip(), "answer_field"
    b = _BOXED.findall(text)
    if b:
        return b[-1].strip(), "boxed"
    for line in reversed(text.strip().splitlines()):
        if line.strip():
            return line.strip(), "last_line_weak"
    return "", "empty"


# ---------------------------------------------------------------------------
# Type-aware scoring (the D1 replacement for substring matching)
# ---------------------------------------------------------------------------

def _whole_token_present(needle: str, haystack: str) -> bool:
    """True if `needle` appears in `haystack` on token boundaries (not as a sub-substring)."""
    needle = needle.strip()
    if not needle:
        return False
    return re.search(r"(?<![\w$%]){}(?![\w$%])".format(re.escape(needle)), haystack) is not None


def score_answer(model_text: str, record: dict, numeric_rel: float = 0.01,
                 numeric_abs: float = 0.5) -> dict:
    """
    Score a model generation against a v2 record. Never uses bare substring on raw GT.

    Returns: {correct: bool|None, method: str, extracted: str, needs_judge: bool}
      - correct=None means "cannot auto-score" (freeform, or no usable answer) -> needs_judge.
    """
    atype = record.get("answer_type", "freeform")
    extracted, emethod = extract_final_answer(model_text)
    ex = normalize(extracted)
    weak = emethod in ("last_line_weak", "empty")

    if atype == "freeform":
        return {"correct": None, "method": "freeform_needs_judge", "extracted": extracted,
                "needs_judge": True}

    if not ex:
        return {"correct": None, "method": "no_answer_extracted", "extracted": "",
                "needs_judge": True}

    if atype == "numeric":
        gt_num = parse_number(record.get("ground_truth", ""))
        if gt_num is None and record.get("accepted_answers"):
            gt_num = parse_number(record["accepted_answers"][0])
        ex_num = parse_number(ex)
        if gt_num is None or ex_num is None:
            return {"correct": None, "method": "numeric_parse_failed", "extracted": extracted,
                    "needs_judge": True}
        ok = numbers_match(ex_num, gt_num, numeric_rel, numeric_abs)
        return {"correct": bool(ok), "method": "numeric" + ("_weak" if weak else ""),
                "extracted": extracted, "needs_judge": False}

    # categorical: normalized exact match OR accepted answer present as a whole token in the
    # (short) extracted answer span. Whole-token guard kills the "A"/"Up"/"In" false positives.
    accepted = record.get("accepted_answers", []) or [normalize(record.get("ground_truth", ""))]
    for acc in accepted:
        if not acc:
            continue
        if ex == acc or _whole_token_present(acc, ex):
            return {"correct": True, "method": "categorical" + ("_weak" if weak else ""),
                    "extracted": extracted, "needs_judge": False}
    return {"correct": False, "method": "categorical" + ("_weak" if weak else ""),
            "extracted": extracted, "needs_judge": False}


# ---------------------------------------------------------------------------
# Solution-concept tagging (D5 first pass; flags low confidence)
# ---------------------------------------------------------------------------

_SUBCAT_TO_CONCEPT = {
    # dominant strategy / dominance
    "prisoners_dilemma": "dominant_strategy", "dominated_strategies": "dominant_strategy",
    "iterated_dominance": "dominant_strategy", "second_price_sealed": "dominant_strategy",
    "english_auction": "dominant_strategy",
    # nash / mixed
    "matching_pennies": "nash_mixed", "mixed_strategy": "nash_mixed",
    "mixed_strategy_equilibrium": "nash_mixed", "mixed_strategy_general": "nash_mixed",
    "battle_of_sexes": "nash", "stag_hunt": "nash", "chicken": "nash",
    "pure_coordination": "nash", "three_by_three_matrix": "nash", "asymmetric_matrix": "nash",
    "zero_sum": "nash", "asymmetric_payoffs": "nash", "pareto_efficiency": "nash",
    "focal_points": "nash",
    # backward induction / sequential
    "backward_induction": "backward_induction", "finite_horizon": "backward_induction",
    "centipede": "backward_induction", "credible_threats": "backward_induction",
    "simple_extensive": "backward_induction", "entry_deterrence": "backward_induction",
    "stackelberg": "backward_induction", "commitment": "backward_induction",
    "commitment_device": "backward_induction", "multi_stage_sequential": "backward_induction",
    "war_of_attrition": "backward_induction", "k_level_reasoning": "iterated_best_response",
    "beauty_contest": "iterated_best_response",
    # bayesian / incomplete info
    "bayesian_updating": "bayesian", "imperfect_information": "bayesian",
    "winners_curse": "bayesian", "common_value": "bayesian", "signaling": "bayesian",
    "screening": "bayesian", "lemons": "bayesian", "deception_detection": "bayesian",
    "information_value": "bayesian", "global_games": "bayesian",
    # repeated-game / folk-theorem / best-response-to-stated
    "iterated_prisoners_dilemma": "best_response_to_stated_opponent",
    "tit_for_tat": "best_response_to_stated_opponent",
    "grim_trigger": "best_response_to_stated_opponent",
    "infinite_horizon": "folk_theorem", "reputation_building": "folk_theorem",
    "reputation": "folk_theorem", "multi_player_repeated": "folk_theorem",
    # mechanism / matching
    "mechanism_participation": "mechanism_design", "revenue_equivalence": "mechanism_design",
    "mechanism_revelation": "mechanism_design", "mechanism_design": "mechanism_design",
    # bargaining
    "nash_bargaining": "bargaining", "rubinstein": "bargaining",
    "alternating_offers": "bargaining", "ultimatum": "backward_induction",
    # non-strategic controls
    "probability": "computation", "combinatorics": "computation", "arithmetic": "computation",
    "algebra": "computation", "optimization": "computation", "estimation": "computation",
    "logical_deduction": "computation", "pattern_recognition": "computation",
}


# keyword fallback (handles heldout's Title-Case prose subcategories, e.g. "Signaling Game")
_CONCEPT_KEYWORDS = [
    ("bayesian", ("signaling", "screening", "lemons", "winner's curse", "winners curse",
                  "common value", "bayesian", "incomplete information", "asymmetric information",
                  "opponent type", "interdependent value", "due diligence", "adverse selection")),
    ("backward_induction", ("stackelberg", "entry deter", "centipede", "chain store", "backward",
                            "commitment", "sunk cost", "capacity", "trust game", "credible threat")),
    ("bargaining", ("ultimatum", "nash demand", "bargaining", "rubinstein", "alternating offer",
                    "salary negotiation", "negotiation with", "batna")),
    ("nash_mixed", ("zero-sum", "zero sum", "mixed strategy", "matching pennies", "blotto",
                    "inspection game", "war of attrition")),
    ("folk_theorem", ("tit-for-tat", "tit for tat", "grim trigger", "repeated", "public goods",
                      "reputation", "oligopoly pricing", "renegotiation")),
    ("nash", ("coordination", "matrix game", "pareto", "battle of", "stag hunt", "chicken")),
    ("dominant_strategy", ("second-price", "second price", "english auction", "dominance",
                           "dominated")),
    ("mechanism_design", ("first-price", "first price", "auction", "mechanism", "dutch", "all-pay")),
    ("computation", ("probability", "arithmetic", "algebra", "combinator", "optimization",
                     "estimation", "deduction", "pattern")),
]


def tag_solution_concept(subcategory: str, category: str) -> tuple[str, str]:
    """Return (solution_concept, confidence). Exact subcat map -> medium; keyword -> low; else low."""
    sc = _SUBCAT_TO_CONCEPT.get(subcategory)
    if sc:
        return sc, "medium"
    if category == "non_strategic_control":
        return "computation", "medium"
    sub = normalize(subcategory)
    for concept, kws in _CONCEPT_KEYWORDS:
        if any(k in sub for k in kws):
            return concept, "low"          # keyword-inferred -> still flag for D5 review
    return "heuristic", "low"
