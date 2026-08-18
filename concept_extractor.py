"""
concept_extractor.py

Uses the loaded model itself to pull 3-8 short concept/entity labels out of
an exchange, so the memory graph doesn't need a separate NLP/embedding
dependency. Falls back to a cheap regex heuristic if the model's output
isn't parseable JSON (small quantized models occasionally wobble on format).
"""

from __future__ import annotations

import json
import re
from typing import List, Protocol


EXTRACTION_PROMPT_TEMPLATE = """Extract the 3 to 8 most important concepts, entities, or topics \
from the exchange below. Respond with ONLY a JSON array of short strings (1-4 words each), \
nothing else. No explanation, no markdown fences.

Exchange:
User: {user_msg}
Assistant: {assistant_msg}

JSON array:"""


class Generator(Protocol):
    def __call__(self, prompt: str, max_tokens: int) -> str: ...


def extract_concepts(user_msg: str, assistant_msg: str, generate_fn: Generator) -> List[str]:
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(
        user_msg=user_msg.strip()[:2000],
        assistant_msg=assistant_msg.strip()[:2000],
    )
    raw = generate_fn(prompt, max_tokens=150)
    concepts = _parse_json_array(raw)
    if concepts:
        return concepts
    return _regex_fallback(user_msg + " " + assistant_msg)


def _parse_json_array(raw: str) -> List[str]:
    raw = raw.strip()
    # strip markdown fences if the model added them anyway
    raw = re.sub(r"^```(json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    out = []
    for item in parsed:
        if isinstance(item, str) and 0 < len(item) <= 60:
            out.append(item.strip())
    return out[:8]


_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "and", "or", "but", "if", "then", "so", "for", "to", "of", "in", "on",
    "at", "by", "with", "as", "it", "this", "that", "i", "you", "we", "they",
    "he", "she", "do", "does", "did", "can", "could", "will", "would",
    "should", "have", "has", "had", "not", "what", "how", "why", "when",
}


def _regex_fallback(text: str, max_concepts: int = 6) -> List[str]:
    """Crude noun-ish phrase heuristic: capitalized words and multi-word runs
    of non-stopword tokens. Only used if the model's JSON output fails to parse."""
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-']*", text)
    candidates = []
    buf = []
    for w in words:
        lw = w.lower()
        if lw in _STOPWORDS or len(w) < 3:
            if buf:
                candidates.append(" ".join(buf))
                buf = []
            continue
        buf.append(lw)
        if len(buf) == 3:
            candidates.append(" ".join(buf))
            buf = []
    if buf:
        candidates.append(" ".join(buf))

    seen = []
    for c in candidates:
        if c not in seen:
            seen.append(c)
        if len(seen) >= max_concepts:
            break
    return seen
