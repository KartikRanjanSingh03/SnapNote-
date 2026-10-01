"""Study-notes generation for SnapNote.

Two backends, one interface:

* ``"extractive"`` — pure-Python extractive summarizer. Works fully offline,
  zero dependencies, zero downloads. This is the **dev fallback**: it is
  honest and deterministic, but it is not an LLM.
* ``"llm"`` — a local Llama-class model through ``transformers``. This is the
  **dev path** (runs on laptop CPU; slow, needs the model download). The real
  NPU path is a compiled Llama via Qualcomm AI Hub — recipe
  ``llama_v3_2_3b_instruct`` (verified in
  https://github.com/qualcomm/ai-hub-models, see README.md) — deployed to the
  Hexagon NPU; that deployment is documented in README.md, not executed here.

No performance numbers are stated anywhere in this module. Benchmarks come
from ``scripts/aihub_profile.py`` against AI Hub cloud-hosted devices.
"""

from __future__ import annotations

import abc
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Prompt templates (used by the "llm" backend; kept as data so they are
# auditable and reusable by the on-device Llama deployment too).
# ---------------------------------------------------------------------------

NOTES_SYSTEM_PROMPT = (
    "You turn a lecture transcript into structured study notes. "
    "Be faithful to the transcript: do not invent facts. "
    "Use exactly the four sections below, in this order."
)

NOTES_USER_TEMPLATE = """Transcript:
\"\"\"
{transcript}
\"\"\"

Produce study notes with exactly these sections:

## Key points
- 5-8 bullet points of the most important ideas.

## Definitions
- Term — one-line definition, only for terms the transcript defines.

## Action items
- Things the lecturer asked students to do, remember, or revise.

## Quiz
- 3-5 short questions a student could use to self-test.
"""

# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------


@dataclass
class Notes:
    """Structured study notes."""

    key_points: List[str] = field(default_factory=list)
    definitions: List[Tuple[str, str]] = field(default_factory=list)  # (term, definition)
    action_items: List[str] = field(default_factory=list)
    quiz: List[str] = field(default_factory=list)
    backend: str = ""

    def to_markdown(self) -> str:
        lines = ["# Study Notes", ""]
        lines.append("## Key points")
        lines += [f"- {p}" for p in self.key_points] or ["- (none found)"]
        lines.append("")
        lines.append("## Definitions")
        lines += [f"- **{t}** — {d}" for t, d in self.definitions] or ["- (none found)"]
        lines.append("")
        lines.append("## Action items")
        lines += [f"- {a}" for a in self.action_items] or ["- (none found)"]
        lines.append("")
        lines.append("## Quiz")
        lines += [f"{i}. {q}" for i, q in enumerate(self.quiz, 1)] or ["(none generated)"]
        lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, object]:
        return {
            "key_points": self.key_points,
            "definitions": [{"term": t, "definition": d} for t, d in self.definitions],
            "action_items": self.action_items,
            "quiz": self.quiz,
            "backend": self.backend,
        }


# ---------------------------------------------------------------------------
# Extractive backend (pure Python, offline, always available)
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because been
    before being below between both but by can did do does doing down during each
    few for from further had has have having he her here hers herself him his how
    i if in into is it its itself me more most my myself no nor not of off on once
    only or other ought our ours ourselves out over own same she should so some
    such than that the their theirs them themselves then there these they this
    those through to too under until up very was we were what when where which
    while who whom why with would you your yours yourself yourselves
    """.split()
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
_WORD = re.compile(r"[a-zA-Z']+")

_DEFINITION_PATTERNS = [
    re.compile(r"(?P<term>[A-Z][\w\- ]{1,60}?)\s+is defined as\s+(?P<defn>[^.]{5,200})", re.I),
    re.compile(r"(?P<term>[A-Z][\w\- ]{1,60}?)\s+refers to\s+(?P<defn>[^.]{5,200})", re.I),
    re.compile(r"(?P<term>[A-Z][\w\- ]{1,60}?)\s+means\s+(?P<defn>[^.]{5,200})", re.I),
    re.compile(r"(?P<term>[A-Z][\w\- ]{1,60}?)\s+is the (?P<kind>process|term|name for|measure of)\b(?P<defn>[^.]{5,200})", re.I),
]

_ACTION_CUES = re.compile(
    r"\b(remember|note that|make sure|don't forget|do not forget|you should|"
    r"practice|revise|keep in mind|important|exam|test you|homework|assignment)\b",
    re.I,
)


def _split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = _SENTENCE_SPLIT.split(text)
    return [p.strip() for p in parts if len(p.strip()) > 1]


def _word_freqs(sentences: List[str]) -> Dict[str, float]:
    freqs: Dict[str, int] = {}
    for s in sentences:
        for w in _WORD.findall(s.lower()):
            if w not in _STOPWORDS and len(w) > 2:
                freqs[w] = freqs.get(w, 0) + 1
    if not freqs:
        return {}
    peak = max(freqs.values())
    return {w: c / peak for w, c in freqs.items()}


def _score_sentence(sentence: str, freqs: Dict[str, float]) -> float:
    words = [w for w in _WORD.findall(sentence.lower()) if w not in _STOPWORDS]
    if not words:
        return 0.0
    # Average normalized frequency; slight preference for mid-length sentences.
    base = sum(freqs.get(w, 0.0) for w in words) / len(words)
    length_bonus = 1.0 if 8 <= len(words) <= 40 else 0.7
    return base * length_bonus


def _clean_term(term: str) -> str:
    term = term.strip().strip(" \"'“”")
    # Drop leading determiners captured by the regex, e.g. "the process".
    term = re.sub(r"^(the|a|an)\s+", "", term, flags=re.I)
    return term[:60].strip()


class ExtractiveNotesBackend:
    """Dev fallback: keyword-scoring extractive summarizer, pure Python."""

    name = "extractive"

    def __init__(self, max_key_points: int = 7, max_quiz: int = 5):
        self.max_key_points = max_key_points
        self.max_quiz = max_quiz

    def generate(self, transcript: str) -> Notes:
        sentences = _split_sentences(transcript)
        notes = Notes(backend=self.name)
        if not sentences:
            return notes

        freqs = _word_freqs(sentences)
        ranked = sorted(
            ((i, _score_sentence(s, freqs)) for i, s in enumerate(sentences)),
            key=lambda t: t[1],
            reverse=True,
        )

        # --- key points: top-scoring sentences, restored to original order ---
        picked = sorted(i for i, _ in ranked[: self.max_key_points])
        seen = set()
        for i in picked:
            s = sentences[i]
            key = s.lower()
            if key not in seen:
                seen.add(key)
                notes.key_points.append(s)

        # --- definitions: explicit definitional phrasing only (never invented) ---
        for s in sentences:
            for pat in _DEFINITION_PATTERNS:
                m = pat.search(s)
                if m:
                    term = _clean_term(m.group("term"))
                    # The term must be a real capitalized term ("Photosynthesis"),
                    # not a relative pronoun the regex happened to catch ("which").
                    if not term or not term[0].isupper():
                        continue
                    defn = m.group("defn").strip().rstrip(".")
                    if term and defn and (term, defn) not in notes.definitions:
                        notes.definitions.append((term, defn))
                    break

        # --- action items: lecturer directives ---
        for s in sentences:
            if _ACTION_CUES.search(s) and s not in notes.action_items:
                notes.action_items.append(s)

        # --- quiz: questions derived from what the transcript actually says ---
        for term, _ in notes.definitions[: self.max_quiz]:
            notes.quiz.append(f"What is {term}?")
        remaining = self.max_quiz - len(notes.quiz)
        for s in notes.key_points:
            if remaining <= 0:
                break
            # Skip sentences already covered by a definition question.
            if any(t.lower() in s.lower() for t, _ in notes.definitions):
                continue
            short = s if len(s) <= 140 else s[:137].rsplit(" ", 1)[0] + "..."
            notes.quiz.append(f"In your own words, explain: {short}")
            remaining -= 1

        return notes


# ---------------------------------------------------------------------------
# LLM backend (dev path: local Llama-class model via transformers)
# ---------------------------------------------------------------------------


class LLMNotesBackend:
    """Dev path: local Llama-class model via Hugging Face transformers.

    Runs on laptop CPU in this repo (slow; needs a model download). The
    production/NPU path is the same prompt templates run against a compiled
    Llama via Qualcomm AI Hub — recipe ``llama_v3_2_3b_instruct`` — deployed
    to the Hexagon NPU. See README.md "Snapdragon deployment guide".
    """

    name = "llm"

    def __init__(self, model_id: Optional[str] = None, max_new_tokens: int = 512):
        # model_id is required explicitly: we do not silently pull weights.
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return self._pipe
        if not self.model_id:
            raise RuntimeError(
                "LLMNotesBackend needs an explicit Hugging Face model id, e.g.\n"
                "    NotesGenerator(backend='llm', model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0')\n"
                "This downloads weights and runs on CPU — the dev path only."
            )
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "transformers is not installed. Install it with:\n"
                "    pip install transformers torch\n"
                "(optional dependency — the extractive backend needs nothing)."
            ) from exc
        self._pipe = pipeline("text-generation", model=self.model_id)
        return self._pipe

    def generate(self, transcript: str) -> Notes:
        pipe = self._load()
        prompt = NOTES_SYSTEM_PROMPT + "\n\n" + NOTES_USER_TEMPLATE.format(transcript=transcript)
        out = pipe(prompt, max_new_tokens=self.max_new_tokens, do_sample=False,
                   return_full_text=False)[0]["generated_text"]
        return _parse_llm_notes(out)

    @property
    def prompt_template(self) -> str:  # reused by the on-device Llama path
        return NOTES_USER_TEMPLATE


def _parse_llm_notes(text: str) -> Notes:
    """Best-effort parse of the four-section LLM output back into Notes."""
    notes = Notes(backend="llm")
    sections = re.split(r"^##\s+", text, flags=re.M | re.I)
    current = None
    buckets: Dict[str, List[str]] = {"key points": [], "definitions": [], "action items": [], "quiz": []}
    for part in sections:
        head, _, body = part.partition("\n")
        key = head.strip().lower()
        current = key if key in buckets else None
        if current:
            for line in body.splitlines():
                line = re.sub(r"^[\-\*\d\.\)\s]+", "", line).strip()
                if line:
                    buckets[current].append(line)
    notes.key_points = buckets["key points"]
    notes.action_items = buckets["action items"]
    notes.quiz = buckets["quiz"]
    for item in buckets["definitions"]:
        term, sep, defn = item.partition("—")
        if not sep:
            term, sep, defn = item.partition("-")
        notes.definitions.append((term.strip(" *"), defn.strip() if sep else item))
    return notes


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------


class NotesGenerator:
    """Generate structured study notes from a transcript."""

    BACKENDS = {
        "extractive": ExtractiveNotesBackend,
        "llm": LLMNotesBackend,
    }

    def __init__(self, backend: str = "extractive", **kwargs):
        if backend not in self.BACKENDS:
            raise ValueError(f"Unknown backend {backend!r}; choose from {sorted(self.BACKENDS)}")
        self.backend_name = backend
        self.backend = self.BACKENDS[backend](**kwargs)

    def generate(self, transcript: str) -> Notes:
        if not transcript or not transcript.strip():
            raise ValueError("Transcript is empty — nothing to summarize.")
        return self.backend.generate(transcript)
