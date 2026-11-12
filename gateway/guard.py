"""Backstop for the formatter: catch a reply that answers or obeys the dictation
instead of cleaning it ("Hi everyone." for "...say hi to everyone", a poem for
"write a poem"). The prompt already forbids this; this catches the rare slip.

Only two clear signals are used, so normal cleanup and self-corrections pass:
- the result is far longer than what was said (an answer, a poem), or
- a longer Latin-script dictation without any correction words lost most of its
  words (a reply that only borrows a couple of them).
"""

from __future__ import annotations

import re

_WORD = re.compile(r"[\w']+", re.UNICODE)
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_CORRECTION_CUES = (
    "actually", "i mean", "sorry", "no", "wait", "scratch that", "make that", "let it be",
    "rather", "nahi", "rehne do", "ek minute", "matlab",
)


def _words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def looks_like_reply(source: str, output: str) -> bool:
    said = _words(source)
    wrote = _words(output)
    if not said or not wrote:
        return False
    # An answer or a generated text: much longer than the dictation itself.
    if len(wrote) > max(2 * len(said), len(said) + 12):
        return True
    # A reply that keeps only a few of the speaker's words. Needs comparable scripts and
    # enough words to judge, and no self-correction (which legitimately drops words).
    if _DEVANAGARI.search(source) or len(said) < 8:
        return False
    lowered = " " + " ".join(said) + " "
    if any(f" {cue} " in lowered for cue in _CORRECTION_CUES):
        return False
    kept = sum(1 for w in set(said) if w in set(wrote))
    return kept / len(set(said)) < 0.35


def plain(source: str) -> str:
    """The dictation itself, lightly tidied: used instead of a rejected reply."""
    text = re.sub(r"\s+", " ", source).strip()
    return text[:1].upper() + text[1:] if text else text
