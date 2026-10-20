"""Write Hindi in Latin letters without changing the words."""

from __future__ import annotations

import re

_INDEPENDENT = {
    "अ": "a",
    "आ": "a",
    "इ": "i",
    "ई": "i",
    "उ": "u",
    "ऊ": "u",
    "ए": "e",
    "ऐ": "ai",
    "ओ": "o",
    "औ": "au",
    "ऋ": "ri",
    "ॐ": "om",
}
_CONSONANTS = {
    "क": "k",
    "ख": "kh",
    "ग": "g",
    "घ": "gh",
    "ङ": "ng",
    "च": "ch",
    "छ": "chh",
    "ज": "j",
    "झ": "jh",
    "ञ": "ny",
    "ट": "t",
    "ठ": "th",
    "ड": "d",
    "ढ": "dh",
    "ण": "n",
    "त": "t",
    "थ": "th",
    "द": "d",
    "ध": "dh",
    "न": "n",
    "प": "p",
    "फ": "ph",
    "ब": "b",
    "भ": "bh",
    "म": "m",
    "य": "y",
    "र": "r",
    "ल": "l",
    "व": "v",
    "श": "sh",
    "ष": "sh",
    "स": "s",
    "ह": "h",
    "ळ": "l",
}
_NUKTA = {
    "क": "q",
    "ख": "kh",
    "ग": "gh",
    "ज": "z",
    "ड": "d",
    "ढ": "dh",
    "फ": "f",
}
_MATRAS = {
    "ा": "a",
    "ि": "i",
    "ी": "i",
    "ु": "u",
    "ू": "u",
    "े": "e",
    "ै": "ai",
    "ो": "o",
    "ौ": "au",
    "ृ": "ri",
    "ॅ": "e",
    "ॉ": "o",
}
_HALANT = "्"
_NUKTA_MARK = "़"
_NASAL = {"ं", "ँ"}
_DIGITS = {chr(0x0966 + i): str(i) for i in range(10)}


def romanize(text: str) -> str:
    """Turn Devanagari into Latin letters. English letters stay as they are."""
    chars = list(text)
    out: list[str] = []
    index = 0
    while index < len(chars):
        char = chars[index]
        if char in _INDEPENDENT:
            out.append(_INDEPENDENT[char])
            index += 1
            _take_nasal(chars, index, out)
            if index < len(chars) and chars[index] in _NASAL:
                index += 1
            continue
        if char in _CONSONANTS:
            base = _CONSONANTS[char]
            index += 1
            if index < len(chars) and chars[index] == _NUKTA_MARK:
                base = _NUKTA.get(char, base)
                index += 1
            if index < len(chars) and chars[index] == _HALANT:
                out.append(base)
                index += 1
                continue
            if index < len(chars) and chars[index] in _MATRAS:
                out.append(base + _MATRAS[chars[index]])
                index += 1
            elif _drop_inherent_a(chars, index):
                out.append(base)
            else:
                out.append(base + "a")
            if index < len(chars) and chars[index] in _NASAL:
                out.append("n")
                index += 1
            continue
        if char in _NASAL:
            out.append("n")
            index += 1
            continue
        if char in _DIGITS:
            out.append(_DIGITS[char])
            index += 1
            continue
        if char in {"।", "॥"}:
            out.append(".")
            index += 1
            continue
        if "\u0900" <= char <= "\u097f":
            index += 1
            continue
        out.append(char)
        index += 1
    return re.sub(r" {2,}", " ", "".join(out)).strip()


def _take_nasal(chars: list[str], index: int, out: list[str]) -> None:
    return


def _drop_inherent_a(chars: list[str], index: int) -> bool:
    """Drop the extra a at the end of a word, and the a in milna-style words."""
    while index < len(chars) and chars[index] in _NASAL:
        index += 1
    if index >= len(chars) or chars[index].isspace() or chars[index] in ".,!?;:":
        return True
    if chars[index] not in _CONSONANTS:
        return False
    look = index + 1
    if look < len(chars) and chars[look] == _NUKTA_MARK:
        look += 1
    if look < len(chars) and chars[look] == _HALANT:
        return False
    return look < len(chars) and chars[look] in _MATRAS
