"""Turn spoken point markers into a numbered list."""

from __future__ import annotations

import re

_POINT = re.compile(
    r"(?i)\b(?:"
    r"point\s+(\d+)"
    r"|pehla|pehli|doosra|doosri|teesra|teesri|chautha|chauthi|paanchva|paanchvi"
    r"|पहला|पहली|दूसरा|दूसरी|तीसरा|तीसरी|चौथा|चौथी"
    r")\b"
)


def shape_points(text: str) -> str:
    matches = list(_POINT.finditer(text))
    if not matches:
        return text
    intro = text[: matches[0].start()].strip(" \n.;")
    lines: list[str] = []
    if intro:
        lines.append(intro if intro.endswith((".", "!", "?")) else intro + ".")
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end() : end].strip(" \n.;")
        if not body:
            continue
        number = match.group(1) or str(index + 1)
        if not body.endswith((".", "!", "?")):
            body += "."
        lines.append(f"{number}. {body}")
    return "\n".join(lines) if lines else text
