"""The correction checker used by eval/run_eval.py."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from run_eval import check  # noqa: E402

CASES = {c["id"]: c for c in json.loads((Path(__file__).resolve().parents[1] / "eval" / "corrections.json").read_text())}


def test_a_correct_correction_passes():
    assert check(CASES["en-or-let-it-be"], "Let's connect at 8.") == []


def test_leftover_earlier_choice_fails():
    problems = check(CASES["en-or-let-it-be"], "Let's connect at 6 or 7, or let it be, 8.")
    assert any("'6'" in p for p in problems) and any("let it be" in p for p in problems)


def test_must_keep_cases_catch_over_correction():
    assert check(CASES["keep-two-choices"], "Let's meet at 6 or 7, whichever works for you.") == []
    assert check(CASES["keep-two-choices"], "Let's meet at 7.") != []


def test_every_case_is_well_formed():
    for case in CASES.values():
        assert case["text"] and isinstance(case["must_have"], list) and isinstance(case["must_not"], list)
