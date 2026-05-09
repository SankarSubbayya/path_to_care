"""Unit tests for harness/reward.py. Output goes to evidence/reward_unit_tests.txt
when run via `python -m harness.test_reward`."""
from __future__ import annotations

import io
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

from harness.reward import reward, is_false_negative_red_to_green

EVIDENCE = "evidence/reward_unit_tests.txt"


def assert_eq(label: str, got, want) -> bool:
    ok = got == want
    print(f"  {'OK' if ok else 'FAIL'}  {label}: got={got!r} want={want!r}")
    return ok


def run() -> int:
    print("== reward exact / adjacent / off-by-2 ==")
    results = [
        assert_eq("R(red, red)", reward("red", "red"), 1.0),
        assert_eq("R(yellow, yellow)", reward("yellow", "yellow"), 1.0),
        assert_eq("R(green, green)", reward("green", "green"), 1.0),
        assert_eq("R(yellow, red) adjacent", reward("yellow", "red"), 0.5),
        assert_eq("R(red, yellow) adjacent", reward("red", "yellow"), 0.5),
        assert_eq("R(green, yellow) adjacent", reward("green", "yellow"), 0.5),
        assert_eq("R(yellow, green) adjacent", reward("yellow", "green"), 0.5),
        assert_eq("R(green, red) off-by-2", reward("green", "red"), 0.0),
        assert_eq("R(red, green) off-by-2", reward("red", "green"), 0.0),
    ]
    print("== case insensitivity / whitespace ==")
    results += [
        assert_eq("R('Red', 'RED')", reward("Red", "RED"), 1.0),
        assert_eq("R(' yellow ', 'red')", reward(" yellow ", "red"), 0.5),
    ]
    print("== false-negative Red->Green flag ==")
    results += [
        assert_eq("FN(green, red) is True", is_false_negative_red_to_green("green", "red"), True),
        assert_eq("FN(yellow, red) is False", is_false_negative_red_to_green("yellow", "red"), False),
        assert_eq("FN(green, green) is False", is_false_negative_red_to_green("green", "green"), False),
    ]
    print()
    failed = sum(1 for r in results if not r)
    verdict = "PASS" if failed == 0 else "FAIL"
    print(f"verdict: {verdict} ({len(results) - failed}/{len(results)} tests passing)")
    return 0 if failed == 0 else 1


def main() -> int:
    Path(os.path.dirname(EVIDENCE)).mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = run()
    out = buf.getvalue()
    with open(EVIDENCE, "w") as f:
        f.write(out)
    print(out, end="")
    return rc


if __name__ == "__main__":
    sys.exit(main())
