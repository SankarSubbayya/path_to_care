"""Validate the adversarial test set distribution. Writes
evidence/test_set_distribution.txt so the verify-gate has proof."""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

CASES_PATH = "data/cases.jsonl"
EVIDENCE = "evidence/test_set_distribution.txt"


def main() -> int:
    Path(os.path.dirname(EVIDENCE)).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(CASES_PATH):
        with open(EVIDENCE, "w") as f:
            f.write(f"verdict: FAIL\nreason: {CASES_PATH} does not exist\n")
        print(f"FAIL: {CASES_PATH} missing")
        return 1

    cases = [json.loads(line) for line in open(CASES_PATH) if line.strip()]
    by_urgency = Counter(c["ground_truth_urgency"] for c in cases)
    adv_count = sum(1 for c in cases if c.get("adversarial_features"))

    expected = {"red": 10, "yellow": 10, "green": 10}
    ok = (
        len(cases) == 30
        and dict(by_urgency) == expected
        and adv_count >= 5  # at least some perturbations applied
    )

    lines = [
        f"verdict: {'PASS' if ok else 'FAIL'}",
        f"total_cases: {len(cases)}",
        f"by_urgency: {dict(by_urgency)}",
        f"expected_by_urgency: {expected}",
        f"cases_with_adversarial_features: {adv_count}",
        f"unique_case_ids: {len(set(c['case_id'] for c in cases))}",
    ]
    print("\n".join(lines))
    with open(EVIDENCE, "w") as f:
        f.write("\n".join(lines) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
