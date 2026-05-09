"""End-to-end orchestrator smoke on the first case. Writes
evidence/orchestrator_smoke.txt with the full CaseTrace."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

EVIDENCE = "evidence/orchestrator_smoke.txt"


def main() -> int:
    Path(os.path.dirname(EVIDENCE)).mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    try:
        from orchestrator.agent import run_case
        cases = [json.loads(line) for line in open("data/cases.jsonl") if line.strip()]
        case = cases[0]
        t0 = time.time()
        trace = run_case(case)
        elapsed = time.time() - t0

        verdict = "PASS" if trace.urgency in ("red", "yellow", "green") else "FAIL"
        lines = [
            f"verdict: {verdict}",
            f"case_id: {trace.case_id}",
            f"truth:   {case['ground_truth_urgency']}",
            f"pred:    {trace.urgency}",
            f"elapsed_seconds: {elapsed:.1f}",
            f"safety_escalation: {trace.safety_escalation}",
            "",
            f"image_top3: {trace.image_top3}",
            f"image_parse_ok: {trace.image_parse_ok}",
            "",
            f"soap_parse_ok: {trace.soap_parse_ok}",
            f"soap_fields:   {json.dumps(trace.soap_fields, indent=2, ensure_ascii=False)[:500]}",
            "",
            f"village_blurb: {trace.village_blurb}",
            "",
            f"reasoning:       {trace.reasoning[:500]!r}",
            f"red_flags_noted: {trace.red_flags_noted}",
            f"patient_framing: {trace.patient_framing[:300]!r}",
        ]
    except Exception as e:
        import traceback
        lines = ["verdict: FAIL", f"exception: {type(e).__name__}: {e}", traceback.format_exc()]

    with open(EVIDENCE, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if lines[0].endswith("PASS") else 1


if __name__ == "__main__":
    sys.exit(main())
