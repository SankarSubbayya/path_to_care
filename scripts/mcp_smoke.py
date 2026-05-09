"""Exercise each MCP module on one canonical case to confirm they respond.

Writes evidence/mcp_smoke.txt. Runs against the first case in cases.jsonl,
exercising image_classifier, soap_extractor, village_context, and
triage_reasoner end-to-end (in process — they share loaded models).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

EVIDENCE = "evidence/mcp_smoke.txt"


def main() -> int:
    Path(os.path.dirname(EVIDENCE)).mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    try:
        from mcp.image_classifier.server import classify
        from mcp.soap_extractor.server import extract
        from mcp.village_context.server import get_context, practical_urgency_blurb
        from mcp.triage_reasoner.server import triage

        cases = [json.loads(line) for line in open("data/cases.jsonl") if line.strip()]
        case = cases[0]
        lines.append(f"case_id: {case['case_id']}")
        lines.append(f"truth:   {case['ground_truth_urgency']}")
        lines.append("")

        t0 = time.time()
        img = classify(case["image_description"], case_id=case["case_id"])
        lines.append(f"-- image_classifier ({time.time()-t0:.1f}s) --")
        lines.append(f"  parse_ok: {img.parse_ok}")
        lines.append(f"  top3:     {img.top3}")
        lines.append("")

        t0 = time.time()
        soap = extract(case["narrative"], case_id=case["case_id"])
        lines.append(f"-- soap_extractor ({time.time()-t0:.1f}s) --")
        lines.append(f"  parse_ok: {soap.parse_ok}")
        lines.append(f"  fields:   {json.dumps(soap.fields, indent=2, ensure_ascii=False)[:600]}")
        lines.append("")

        t0 = time.time()
        ctx = get_context(case)
        blurb = practical_urgency_blurb(case)
        lines.append(f"-- village_context ({time.time()-t0:.2f}s) --")
        lines.append(f"  context_keys: {sorted(ctx.keys())}")
        lines.append(f"  blurb:        {blurb}")
        lines.append("")

        t0 = time.time()
        tr = triage(
            image_top3=img.top3,
            soap_fields=soap.fields,
            village_context=blurb,
            case_id=case["case_id"],
        )
        lines.append(f"-- triage_reasoner ({time.time()-t0:.1f}s) --")
        lines.append(f"  parse_ok:        {tr.parse_ok}")
        lines.append(f"  urgency:         {tr.urgency}")
        lines.append(f"  reasoning:       {tr.reasoning[:400]!r}")
        lines.append(f"  red_flags_noted: {tr.red_flags_noted}")
        lines.append(f"  patient_framing: {tr.patient_framing[:300]!r}")
        lines.append("")

        all_parse_ok = img.parse_ok and soap.parse_ok and tr.parse_ok
        verdict = "PASS" if all_parse_ok else "PARTIAL"
        lines.insert(0, f"verdict: {verdict}")
    except Exception as e:
        import traceback
        lines.insert(0, "verdict: FAIL")
        lines.append(f"exception: {type(e).__name__}: {e}")
        lines.append(traceback.format_exc())

    with open(EVIDENCE, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if lines[0].endswith("PASS") else (0 if "PARTIAL" in lines[0] else 1)


if __name__ == "__main__":
    sys.exit(main())
