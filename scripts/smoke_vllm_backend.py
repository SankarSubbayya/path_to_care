"""Smoke test for the vLLM inference backend.

Sets PTC_INFERENCE=vllm BEFORE importing core.llm so the dispatcher routes
to core._llm_vllm. Calls gemma4() then chat_text() against the running
vLLM container (default http://localhost:8000/v1, key ptc-demo-2026-amd).
Writes evidence/vllm_backend_smoke.txt.

Prereq: the vLLM container is running. Start with:
  bash scripts/vllm_serve.sh
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# IMPORTANT: set the env var BEFORE importing core.llm.
os.environ["PTC_INFERENCE"] = "vllm"

EVIDENCE = "evidence/vllm_backend_smoke.txt"


def main() -> int:
    Path(os.path.dirname(EVIDENCE)).mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    try:
        from core import llm
        lines.append(f"INFERENCE_BACKEND: {llm.INFERENCE_BACKEND}")
        assert llm.INFERENCE_BACKEND == "vllm", "expected vllm dispatch"

        handle = llm.gemma4()
        lines.append(f"handle.model:         {handle.model}")
        lines.append(f"handle.is_multimodal: {handle.is_multimodal}")

        prompt = (
            "List up to three plausible skin conditions for a swollen, reddened "
            "ankle with a small puncture wound. Confidences in [0,1]. Do not "
            "diagnose. One per line."
        )
        t0 = time.time()
        out = llm.chat_text(handle, prompt, max_new_tokens=120)
        elapsed = time.time() - t0

        # Cardinal-rule sanity check (no diagnostic phrases).
        diagnostic_phrases = ["you have", "the diagnosis is", "diagnosed with"]
        violations = [p for p in diagnostic_phrases if p in out.lower()]

        lines.insert(0, "verdict: PASS")
        lines.append(f"generate_seconds: {elapsed:.2f}")
        lines.append(f"output_len_chars: {len(out)}")
        lines.append(f"output: {out!r}")
        lines.append(f"diagnostic_phrases_found: {violations}")
        if violations:
            lines[0] = "verdict: WARN_DIAGNOSTIC_LANGUAGE"
    except Exception as e:
        import traceback
        lines = ["verdict: FAIL", f"exception: {type(e).__name__}: {e}",
                 traceback.format_exc()]

    with open(EVIDENCE, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if lines[0].startswith("verdict: PASS") else 1


if __name__ == "__main__":
    sys.exit(main())
