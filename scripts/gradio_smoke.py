"""Phase 7 smoke for the Gradio frontend: import the app, assert the
`demo` object is a valid gradio.Blocks, and write evidence/gradio_smoke.txt.

Doesn't actually launch the server — that needs the GPU which we want free
for the eval / training jobs. The launch path is exercised end-to-end during
the HF Space deploy."""
from __future__ import annotations

import os
import sys
from pathlib import Path

EVIDENCE = "evidence/gradio_smoke.txt"


def main() -> int:
    Path(os.path.dirname(EVIDENCE)).mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    try:
        import gradio as gr
        lines.append(f"gradio_version: {gr.__version__}")
        # Import the app. Will fail loudly if any orchestrator/MCP module
        # has a syntax error or import error.
        from frontend.app import demo, DISCLAIMER_PATIENT, run, CANNED_RAJAN_NARRATIVE
        assert isinstance(demo, gr.Blocks), f"demo is {type(demo)}, expected gradio.Blocks"
        assert "decision-support" in DISCLAIMER_PATIENT.lower(), "disclaimer missing key phrase"
        assert callable(run), "run() must be callable"
        assert "rusty nail" in CANNED_RAJAN_NARRATIVE, "canned dialogue should reference the Rajan case"
        lines.insert(0, "verdict: PASS")
        lines.append("demo object: gradio.Blocks ✓")
        lines.append("disclaimer present ✓")
        lines.append("run() is callable ✓")
        lines.append("Rajan dialogue seeded ✓")
        lines.append(f"app_module: frontend.app")
    except Exception as e:
        import traceback
        lines = ["verdict: FAIL", f"exception: {type(e).__name__}: {e}", traceback.format_exc()]
    with open(EVIDENCE, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if lines[0].endswith("PASS") else 1


if __name__ == "__main__":
    sys.exit(main())
