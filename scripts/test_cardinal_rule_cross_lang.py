"""Cross-language equivalence test: assert that core/cardinal_rule.enforce
(Python) and frontend-next/src/lib/cardinal-rule.ts (TypeScript) produce
character-for-character identical output on a shared set of inputs.

Why this matters: the regex rewriter is a safety component. The Python
implementation gates eval-time orchestrator output (run_case in
orchestrator/agent.py), and the TS implementation gates Next.js's
/api/triage HTTP responses. If they drift, the patient sees one thing in
the Gradio Space and another in the Next.js Space — and worse, evidence
of cardinal-rule violations only shows up in one of the two paths.

How:
  1. Build a list of probe strings (the same ones the unit tests use).
  2. Call core.cardinal_rule.enforce on each.
  3. Spawn `node` to run a tiny TS-eval script that calls the same
     enforceCardinalRule from src/lib/cardinal-rule.ts and prints the
     results as JSON.
  4. Diff the two outputs character-for-character.
  5. Exit non-zero on any mismatch.

Usage:
  PYTHONPATH=. .venv/bin/python scripts/test_cardinal_rule_cross_lang.py

Evidence:
  evidence/cardinal_rule_cross_lang.txt — verdict + per-case pass/fail
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

# Match the TS unit-test cases (frontend-next/src/lib/__tests__/cardinal-rule.test.ts)
PROBES = [
    "you have cellulitis",
    "You have a fever",
    "YOU   HAVE  high fever",
    "the diagnosis is sepsis",
    "Diagnosis is contact dermatitis",
    "diagnosed with impetigo",
    "I confirm the rash",
    "This is definitely an infection",
    "This is clearly cellulitis",
    "This is definitely cellulitis",
    "certainly an infection",
    # Clean inputs (must round-trip unchanged)
    "Signs suggest cellulitis. The patient should be evaluated within 24 hours.",
    "Apply over-the-counter cream and watch for spreading.",
    # The real Y09 string from logs/cardinal_rule_rewrites.log
    ("These spots are spreading and you have a fever, so it is important "
     "to see a nurse in the next day or two."),
    # Idempotence probe (already-rewritten output)
    "signs suggest cellulitis",
    # Edge cases
    "",
    "  ",
    "you  have   triple   space",   # internal whitespace handling
    "Diagnosis is X.\nYou have Y.",  # multiline with two patterns
]

EVIDENCE_PATH = "evidence/cardinal_rule_cross_lang.txt"


def run_python(probes: list[str]) -> list[str]:
    from core.cardinal_rule import enforce
    return [enforce(p) for p in probes]


def run_typescript(probes: list[str]) -> list[str]:
    """Spawn a tiny inline TS evaluator. We avoid ts-node by transpiling on
    the fly via tsx (already in node_modules from create-next-app), with a
    fallback to building a temporary .mjs that imports the compiled module
    if tsx isn't available."""
    # The TS file uses ESM `import`. We write a small wrapper to a tmp
    # .ts file under frontend-next/, then run it via `npx tsx <file>`.
    fe_root = Path("frontend-next").resolve()
    if not fe_root.exists():
        raise RuntimeError(f"frontend-next dir not found at {fe_root}")

    runner_path = fe_root / "scripts" / "_xlang_eval.ts"
    runner_path.parent.mkdir(exist_ok=True)
    runner_src = (
        "import { enforceCardinalRule } from '../src/lib/cardinal-rule.ts';\n"
        "const probes: string[] = JSON.parse(process.argv[2]);\n"
        "const out = probes.map((p) => enforceCardinalRule(p).text);\n"
        "process.stdout.write(JSON.stringify(out));\n"
    )
    runner_path.write_text(runner_src)

    env = {**os.environ, "PATH": "/usr/bin:/usr/local/bin"}
    env.pop("ELECTRON_RUN_AS_NODE", None)
    cmd = [
        "/usr/bin/npx", "--yes", "--prefix", str(fe_root), "tsx",
        str(runner_path), json.dumps(probes),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env, cwd=str(fe_root))
    if proc.returncode != 0:
        raise RuntimeError(
            f"tsx runner failed (rc={proc.returncode}):\n"
            f"stdout: {proc.stdout[:500]}\n"
            f"stderr: {proc.stderr[:500]}"
        )
    return json.loads(proc.stdout.strip())


def main() -> int:
    Path(os.path.dirname(EVIDENCE_PATH)).mkdir(parents=True, exist_ok=True)
    py = run_python(PROBES)
    ts = run_typescript(PROBES)
    assert len(py) == len(ts) == len(PROBES), "lengths diverged"

    lines: list[str] = []
    mismatches: list[int] = []
    for i, (p, a, b) in enumerate(zip(PROBES, py, ts)):
        ok = a == b
        if not ok:
            mismatches.append(i)
        lines.append(
            f"  [{i:02d}] {'OK' if ok else 'MISMATCH'}  input={p!r}\n"
            f"        py={a!r}\n"
            f"        ts={b!r}"
        )

    verdict = "PASS" if not mismatches else "FAIL"
    summary = (
        f"verdict: {verdict}\n"
        f"probes: {len(PROBES)}\n"
        f"matches: {len(PROBES) - len(mismatches)}\n"
        f"mismatches: {len(mismatches)}\n"
    )
    body = summary + "\n" + "\n".join(lines) + "\n"
    Path(EVIDENCE_PATH).write_text(body)
    print(body)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
