# Long-running conventions for Path to Care

These rules govern *how* you build. The project context (what you build, the cardinal rule, the 24-hour plan, model choices) lives in the project-root `CLAUDE.md` — read both.

## Always start a session here

1. Read `PROGRESS.md`. It is your handoff note from the previous session.
2. `git log --oneline -10` to see what was just committed.
3. Read `test-results.json` to see which features are still false.
4. Read the project-root `CLAUDE.md` 24-hour table to confirm what phase you're in.
5. If something is broken on disk (import errors, missing files), fix that *before* starting the next feature.

## One feature at a time

Work on exactly one feature from `test-results.json` per pass. Finish it (evidence opened, gate flipped to true) before starting another. If a new ask comes mid-flight, log it in `PROGRESS.md` under `## Next`, finish current.

## Proof before passing

A feature flips from `false` to `true` only after you have:

1. Produced the evidence artifact (an eval run, a training log, a smoke test, an HTTP 200 from the deployed Space).
2. Opened it with the Read tool.
3. Confirmed it shows what it should — not just that the file exists.

The `verify-gate` hook will deny writes to `test-results.json` until evidence has been opened this turn. Do not work around it.

## Evidence kinds for this project

- `evidence/*.txt` — smoke test transcripts, package versions, distribution checks, deployment URL+status, manual sanity outputs.
- `results/*.json` — eval harness output (`baseline_metrics.json`, `tuned_metrics.json`, per-case scores).
- `logs/*.log` — LoRA training loss curves, server stdout, long-running command output.
- `adapters/*/adapter_config.json` — confirmation that a LoRA run actually produced weights.
- `data/cases.jsonl` — the adversarial test set (line-count and label distribution are part of the proof).

## Plausibility is not correctness

A diff that looks reasonable + a metrics file that loaded without error is *not* sufficient. The metrics must be **inspected** — does `tuned_metrics.json` actually show a delta over `baseline_metrics.json`? Does the training log's final loss look sane? Did the deployed Space return a meaningful response, not a 500?

If you find yourself assuming something probably works, stop and look for proof.

## Keep PROGRESS.md current

After each completed feature, update `PROGRESS.md`: tick what's done, note what you learned (especially failure modes), state what's next. Future sessions read this cold.

## Commit often

The `Stop` hook commits tracked changes at session end, but `git add` new source files yourself and commit at meaningful checkpoints with descriptive messages. Sleeping or context-resetting on uncommitted code is how 24-hour builds become 0-hour builds.

## Phase boundaries: run the evaluator

After the last feature in a phase flips true, run the evaluator subagent (`.claude/agents/evaluator.md`) on the phase's diff and evidence. It returns `PASS` or `NEEDS_WORK`. `NEEDS_WORK` findings become the first todos of the next phase. Do not skip this — the evaluator is the answer to "did I just lie to myself."

## If you're told to stop

`OPERATOR STEERING:` messages come via the steer hook. Treat them as higher priority than your current plan. `AGENT_STOP` halts everything until removed.

## Don't add what wasn't asked

The 24-hour plan in project-root `CLAUDE.md` is the scope. Anything not in the table is out unless explicitly added. Especially: do not add new MCP servers, new fine-tunes, new datasets, new languages, new auth flows. The clock is the constraint.
