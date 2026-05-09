#!/usr/bin/env bash
# Denies any Write/Edit to test-results.json unless the agent has opened at
# least one evidence file (eval JSON, training log, smoke output, etc.) since
# the gate last fired. This forces "claim of done" to follow "look at proof."
#
# Teaching example, not a security boundary: only hooks Write/Edit (Bash sed/jq
# can rewrite the file unchecked); evidence reads unlock any result row, not
# the corresponding one.
log="${VERIFY_READ_LOG:-./.claude/.evidence-reads}"
results="${RESULTS_FILE:-test-results.json}"

input=$(cat)
target=$(printf '%s' "$input" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("tool_input",{}).get("file_path",""))' 2>/dev/null)

# Only guard the results file (anchor on path separator so e.g. tuned-results.json doesn't match)
case "$target" in "$results"|*/"$results") ;; *) exit 0 ;; esac

if [ ! -s "$log" ]; then
  cat <<'JSON'
{"decision":"block","reason":"Cannot modify test-results.json: no Path-to-Care evidence file has been Read this turn. Open the relevant artifact first (evidence/*.txt, results/*.json, logs/*.log, adapters/*/adapter_config.json, or data/cases.jsonl), confirm it shows what it should, then retry."}
JSON
  exit 0
fi
# consume the evidence so the next change needs fresh proof
: > "$log"
