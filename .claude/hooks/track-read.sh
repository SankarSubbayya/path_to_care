#!/usr/bin/env bash
# Records when Claude opens an evidence file. verify-gate.sh consults this
# before allowing test-results.json to flip a feature true.
#
# Path-to-Care evidence kinds:
#   evidence/*.txt          — smoke tests, distribution checks, deployment confirmations
#   results/*.json          — eval harness outputs (baseline_metrics, tuned_metrics, ...)
#   logs/*.log              — training logs, server logs
#   adapters/*/adapter_config.json — proof a LoRA training run produced weights
#   data/cases.jsonl        — adversarial test set
log="${VERIFY_READ_LOG:-./.claude/.evidence-reads}"
path=$(cat | python3 -c 'import json,sys; print(json.load(sys.stdin).get("tool_input",{}).get("file_path",""))' 2>/dev/null)
case "$path" in
  */evidence/*.txt|*/results/*.json|*/logs/*.log|*/adapters/*/adapter_config.json|*/data/cases.jsonl)
    [ -f "$path" ] && echo "$path" >> "$log" ;;
esac
exit 0
