#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/opt/hermes-agent/venv/bin/python3}"
HYGIENE_SCRIPT="${HYGIENE_SCRIPT:-/root/.hermes/scripts/memory-hygiene.py}"

output="$($PYTHON_BIN "$HYGIENE_SCRIPT" --dry-run 2>&1)" || {
  printf 'Memory hygiene dry-run failed:\n%s\n' "$output"
  exit 1
}

summary="$(printf '%s\n' "$output" | grep -E 'Hygiene job complete\.' | tail -n 1 || true)"

if printf '%s\n' "$summary" | grep -Eq 'would soft-delete [1-9][0-9]* points|would update [1-9][0-9]* points'; then
  printf 'Memory hygiene audit found proposed changes:\n%s\n\n' "$summary"
  printf '%s\n' "$output" | grep -E '\[DRY-RUN\]|Decision:|Analyzing group|Hygiene job complete' || true
fi
