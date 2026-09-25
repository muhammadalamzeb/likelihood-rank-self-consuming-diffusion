#!/usr/bin/env bash
# Cross-platform (Linux/macOS) wrapper around reproduce_w2.py
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -x .venv/bin/python ]]; then
  PY=.venv/bin/python
else
  PY=python3
fi
exec "$PY" scripts/reproduce_w2.py
