#!/usr/bin/env bash
# Create the .venv and install dependencies. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"

if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.11 .venv
    uv pip install --python .venv -r requirements.txt
else
    python3 -m venv .venv
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/pip install -r requirements.txt
fi

echo
echo "Done. Activate with:  source .venv/bin/activate"
