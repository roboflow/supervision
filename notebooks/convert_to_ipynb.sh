#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for py in "$SCRIPT_DIR"/*.py; do
    echo "Converting: $(basename "$py")"
    jupytext --to ipynb "$py"
done
