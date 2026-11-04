#!/usr/bin/env bash
# Usage: ./run_all.sh [--device cuda|cpu|mps] [--why]
set -e
cd "$(dirname "$0")"
PY=${PY:-python}
for f in repro_split_heads.py repro_dim_equality.py repro_xsa.py repro_paper_patterns.py; do
  echo "=================== $f"
  $PY "$f" "$@" 2>/dev/null
done
