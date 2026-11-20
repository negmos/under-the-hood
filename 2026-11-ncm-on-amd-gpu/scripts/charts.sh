#!/usr/bin/env bash
# Rebuild every chart for the post from results/. Never edit charts by hand.
# Needs a venv with matplotlib and rouge_score (the ncm venv on the EVO-X2), and src/data for chart 4.
# Pass --skip-length to build charts 1-3 only (e.g. on a machine without src/data).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$HERE/analysis"

python3 "$HERE/scripts/make_charts.py" "$@"
