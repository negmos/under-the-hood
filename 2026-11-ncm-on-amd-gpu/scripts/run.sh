#!/usr/bin/env bash
# Run one experiment with hardware sampling and environment capture.
# Usage: bash scripts/run.sh [run-name] [note] [extra options, e.g. --bf16]
#   Quality run (evaluate.py, all test articles): bash scripts/run.sh evo-x2-xsum-bf16 "note" --bf16
#   Timing run (needs env/timing.patch):          bash scripts/run.sh evo-x2-ncm-time-1 "note" --bf16 --time --time-passes
#   T5 timing run (scripts/time_t5.py):           bash scripts/run.sh evo-x2-t5-stop-1 "note" --t5 --bf16 --mode stop \
#                                                     --articles ../results/evo-x2-ncm-time-1/articles.txt
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS="$HERE/../tools"
RUN="${1:-$(hostname)-$(date +%Y%m%d-%H%M%S)}"
NOTE="${2:-}"
OUT="$HERE/results/$RUN"

# env.json records the packages of the Python that runs collect_env.py.
if [[ "${VIRTUAL_ENV:-}" != "$HOME/venvs/ncm" ]]; then
    echo "Activate the NCM venv first: source ~/venvs/ncm/bin/activate" >&2
    exit 1
fi

mkdir -p "$OUT"

python3 "$TOOLS/collect_env.py" --out "$OUT/env.json" --note "$NOTE"

python3 "$TOOLS/sample_hw.py" --out "$OUT/hw.csv" --interval 1 &
SAMPLER=$!
# Wait for the sampler to exit, so hw.csv is complete before run.sh returns (and before any commit).
trap 'kill "$SAMPLER" 2>/dev/null || true; wait "$SAMPLER" 2>/dev/null || true' EXIT

# ---- The experiment: XSum quality evaluation, or timing when --time is given. --
cd "$HERE/src"

# Which NCM code ran: the commit, and any local changes (empty when unpatched).
git rev-parse HEAD > "$OUT/src-commit.txt"
git diff > "$OUT/src.diff"

COMMON=(--config config/eval/xsum.yaml --checkpoint-dir artifacts/xsum --data-dir data/xsum)

if [[ " ${*:3} " == *" --t5 "* ]]; then
    # T5 baseline for question A (scripts/time_t5.py): one article per request, same protocol.
    python3 "$HERE/scripts/time_t5.py" --config config/eval/xsum.yaml --data-dir data/xsum \
        --out "$OUT/timing.csv" "${@:3}" 2>&1 | tee "$OUT/timing.txt"
elif [[ " ${*:3} " == *" --time "* ]]; then
    # Timing (env/timing.patch): one article per request, run by eval_shard.py directly.
    # Writes timing.csv, articles.txt and timing-meta.json into $OUT.
    python3 scripts/eval_shard.py "${COMMON[@]}" --out "$OUT/timing.csv" "${@:3}" 2>&1 | tee "$OUT/timing.txt"
else
    # Quality: all test articles, generated and scored by evaluate.py.
    python3 evaluate.py "${COMMON[@]}" --out "$OUT/eval/xsum" "${@:3}" 2>&1 | tee "$OUT/eval.txt"
fi
# ------------------------------------------------------------------------------

echo "Results in $OUT"
