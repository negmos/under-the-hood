"""ROUGE of the summaries in timing CSVs, all against the same references.

Every timing run (NCM through env/timing.patch, T5 through scripts/time_t5.py)
writes its summaries into timing.csv. This scores them against NCM's own decoded
reference summaries, taken from a full evaluation run's shard file, with the same
scorer and settings as NCM's eval_merge.py (rouge_score, stemming, F-measure,
mean x 100). So every model is scored on the same articles and the same text.

Run from the experiment folder, in the venv that has rouge_score:
  python scripts/rouge_timed.py results/evo-x2-a-ncm-1/timing.csv results/evo-x2-a-t5stop-1/timing.csv
"""
import argparse
import csv
import pickle

from rouge_score import rouge_scorer

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("csv", nargs="+", help="timing.csv files")
ap.add_argument("--refs", default="results/evo-x2-xsum-bf16/eval/xsum/shard_0.pkl",
                help="shard file of a full evaluation run: its 'refs' are NCM's decoded references")
a = ap.parse_args()

shard = pickle.load(open(a.refs, "rb"))  # written by our own evaluation run
refs = dict(zip(shard["idxs"], shard["refs"]))
sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

print(f"{'file':55s} {'n':>5} {'R-1':>6} {'R-2':>6} {'R-L':>6} {'words':>6}")
for path in a.csv:
    rows = list(csv.DictReader(open(path)))
    R = {k: [] for k in ("rouge1", "rouge2", "rougeL")}
    for r in rows:
        v = sc.score(refs[int(r["article"])], r["summary"])
        for k in R:
            R[k].append(v[k].fmeasure)
    n = len(rows)
    words = sum(len(r["summary"].split()) for r in rows) / n
    print(f"{path:55s} {n:5d} {100 * sum(R['rouge1']) / n:6.2f} {100 * sum(R['rouge2']) / n:6.2f} "
          f"{100 * sum(R['rougeL']) / n:6.2f} {words:6.1f}")
