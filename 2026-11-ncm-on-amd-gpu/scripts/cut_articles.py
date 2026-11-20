"""Question C: do articles that fill NCM's 512-token window get lower ROUGE than shorter ones?

Each stored XSum test entry is 512 article positions (short articles padded with
special ids) followed by 48 positions holding the reference summary; NCM reads only
the first 512 (src/scripts/eval_shard.py, prefix()). The authors cut long articles
before storing them, so an article's original length is gone: what's left is how many
real tokens it has in those 512 positions. An article that fills all 512 was at least
that long and most likely cut. This scores each summary of a full evaluation run
against its reference (same scorer and settings as NCM's eval_merge.py) and compares
articles that fill the window with shorter ones, overall and by length band.

Correlation only: long articles may also be harder to summarize, cut or not.

Run from the experiment folder on the machine that has the data, in the venv with rouge_score:
  python scripts/cut_articles.py
"""
import argparse
import math
import pickle
import statistics as st

from rouge_score import rouge_scorer

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--articles", default="src/data/xsum/passages.pickle", help="NCM's XSum test articles")
ap.add_argument("--shard", default="results/evo-x2-xsum-fp32/eval/xsum/shard_0.pkl", help="a full evaluation run")
ap.add_argument("--cut", type=int, default=512, help="NCM's prefix length")
a = ap.parse_args()

items = pickle.load(open(a.articles, "rb"))  # the authors' published test set, as used by the evaluation runs
shard = pickle.load(open(a.shard, "rb"))  # written by our own evaluation run
if isinstance(items[0], str):
    raise SystemExit("expected token-id lists in the articles file")
lengths = [sum(int(t) < 50256 for t in x[:a.cut]) for x in items]  # real article tokens in the window NCM reads

sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
rl = {i: 100 * sc.score(r, h)["rougeL"].fmeasure for i, h, r in zip(shard["idxs"], shard["hyps"], shard["refs"])}


def row(label, idx):
    v = [rl[i] for i in idx]
    if len(v) < 2:
        return f"{label:24s} {len(v):6d}"
    half = 1.96 * st.stdev(v) / math.sqrt(len(v))  # 95% interval of the mean
    return f"{label:24s} {len(v):6d} {100 * len(v) / len(rl):6.1f}% {st.mean(v):7.2f}  ± {half:.2f}"


q = st.quantiles(lengths, n=10)
print(f"{len(items)} test articles; real GPT-2 tokens in the first {a.cut} positions: p10 {q[0]:.0f}, "
      f"p50 {st.median(lengths):.0f}, p90 {q[-1]:.0f}")
print(f"\n{'group':24s} {'n':>6} {'share':>7} {'ROUGE-L':>7}  95% interval")
print(row(f"shorter (< {a.cut})", [i for i in rl if lengths[i] < a.cut]))
print(row(f"fills window (= {a.cut})", [i for i in rl if lengths[i] == a.cut]))
print(row("all", list(rl)))
print()
step = a.cut // 4
for lo in range(0, a.cut, step):
    hi = min(lo + step, a.cut - 1)
    print(row(f"{lo + 1}-{hi} tokens", [i for i in rl if lo + 1 <= lengths[i] <= hi]))
print(row(f"{a.cut} (fills window)", [i for i in rl if lengths[i] == a.cut]))
