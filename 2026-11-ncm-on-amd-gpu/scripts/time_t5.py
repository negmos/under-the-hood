"""Time an autoregressive summarizer (T5) one request at a time, for question A.

Same protocol as NCM's timing mode (env/timing.patch): the same test articles
(articles.txt from an NCM timing run), batch size 1, warm-up by time, BF16
mixed precision, torch.cuda.synchronize() around the timed GPU work.
Timed span: input already on the GPU -> summary text, as for NCM.

Two output modes:
  stop:  greedy, stops at the end-of-sequence token (up to --max-new-tokens)
  fixed: greedy, always exactly --fixed-tokens new tokens (default 48)

Run from NCM's src/ folder (the data paths are NCM's), normally through run.sh:
  bash scripts/run.sh evo-x2-t5-stop-1 "note" --t5 --bf16 --mode stop \
      --articles ../results/evo-x2-ncm-time-1/articles.txt
"""
import argparse
import csv
import json
import os
import pickle
import time

import torch
from omegaconf import OmegaConf
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--t5", action="store_true", help="selects this script in run.sh; ignored here")
ap.add_argument("--model", default="pki/t5-small-finetuned_xsum", help="Hugging Face model id")
ap.add_argument("--revision", default=None, help="model revision (commit) to pin; the one used is recorded either way")
ap.add_argument("--config", default="config/eval/xsum.yaml", help="NCM eval config, for the data file names")
ap.add_argument("--data-dir", default="data/xsum")
ap.add_argument("--articles", required=True, help="articles.txt from an NCM timing run (one test index per line)")
ap.add_argument("--mode", choices=["stop", "fixed"], required=True)
ap.add_argument("--max-new-tokens", type=int, default=128, help="stop mode: upper limit, well above XSum summary lengths")
ap.add_argument("--fixed-tokens", type=int, default=48, help="fixed mode: exact number of new tokens")
ap.add_argument("--input-len", type=int, default=512, help="article cut or padded to this many T5 tokens, like NCM's 512")
ap.add_argument("--prefix", default="summarize: ", help="task prefix T5 summarizers are usually trained with")
ap.add_argument("--bf16", action="store_true", help="generate under BF16 mixed precision (torch.autocast)")
ap.add_argument("--warmup-s", type=float, default=60.0, help="seconds of warm-up requests before measuring (discarded)")
ap.add_argument("--out", required=True, help="CSV path; articles.txt and timing-meta.json go next to it")
a = ap.parse_args()

cfg = OmegaConf.load(a.config)
items = pickle.load(open(os.path.join(a.data_dir, cfg.data.test), "rb"))
refs = pickle.load(open(os.path.join(a.data_dir, cfg.data.test_refs), "rb"))
arts = [int(line) for line in open(a.articles) if line.strip()]

# NCM stores XSum articles and references as GPT-2 token ids: 50256 is <|endoftext|>, 50257 the added <|pad|>.
# Each stored test entry is 512 article positions (short articles padded) followed by 48 positions that hold
# the reference summary. NCM's prefix() reads only the first prefix_len (512) positions, so T5 must too:
# decoding the whole entry would hand T5 the reference summary right after short articles.
gpt2 = AutoTokenizer.from_pretrained("gpt2")
EOT = 50256
PREFIX_LEN = int(cfg.model.prefix_len)


def article_text(x):
    """Article as text: exactly the positions NCM reads (src/scripts/eval_shard.py, prefix(): the first 512)."""
    if isinstance(x, str):
        return x
    return gpt2.decode([int(t) for t in x[:PREFIX_LEN] if int(t) < EOT])


def ref_text(x):
    """Reference as text, as NCM's dec(): cut at the first end-of-text, drop special ids."""
    if isinstance(x, str):
        return x
    ids = [int(t) for t in x]
    if EOT in ids:
        ids = ids[:ids.index(EOT)]
    return gpt2.decode([t for t in ids if t < EOT])

print(f"  first article, decoded: {article_text(items[arts[0]])[:160]!r}", flush=True)
print(f"  its reference:          {ref_text(refs[arts[0]])[:160]!r}", flush=True)

tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
model = AutoModelForSeq2SeqLM.from_pretrained(a.model, revision=a.revision, use_safetensors=True).cuda().eval()

if a.mode == "stop":
    gen_kw = dict(max_new_tokens=a.max_new_tokens)
else:
    gen_kw = dict(max_new_tokens=a.fixed_tokens, min_new_tokens=a.fixed_tokens)
gen_kw.update(do_sample=False, num_beams=1)


def tick():
    torch.cuda.synchronize()
    return time.perf_counter()


def one(i):
    enc = tok(a.prefix + article_text(items[i]), max_length=a.input_len, truncation=True, padding="max_length", return_tensors="pt")
    ids, mask = enc["input_ids"].cuda(), enc["attention_mask"].cuda()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=a.bf16):
        t0 = tick()
        out = model.generate(input_ids=ids, attention_mask=mask, **gen_kw)
        t1 = tick()
    seq = out[0].tolist()[1:]  # drop the decoder start token
    text = tok.decode(seq, skip_special_tokens=True)
    t2 = time.perf_counter()
    ms = lambda s, e: round((e - s) * 1e3, 3)
    return {
        "article": i, "ts": round(time.time(), 3),
        "total_ms": ms(t0, t2), "generate_ms": ms(t0, t1), "dec_ms": ms(t1, t2),
        "out_tokens": len(seq), "stopped_eos": int(bool(seq) and seq[-1] == tok.eos_token_id),
        "mem_alloc_mb": round(torch.cuda.memory_allocated() / 2**20, 1),
        "mem_reserved_mb": round(torch.cuda.memory_reserved() / 2**20, 1),
        "summary": text,
    }


out_dir = os.path.dirname(os.path.abspath(a.out))
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "articles.txt"), "w") as f:
    f.write("\n".join(map(str, arts)) + "\n")

w0, nw = time.perf_counter(), 0
while time.perf_counter() - w0 < a.warmup_s:
    one(arts[nw % len(arts)])
    nw += 1
warm_s = time.perf_counter() - w0
print(f"  warm-up: {nw} requests in {warm_s:.1f} s (discarded)", flush=True)

fields = ["req", "article", "ts", "total_ms", "generate_ms", "dec_ms", "out_tokens", "stopped_eos",
          "mem_alloc_mb", "mem_reserved_mb", "summary"]
hyps = []
with open(a.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for k, i in enumerate(arts):
        row = one(i)
        hyps.append(row["summary"])
        w.writerow({"req": k, **row})
        if k % 100 == 0:
            print(f"  timed {k}/{len(arts)}", flush=True)

from rouge_score import rouge_scorer  # same scorer and settings as NCM's eval_merge.py

sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
R = {k: [] for k in ("rouge1", "rouge2", "rougeL")}
for h, i in zip(hyps, arts):
    v = sc.score(ref_text(refs[i]), h)
    for k in R:
        R[k].append(v[k].fmeasure)
rouge = {k: round(100 * sum(v) / len(v), 2) for k, v in R.items()}
print(f"ROUGE-1/2/L on the {len(arts)} timed articles: {rouge['rouge1']} / {rouge['rouge2']} / {rouge['rougeL']}", flush=True)

with open(os.path.join(out_dir, "timing-meta.json"), "w") as f:
    json.dump({"model": a.model, "model_commit": getattr(model.config, "_commit_hash", None),
               "mode": a.mode, "generate_kwargs": gen_kw, "generation_config": model.generation_config.to_dict(),
               "prefix": a.prefix, "input_len": a.input_len, "articles": len(arts), "articles_from": a.articles,
               "batch_size": 1, "bf16": a.bf16, "warmup_s": round(warm_s, 1), "warmup_requests": nw,
               "rouge": rouge}, f, indent=2, default=str)
print(f"timed {len(arts)} requests -> {a.out}", flush=True)
