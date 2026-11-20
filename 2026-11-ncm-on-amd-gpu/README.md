# Taking NCM for a spin

Code, results and charts behind the field test **[Is AI latency really fixed? Taking NCM for a spin](https://blog.negmbots.com/field-tests/2026-11-ncm-on-amd-gpu/)** on *Load Bearing*.

The question: can a model give you a cost per request you can forecast and plan for, and where does that predictability actually come from? I ran the released code and model of [Next-Concept Modeling](https://arxiv.org/abs/2610.14162) (NCM, Stanford, 2026) for news summarization (XSum) on an AMD Radeon 8060S iGPU, one request at a time, and compared it with a small public summarizer that writes word by word ([T5-small fine-tuned on XSum](https://huggingface.co/pki/t5-small-finetuned_xsum)).

## What's here

| Folder | Contents |
|---|---|
| `env/` | three small patches to the NCM code (`bf16.patch`, `timing.patch`, `compile.patch`), and the exact Python packages (`requirements.lock`) |
| `scripts/` | `run.sh` (every run goes through it; it records versions and samples the hardware every second), the T5 timing script, the ROUGE and article-length analyses, and the chart scripts |
| `results/` | one folder per run: `env.json` (versions), `hw.csv` (clocks, power, temperatures), and the measurements (`timing.csv`, `eval.txt`) |
| `analysis/` | the charts in the post, light and dark, made by `scripts/charts.sh` from `results/` |
| `../tools/` | the two shared tools `run.sh` calls: `collect_env.py` and `sample_hw.py` |

Not included: the NCM code itself (clone it, below), the model checkpoints and the XSum test set (download them, below), and evaluation outputs that contain XSum's reference summaries.

## How to rerun it

On a Linux machine with a ROCm-capable AMD GPU. From this folder:

```bash
# 1. NCM's code at the commit I used, with my patches
git clone https://github.com/HazyResearch/ncm src
git -C src checkout c70274273f489abc73832b2b98584a8483b6b203
# bf16.patch: adds --bf16 (the released code runs in float32)
# timing.patch: adds --time (per-request timing)
# compile.patch: adds --compile (torch.compile)
git -C src apply ../env/bf16.patch ../env/timing.patch ../env/compile.patch

# 2. A Python environment (run.sh expects it at ~/venvs/ncm)
uv venv ~/venvs/ncm && source ~/venvs/ncm/bin/activate
uv pip install -r env/requirements.lock --extra-index-url https://download.pytorch.org/whl/rocm10.0

# 3. The XSum checkpoint and test set download
cd src
hf download hazyresearch/ncm-checkpoints --include "xsum/*" --local-dir artifacts
hf download hazyresearch/ncm-datasets --repo-type dataset --include "xsum/*" --local-dir data
cd ..

# 4. Runs (each writes results/<name>/)
bash scripts/run.sh my-xsum-fp32 "quality, float32"                       # ~40 min on my machine: ROUGE on all 11,334 articles
bash scripts/run.sh my-ncm-time "NCM latency" --bf16 --time               # ~3 min on my machine: 1,000 requests, batch size 1
bash scripts/run.sh my-t5-stop "T5 latency" --t5 --bf16 --mode stop \
    --articles ../results/evo-x2-ncm-time-1/articles.txt                  # Same 1,000 articles against T5
bash scripts/charts.sh                                                    # rebuild the charts
```

Options: `--time-passes` adds per-round timing inside NCM, `--mode fixed` forces T5 to 48 tokens, `--compile` compiles NCM's transformer blocks.

## Design notes

- **Every timed span is wrapped in `torch.cuda.synchronize()`.** Since GPU work is asynchronous; Python queues the work and move on, this forces the code to wait until the GPU is finished.
- **One request at a time, after 60 s of discarded warm-up.** The first requests are usually paying for loading, GPU's warm-up, and short power boost on the machine. This allows keeping these out of the numbers measured.
- **The same 1,000 articles for every model, 3 runs each, in rotating order,** with each run committed before the next and a pause in between, so no model always runs on a cold or a hot machine cycle.
- **T5's input is padded to 512 tokens like NCM's,** so only the output length varies between the two. While this is not how to properly serve T5; it isolates the question and provide a fair comparison.
- **All comparisons run in BF16 without compilation,** for fairness as compilation adds GPU-specific optimizations that may vary from one model to another. One separate set of runs shows what compiling NCM adds.
- **Both models are scored against the same references with the same scorer.**

## How this was built

Usually I work my way into designing the whole experiment by identifying first what I think the value add we shall confirm/validate/find. The experiment components are then researched (e.g. which models to use, which HW/SW can run it, would the output actually scale on a larger machine set), then I start putting together the steps, expected outcome, some smoke tests, ...etc.
The whole experiment code is then executed and code snippets and output text is enhanced/reworded with AI using Opencode and locally running opensource LLM model.

## License

- MIT for the code (`LICENSE`)
- CC BY 4.0 for results and charts (`LICENSE-DATA`)
