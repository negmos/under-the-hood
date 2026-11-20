"""Every chart for the post, rebuilt from results/ (chart 4 also reads NCM's test articles).

Called by scripts/charts.sh; never edit the SVGs by hand. Each chart is written twice,
analysis/<name>.svg (light) and analysis/<name>-dark.svg, for the blog's two themes.

  1 latency-cdf          question A: latency per request, NCM vs T5 stop vs T5 fixed (9 runs)
  2 t5-latency-length    question A: T5's latency against its output length, with NCM's median
  3 pass-ratios          rule 2: time per pass relative to the last pass (3 runs)
  4 rouge-by-length      question C: ROUGE-L by real article tokens in the 512 window

Colours: the blog's CSS tokens for surface and text; series NCM / T5 stop / T5 fixed
are blue, orange and aqua, checked for colour-blind safety in both themes (every pair
stays distinguishable; orange and aqua are below 3:1 contrast on the light paper, so
every series is also named by a direct label).

Run from anywhere, in a venv with matplotlib (and rouge_score for chart 4):
  python scripts/make_charts.py                  # all four (chart 4 needs src/data)
  python scripts/make_charts.py --skip-length    # charts 1-3 only
"""
import argparse
import csv
import math
import os
import pickle
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "analysis")

THEMES = {
    "light": dict(surface="#F4F6F8", ink="#16233D", soft="#4A5A75", rule="#D3DCE6",
                  series=["#2C5BD6", "#eb6834", "#1baf7a"]),
    "dark": dict(surface="#0F1E3A", ink="#DCE6F5", soft="#9DB0CF", rule="#26406B",
                 series=["#3987e5", "#d95926", "#199e70"]),
}
NCM, T5STOP, T5FIXED = 0, 1, 2

plt.rcParams.update({
    "font.family": ["DejaVu Sans"],  # ships with matplotlib, so the layout is measured with the font drawn
    "font.size": 11,
    "svg.fonttype": "path",  # text drawn as shapes: every reader sees the same (an <img> can't use page fonts)
})


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def pct(v, q):
    v = sorted(v)
    i = (len(v) - 1) * q
    lo = int(i)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (i - lo)


def figure(t, title, subtitle, size=(8.2, 4.9)):
    fig, ax = plt.subplots(figsize=size)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["soft"])
    ax.tick_params(colors=t["soft"], labelcolor=t["ink"])
    ax.grid(True, color=t["rule"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.text(0, 1.13, title, transform=ax.transAxes, color=t["ink"], fontsize=13.5, fontweight="bold")
    ax.text(0, 1.05, subtitle, transform=ax.transAxes, color=t["soft"], fontsize=10)
    return fig, ax


def axis_labels(ax, t, x, y):
    ax.set_xlabel(x, color=t["ink"])
    ax.set_ylabel(y, color=t["ink"])


def save(fig, name, theme):
    path = os.path.join(OUT, f"{name}{'' if theme == 'light' else '-dark'}.svg")
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("  wrote", os.path.relpath(path, HERE))


def runs(key):
    return [read(os.path.join(RES, f"evo-x2-a-{key}-{r}", "timing.csv")) for r in (1, 2, 3)]


# ---- 1. Latency per request -------------------------------------------------------------
def chart_latency_cdf(theme):
    t = THEMES[theme]
    fig, ax = figure(t, "NCM's latency is flat; T5's depends on how long its answer gets",
                     "Latency per request, 1,000 XSum articles, batch size 1, BF16, Radeon 8060S. "
                     "One line per run, 3 runs each.")
    series = [("ncm", "NCM (always 48 positions)", NCM),
              ("t5stop", "T5, stops on its own", T5STOP),
              ("t5fixed", "T5, forced to 48 tokens", T5FIXED)]
    tail = None
    for key, label, slot in series:
        for k, rows in enumerate(runs(key)):
            v = sorted(float(x["total_ms"]) for x in rows)
            y = [100 * (i + 1) / len(v) for i in range(len(v))]
            ax.step(v, y, where="post", color=t["series"][slot], linewidth=1.7,
                    label=label if k == 0 else None)
            if key == "t5stop" and (tail is None or v[-1] > tail):
                tail = v[-1]
    # direct labels, placed next to each family of curves
    first = {key: sorted(float(x["total_ms"]) for x in runs(key)[0]) for key, _, _ in series}
    ax.annotate("NCM", (pct(first["ncm"], 0.25), 25), xytext=(10, -2), textcoords="offset points",
                color=t["ink"], fontsize=10.5, fontweight="bold")
    ax.annotate("T5, stops on its own", (pct(first["t5stop"], 0.75), 75), xytext=(10, -4),
                textcoords="offset points", color=t["ink"], fontsize=10.5, fontweight="bold")
    ax.annotate("T5, forced to 48 tokens", (pct(first["t5fixed"], 0.5), 50), xytext=(-12, 0),
                textcoords="offset points", ha="right", color=t["ink"], fontsize=10.5, fontweight="bold")
    ax.annotate(f"the longest answers: repetition loops,\none up to the 128-token cap ({tail:.0f} ms)", (tail, 100),
                xytext=(-8, -40), textcoords="offset points", ha="right", color=t["soft"], fontsize=9.5,
                arrowprops=dict(arrowstyle="-", color=t["soft"], linewidth=0.8))
    ax.set_xscale("log")
    ticks = [60, 80, 100, 150, 200, 300, 400]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax.set_xlim(60, 450)
    ax.set_ylim(0, 102)
    axis_labels(ax, t, "Latency per request (ms, log scale)", "Share of requests (%)")
    leg = ax.legend(loc="lower right", frameon=False, fontsize=9.5)
    for txt in leg.get_texts():
        txt.set_color(t["ink"])
    save(fig, "latency-cdf", theme)


# ---- 2. T5's latency against its output length -------------------------------------------
def chart_t5_length(theme):
    t = THEMES[theme]
    rows = runs("t5stop")[0]
    x = [int(r["out_tokens"]) for r in rows]
    y = [float(r["total_ms"]) for r in rows]
    fit_x = [a for a in x if a < 128]
    fit_y = [b for a, b in zip(x, y) if a < 128]
    mx, my = st.mean(fit_x), st.mean(fit_y)
    slope = sum((a - mx) * (b - my) for a, b in zip(fit_x, fit_y)) / sum((a - mx) ** 2 for a in fit_x)
    icpt = my - slope * mx
    ncm = st.median(float(r["total_ms"]) for rr in runs("ncm") for r in rr)
    cross = (ncm - icpt) / slope
    long = [(a, b) for a, b in zip(x, y) if a > 64]

    fig, ax = figure(t, "T5 pays about 3 ms per token; NCM pays the same every time",
                     f"T5 stopping on its own (run 1, 1,000 requests) against NCM's median ({ncm:.0f} ms). "
                     "Batch size 1, BF16.")
    ax.scatter(x, y, s=16, color=t["series"][T5STOP], alpha=0.4, linewidths=0, label="T5, one dot per request")
    xs = [5, 130]
    ax.plot(xs, [icpt + slope * a for a in xs], color=t["series"][T5STOP], linewidth=2,
            label=f"T5 fit: {icpt:.1f} ms + {slope:.2f} ms per token")
    ax.axhline(ncm, color=t["series"][NCM], linewidth=2, label="NCM median, any article")
    ax.axvline(cross, color=t["soft"], linewidth=0.9, linestyle=(0, (2, 3)))
    ax.annotate(f"T5 faster below ~{cross:.0f} tokens,\nslower above", (cross, 20), xytext=(8, 0),
                textcoords="offset points", color=t["ink"], fontsize=10)
    ax.annotate("NCM", (128, ncm), xytext=(0, 6), textcoords="offset points", ha="right", color=t["ink"],
                fontsize=10.5, fontweight="bold")
    if long:
        ax.annotate(f"{len(long)} long answers ({min(a for a, _ in long)}–{max(a for a, _ in long)} tokens): "
                    "repetition\nloops and Welsh-language articles; the longest hit the 128-token cap",
                    (max(a for a, _ in long), max(b for _, b in long)), xytext=(-10, 4), textcoords="offset points",
                    ha="right", va="bottom", color=t["soft"], fontsize=9.5)
    ax.set_xlim(0, 135)
    ax.set_ylim(0, 440)
    axis_labels(ax, t, "Tokens T5 wrote (including the end token)", "Latency per request (ms)")
    leg = ax.legend(loc="lower right", frameon=False, fontsize=9.5)
    for txt in leg.get_texts():
        txt.set_color(t["ink"])
    save(fig, "t5-latency-length", theme)


# ---- 3. Rule 2: time per pass relative to the last pass -----------------------------------
def chart_pass_ratios(theme):
    t = THEMES[theme]
    codes = [4, 6, 10, 14, 20, 26, 32, 40, 48]
    per_run = []
    for r in (1, 2, 3):
        rows = read(os.path.join(RES, f"evo-x2-ncm-time-{r}", "timing.csv"))
        passes = list(zip(*[[float(v) for v in x["pass_ms"].split(";")] for x in rows]))
        med = [st.median(p) for p in passes]
        per_run.append([m / med[-1] for m in med])
    idx = list(range(len(codes)))
    mid = [st.median(c) for c in zip(*per_run)]
    lo = [m - min(c) for m, c in zip(mid, zip(*per_run))]
    hi = [max(c) - m for m, c in zip(mid, zip(*per_run))]

    fig, ax = figure(t, "Every pass costs about the same, small level or big",
                     "NCM on XSum: each pass's median time relative to the last pass, 1,000 requests per run. "
                     "Bars: middle of 3 runs; whiskers: their range.")
    ax.bar(idx, mid, width=0.62, color=t["series"][NCM], zorder=2)
    ax.errorbar(idx, mid, yerr=[lo, hi], fmt="none", ecolor=t["ink"], elinewidth=1, capsize=3, zorder=3)
    ax.plot(idx, [c / codes[-1] for c in codes], color=t["soft"], linewidth=1.4, linestyle=(0, (5, 3)),
            zorder=4)
    ax.plot(idx, [(512 + c) / (512 + codes[-1]) for c in codes], color=t["soft"], linewidth=1.4,
            linestyle=(0, (1, 2)), zorder=4)
    ax.axhline(0.9, color=t["soft"], linewidth=0.8, zorder=1)
    ax.annotate("if the cost followed the level's codes (4 of 48 … 48 of 48)", (2, 10 / 48), xytext=(6, -14),
                textcoords="offset points", color=t["ink"], fontsize=9.5)
    ax.annotate("if it followed article + level positions (516 … 560)", (0.5, 1.0), xytext=(0, 20),
                textcoords="offset points", color=t["ink"], fontsize=9.5)
    ax.text(len(codes) - 0.5, 0.9, "rule 2: 'not cheaper' if ≥ 90% ", ha="right", va="bottom",
            color=t["soft"], fontsize=9)
    ax.set_xticks(idx)
    ax.set_xticklabels([f"{i}\n{c} codes" for i, c in zip(idx, codes)], fontsize=9.5)
    ax.set_ylim(0, 1.3)
    ax.grid(axis="x", visible=False)
    axis_labels(ax, t, "Pass (level)", "Time relative to the last pass")
    save(fig, "pass-ratios", theme)


# ---- 4. Question C: ROUGE-L by article length ---------------------------------------------
def length_bands(cut=512):
    from rouge_score import rouge_scorer  # only needed here

    items = pickle.load(open(os.path.join(HERE, "src", "data", "xsum", "passages.pickle"), "rb"))
    shard = pickle.load(open(os.path.join(RES, "evo-x2-xsum-fp32", "eval", "xsum", "shard_0.pkl"), "rb"))
    lengths = [sum(int(tok) < 50256 for tok in x[:cut]) for x in items]
    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rl = {i: 100 * sc.score(r, h)["rougeL"].fmeasure for i, h, r in zip(shard["idxs"], shard["hyps"], shard["refs"])}
    step = cut // 4
    bands = [(f"{lo + 1}–{min(lo + step, cut - 1)}", [i for i in rl if lo + 1 <= lengths[i] <= min(lo + step, cut - 1)])
             for lo in range(0, cut, step)]
    bands.append((f"{cut}\n(fills it)", [i for i in rl if lengths[i] == cut]))
    out = []
    for label, idx in bands:
        v = [rl[i] for i in idx]
        out.append((label, len(v), st.mean(v), 1.96 * st.stdev(v) / math.sqrt(len(v))))
    return out, st.mean(rl.values()), len(rl)


def chart_rouge_length(theme, data):
    t = THEMES[theme]
    bands, overall, total = data
    fig, ax = figure(t, "Long articles get worse summaries; a third don't fit the window",
                     "NCM on all 11,334 XSum test articles (float32 run), ROUGE-L by real article tokens "
                     "in its 512-token window. Whiskers: 95% interval.")
    idx = list(range(len(bands)))
    ax.bar(idx, [b[2] for b in bands], width=0.62, color=t["series"][NCM], zorder=2)
    ax.errorbar(idx, [b[2] for b in bands], yerr=[b[3] for b in bands], fmt="none", ecolor=t["ink"],
                elinewidth=1, capsize=3, zorder=3)
    for i, b in enumerate(bands):
        ax.text(i, b[2] + b[3] + 0.6, f"{b[2]:.1f}", ha="center", va="bottom", color=t["ink"], fontsize=10)
    ax.axhline(overall, color=t["soft"], linewidth=1.2, linestyle=(0, (5, 3)), zorder=1)
    ax.text(-0.45, overall + 0.5, f"all articles: {overall:.1f}", color=t["soft"], fontsize=9.5)
    ax.axvline(len(bands) - 1.5, color=t["soft"], linewidth=0.9, linestyle=(0, (2, 3)))
    last = bands[-1]
    ax.annotate(f"{100 * last[1] / total:.0f}% of articles fill the\nwindow and were most likely cut",
                (len(bands) - 1.5, 6), xytext=(-8, 0), textcoords="offset points", ha="right",
                color=t["ink"], fontsize=10)
    ax.set_xticks(idx)
    ax.set_xticklabels([f"{b[0]}\nn = {b[1]:,}" for b in bands], fontsize=9.5)
    ax.set_ylim(0, 38)
    ax.grid(axis="x", visible=False)
    axis_labels(ax, t, "Real article tokens in NCM's 512-token window", "ROUGE-L (higher is better)")
    save(fig, "rouge-by-length", theme)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-length", action="store_true", help="skip chart 4 (needs src/data and rouge_score)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    data = None if a.skip_length else length_bands()
    for theme in THEMES:
        chart_latency_cdf(theme)
        chart_t5_length(theme)
        chart_pass_ratios(theme)
        if data:
            chart_rouge_length(theme, data)


if __name__ == "__main__":
    main()
