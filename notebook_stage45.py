"""Add (or replace) the Stage 4 and Stage 5 sections of main.ipynb.

These cells only READ results/*.json (the heavy runs are `stage4_neural.py`, `stage4_english.py`, `stage5_domain.py`),
so they are executed on their own in a temporary notebook and copied back with their outputs: no 1-hour full re-run.
"""
import json
import subprocess
import sys

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.strip()}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip()}
src = lambda c: c["source"] if isinstance(c["source"], str) else "".join(c["source"])

new = [
md("""
# Stage 4 — Tokenisation and the neural comparison

**Already done in earlier stages:** the tokenisers and BPE (2.5), bits-per-character (3.4), and every tokenisation at its own best order (3.7).

**New here:** a small *neural* language model trained from scratch on exactly the same tokens as the n-gram, and the question the plan calls the centrepiece: **how do the two compare as the training data grows?**

- An **n-gram** model *counts*. It can only reuse word sequences it has literally seen.
- A **neural** model *learns numbers* (an "embedding") for each token, so tokens used in similar ways end up with similar numbers. That lets it guess sensibly about sequences it never saw. The price: it needs many examples before those numbers mean anything.
- **LSTM**: reads the sentence one token at a time, carrying a memory forward. **Transformer**: looks back at all earlier tokens at once and decides which ones matter ("attention"). This is the design behind today's large models.

The heavy training lives in `stage4_neural.py` and `stage4_english.py`; these cells read the saved results.
"""),
code(r'''
import json
from pathlib import Path

import matplotlib.pyplot as plt

s4 = json.loads(Path("results/stage4_scaling.json").read_text())
ARCH_LABEL = {"ngram": "n-gram (modified Kneser-Ney)", "lstm": "LSTM", "transformer": "tiny transformer"}
sizes = sorted(int(k) for k in s4["scaling"])
print(f"training sizes measured: {sizes}  (all models use bpe4k tokens and are graded on the same {s4['scaling'][str(sizes[0])]['tokens']:,} dev tokens)")
'''),
md("""
## 4.1 The scaling experiment
Same tokens (bpe4k), same dev set, so perplexity **is** comparable here. Lower is better. The n-gram uses its best order (2-6) at each size; the neural models stop training when their dev score stops improving.
"""),
code(r'''
print(f"{'sentences':>10} {'n-gram':>9} {'LSTM':>9} {'transformer':>12}   best")
for n in sizes:
    row = s4["scaling"][str(n)]
    scores = {a: row[a]["pp"] for a in ARCH_LABEL if a in row}
    best = min(scores, key=scores.get)
    print(f"{n:>10,} {scores.get('ngram', float('nan')):>9.1f} {scores.get('lstm', float('nan')):>9.1f} {scores.get('transformer', float('nan')):>12.1f}   {ARCH_LABEL[best]}")

print("\nHow far behind the n-gram is each neural model? (1.00 = level; below 1 = the neural model is ahead)")
for n in sizes:
    row = s4["scaling"][str(n)]
    print(f"{n:>10,}   " + "   ".join(f"{ARCH_LABEL[a]}: {row[a]['pp'] / row['ngram']['pp']:.2f}x" for a in ("lstm", "transformer") if a in row))
'''),
code(r'''
COLORS = {"ngram": "#2a78d6", "lstm": "#eb6834", "transformer": "#1baf7a"}  # validated colour-blind-safe order
MARKERS = {"ngram": "o", "lstm": "s", "transformer": "^"}                  # shape repeats identity, not colour alone
fig, ax = plt.subplots(figsize=(8, 4.6), facecolor="#fcfcfb")
ax.set_facecolor("#fcfcfb")
ends = {}
for arch in ARCH_LABEL:
    xs = [n for n in sizes if arch in s4["scaling"][str(n)]]
    ys = [s4["scaling"][str(n)][arch]["pp"] for n in xs]
    ax.plot(xs, ys, color=COLORS[arch], marker=MARKERS[arch], markersize=8, linewidth=2, label=ARCH_LABEL[arch],
            markeredgecolor="#fcfcfb", markeredgewidth=2)
    ends[arch] = (xs[-1], ys[-1])
label_y, last = {}, 0
for arch, (x, y) in sorted(ends.items(), key=lambda kv: kv[1][1]):  # nudge labels apart when two lines end close together
    label_y[arch] = last = max(y, last * 1.13)
for arch, (x, y) in ends.items():
    ax.annotate(f"{ARCH_LABEL[arch].split(' (')[0]}  {y:.0f}", (x, y), xytext=(x * 1.12, label_y[arch]), textcoords="data",
                va="center", fontsize=9, color="#3d3d3a")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks(sizes); ax.set_xticklabels([f"{n:,}" for n in sizes])
ax.set_xlim(sizes[0] * 0.8, sizes[-1] * 3.2)
ax.minorticks_off()
ax.set_xlabel("training sentences", color="#3d3d3a"); ax.set_ylabel("dev perplexity (lower is better)", color="#3d3d3a")
ax.set_title("Perplexity against training data: counting vs neural (bpe4k tokens)", loc="left", fontsize=11, color="#1a1a19")
ax.grid(True, which="major", color="#e6e5e0", linewidth=0.8); ax.set_axisbelow(True)
for side in ("top", "right"): ax.spines[side].set_visible(False)
for side in ("left", "bottom"): ax.spines[side].set_color("#c3c2b7")
ax.tick_params(colors="#73726c")
ax.legend(frameon=False, fontsize=9, labelcolor="#3d3d3a")
fig.tight_layout()
Path("results").mkdir(exist_ok=True)
fig.savefig("results/stage4_scaling.png", dpi=160)
plt.show()
'''),
md("""
## 4.2 Side experiment: how big should the neural model be?
LSTM at one training size, four widths. "Width" is how many numbers the model uses to describe each token and its memory. Bigger can learn more, but with limited data it can also just memorise.
"""),
code(r'''
print(f"{'width':>6} {'parameters':>12} {'perplexity':>11} {'bits/char':>10} {'epochs':>7} {'seconds':>8}")
for dim, r in sorted(s4["width"].items(), key=lambda kv: int(kv[0])):
    print(f"{dim:>6} {r['params']:>12,} {r['pp']:>11.1f} {r['bpc']:>10.3f} {r['epochs']:>7} {r['seconds']:>8}")
if s4["width"]:
    best = min(s4["width"], key=lambda d: s4["width"][d]["pp"])
    print(f"\nbest width at this data size: {best}")
'''),
md("""
## 4.3 Side experiment: which tokens should the neural model read?
Same LSTM, same data, four BPE vocabularies. **Perplexity is not comparable across rows** (different tokens), so rank on **bits per character**. The n-gram at its best order is shown beside it.
"""),
code(r'''
print(f"{'tokeniser':<10} {'LSTM bits/char':>15} {'n-gram bits/char':>17} {'LSTM perplexity':>16} {'n-gram order':>13}")
for name, r in s4["tokenizer"].items():
    print(f"{name:<10} {r['lstm']['bpc']:>15.3f} {r['ngram']['bpc']:>17.3f} {r['lstm']['pp']:>16.1f} {r['ngram']['order']:>13}")
'''),
md("""
## 4.4 Explore: the same pipeline on English
Same number of sentences from the same crawl, punctuation-split tokens, modified Kneser-Ney at order 5.

**Caveat:** the English side is the mined "translation", and Stage 0 showed many pairs are misaligned. So this is *English text of similar size from the same source*, not the same content. The type/token ratio is measured on the **first million tokens of each**, because that ratio always falls as a corpus grows.
"""),
code(r'''
en = json.loads(Path("results/stage4_english.json").read_text())
LABELS = [("sentences", "training sentences", ",.0f"), ("tokens", "tokens", ",.0f"), ("types", "different words (types)", ",.0f"),
          ("types_in_first_million_tokens", "different words in first 1M tokens", ",.0f"),
          ("type_token_ratio_first_million", "type/token ratio (first 1M)", ".4f"), ("seen_once_share", "share of words seen only once", ".1%"),
          ("avg_word_length", "average word length (characters)", ".2f"), ("dev_oov", "unknown words in dev", ".2%"),
          ("perplexity", "perplexity", ",.1f"), ("bits_per_char", "bits per character", ".3f")]
print(f"{'':<36} {'Ewe':>12} {'English':>12}")
for key, label, fmt in LABELS:
    print(f"{label:<36} {format(en['ewe'][key], fmt):>12} {format(en['en'][key], fmt):>12}")
more = en["ewe"]["types_in_first_million_tokens"] / en["en"]["types_in_first_million_tokens"] - 1
print(f"\nIn the same number of tokens, Ewe shows {more:.0%} more different words than English.")
'''),
md("""
## Stage 4 summary — what happened
"""),
code(r'''
first, last = s4["scaling"][str(sizes[0])], s4["scaling"][str(sizes[-1])]
print(f"At {sizes[0]:,} sentences:   " + " | ".join(f"{ARCH_LABEL[a].split(' (')[0]} {first[a]['pp']:.0f}" for a in ARCH_LABEL if a in first))
print(f"At {sizes[-1]:,} sentences: " + " | ".join(f"{ARCH_LABEL[a].split(' (')[0]} {last[a]['pp']:.0f}" for a in ARCH_LABEL if a in last))
for arch in ("lstm", "transformer"):
    cross = next((n for n in sizes if arch in s4["scaling"][str(n)] and s4["scaling"][str(n)][arch]["pp"] < s4["scaling"][str(n)]["ngram"]["pp"]), None)
    print(f"{ARCH_LABEL[arch]}: " + (f"overtakes the n-gram at {cross:,} sentences" if cross else "never overtakes the n-gram in the range we measured"))
'''),
md("""
# Stage 5 — Adapting a pretrained model to a domain

A large pretrained model already knows general English. **Domain adaptation** = keep training it on text from one field so it gets better at that field. The risk is **catastrophic forgetting**: getting worse at everything else. We measure both.

| Method | What is trained |
|---|---|
| **Full fine-tuning** | Every weight in the model moves. Most powerful, most memory, one full-size copy per domain. |
| **LoRA** | The model is frozen. Small add-on matrices are trained beside some layers. **Rank** = how big those add-ons are. |
| **DoRA** | A LoRA variant that separates "which direction to change" from "how much". |

**The fan-out:** 2 domains (health = PubMed abstracts, agriculture = farming Q&A) × 5 methods, on Qwen2.5-0.5B, plus one repeat on a second, smaller model. Every model is scored on held-out **health**, **agriculture** and **general** (Wikipedia) text. All text is English: no small open model knows Ewe well enough for adaptation to mean anything.

The training lives in `stage5_domain.py`; these cells read the saved results.
"""),
code(r'''
s5 = json.loads(Path("results/stage5_domain.json").read_text())
print("settings:", s5["settings"], "\n")

def pick(runs, domain, name):
    """The run for this domain + method at the chosen learning rate (full fine-tuning has its own fixed rate)."""
    lr = s5["settings"].get("full_lr", 1e-5) if name == "full" else s5.get("lr_choice", {}).get("chosen")
    return runs.get(f"{domain}/{name}@{lr:g}") if lr else None

for name, d in s5["data"].items():
    print(f"{name:<12} train: {d['train_docs']:>5,} documents / {d['train_words']:>9,} words | test: {d['test_docs']} documents")
    print(f"{'':<12} e.g. {d['example'][:150]}...\n")
'''),
md("""
## 5.0 Choosing the learning rate (on validation, never on test)
The learning rate is how big a step the model takes each time it corrects itself. Our first attempt used 1e-4 and gave a warning sign: LoRA rank 8 ended up **worse** on health text than rank 2, which should not happen if training is going well. Too big a step makes a model overshoot. So we tried three rates with the same setup (LoRA rank 8, health) and kept the one with the lowest perplexity on the **validation** split.
"""),
code(r'''
main_model = next(iter(s5["runs"]))
choice = s5.get("lr_choice")
if choice:
    base_h = s5["runs"][main_model]["base"]["health"]
    print(f"{'learning rate':>14} {'validation ppl':>15} {'health test':>12} {'general test':>13}")
    for lr, v in choice["validation_perplexity_by_lr"].items():
        r = s5["runs"][main_model][f"health/lora_r8@{lr}"]
        mark = "  <- chosen" if float(lr) == choice["chosen"] else ""
        print(f"{lr:>14} {v:>15.2f} {r['health']:>12.2f} {r['general']:>13.2f}{mark}")
    print(f"\n(base model, untouched: health {base_h:.2f}, general {s5['runs'][main_model]['base']['general']:.2f})")
else:
    print("learning-rate sweep not finished yet")
'''),
md("""
## 5.1 Before and after, for every method
Numbers are **perplexity** (lower is better); the percentage is the change from the untouched base model. The same model and tokeniser are used before and after, so these perplexities are comparable.

- **Trained-on column going down** = adaptation worked.
- **General column going up** = catastrophic forgetting, measured.
"""),
code(r'''
METHOD_LABEL = {"lora_r2": "LoRA rank 2", "lora_r8": "LoRA rank 8", "lora_r32": "LoRA rank 32", "dora_r8": "DoRA rank 8", "full": "full fine-tuning"}
EVALS = ["health", "agriculture", "general"]

def show(model):
    runs = s5["runs"][model]
    base = runs["base"]
    print(f"MODEL: {model.split('/')[-1]}")
    print(f"  {'':<28}" + "".join(f"{e:>22}" for e in EVALS) + f"{'trained weights':>18}{'minutes':>9}")
    print(f"  {'base model (untouched)':<28}" + "".join(f"{base[e]:>22.2f}" for e in EVALS))
    for domain in ("health", "agriculture"):
        print(f"  adapted to {domain.upper()}:")
        for name, label in METHOD_LABEL.items():
            r = pick(runs, domain, name)
            if r:
                cells = "".join(f"{r[e]:>13.2f} ({(r[e] / base[e] - 1):>+6.1%})" for e in EVALS)
                weights = f"{r['trainable_millions']:.1f}M ({r['trainable_pct']:.2f}%)" if r.get("trainable_millions") else ""
                print(f"    {label:<26}{cells}{weights:>18}{r['train_seconds'] / 60:>9.1f}")
    print()

for model in s5["runs"]:
    show(model)
'''),
md("""
## 5.2 The 2×2 table the plan asks for
Base vs adapted, in-domain vs general, for the middle setting (LoRA rank 8).
"""),
code(r'''
model = next(iter(s5["runs"]))
runs, base = s5["runs"][model], s5["runs"][model]["base"]
for domain in ("health", "agriculture"):
    r = pick(runs, domain, "lora_r8")
    if not r:
        continue
    print(f"Domain: {domain}   ({model.split('/')[-1]}, LoRA rank 8)")
    print(f"  {'':<12} {'in-domain':>12} {'general':>12}")
    print(f"  {'base':<12} {base[domain]:>12.2f} {base['general']:>12.2f}")
    print(f"  {'adapted':<12} {r[domain]:>12.2f} {r['general']:>12.2f}")
    print(f"  {'change':<12} {r[domain] / base[domain] - 1:>+12.1%} {r['general'] / base['general'] - 1:>+12.1%}\n")
'''),
md("""
## Stage 5 summary — what happened
"""),
code(r'''
for domain in ("health", "agriculture"):
    rows = [(METHOD_LABEL[n], pick(runs, domain, n)) for n in METHOD_LABEL if pick(runs, domain, n)]
    if not rows:
        continue
    best = min(rows, key=lambda x: x[1][domain])
    least = min(rows, key=lambda x: x[1]["general"])
    print(f"{domain}: best in-domain = {best[0]} ({base[domain]:.2f} -> {best[1][domain]:.2f}); "
          f"least forgetting = {least[0]} (general {base['general']:.2f} -> {least[1]['general']:.2f})")
'''),
]

# run the new cells on their own, then splice them (with outputs) into main.ipynb
tmp = {"cells": new, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(tmp["cells"]):
    c["id"] = f"s45-{i:02d}"
json.dump(tmp, open("_stage45.ipynb", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
allow = ["--allow-errors"] if "--allow-errors" in sys.argv else []
subprocess.run(["uv", "run", "--with", "nbconvert", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", *allow, "_stage45.ipynb"], check=True)
done = json.load(open("_stage45.ipynb", encoding="utf-8"))["cells"]

nb = json.load(open("main.ipynb", encoding="utf-8"))
cells = nb["cells"]
start = next((i for i, c in enumerate(cells) if c["cell_type"] == "markdown" and src(c).startswith("# Stage 4 ")), len(cells))
nb["cells"] = cells[:start] + done
for i, c in enumerate(nb["cells"]):
    c["id"] = f"c{i:03d}"
json.dump(nb, open("main.ipynb", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
import os; os.remove("_stage45.ipynb")
print(f"main.ipynb now has {len(nb['cells'])} cells (Stage 4 and 5 sections refreshed)")
