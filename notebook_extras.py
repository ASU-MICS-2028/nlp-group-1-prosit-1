"""Put every follow-up experiment into main.ipynb, in the stage it belongs to.

Each section only READS results/* (the heavy runs are the stage*_*.py scripts), so this executes in seconds and can be
re-run whenever a results file changes. Sections are tagged with ids starting "x-"; re-running replaces them.

Run: uv run python notebook_extras.py
"""
import json
import os
import subprocess

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.strip()}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip()}
src = lambda c: c["source"] if isinstance(c["source"], str) else "".join(c["source"])

COMMON = r'''
import json
from pathlib import Path
from IPython.display import Markdown, display

def load(name):
    p = Path("results") / name
    return json.loads(p.read_text()) if p.exists() else None

def show_md(name):
    p = Path("results") / name
    display(Markdown(p.read_text(encoding="utf-8") if p.exists() else f"*{name} not produced yet*"))
'''

SECTIONS = {
# ------------------------------------------------------------------ after Stage 0 summary
"after:## Stage 0 summary": [
md("""
# Stage 0 follow-ups: testing every data decision

Stage 0 made choices (which split ratio, which cleaning steps, which filter). Each one is tested here by changing it and measuring what happens to the same model (modified Kneser-Ney, order 5). Scripts: `stage0_splits.py`, `stage0_ablation.py`, `stage0_filter_test.py`.
"""),
code(COMMON + r'''
show_md("split_ratios.md")
'''),
md("""
**Reading the split table.** Each ratio has its own dev/test, so those columns are different exams and cannot be compared across rows. The *common test* (the 98/1/1 test set, hash bucket 0, which no ratio trains on) is the fair exam.

**The trap.** On all tokens, the 1% split beats the 5% split on the common exam. With tiny training data 15% of exam words are unknown, and the model's reserved share for unknown words is so large (13.9%) that each unknown word costs only 2.8 bits, cheaper than a known word. Score known words only and the truth appears: the 1% model is the worst of all. Unknown-word handling can flatter perplexity; this is that warning in our own table.

**Dev size.** Eight random draws of the dev set at each size: 100 sentences wobble by 40 points, 10,000 by 3. That is why 5% (about 15,000) was right.
"""),
code(r'''
r = load("split_ratios.json")
print(f"{'split':>8} {'train':>8} {'common test (all tokens)':>25} {'common test (known words)':>26} {'bits/char':>10} {'OOV':>7}")
for f, row in r["ratios"].items():
    print(f"{f:>8} {row['train']:>8,} {row['common_pp']:>25.1f} {row['common_pp_known']:>26.1f} {row['common_bpc']:>10.3f} {row['common_oov']:>7.2%}")
'''),
md("""
## Leave one preparation step out
Same model, same dev exam where the step does not change which sentences exist. The random-split row is the one to stare at: the *only* change that makes the score better, and it does so by leaking near-copies of dev sentences into training.
"""),
code(r'''
show_md("ablation.md")
r = load("ablation.json")
base = r["baseline (all steps)"]["dev_pp"]
for k, v in r.items():
    print(f"{k:<28} dev perplexity {v['dev_pp']:>7.1f}  ({v['dev_pp'] / base - 1:+6.1%} vs baseline)")
'''),
md("""
## Grading the filter with a language identifier
GlotLID (Kargaran et al., 2023) recognises about 2,100 languages including Ewe, Ga, Fon, Twi and Adangme. It is used here only to *grade* our letter filter, as a stand-in for the native-speaker audit the plan wanted; it is not our filter. The standard fastText identifier does not know Ewe at all.
"""),
code(r'''
show_md("filter_test.md")
'''),
],
# ------------------------------------------------------------------ after Stage 2 summary
"after:## Stage 2 summary": [
md("""
## Stage 2 follow-up: ten generated sentences per order
The plan asks for ten; the cells above show five. `[COPIED]` marks a sentence that appears word for word in the training data.
"""),
code(COMMON + r'''
show_md("generated_samples.md")
'''),
],
# ------------------------------------------------------------------ after Stage 3 summary
"after:## Stage 3 summary": [
md("""
# Stage 3 follow-ups

## Is smoothing a real gain, or does it just hide the zeros?
Plain counting (MLE) gives infinity to any sentence with an unseen pair, so "finite beats infinite" proves little on its own. The honest test: on the dev sentences where **every** pair was seen in training, plain counting works, and we can ask what smoothing costs there.
"""),
code(COMMON + r'''
r = load("dev_and_smoothing_checks.json")["smoothing_vs_mle"]
print(f"dev sentences with no unseen pair: {r['sentences_all_seen']:,}; with at least one: {r['sentences_with_unseen']:,}\n")
print(f"{'method':<22} {'fully-seen sentences':>21} {'the rest':>10} {'all dev':>9}")
print(f"{'plain counting (MLE)':<22} {r['mle_on_seen']:>21.1f} {'infinity':>10} {'infinity':>9}")
for m, label in (("add_k", "add-k"), ("jm", "interpolation"), ("katz", "Katz backoff"), ("kn", "Kneser-Ney"), ("mkn", "modified Kneser-Ney")):
    print(f"{label:<22} {r[m]['seen']:>21.1f} {r[m]['unseen']:>10.1f} {r[m]['all']:>9.1f}")
print("\nSmoothing costs a little where counting works (73.7 -> 87.1) and gains everything where it fails (81% of sentences). A trade, not a trick.")
'''),
md("""
## Should the dev set become training data at the end?
After all tuning is done, the dev set is no longer needed for choosing settings, so it can join the training data for a final deployed model. Test-set score, train only vs train + dev:
"""),
code(r'''
r = load("dev_and_smoothing_checks.json")["dev_into_train"]
for k, v in r.items():
    print(f"{k:<26} test perplexity {v['test_pp']:.1f} | known words {v['test_pp_known']:.1f} | bits/char {v['test_bpc']:.3f}")
print("\nA 2.5% gain from 5.5% more sentences, right on the data curve. Only valid AFTER tuning: fold dev in first and there is nothing left to tune on except test.")
'''),
md("""
## Charging word-level models fairly for unknown words, and pushing small tokenisers higher
Section 3.7 left a caveat: a word model pays one flat cost for an unknown word while BPE and character models must spell it. Here the word model is charged the cost of spelling each unknown word with a character model (order 8), so the comparison is fair. Also: bpe1k, bpe2k and char were still improving at their highest tested order, so they get higher ones.
"""),
code(r'''
show_md("stage3_extra.md")
'''),
md("""
## The same smoothing ladder on English
The statistical half of "both high and low resource languages": every method on the English side of the corpus, same recipe. (Script: `stage4_english.py`.)
"""),
code(r'''
r = load("stage4_english.json")
print(f"{'method':<8} {'Ewe':>10} {'English':>10}      {'Kneser-Ney by order':<22} {'Ewe':>8} {'English':>8}")
methods = list(r["ewe"]["ladder"]["methods"])
orders = list(r["ewe"]["ladder"]["kn_by_order"])
for i in range(max(len(methods), len(orders))):
    left = f"{methods[i]:<8} {r['ewe']['ladder']['methods'][methods[i]]['pp']:>10,.1f} {r['en']['ladder']['methods'][methods[i]]['pp']:>10,.1f}" if i < len(methods) else " " * 30
    right = f"order {orders[i]:<16} {r['ewe']['ladder']['kn_by_order'][orders[i]]:>8.1f} {r['en']['ladder']['kn_by_order'][orders[i]]:>8.1f}" if i < len(orders) else ""
    print(f"{left}      {right}")
print("\nSame ranking in both languages (add-k and Good-Turing far behind; Kneser-Ney best and improving with order). English is easier at the same size: 42 vs 51.")
'''),
],
}

# Stage 4 and 5 additions are appended to the END of those stages' own sections (built by notebook_stage45.py)
SECTIONS["before:## Stage 4 summary"] = [
md("""
## 4.5 Transformer side experiments
The same width and tokeniser sweeps as 4.2 and 4.3, for the transformer, at 10,000 sentences.
"""),
code(COMMON + r'''
r = load("stage4_scaling.json")
print("width (10k sentences):")
print(f"  {'width':>6} {'LSTM':>8} {'transformer':>12}")
for d in ("64", "128", "256", "512"):
    l, t = r["width"].get(d), r.get("width_transformer", {}).get(d)
    print(f"  {d:>6} {l['pp'] if l else float('nan'):>8.1f} {t['pp'] if t else float('nan'):>12.1f}")
print("\ntokeniser (10k sentences, bits/char):")
print(f"  {'tokeniser':<8} {'n-gram':>8} {'LSTM':>8} {'transformer':>12}")
for name in ("bpe1k", "bpe2k", "bpe4k", "bpe16k"):
    l, t = r["tokenizer"].get(name), r.get("tokenizer_transformer", {}).get(name)
    print(f"  {name:<8} {l['ngram']['bpc'] if l else float('nan'):>8.3f} {l['lstm']['bpc'] if l else float('nan'):>8.3f} {t['transformer']['bpc'] if t else float('nan'):>12.3f}")
'''),
md("""
## 4.6 Did the LSTM stall at 100k because it was too small?
A 512-wide LSTM (6.3M weights, 3x the original) at the two largest sizes.
"""),
code(r'''
r = load("stage4_scaling.json")
print(f"{'sentences':>10} {'n-gram':>8} {'LSTM 256':>10} {'LSTM 512':>10} {'transformer':>12}")
for n in ("30000", "100000"):
    row = r["scaling"][n]
    print(f"{int(n):>10,} {row['ngram']['pp']:>8.1f} {row['lstm']['pp']:>10.1f} {row['lstm_512']['pp'] if 'lstm_512' in row else float('nan'):>10.1f} {row['transformer']['pp']:>12.1f}")
'''),
md("""
## 4.7 The scaling curve on English, two ways
**Same-source English** (the English side of our split, 265k sentences, same recipe as Ewe): does the crossover point move for a high-resource language?
**WikiText-103** (103M words of Wikipedia): what does *actually* having a lot of data buy? Our own counter goes to 300k sentences; beyond that KenLM (checked against our code) does the counting. Scripts: `stage4_neural.run_english`, `stage4_wikitext.py`.
"""),
code(r'''
en = load("stage4_scaling_english.json")
ewe = load("stage4_scaling.json")
if en:
    print("Same-source English vs Ewe (bpe4k tokens; perplexities comparable within a language only):")
    print(f"{'sentences':>10} {'Ewe n-gram':>11} {'Ewe LSTM':>9} {'Ewe transf.':>12} | {'EN n-gram':>10} {'EN LSTM':>8} {'EN transf.':>11}")
    for n in sorted(set(en["scaling"]) | set(ewe["scaling"]), key=int):
        a, b = ewe["scaling"].get(n, {}), en["scaling"].get(n, {})
        g = lambda row, k: f"{row[k]['pp']:.1f}" if k in row else "-"
        print(f"{int(n):>10,} {g(a,'ngram'):>11} {g(a,'lstm'):>9} {g(a,'transformer'):>12} | {g(b,'ngram'):>10} {g(b,'lstm'):>8} {g(b,'transformer'):>11}")
wt = load("wikitext.json")
if wt:
    print("\nWikiText-103, modified Kneser-Ney order 5, whitespace tokens, 20k-sentence dev slice:")
    print(f"{'sentences':>10} {'KenLM ppl':>10} {'known only':>11} {'OOV':>7} {'our code':>9}")
    for n, row in wt["ngram"].items():
        print(f"{int(n):>10,} {row['kenlm_pp']:>10.1f} {row['kenlm_pp_known']:>11.1f} {row['oov_rate']:>7.2%} {row.get('own_pp', float('nan')):>9.1f}")
    if "neural" in wt:
        print("\nWikiText-103, n-gram vs neural (bpe4k tokens):")
        print(f"{'sentences':>10} {'n-gram':>8} {'LSTM':>8} {'transformer':>12}")
        for n, row in wt["neural"].items():
            g = lambda k: f"{row[k]['pp']:.1f}" if k in row else "-"
            print(f"{int(n):>10,} {g('ngram'):>8} {g('lstm'):>8} {g('transformer'):>12}")
'''),
]
SECTIONS["before:## Stage 5 summary"] = [
md("""
## 5.3 Are the differences real? Repeats with other random seeds
Each Section C setting was first run once. Two more seeds on the six most-quoted settings show how much a number moves when only the random seed changes. Differences smaller than that spread are noise.
"""),
code(COMMON + r'''
s5 = load("stage5_domain.json")
runs = s5["runs"]["data/models/Qwen2.5-0.5B"]
import re
groups = {}
for key, r in runs.items():
    m = re.match(r"(\w+)/(\w+)@([\d.e-]+)(?:#seed(\d))?$", key)
    if m and float(m.group(3)) == s5["lr_choice"]["chosen"] or (m and m.group(2) == "full"):
        groups.setdefault((m.group(1), m.group(2)), []).append(r)
print(f"{'domain':<12} {'method':<9} {'runs':>4} {'in-domain: min - max':>22} {'general: min - max':>20}")
for (domain, method), rs in sorted(groups.items()):
    if len(rs) > 1:
        ind = [r[domain] for r in rs]; gen = [r["general"] for r in rs]
        print(f"{domain:<12} {method:<9} {len(rs):>4} {min(ind):>10.2f} - {max(ind):<9.2f} {min(gen):>9.2f} - {max(gen):<8.2f}")
'''),
md("""
## 5.4 What the adapted models actually write
Same prompt, greedy decoding, base model vs each adapter. Adaptation shows up as register and vocabulary, not new facts.
"""),
code(r'''
show_md("stage5_generation.md")
'''),
]


def main():
    nb = json.load(open("main.ipynb", encoding="utf-8"))
    cells = [c for c in nb["cells"] if not str(c.get("id", "")).startswith("x-")]  # drop previous extras
    new_cells = []
    for anchor, block in SECTIONS.items():
        mode, text = anchor.split(":", 1)
        idx = next((i for i, c in enumerate(cells) if c["cell_type"] == "markdown" and src(c).startswith(text)), None)
        if idx is None:
            print(f"anchor not found, skipping: {text!r}")
            continue
        if mode == "after":  # after the anchor markdown AND its code cell
            pos = idx + 1
            while pos < len(cells) and cells[pos]["cell_type"] == "code":
                pos += 1
        else:
            pos = idx
        for j, c in enumerate(block):
            c["id"] = f"x-{abs(hash(anchor)) % 10**6}-{j:02d}"
        cells[pos:pos] = block
        new_cells += block
    # execute only the new cells, in a scratch notebook, then copy their outputs back
    tmp = {"cells": [dict(c) for c in new_cells], "metadata": nb["metadata"], "nbformat": 4, "nbformat_minor": 5}
    json.dump(tmp, open("_extras.ipynb", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    subprocess.run(["uv", "run", "--with", "nbconvert", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", "--allow-errors", "_extras.ipynb"],
                   check=True, capture_output=True)
    done = {c["id"]: c for c in json.load(open("_extras.ipynb", encoding="utf-8"))["cells"]}
    os.remove("_extras.ipynb")
    errors = 0
    for c in cells:
        if c["id"] in done:
            c.update(done[c["id"]])
            errors += sum(o.get("output_type") == "error" for o in c.get("outputs", []))
    nb["cells"] = cells
    json.dump(nb, open("main.ipynb", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"main.ipynb: {len(cells)} cells; {len(new_cells)} follow-up cells refreshed; {errors} cell errors")


if __name__ == "__main__":
    main()
