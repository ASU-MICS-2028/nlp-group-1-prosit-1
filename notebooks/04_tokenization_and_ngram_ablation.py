# %% [markdown]
"""
# 04_tokenization_and_ngram_ablation
"""

# %% [markdown]
"""
# 04. Deep Learning Exploration: Tokenization Spectrum & N-Gram Lookback Ablation (N=1 to N=6)

**Student Name**: Eric Elikplim Sunu  
**Course**: ICS554 Natural Language Processing · Ashesi University  
**Objective**: Rigorously investigate how tokenization strategies and n-gram orders scale, documenting the empirical "breaking point" where statistical models collapse due to sparsity, and testing our multi-source dataset harmonization pipeline for **Ewe (Èʋegbe)**.

---
### Experimental Matrix
1. **Tokenization Strategies**: Character-level, Whitespace, Unicode NFC Word, Ewe Rule Stemmer, and Subword Byte-Pair Encoding (BPE).
2. **Lookback Horizon ($N=1$ to $N=6$)**: Measuring vocabulary size $|V|$, total sequence lengths, zero-count sparsity rate, test perplexity, and qualitative generation.
3. **Multi-Source Dataset Harmonization**: Ingesting up to 4 disparate Ewe data sources, applying Unicode normalization, removing duplicates, and creating clean train/val/test splits.
"""

# %%
import sys
import random
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure repo root is on python path
REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tokenizers import (
    WhitespaceTokenizer,
    UnicodeWordTokenizer,
    CharacterTokenizer,
    EweRuleStemmerTokenizer,
    SimpleBPETokenizer,
)
from src.experiment_runner import run_ngram_experiment
from src.data_pipeline import merge_and_harmonize_datasets

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
sns.set_theme(style="whitegrid", palette="muted")


# %% [markdown]
"""
## 1. Load Experimental Ewe Corpus
We load our curated Ewe sentences covering daily greetings, news, and proverbs.
"""

# %%
ewe_corpus = [
    "Woezɔ loo, miawo katã míedi ŋutifafa le dukɔa me",
    "Efoa nyuie mah? Nyee, mefo nyuie, akpe kaka",
    "Kofi yi suku le Keta egbe ŋdi kaba",
    "Ama fle nuɖuɖu vivi le asime le Ho",
    "Míeyi aƒeme kaba elabena tsi le dzadzam le Aflao",
    "Nufiala fia nu nusrɔ̃lawo nyuie le suku me",
    "Mia dogo le etsɔ me ne Mawu lɔ̃",
    "Devi sia nya nu ŋutɔ le eƒe nusɔsrɔ̃ me",
    "Míedi be míawɔ dɔ le ɖekawɔwɔ me le Ghana ha",
    "Ɖo to nyuie ne nàse nya si gblɔm wole na mi",
    "Agbledeŋu nye dɔ vevi aɖe le miaƒe nutoa me",
    "Míele kuku ɖem na mi be miagbɔ kaba",
    "Ŋutsu la kple nyɔnu la woyi agble me le Kpalime",
    "Mía kplɔlawo le dɔ wɔm be dukɔa nade ŋgɔ",
    "Akpe na mi katã ɖe miaƒe kpekpeɖeŋu ta le dɔa me",
    "Nusrɔ̃lawo katã di be yewoawɔ dɔ nyuie le suku"
]

split_idx = int(0.8 * len(ewe_corpus))
train_texts = ewe_corpus[:split_idx]
test_texts = ewe_corpus[split_idx:]
print(f"Train samples: {len(train_texts)}, Test samples: {len(test_texts)}")


# %% [markdown]
"""
## 2. Inspecting Tokenization Behaviors Across Ewe Text
Notice how different tokenizers treat Ewe characters (`ɖ`, `ƒ`, `ɣ`, `ŋ`, `ɔ`, `ɛ`, `ʋ`) and tone diacritics.
"""

# %%
sample_phrase = "Woezɔ loo! Nusrɔ̃lawo le suku me. Efoa nyuie mah?"

bpe_tok = SimpleBPETokenizer(num_merges=15)
bpe_tok.train(ewe_corpus)

tokenizers = {
    "Whitespace": WhitespaceTokenizer(),
    "Unicode Word": UnicodeWordTokenizer(),
    "Character": CharacterTokenizer(),
    "Ewe Stemmer": EweRuleStemmerTokenizer(),
    "Byte-Pair Encoding (BPE)": bpe_tok,
}

print(f"Sample Phrase: '{sample_phrase}'\n")
for name, tok in tokenizers.items():
    t_out = tok.tokenize(sample_phrase)
    print(f"{name:25s} -> Tokens ({len(t_out)}): {t_out[:10]}")


# %% [markdown]
"""
## 3. Sweeping N-Gram Orders ($N=1$ to $N=6$) Across Tokenizers
We evaluate how sparsity increases and perplexity evolves as the lookback horizon scales from Unigram ($N=1$) to 6-gram ($N=6$).
"""

# %%
import json
results_json_path = REPO_ROOT / "reports" / "results_unified_all_tokenizers.json"

if results_json_path.exists():
    print(f"Loading full Grand Unified Mega-Corpus results from {results_json_path.name}...")
    with open(results_json_path) as f:
        data = json.load(f)
    all_results = []
    for tok_name, orders in data["results"].items():
        all_results.extend(orders)
    df_results = pd.DataFrame(all_results)
    print(f"Loaded {len(df_results)} benchmark rows across {df_results['tokenizer'].nunique()} tokenizers.")
else:
    print("Running live sweep on local sample...")
    all_results = []
    for name, tok in tokenizers.items():
        print(f"Running sweep for: {name}...")
        res = run_ngram_experiment(train_texts, test_texts, tok, max_order=6, smoothing="laplace")
        all_results.extend(res)
    df_results = pd.DataFrame(all_results)

print(df_results[['tokenizer', 'order', 'vocab_size', 'sparsity_pct', 'perplexity']].to_string(index=False))


# %% [markdown]
"""
## 4. Visualizing the "Breaking Point"
Notice where test sparsity approaches 100% and perplexity inverts or spikes.
"""

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# 1. Sparsity Rate vs N-Gram Order
sns.lineplot(
    data=df_results,
    x="order",
    y="sparsity_pct",
    hue="tokenizer",
    marker="o",
    ax=axes[0],
)
axes[0].set_title("Test Set Sparsity Rate (% Unseen N-Grams)", fontsize=13, weight="bold")
axes[0].set_xlabel("N-Gram Order (N)", fontsize=11)
axes[0].set_ylabel("Unseen N-Gram Rate (%)", fontsize=11)
axes[0].set_ylim(-5, 105)

# 2. Perplexity vs N-Gram Order (for Word-level and BPE)
df_word = df_results[df_results["tokenizer"].isin(["Unicode Word", "Byte-Pair Encoding (BPE)", "Ewe Stemmer"])]
sns.lineplot(
    data=df_word,
    x="order",
    y="perplexity",
    hue="tokenizer",
    marker="s",
    ax=axes[1],
)
axes[1].set_title("Test Perplexity vs. N-Gram Order", fontsize=13, weight="bold")
axes[1].set_xlabel("N-Gram Order (N)", fontsize=11)
axes[1].set_ylabel("Perplexity (Lower is better)", fontsize=11)

plt.tight_layout()
plt.savefig(REPO_ROOT / "figures" / "ngram_ablation_breaking_point.png", dpi=300)
plt.show()


# %% [markdown]
"""
## 5. Text Generation Breakdown Across Orders (Unigram to 6-Gram)
Notice how generation transitions from random unigram word soup to fluent bigrams/trigrams, and then to rigid memorization at $N=5, 6$.
"""

# %%
word_results = df_results[df_results["tokenizer"] == "Unicode Word"]
print("=== Generation Evolution in Ewe (Unicode Word Tokenizer) ===\n")
for _, row in word_results.iterrows():
    print(f"[{row['order_name']} (Sparsity: {row['sparsity_pct']}%)]: {row['sample_generation']}")


# %% [markdown]
"""
## 6. Multi-Source Dataset Harmonization Pipeline
When you collect datasets from 4 different locations (e.g. news, conversational text, Menyo-20k, cultural texts), use our pipeline to normalize (NFC), deduplicate, and merge them into a single clean corpus.
"""

# %%
# Example demonstration with simulated multi-source files
source_dir = REPO_ROOT / "data" / "raw" / "low_resource"
processed_dir = REPO_ROOT / "data" / "processed" / "low_resource"

# Create demonstration source files if not yet provided
src1 = source_dir / "source1_conversational.txt"
src2 = source_dir / "source2_news.txt"

if not src1.exists():
    with open(src1, "w", encoding="utf-8") as f:
        f.write("Woezɔ loo\nEfoa nyuie mah?\nNyee, mefo nyuie, akpe\nMia dogo le etsɔ me\n")

if not src2.exists():
    with open(src2, "w", encoding="utf-8") as f:
        f.write("Kofi yi suku le Keta\nAma fle nuɖuɖu le asime\nMiawo katã míedi ŋutifafa le dukɔa me\n")

sources_to_merge = [f for f in source_dir.glob("*.txt") if f.name != "README.md"]
print(f"Found {len(sources_to_merge)} source files to harmonize: {[s.name for s in sources_to_merge]}")

summary = merge_and_harmonize_datasets(
    source_files=sources_to_merge,
    output_dir=processed_dir,
    min_words=2,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
)

