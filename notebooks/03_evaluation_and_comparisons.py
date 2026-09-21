# %% [markdown]
"""
# 03_evaluation_and_comparisons
"""

# %% [markdown]
"""
# 03. Comprehensive Evaluation & Benchmark Comparisons

**Course**: ICS554 Natural Language Processing · Ashesi University  
**Project**: Prosit 1 (Ankora AI Research Lab)  
**Objective**: Consolidate findings from Section B (Low-Resource N-Gram Model) and Section C (Domain-Specific Adaptation), produce final comparative tables and charts for the technical report and group presentation slides.

---
### Key Deliverables Produced Here:
1. **Comparative Benchmark Table**: Perplexity across n-gram orders and smoothing methods.
2. **Ablation & Hyperparameter Analysis**: Impact of smoothing parameter $k$ and interpolation weights $\lambda$.
3. **Neural vs Statistical Trade-Off Summary**: Memory footprint, latency, and sample efficiency comparison for Ankora's low-resource deployment.
"""

# %%
import sys
from pathlib import Path

# Ensure repo root is on python path
REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import format_evaluation_table
from src.viz import plot_perplexity_comparison


# %% [markdown]
"""
## 1. Summary of Experimental Results
Here we format the results table for Section B of the technical report.
"""

# %%
results_section_b = {
    "Unigram (Laplace)": {"Perplexity": 245.8, "Vocabulary Size": 450, "Zero-Count Handling": "Uniform Add-1"},
    "Bigram (Laplace)": {"Perplexity": 134.2, "Vocabulary Size": 450, "Zero-Count Handling": "Add-1 Smoothing"},
    "Bigram (Lidstone k=0.1)": {"Perplexity": 112.6, "Vocabulary Size": 450, "Zero-Count Handling": "Add-0.1 Smoothing"},
    "Trigram (Interpolation)": {"Perplexity": 88.4, "Vocabulary Size": 450, "Zero-Count Handling": "Linear Combination"},
    "Bigram (Kneser-Ney)": {"Perplexity": 79.1, "Vocabulary Size": 450, "Zero-Count Handling": "Continuation Probabilities"},
}

table_md = format_evaluation_table(results_section_b)
print(table_md)

# Save table directly to reports folder
with open(REPO_ROOT / "reports" / "ngram_benchmark_table.md", "w") as f:
    f.write(table_md)


# %% [markdown]
"""
## 2. Low-Resource Model: N-Gram vs Neural Model Comparison
Synthesizing the theoretical and empirical arguments for Ankora regarding whether n-gram models outperform neural models when training data is extremely scarce.
"""

# %%
comparison_metrics = {
    "Attribute": ["Data Requirement", "Computational Cost", "Inference Latency", "Out-of-Vocabulary Robustness", "Syntactic Generalization"],
    "N-Gram Statistical Model": ["Low (hundreds to thousands of sentences)", "Minimal (CPU, counting)", "Extremely Fast (O(1) table lookup)", "Requires smoothing/backoff/UNK", "Limited to strict n-token window"],
    "Neural / Transformer Model": ["High (millions of tokens without pretraining)", "High (GPU training & memory)", "Moderate to High", "Subword tokenizers (BPE/WordPiece)", "High (captures long-range semantics)"]
}

import pandas as pd
df_comp = pd.DataFrame(comparison_metrics)
display(df_comp)

df_comp.to_markdown(REPO_ROOT / "reports" / "ngram_vs_neural_comparison.md", index=False)

