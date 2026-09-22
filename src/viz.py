"""
Visualization utilities for language modeling, perplexity analysis, and training curves.
"""

from pathlib import Path
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import seaborn as sns


# Set aesthetic style
sns.set_theme(style="whitegrid", palette="muted")


def plot_ngram_frequency(
    frequencies: Dict[str, int],
    top_k: int = 20,
    title: str = "Top N-Gram Frequencies",
    save_path: Optional[Path] = None,
):
    """
    Plots a horizontal bar chart of the most frequent n-grams.
    """
    sorted_items = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)[:top_k]
    words = [k if isinstance(k, str) else " ".join(k) for k, v in sorted_items]
    counts = [v for k, v in sorted_items]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts, y=words, color="steelblue")
    plt.title(title, fontsize=14, weight="bold")
    plt.xlabel("Frequency", fontsize=12)
    plt.ylabel("Token / N-Gram", fontsize=12)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_perplexity_comparison(
    models: List[str],
    perplexities: List[float],
    title: str = "Perplexity Comparison Across Language Models",
    save_path: Optional[Path] = None,
):
    """
    Plots a bar chart comparing test perplexities across various model variants.
    Lower is better.
    """
    plt.figure(figsize=(10, 5))
    # One series, one color: a different hue per bar would suggest the bars are different kinds of thing
    bars = plt.bar(models, perplexities, color="#2a78d6")
    plt.title(title, fontsize=13, loc="left")
    plt.ylabel("Perplexity (Lower is better)", fontsize=12)
    plt.xticks(rotation=25, ha="right")
    ax = plt.gca()
    ax.grid(axis="x", visible=False)  # bars only need horizontal reference lines
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.6)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + max(perplexities) * 0.01,
            f"{yval:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()
