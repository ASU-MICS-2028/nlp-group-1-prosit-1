"""Ten generated sentences per order, as the plan asks (the notebook shows five). Writes results/generated_samples.md."""
import random
from pathlib import Path

from stage2_ngram import generate, train

train_s = Path("data/splits/90-5-5/train.ewe.txt").read_text(encoding="utf-8").splitlines()
seen = set(train_s)
rng = random.Random(0)
md = "# Generated sentences, ten per order (plain sampling from MLE counts; [COPIED] = appears verbatim in training)\n"
for n in (1, 2, 3, 4, 5):
    m = train(train_s, n)
    md += f"\n## order {n}\n\n"
    for _ in range(10):
        s = generate(m, rng)
        md += f"- {'[COPIED] ' if s in seen else ''}{s[:160]}\n"
    del m
    print(f"order {n} done", flush=True)
Path("results/generated_samples.md").write_text(md, encoding="utf-8")
