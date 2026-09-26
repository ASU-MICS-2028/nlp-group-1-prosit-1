"""Re-score the agriculture models on a leak-free test set (no test question also in train/valid). Scoring only."""
import json
import re
import sys
from pathlib import Path

from stage5_domain import BATCH, MAX_LEN, mlx

A = Path("data/domain/adapters")
RUNS = {("data/models/Qwen2.5-0.5B", "base"): None,
        **{("data/models/Qwen2.5-0.5B", n): A / "Qwen2.5-0.5B/agriculture" / f"{n}@{'1e-05'}" for n in ("lora_r2", "lora_r8", "lora_r32", "dora_r8", "full")},
        ("data/models/SmolLM2-135M", "base"): None, ("data/models/SmolLM2-135M", "lora_r8"): A / "SmolLM2-135M/agriculture/lora_r8@1e-05"}
out = {}
for (model, name), adapter in RUNS.items():
    args = ["--data", "data/domain/agriculture_leakfree", "--test", "--test-batches", "-1", "--batch-size", str(BATCH),
            "--max-seq-length", str(MAX_LEN), "--adapter-path", str(adapter) if adapter else ""]
    ppl = float(re.search(r"Test ppl ([\d.]*\d)", mlx(model, *args)).group(1))
    out[f"{model.split('/')[-1]}/{name}"] = ppl
    print(f"{model.split('/')[-1]:<14} {name:<9} agriculture (leak-free) perplexity {ppl:.3f}", flush=True)
Path("results/agriculture_leakfree.json").write_text(json.dumps(out, indent=1))
