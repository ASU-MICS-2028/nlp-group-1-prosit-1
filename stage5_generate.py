"""What does domain adaptation change in the text the model writes? Same prompts, base model vs each adapter.
Writes results/stage5_generation.md. Greedy decoding (temperature 0) so the comparison is repeatable."""
import re
import subprocess
import sys
from pathlib import Path

MODEL = "data/models/Qwen2.5-0.5B"
ADAPTERS = {"base model": None, "health adapter (LoRA r8)": "data/domain/adapters/Qwen2.5-0.5B/health/lora_r8@1e-05",
            "agriculture adapter (LoRA r8)": "data/domain/adapters/Qwen2.5-0.5B/agriculture/lora_r8@1e-05"}
PROMPTS = ["The patient was admitted with", "To improve soil fertility, farmers should", "The city of Accra is"]


def gen(prompt, adapter):
    args = [sys.executable, "-m", "mlx_lm", "generate", "--model", MODEL, "--prompt", prompt, "--max-tokens", "45", "--temp", "0", "--ignore-chat-template"]
    if adapter:
        args += ["--adapter-path", adapter]
    out = subprocess.run(args, capture_output=True, text=True).stdout
    body = out.split("==========")[1] if "==========" in out else out
    return " ".join(body.split())


md = "# What the adapted models write\n\nGreedy decoding, 45 new tokens, same prompt to each model. Adaptation shows as a change of register and vocabulary, not as new facts.\n"
for prompt in PROMPTS:
    md += f"\n## Prompt: \"{prompt}\"\n\n"
    for name, adapter in ADAPTERS.items():
        text = gen(prompt, adapter)
        md += f"- **{name}:** {text}\n"
        print(f"[{name}] {prompt} ... {text[:120]}", flush=True)
Path("results/stage5_generation.md").write_text(md, encoding="utf-8")
