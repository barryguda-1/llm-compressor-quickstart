"""
Optional: compare the original vs quantized model using the HF transformers
pipeline (no vLLM needed):

  1. Token-by-token generation comparison
  2. On-disk weight-size comparison (confirm quantization shrank the model)

Useful sanity check if you have a local GPU but no vLLM. For a cloud version
that needs no local hardware, run `modal run modal_verify.py` instead.
"""

from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

ORIGINAL = "JetBrains/Mellum2-12B-A2.5B-Thinking"
QUANTIZED = Path("outputs") / "Mellum2-12B-A2.5B-Thinking-FP8"
PROMPT = "Is 1024 a power of 2? Explain briefly."


def weight_size_gb(path) -> float:
    """Sum of .safetensors + .bin weight files under a model dir (GB)."""
    total = 0
    for p in Path(path).rglob("*"):
        if p.is_file() and p.suffix in (".safetensors", ".bin"):
            total += p.stat().st_size
    return total / 1e9


def generate(model_id_or_path: str) -> str:
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path, dtype=torch.float16, device_map="auto"
    )
    inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def main():
    print(f"PROMPT: {PROMPT!r}\n")

    print("Loading ORIGINAL model...")
    print("ORIGINAL  ->", generate(ORIGINAL))
    print("\nLoading QUANTIZED model...")
    print("QUANTIZED ->", generate(str(QUANTIZED)))

    orig_dir = snapshot_download(repo_id=ORIGINAL)
    orig_size = weight_size_gb(orig_dir)
    quant_size = weight_size_gb(QUANTIZED)
    shrink = (1 - quant_size / orig_size) * 100 if orig_size else 0.0

    print("\nWeight size on disk:")
    print(f"  ORIGINAL  : {orig_size:.3f} GB")
    print(f"  QUANTIZED : {quant_size:.3f} GB")
    print(f"  SHRINK    : {shrink:.1f}% smaller")


if __name__ == "__main__":
    main()
