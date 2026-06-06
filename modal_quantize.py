"""
Quantize TinyLlama (or any HF causal LM) on Modal with an H100 GPU.

Why Modal? FP8 quantization requires an Ada Lovelace / Hopper / Blackwell GPU.
This script gives you on-demand H100 access without local hardware.

Run:
    modal run modal_quantize.py
    modal run modal_quantize.py --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0
    modal run modal_quantize.py --scheme FP8_BLOCK

Cost estimate (TinyLlama, H100 @ ~$3.50/hr):
    - Model download (~2 GB):  ~$0.05
    - Quantization (~30 s):    ~$0.05
    - Sanity-check generation: ~$0.01
    - Volume storage (~2 GB):  ~$0.02/month
    -----------------------    --------
    Total:                     ~$0.15 per run

Setup (one-time):
    pip install modal
    modal token new
"""

import subprocess
import sys
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Modal resources
# ---------------------------------------------------------------------------

GPU = "H100"
TIMEOUT_SECONDS = 30 * 60  # 30 min hard cap
RESULTS_VOLUME_NAME = "llm-compressor-results"

# Image: lean Debian base + everything LLM Compressor needs.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "llmcompressor==0.11.0",
        "compressed-tensors==0.16.0",
        "transformers>=4.45.0",
        "accelerate>=1.0.0",
        "safetensors",
        "sentencepiece",
        "hf-transfer",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App(image=image, name="llm-compressor-quickstart")
results_vol = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)


# ---------------------------------------------------------------------------
# Remote function — runs on H100
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT_SECONDS,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/results": results_vol,
    },
)
def quantize(model_id: str, scheme: str) -> dict:
    """Run LLM Compressor on `model_id` and write checkpoint to /results."""
    from compressed_tensors.offload import dispatch_model
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f"Applying scheme: {scheme}")
    recipe = QuantizationModifier(
        targets="Linear",
        scheme=scheme,
        ignore=["lm_head"],
    )
    oneshot(model=model, recipe=recipe)

    print("Running sanity-check generation...")
    dispatch_model(model)
    inputs = tokenizer("Hello, my name is", return_tensors="pt").input_ids.to(
        model.device
    )
    output = model.generate(inputs, max_new_tokens=30)
    sample = tokenizer.decode(output[0])
    print(f"SAMPLE: {sample}")

    save_name = f"{model_id.split('/')[-1]}-{scheme.replace('_', '-')}"
    save_dir = Path("/results") / save_name
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to: {save_dir}")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    # Persist the volume so the local entrypoint can read it back.
    hf_cache_vol.commit()
    results_vol.commit()

    files = sorted(
        str(p.relative_to(save_dir)) for p in save_dir.rglob("*") if p.is_file()
    )
    return {"save_name": save_name, "files": files, "sample": sample}


# ---------------------------------------------------------------------------
# Local entrypoint — orchestrates the remote run and downloads results
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    scheme: str = "FP8_DYNAMIC",
    output_dir: str = "outputs",
):
    print(f"Submitting job to Modal (GPU=H100, model={model_id}, scheme={scheme})...")
    result = quantize.remote(model_id, scheme)
    print(f"\nRemote sample generation:\n  {result['sample']}\n")

    local_dir = Path(output_dir) / result["save_name"]
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(result['files'])} files to {local_dir}...")

    # Pull the entire checkpoint directory off the volume via the Modal CLI.
    # (Volume.batch_download was removed in Modal 1.0; the CLI is now the
    # recommended way to download a directory from a Volume.)
    subprocess.run(
        [
            sys.executable, "-m", "modal", "volume", "get",
            RESULTS_VOLUME_NAME, result["save_name"], str(local_dir),
        ],
        check=True,
    )

    for rel_path in result["files"]:
        print(f"  {local_dir / rel_path}")

    print("\nDone. Next step: modal run modal_inference.py")
