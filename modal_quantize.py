"""
Quantize TinyLlama (or any HF causal LM) on Modal with an H100 GPU.

Why Modal? FP8 quantization requires an Ada Lovelace / Hopper / Blackwell GPU.
This script gives you on-demand H100 access without local hardware.

Run:
    modal run modal_quantize.py
    modal run modal_quantize.py --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0
    modal run modal_quantize.py --scheme FP8_BLOCK

    # Re-download a checkpoint without re-paying for quantization:
    modal run modal_quantize.py --download-only

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
TIMEOUT_SECONDS = 120 * 60  # 2 hr hard cap
RESULTS_VOLUME_NAME = "llm-compressor-results"

# Image: lean Debian base + everything LLM Compressor needs.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "llmcompressor==0.12.0",
        "compressed-tensors==0.17.1",
        "transformers>=5.8.1",
        "torchvision",
        "accelerate>=1.13.0",
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
# Local helpers
# ---------------------------------------------------------------------------


def _save_name(model_id: str, scheme: str) -> str:
    """Build the on-disk directory name for a quantized checkpoint."""
    return f"{model_id.split('/')[-1]}-{scheme.replace('_', '-')}"


def pull_from_volume(save_name: str, output_dir: str = "outputs") -> Path:
    """Download a previously-quantized checkpoint from the Modal results volume.

    Decoupled from `quantize` because the transfer is slow (multi-GB) and you
    often want to re-run it on its own -- e.g. after an interrupted download or
    to pull a checkpoint produced by an earlier run:

        modal run modal_quantize.py --download-only \\
            --model-id EssentialAI/rnj-1-instruct --scheme FP8_DYNAMIC
    """
    # Pass `output_dir` (the parent) -- NOT `output_dir / save_name` -- because
    # `modal volume get SRC DEST` appends the remote dir name to DEST itself.
    # Passing the nested path produced `outputs/<name>/<name>/...`.
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    local_dir = Path(output_dir) / save_name
    print(f"Downloading checkpoint from volume to {local_dir}...")
    # Volume.batch_download was removed in Modal 1.0; the CLI is now the
    # recommended way to download a directory from a Volume. `--force` makes
    # the call idempotent so interrupted downloads can be resumed.
    subprocess.run(
        [
            sys.executable, "-m", "modal", "volume", "get",
            RESULTS_VOLUME_NAME, save_name, str(output_dir),
            "--force",
        ],
        check=True,
    )
    return local_dir


# ---------------------------------------------------------------------------
# Remote function — runs on H100
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("huggingface")],
    gpu=GPU,
    timeout=TIMEOUT_SECONDS,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/results": results_vol,
    },
)
def quantize(model_id: str, scheme: str) -> dict:
    """Run LLM Compressor on `model_id` and write checkpoint to /results."""
    import os
    from compressed_tensors.offload import dispatch_model
    from compressed_tensors.utils import save_mtp_tensors_to_checkpoint
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration  # multimodal class
    import torch,os

    print(f"Loading model: {model_id}")
    token = os.environ["HF_TOKEN"]
    # Load Both text + vision via the multimodal class
    model = Qwen3_5ForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16, token=token)
    processor = AutoProcessor.from_pretrained(model_id, token=token)
    print(f"Applying scheme: {scheme}")
    recipe = QuantizationModifier(
        targets="Linear",
        scheme=scheme,
        ignore=[
            "lm_head",                # large vocab (248k), sensitive
            "re:.*visual.*",          # vision tower -> kept at BF16
            "re:.*linear_attn.*",     # hybrid Gated-DeltaNet projections
        ],
    )
    oneshot(model=model, recipe=recipe)

    # Save BEFORE the sanity-check dispatch. dispatch_model() offloads layers
    # to CPU, which makes save_pretrained() stall trying to re-materialize
    # shards one at a time (this was hitting the 30-min Modal timeout).
    save_name = _save_name(model_id, scheme)
    save_dir = Path("/results") / save_name
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to: {save_dir}")
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)

    ## Carry MTP tensors over from the source checkpoint
    #    (Qwen3_5ForConditionalGeneration excludes them on load).
    #    Some checkpoints (e.g. Ornith) ship no MTP tensors — skip gracefully.
    try:
        save_mtp_tensors_to_checkpoint(source_model=model_id, dest_dir=save_dir)
    except ValueError as e:
        print(f"Skipping MTP copy — source has no MTP tensors: {e}")
    print("Running sanity-check generation...")
    sample = "<sanity check skipped>"
    try:
        dispatch_model(model)
        # Text-only sanity check: use the tokenizer (not the multimodal processor),
        # which would otherwise treat the string as an image source.
        tok = processor.tokenizer
        inputs = tok("Hello, my name is", return_tensors="pt").input_ids.to(model.device)
        output = model.generate(inputs, max_new_tokens=30)
        sample = tok.decode(output[0], skip_special_tokens=True)
        print(f"SAMPLE: {sample}")
    except Exception as e:
        # The sanity check is informational only — never block the checkpoint
        # from being committed (an earlier bug here lost whole quantization runs).
        print(f"SAMPLE: skipped (sanity check failed: {e!r})")

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
    model_id: str = "EssentialAI/rnj-1-instruct",
    scheme: str = "FP8_DYNAMIC",
    output_dir: str = "outputs",
    download_only: bool = False,
):
    save_name = _save_name(model_id, scheme)

    if download_only:
        local_dir = pull_from_volume(save_name, output_dir)
        for f in sorted(local_dir.rglob("*")):
            if f.is_file():
                print(f"  {f}")
        print(f"\nDownloaded to {local_dir}")
        return

    print(f"Submitting job to Modal (GPU=H100, model={model_id}, scheme={scheme})...")
    result = quantize.remote(model_id, scheme)
    print(f"\nRemote sample generation:\n  {result['sample']}\n")

    print(f"Saved {len(result['files'])} file(s) to volume:")
    for rel_path in result["files"]:
        print(f"  /results/{result['save_name']}/{rel_path}")

    print(
        "\nDone. Checkpoint lives on the Modal volume. To pull a local copy later:"
        "\n  make download-modal"
        "\nNext step: make verify-modal"
    )
