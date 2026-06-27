"""
Quantize a Hugging Face causal LM on Modal with an H100 GPU.

Defaults to JetBrains/Mellum2-12B-A2.5B-Thinking, a Qwen3-family MoE model
(64 experts, 8 active), quantized to FP8 (static weights + static activations)
using UltraChat-200K calibration data.

Why Modal? FP8 quantization requires an Ada Lovelace / Hopper / Blackwell GPU.
This script gives you on-demand H100 access without local hardware.

Why llmcompressor >=0.12? MoE models require `load_context()` to linearize
their expert MLPs so the quantizer can actually reach them. Without it, the
quantizer silently skips every expert and only compresses attention layers.

Run:
    make quantize-modal
    modal run modal_quantize.py
    modal run modal_quantize.py --model-id JetBrains/Mellum2-12B-A2.5B-Thinking
    modal run modal_quantize.py --scheme FP8_BLOCK
    modal run modal_quantize.py --num-samples 1024 --max-seq-len 4096

    # Re-download a checkpoint without re-paying for quantization:
    modal run modal_quantize.py --download-only

Cost estimate (Mellum2-12B-A2.5B-Thinking, H100 @ ~$3.50/hr):
    - Model download (~24 GB, cached after first run): ~$0.30
    - Calibration (512 samples x 2048 tokens, 8 experts active): ~$1.50
    - Quantize + save + sanity-check generation:              ~$0.30
    ----------------------------------------------------------: --------
    Total per run:                                            ~$2.10

Setup (one-time):
    pip install modal
    modal token new
"""

import subprocess
import sys
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

GPU = "H100"
TIMEOUT_SECONDS = 180 * 60  # 3 hr hard cap (calibration is slower than dynamic)
RESULTS_VOLUME_NAME = "llm-compressor-results"
MODEL_ID = "JetBrains/Mellum2-12B-A2.5B-Thinking"
SCHEME = "FP8"  # static weights + static activations (calibrated)

# Calibration dataset. UltraChat-200K is the canonical llmcompressor default
# and works well for reasoning/chat models even though it isn't code-specific.
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# MoE ignore patterns for Mellum2 (Qwen3-family). Confirmed by running
# modal_discover.py against the model config:
#   - model.layers.N.mlp.gate -> MellumTopKRouter (must stay full precision,
#     otherwise routing decisions degrade and quality collapses).
#   - lm_head stays full precision (tied to embed_tokens; sensitive).
# NOTE: the expert MLP linears are NOT in this list -- we *want* them
# quantized. They are only reachable at all because of load_context().
IGNORE_PATTERNS = ["lm_head", "re:.*mlp.gate$"]

# Image: lean Debian base + everything LLM Compressor 0.12 needs.
# Pin floors match llmcompressor 0.12's own setup.py constraints.
# torchvision is required even for text-only models: transformers v5's
# monkey_patching.py iterates dir() on every lazy-loaded submodule (including
# vision ones like aria) at from_pretrained time, and any vision import that
# fails aborts the load.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "llmcompressor>=0.12.0",
        "compressed-tensors>=0.17.1",
        "transformers>=5.9.0",
        "accelerate>=1.0.0",
        "datasets>=4.8.4",
        "torchvision",  # required by transformers v5 lazy-import chain
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
            --model-id JetBrains/Mellum2-12B-A2.5B-Thinking --scheme FP8
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
    gpu=GPU,
    timeout=TIMEOUT_SECONDS,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/results": results_vol,
    },
)
def quantize(
    model_id: str,
    scheme: str,
    num_samples: int,
    max_seq_len: int,
) -> dict:
    """Run LLM Compressor on `model_id` and write checkpoint to /results."""
    from datasets import load_dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.utils import load_context
    from compressed_tensors.offload import dispatch_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # 1. Load model under MoE-aware context. load_context() linearizes expert
    #    MLPs so the quantizer can actually see them (otherwise the only
    #    Linear layers visible are attention projections -- confirmed via
    #    modal_discover.py). It also enables disk offload if the model
    #    doesn't fit in RAM, useful for 12B+ checkpoints.
    print(f"Loading model under load_context(): {model_id}")
    with load_context():
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_id)

    # 2. Build the calibration dataset. Canonical llmcompressor pattern from
    #    examples/quantizing_moe/qwen_example.py: apply_chat_template first
    #    (so the model sees inputs shaped like its training distribution),
    #    then tokenize with truncation to max_seq_len.
    print(
        f"Loading calibration data: {DATASET_ID}/{DATASET_SPLIT} "
        f"({num_samples} samples, max_seq_len={max_seq_len})"
    )
    ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
    ds = ds.shuffle(seed=42).select(range(num_samples))

    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
            )
        }

    ds = ds.map(preprocess)

    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=max_seq_len,
            truncation=True,
            add_special_tokens=False,
        )

    ds = ds.map(tokenize, remove_columns=ds.column_names)

    # 3. Configure the recipe. FP8 = per-channel FP8 weights + static FP8
    #    activations (calibrated). QuantizationModifier uses RTN (simple PTQ)
    #    which is near-lossless for FP8 on MoE models. For lower-bit schemes
    #    you'd switch to GPTQModifier or AWQModifier.
    print(f"Applying scheme: {scheme} (ignore={IGNORE_PATTERNS})")
    recipe = QuantizationModifier(
        targets="Linear",
        scheme=scheme,
        ignore=IGNORE_PATTERNS,
    )

    # 4. Run quantization. v0.12 API: dataset=, max_seq_length=,
    #    num_calibration_samples= (older calibration_dataset=/max_seq_len=
    #    names are gone).
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=max_seq_len,
        num_calibration_samples=num_samples,
    )

    # 5. Save BEFORE the sanity-check dispatch. dispatch_model() offloads
    #    layers to CPU, which makes save_pretrained() stall trying to
    #    re-materialize shards one at a time (this was hitting the 30-min
    #    Modal timeout in earlier versions of this script).
    save_name = _save_name(model_id, scheme)
    save_dir = Path("/results") / save_name
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to: {save_dir}")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    # 6. Sanity-check generation. Note: Mellum2-Thinking emits <think>...</think>
    #    blocks before the final answer, so this 30-token sample will look
    #    truncated -- that's expected. Real verification runs in modal_verify.py
    #    with longer max_new_tokens.
    print("Running sanity-check generation...")
    dispatch_model(model)
    inputs = tokenizer("Hello, my name is", return_tensors="pt").input_ids.to(
        model.device
    )
    output = model.generate(inputs, max_new_tokens=30)
    sample = tokenizer.decode(output[0])
    print(f"SAMPLE: {sample}")

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
    model_id: str = MODEL_ID,
    scheme: str = SCHEME,
    num_samples: int = NUM_CALIBRATION_SAMPLES,
    max_seq_len: int = MAX_SEQUENCE_LENGTH,
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

    print(
        f"Submitting job to Modal (GPU={GPU}, model={model_id}, scheme={scheme}, "
        f"num_samples={num_samples}, max_seq_len={max_seq_len})..."
    )
    result = quantize.remote(
        model_id=model_id,
        scheme=scheme,
        num_samples=num_samples,
        max_seq_len=max_seq_len,
    )
    print(f"\nRemote sample generation:\n  {result['sample']}\n")

    print(f"Saved {len(result['files'])} file(s) to volume:")
    for rel_path in result["files"]:
        print(f"  /results/{result['save_name']}/{rel_path}")

    print(
        "\nDone. Checkpoint lives on the Modal volume. To pull a local copy later:"
        "\n  make download-modal"
        "\nNext step: make verify-modal"
    )
