"""
Verify the quantized checkpoint on Modal using vLLM:

  1. Compare token-by-token generation: original BF16 vs FP8 quantized
  2. Compare on-disk weight size: confirm quantization actually shrank the model

Why vLLM and not transformers? Mellum2 is a MoE, and transformers v5's grouped_mm
expert kernel rejects FP8 inputs with `RuntimeError: Expected mat_a to be
Float32, BFloat16 or Float16 matrix, got Float8_e4m3fn`. vLLM has proper FP8
MoE kernels via the compressed-tensors integration.

Run:
    make verify-modal
    modal run modal_verify.py
    modal run modal_verify.py --model-name Mellum2-12B-A2.5B-Thinking-FP8
    modal run modal_verify.py --prompt "Is 1024 a power of 2?"

Needs the checkpoint to already exist in the results volume -- run
`make quantize-modal` first.
"""

import re
from pathlib import Path

import modal

results_vol = modal.Volume.from_name("llm-compressor-results", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

# Same image as modal_inference.py: vLLM's V1 engine needs nvcc/cuDNN for
# torch.compile kernel JIT, and vLLM >=0.20 is required for transformers v5
# support (needed for Mellum2). We deliberately do NOT pin compressed-tensors
# or transformers here -- vLLM has its own strict pins (compressed-tensors
# ==0.17.0 for vllm 0.23) and the checkpoint format is file-compatible with
# the 0.17.1 written by llmcompressor 0.12.
#
# VLLM_MOE_BACKEND=TRITON avoids flashinfer's CUTLASS JIT (which takes 5-10
# min on first run for H100 FP8 MoE kernels). Triton JITs in ~10 sec. For
# production inference you'd remove this and let vLLM pick FLASHINFER_CUTLASS
# (faster at runtime, slower first compile); for verify we want fast feedback.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install(
        "vllm>=0.20.0",
        "accelerate>=1.0.0",
        "torchvision",  # required by transformers v5 lazy-import chain
        "hf-transfer",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "VLLM_MOE_BACKEND": "TRITON",
    })
)

app = modal.App(image=image, name="llm-compressor-quickstart-verify")
GPU = "H100"

THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text: str) -> str:
    """Remove <think>...</think> reasoning traces for a cleaner side-by-side diff."""
    return THINK_PATTERN.sub("", text).strip()


@app.function(
    image=image,
    gpu=GPU,
    timeout=20 * 60,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/results": results_vol,
    },
)
def generate_with_vllm(
    model_src: str,
    prompt: str,
    max_new_tokens: int,
    is_local_path: bool,
) -> str:
    """Load a model in vLLM and generate a single response.

    `is_local_path=True` for the /results checkpoint; False for an HF repo id.
    """
    import time
    from vllm import LLM, SamplingParams

    print(f"[{time.strftime('%H:%M:%S')}] Loading vLLM model: {model_src}", flush=True)
    t0 = time.time()
    llm = LLM(
        model=model_src,
        dtype="auto",
        # NOTE: enforce_eager=True was causing very slow first-token latency
        # on the FP8 MoE (CUTLASS FP8 kernel JIT happens per-call in eager
        # mode). Letting vLLM use torch.compile via enforce_eager=False gives
        # a longer warmup but a much faster actual generate() call.
        enforce_eager=False,
        reasoning_parser="qwen3",
        max_model_len=8192,  # cap for verify (avoids huge KV alloc for 128K ctx)
    )
    print(f"[{time.strftime('%H:%M:%S')}] Model loaded in {time.time()-t0:.1f}s", flush=True)

    messages = [{"role": "user", "content": prompt}]
    sampling = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=max_new_tokens,
    )
    # llm.chat() applies the chat template for us -- safer than calling
    # apply_chat_template ourselves (return type varies across transformers
    # versions).
    print(f"[{time.strftime('%H:%M:%S')}] Starting generation...", flush=True)
    t1 = time.time()
    outputs = llm.chat(messages=[messages], sampling_params=sampling)
    print(f"[{time.strftime('%H:%M:%S')}] Generation took {time.time()-t1:.1f}s", flush=True)
    out = outputs[0].outputs[0]
    reasoning = getattr(out, "reasoning_text", None) or ""
    answer = out.text
    combined = (reasoning + "\n\n" + answer).strip() if reasoning else answer

    # Free GPU memory before returning (the second model load needs it).
    del llm
    import torch
    torch.cuda.empty_cache()
    return combined


@app.function(
    image=image,
    cpu=2,
    timeout=5 * 60,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/results": results_vol,
    },
)
def weight_size_comparison(model_id: str, model_name: str) -> dict:
    """Standalone CPU-only size check. Cheap -- no GPU, no model load."""
    from huggingface_hub import snapshot_download

    def weight_size_gb(path) -> float:
        total = 0
        for p in Path(path).rglob("*"):
            if p.is_file() and p.suffix in (".safetensors", ".bin"):
                total += p.stat().st_size
        return total / 1e9

    orig_dir = snapshot_download(repo_id=model_id)
    orig_size = weight_size_gb(orig_dir)

    quant_dir = f"/results/{model_name}"
    weight_files = (
        list(Path(quant_dir).glob("*.safetensors")) + list(Path(quant_dir).glob("*.bin"))
        if Path(quant_dir).exists()
        else []
    )
    if not weight_files:
        raise FileNotFoundError(
            f"No weight files in {quant_dir}. Re-run `make quantize-modal`."
        )
    quant_size = weight_size_gb(quant_dir)

    hf_cache_vol.commit()

    return {
        "original_size_gb": orig_size,
        "quantized_size_gb": quant_size,
        "shrink_pct": (1 - quant_size / orig_size) * 100 if orig_size else 0.0,
    }


@app.local_entrypoint()
def main(
    model_id: str = "JetBrains/Mellum2-12B-A2.5B-Thinking",
    model_name: str = "Mellum2-12B-A2.5B-Thinking-FP8",
    prompt: str = "Is 1024 a power of 2? Explain your reasoning briefly.",
    max_new_tokens: int = 1024,
    skip_original: bool = False,
):
    quant_dir = f"/results/{model_name}"

    print(f"Step 1/3: Size comparison (CPU-only, ~$0.01)...")
    sizes = weight_size_comparison.remote(
        model_id=model_id, model_name=model_name
    )

    if skip_original:
        print("\n[sKIP_ORIGINAL=true] Skipping BF16 baseline generation.")
        orig_text = "<skipped>"
    else:
        print(f"\nStep 2/3: Generating with ORIGINAL ({model_id}) on vLLM H100...")
        orig_text = generate_with_vllm.remote(
            model_src=model_id,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            is_local_path=False,
        )

    print(f"\nStep 3/3: Generating with QUANTIZED ({quant_dir}) on vLLM H100...")
    quant_text = generate_with_vllm.remote(
        model_src=quant_dir,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        is_local_path=True,
    )

    print("\n" + "=" * 70)
    print(f"PROMPT: {prompt!r}")
    print("=" * 70)
    if not skip_original:
        print(f"\nORIGINAL (BF16, stripped of <think>):")
        print(f"  {strip_think(orig_text)}")
    print(f"\nQUANTIZED (FP8, stripped of <think>):")
    print(f"  {strip_think(quant_text)}")
    print("\n" + "=" * 70)
    print("Weight size on disk:")
    print(f"  ORIGINAL  : {sizes['original_size_gb']:.3f} GB")
    print(f"  QUANTIZED : {sizes['quantized_size_gb']:.3f} GB")
    print(f"  SHRINK    : {sizes['shrink_pct']:.1f}% smaller")
    print("=" * 70)
    print(
        "\nNotes:"
        "\n  - Sampling: temperature=0.6, top_p=0.95, top_k=20 (Mellum2 card)."
        "\n  - <think>...</think> blocks stripped from the diff above for readability."
        "\n  - vLLM is required: transformers v5 can't run FP8 MoE inference"
        "\n    (grouped_mm kernel rejects Float8_e4m3fn)."
    )
