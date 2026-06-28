"""
Verify the quantized checkpoint on Modal:

  1. Compare token-by-token generation: original HF model vs quantized checkpoint
  2. Compare on-disk weight size: confirm quantization actually shrank the model

Run:
    modal run modal_verify.py
    modal run modal_verify.py --model-name TinyLlama-1.1B-Chat-v1.0-FP8-BLOCK
    modal run modal_verify.py --prompt "The capital of France is"

Needs the checkpoint to already exist in the results volume — run
`modal run modal_quantize.py` first.
"""

from pathlib import Path

import modal

results_vol = modal.Volume.from_name("llm-compressor-results", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

# transformers loads the compressed-tensors FP8 checkpoint via its integration
# with the `compressed-tensors` package — no vLLM (and no nvcc) needed here.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "transformers>=5.8.1",
        "compressed-tensors==0.17.1",
        "accelerate>=1.0.0",
        "safetensors",
        "sentencepiece",
        "hf-transfer",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App(image=image, name="llm-compressor-quickstart-verify")
GPU = "H100"
@app.function(
    image=image,
    secrets=[modal.Secret.from_name("huggingface")],
    gpu=GPU,
    timeout=10 * 60,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/results": results_vol,
    },
)
def verify(
    model_id: str,
    model_name: str,
    prompt: str,
    max_new_tokens: int,
) -> dict:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    def weight_size_gb(path) -> float:
        """Sum of .safetensors + .bin weight files under a model dir (GB)."""
        total = 0
        for p in Path(path).rglob("*"):
            if p.is_file() and p.suffix in (".safetensors", ".bin"):
                total += p.stat().st_size
        return total / 1e9

    def generate(model_src: str) -> str:
        # Ornith is a multimodal qwen3_5 model. Load it with the SAME class
        # modal_quantize.py uses to SAVE the checkpoint
        # (Qwen3_5ForConditionalGeneration) so the /results/... quantized dir
        # resolves correctly. AutoProcessor is intentionally avoided -- its
        # image-processor auto-detection raises on this repo's
        # preprocessor_config.json; AutoTokenizer is enough for text-only input
        # (matches the quantize sanity-check pattern). The chat template is
        # applied so the reasoning model runs in its trained <think>...</think> format.
        tokenizer = AutoTokenizer.from_pretrained(model_src)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_src, dtype=torch.bfloat16, device_map="auto"
        )
        messages = [{"role": "user", "content": prompt}]
        # Two-step tokenization (matches the model card quickstart): render the
        # chat template to a STRING first, then tokenize. Passing
        # apply_chat_template's tensor output straight into generate confuses
        # its batch-size inference (AttributeError on .shape).
        templated = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(templated, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        # Decode only the newly generated tokens (skip the prompt).
        input_len = inputs["input_ids"].shape[-1]
        generated = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
        del model
        torch.cuda.empty_cache()
        return generated

    # --- size comparison ---
    print(f"Resolving original snapshot for {model_id}...")
    orig_dir = snapshot_download(repo_id=model_id)
    orig_size = weight_size_gb(orig_dir)

    quant_dir = f"/results/{model_name}"
    # Require actual weight files, not just the directory. A prior quantize
    # run that crashed before `results_vol.commit()` (modal_quantize.py:151)
    # can leave a stale snapshot where Path(...).exists() is True but no
    # .safetensors/.bin ever landed -- transformers would then fail later
    # with a confusing "no file named model.safetensors" error.
    weight_files = (
        list(Path(quant_dir).glob("*.safetensors"))
        + list(Path(quant_dir).glob("*.bin"))
        if Path(quant_dir).exists()
        else []
    )
    if not weight_files:
        raise FileNotFoundError(
            f"No weight files (.safetensors / .bin) found in {quant_dir}. "
            f"The quantization either did not complete or was never run. "
            f"Re-run `modal run modal_quantize.py`."
        )
    quant_size = weight_size_gb(quant_dir)

    # --- generation comparison ---
    print("Loading ORIGINAL model...")
    orig_text = generate(model_id)
    print("Loading QUANTIZED model...")
    quant_text = generate(quant_dir)

    # Persist any freshly downloaded HF cache files for next time.
    hf_cache_vol.commit()

    return {
        "prompt": prompt,
        "original_output": orig_text,
        "quantized_output": quant_text,
        "original_size_gb": orig_size,
        "quantized_size_gb": quant_size,
        "shrink_pct": (1 - quant_size / orig_size) * 100 if orig_size else 0.0,
    }


@app.local_entrypoint()
def main(
    model_id: str = "deepreinforce-ai/Ornith-1.0-9B",
    model_name: str = "Ornith-1.0-9B-FP8-DYNAMIC",
    prompt: str = "The capital of Poland and Australia is",
    max_new_tokens: int = 4096,
):
    print(
        f"Submitting verify job to Modal "
        f"(model_id={model_id}, checkpoint={model_name})..."
    )
    r = verify.remote(
        model_id=model_id,
        model_name=model_name,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )

    print(f"\nPROMPT: {r['prompt']!r}\n")
    print(f"ORIGINAL  -> {r['original_output']}")
    print(f"QUANTIZED -> {r['quantized_output']}")
    print("\nWeight size on disk:")
    print(f"  ORIGINAL  : {r['original_size_gb']:.3f} GB")
    print(f"  QUANTIZED : {r['quantized_size_gb']:.3f} GB")
    print(f"  SHRINK    : {r['shrink_pct']:.1f}% smaller")
