# Mellum2-12B-A2.5B-Thinking FP8 Quantization — Execution Plan

**Status:** Approved, awaiting plan-mode exit to execute
**Target:** `barryke/Mellum2-12B-A2.5B-Thinking-FP8` (FP8 static weights + static activations, calibrated)
**Runner:** Modal H100

## Locked decisions

| Decision | Value |
|---|---|
| Scheme | `FP8` (per-channel weights + **static** activations, calibrated) |
| Calibration data | `HuggingFaceH4/ultrachat_200k`, split `train_sft`, 512 samples, seq_len 2048 |
| Library | `llmcompressor >=0.12.0` (required for `load_context` + MoE linearization) |
| Layer-name inspection | Yes, run discovery first |
| Version scope | Bump everywhere (requirements.txt + environment.yml + 4 Modal images) |
| Runner | Modal H100 |
| Output repo | `barryke/Mellum2-12B-A2.5B-Thinking-FP8` |

## Dependency upgrade cascade (per llmcompressor 0.12 setup.py)

| Package | Current | New | Files affected |
|---|---|---|---|
| `llmcompressor` | `==0.11.0` | `>=0.12.0` | requirements.txt, environment.yml, modal_quantize.py |
| `compressed-tensors` | `==0.16.0` | `>=0.17.1` | requirements.txt, environment.yml, modal_quantize.py, modal_verify.py |
| `transformers` | `>=4.45.0` | `>=5.9.0` | requirements.txt, environment.yml, all 4 Modal images |
| `torch` | (transitive) | `>=2.10.0` | environment.yml |
| `datasets` | not pinned | `>=4.8.4` | requirements.txt, environment.yml, modal_quantize.py |
| `vllm` (Modal only) | `>=0.6.0` | `>=0.10.0` | modal_inference.py |

## Execution phases

### Phase 1: Discovery (CPU-only, ~$0.01)
Create `modal_discover.py` with empty-weights instantiation to print all gate/router layer names and suggest the exact `ignore` pattern. Content saved below in this plan file. Run via `modal run modal_discover.py` and pause for user review before continuing.

### Phase 2: Dependency bumps
Update `requirements.txt`, `environment.yml`, and the 4 Modal image `pip_install` blocks. Verify local `pip install -r requirements.txt` resolves.

### Phase 3: Rewrite `modal_quantize.py`
- New defaults: `MODEL_ID = "JetBrains/Mellum2-12B-A2.5B-Thinking"`, `SCHEME = "FP8"`
- Add `DATASET_ID`, `DATASET_SPLIT`, `NUM_CALIBRATION_SAMPLES = 512`, `MAX_SEQUENCE_LENGTH = 2048`
- Bump `TIMEOUT_SECONDS` 120 → 180 min
- Add `datasets>=4.8.4` to image
- Use `load_context()` for model loading
- Build calibration dataset with canonical preprocess + tokenize pattern
- Use new `oneshot(dataset=, max_seq_length=, num_calibration_samples=)` API
- Final `ignore` list = output from Phase 1
- Thread `num_samples` / `max_seq_len` through entrypoint

### Phase 4: Makefile + cleanup
- Update `MODEL`, `SCHEME`, `OUTPUT_MODEL` defaults
- Add `NUM_SAMPLES ?= 512`, `MAX_SEQ_LEN ?= 2048` knobs
- Update `quantize-modal` recipe to pass new args
- Drop or repoint obsolete `quantize-modal-block`
- Delete stale duplicate `modal_quantize-fp8.py`

### Phase 5: `modal_verify.py`
- Mellum defaults (`model_id`, `model_name`)
- `max_new_tokens` 20 → 512 (to accommodate `<think>` traces)
- Reasoning-appropriate prompt

### Phase 6: `modal_inference.py`
- Mellum defaults (`MODEL_NAME`, `MODEL_DIR`, `PROMPT`)
- **Fix chat-template bug**: wrap prompt via `tokenizer.apply_chat_template(..., add_generation_prompt=True, tokenize=True, return_tensors="pt")` before `llm.generate(...)`
- `vllm>=0.10.0`
- Pass `reasoning_parser="qwen3"` to `LLM(...)`
- Bump `max_tokens` (50 → 512), temperature (0.0 → 0.6) per Mellum card

### Phase 7: `modal_push_hf.py`
- Update `MODEL_ID` / `SCHEME` defaults

### Phase 8: `MODEL_CARD.md` (full rewrite)
- Front matter: `base_model: JetBrains/Mellum2-12B-A2.5B-Thinking`, mellum/moe tags
- Scheme: FP8 static, calibration = UltraChat200K 512 × 2048
- Architecture: 28 layers, hidden 2304, MoE intermediate 896, **64 experts / 8 active**, sliding window 1024, vocab 98,304, GQA 32Q/4KV
- vLLM serve snippet with `--reasoning-parser qwen3 --max-model-len 131072`
- Mellum2 Thinking benchmark values from HF card

### Phase 9: `README.md` (light touch)
- Update "swap the model" example to Mellum2
- Add Mellum2 calibration note alongside TinyLlama narrative
- Refresh cost table

### Phase 10: `scripts/verify.py`
- Update `ORIGINAL` / `QUANTIZED` / `PROMPT` constants (optional, Modal-only user)

### Phase 11: Run quantize
`make quantize-modal` — ~30-45 min on H100, ~$2

### Phase 12: Verify + push
`make verify-modal && make push-hf-modal`

---

## Discovery script content (Phase 1)

Save this to `modal_discover.py`:

```python
"""
Phase 1 discovery: inspect JetBrains/Mellum2-12B-A2.5B-Thinking module names
to confirm the exact MoE gate/router ignore pattern for the quantization recipe.

Uses empty-weights instantiation (accelerate.init_empty_weights) so only
config.json is downloaded -- no 24 GB weight pull, no GPU needed. Essentially
free (~$0.01 for a few seconds of CPU time).

Run:
    modal run modal_discover.py
    modal run modal_discover.py --model-id <org>/<model>
"""

import re
from collections import Counter

import modal

MODEL_ID = "JetBrains/Mellum2-12B-A2.5B-Thinking"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "transformers>=5.9.0",
        "accelerate>=1.0.0",
        "hf-transfer",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App(image=image, name="llm-compressor-quickstart-discover")
hf_cache_vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)


@app.function(
    image=image,
    cpu=2,
    timeout=5 * 60,
    volumes={"/root/.cache/huggingface": hf_cache_vol},
)
def discover(model_id: str) -> dict:
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    print(f"Downloading config.json for {model_id}...")
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    print(f"model_type={getattr(config, 'model_type', '?')}")
    print(f"architectures={getattr(config, 'architectures', '?')}")

    print("Instantiating model graph with empty weights (no weight download)...")
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    patterns_of_interest = ("gate", "router", "shared_expert", "MoE")

    matches = []
    all_linear_suffixes = Counter()
    for name, module in model.named_modules():
        cls = type(module).__name__
        if "Linear" in cls:
            parts = name.split(".")
            suffix = ".".join(parts[-2:]) if len(parts) >= 2 else name
            all_linear_suffixes[suffix] += 1
        if any(p.lower() in name.lower() for p in patterns_of_interest):
            matches.append({"name": name, "class": cls})

    def templatize(name: str) -> str:
        return re.sub(r"\.\d+\.", ".N.", name)

    seen_templates = {}
    for m in matches:
        tmpl = templatize(m["name"])
        if tmpl not in seen_templates:
            seen_templates[tmpl] = m

    print("\n========== ROUTING-RELATED MODULES ==========")
    for tmpl, m in sorted(seen_templates.items()):
        print(f"  {tmpl}  ->  {m['class']}")

    print("\n========== ALL LINEAR SUFFIXES (top 20) ==========")
    for suffix, count in all_linear_suffixes.most_common(20):
        print(f"  {count:5d} x  ...{suffix}")

    gate_like, router_like, shared_expert_like = [], [], []
    for tmpl, m in seen_templates.items():
        if tmpl.endswith("gate_proj") or tmpl.endswith("up_proj") or tmpl.endswith("down_proj"):
            continue
        if "router" in tmpl.lower():
            router_like.append(tmpl)
        elif "shared_expert_gate" in tmpl.lower():
            shared_expert_like.append(tmpl)
        elif tmpl.endswith(".gate") or ".gate." in tmpl:
            gate_like.append(tmpl)

    def to_regex(templates):
        out = []
        for t in templates:
            anchor = ".".join(t.split(".")[-2:]).replace("N", r"\d+")
            out.append(f"re:.*{anchor}$")
        return sorted(set(out))

    suggested_ignore = ["lm_head"] + to_regex(gate_like) + to_regex(router_like) + to_regex(shared_expert_like)

    print("\n========== SUGGESTED IGNORE LIST ==========")
    for p in suggested_ignore:
        print(f"  {p!r},")

    hf_cache_vol.commit()

    return {
        "model_id": model_id,
        "model_type": getattr(config, "model_type", "?"),
        "architectures": getattr(config, "architectures", "?"),
        "routing_modules": [{"template": t, "class": m["class"]} for t, m in sorted(seen_templates.items())],
        "linear_suffixes": dict(all_linear_suffixes.most_common(30)),
        "suggested_ignore": suggested_ignore,
    }


@app.local_entrypoint()
def main(model_id: str = MODEL_ID):
    print(f"Submitting discovery job for {model_id} (CPU-only, no GPU cost)...")
    r = discover.remote(model_id)
    print("\n" + "=" * 60)
    print(f"Model:        {r['model_id']}")
    print(f"model_type:   {r['model_type']}")
    print(f"architectures:{r['architectures']}")
    print("=" * 60)
    print("\nRouting-related modules found:")
    for m in r["routing_modules"]:
        print(f"  {m['template']}  ->  {m['class']}")
    print("\nSuggested ignore list for QuantizationModifier:")
    for p in r["suggested_ignore"]:
        print(f"  {p!r},")
```

## Expected cost & timeline
| Phase | Cost | Time |
|---|---|---|
| 1 Discovery | ~$0.01 | 2 min |
| 2-10 Code changes | $0 | 30 min dev time |
| 11 Quantize | ~$2.00 | 30-45 min H100 |
| 12 Verify + push | ~$0.50 | 10 min |
| **Total** | **~$2.50** | ~1.5 hr wall clock |

## Risks
1. Transformers v5 is a major upgrade — may clash with other projects in shared conda env. Mitigation: Modal images are isolated (safe); local env is the only risk.
2. vLLM version unverified — needs ≥0.10 to load compressed-tensors 0.17 + transformers v5 + qwen3 parser. Will discover at verify time.
3. Existing `outputs/granite-4.1-8b-FP8-DYNAMIC` sample produced on 0.11.0 — won't be re-producible on new stack but the checkpoint still loads (it's just a serialized format).
4. Cost ~15× higher than TinyLlama ($0.15 → ~$2/run). Still cheap but worth knowing if iterating.
