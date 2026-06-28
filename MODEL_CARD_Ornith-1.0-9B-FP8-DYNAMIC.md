---
base_model: deepreinforce-ai/Ornith-1.0-9B
license: mit
language:
  - en
library_name: transformers
pipeline_tag: image-text-to-text
tags:
  - fp8
  - fp8-dynamic
  - quantized
  - compressed-tensors
  - llm-compressor
  - vllm
  - qwen3_5
  - conversational
  - reasoning
inference: false
---

# barryke/Ornith-1.0-9B-FP8-DYNAMIC

FP8 **dynamic-activation, per-channel-weight** quantization of
[`deepreinforce-ai/Ornith-1.0-9B`](https://huggingface.co/deepreinforce-ai/Ornith-1.0-9B),
produced with [LLM Compressor](https://github.com/vllm-project/llm-compressor)
and stored in the [compressed-tensors](https://github.com/neuralmagic/compressed-tensors)
format that vLLM and Transformers ≥ 5.8.1 load directly.

| | |
|---|---|
| **Base model** | [`deepreinforce-ai/Ornith-1.0-9B`](https://huggingface.co/deepreinforce-ai/Ornith-1.0-9B) (Qwen 3.5 9B, multimodal, reasoning) |
| **Quantization scheme** | `FP8_DYNAMIC` (E4M3 weights, E4M3 activations, dynamic per-token scale) |
| **Weight granularity** | per-channel |
| **Activation granularity** | dynamic per-token |
| **Layers quantized** | all `Linear` layers except those listed below |
| **Layers kept at BF16** | `lm_head`, `re:.*visual.*` (vision tower), `re:.*linear_attn.*` (hybrid Gated-DeltaNet projections) |
| **Calibration data** | **none** — dynamic activations need no calibration set |
| **Framework** | `llmcompressor==0.12.0`, `compressed-tensors==0.17.1` |
| **Quantization hardware** | NVIDIA H100 (via Modal) |
| **License** | MIT (inherited from the base model) |

---

## Why this quantization?

`FP8_DYNAMIC` is the simplest scheme that recovers near-full accuracy on most
LLMs while cutting weight size roughly in half:

- **No calibration dataset** required — activation scales are computed on the
  fly per token at inference time.
- **Per-channel weight scales** avoid the accuracy loss that per-tensor weight
  quantization causes on outlier channels.
- Runs at full speed on Ada Lovelace / Hopper / Blackwell GPUs (compute
  capability ≥ 8.9).

The vision tower and the hybrid Gated-DeltaNet attention projections are
sensitive and left in BF16, matching the recipe used inside the quantization
script. The `lm_head` (≈248k-token vocabulary) is also left in BF16.

---

## Size on disk
| Checkpoint | Weights on disk |
|---|---|
| Original (`deepreinforce-ai/Ornith-1.0-9B`, BF16) | <!-- ORIGINAL_SIZE_GB --> GB |
| This repo (FP8-DYNAMIC) | <!-- QUANTIZED_SIZE_GB --> GB |
| **Reduction** | <!-- SHRINK_PCT --> % smaller |

---

## Quantization recipe

Reproduced verbatim from the Modal run that produced this checkpoint:

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=[
        "lm_head",             # 248k-token vocab, sensitive
        "re:.*visual.*",       # vision tower -> kept at BF16
        "re:.*linear_attn.*",  # hybrid Gated-DeltaNet projections
    ],
)
oneshot(model=model, recipe=recipe)
```

The pipeline (load → quantize → save → sanity-check generate) was orchestrated
on a Modal H100 container. End-to-end cost for a 9B model is on the order of
**~$0.30 per run**.

---

## Verification

Side-by-side generation against the original BF16 checkpoint on the same
prompt (greedy decoding). Both checkpoints produced **byte-identical output**,
confirming FP8-DYNAMIC recovery on this workload:

**Prompt:** `The capital of Poland and Australia is`

```
ORIGINAL  -> Thinking Process:

1.  **Analyze the Request:** The user is asking for the capital of Poland and Australia. This is a factual question with two parts.

2.  **Identify the Capitals:**
    *   Poland: Warsaw (Warszawa).
    *   Australia: Canberra.

3.  **Formulate the Answer:** Combine the two facts into a clear sentence.

4.  **Review for Accuracy:**
    *   Is Warsaw the capital of Poland? Yes.
    *   Is Canberra the capital of Australia? Yes.

5.  **Draft the Response:** "The capital of Poland is Warsaw, and the capital of Australia is Canberra."

6.  **Final Polish:** Keep it concise and direct. "The capital of Poland is Warsaw, and the capital of Australia is Canberra." or simply list them. A sentence is better
for flow.

7.  **Output Generation:** (Matches the drafted response)cw
</think>

The capital of Poland is **Warsaw**, and the capital of Australia is **Canberra**.

QUANTIZED -> Thinking Process:

1.  **Analyze the Request:** The user is asking for the capital of Poland and Australia. This is a factual question with two parts.

2.  **Identify the Capitals:**
    *   Poland: Warsaw (Warszawa).
    *   Australia: Canberra.

3.  **Formulate the Answer:** Combine the two facts into a clear sentence.

4.  **Review for Accuracy:**
    *   Is Warsaw the capital of Poland? Yes.
    *   Is Canberra the capital of Australia? Yes.

5.  **Draft the Response:** "The capital of Poland is Warsaw, and the capital of Australia is Canberra."

6.  **Final Polish:** Keep it concise and direct. "The capital of Poland is Warsaw, and the capital of Australia is Canberra." or simply list them. A sentence is better
for flow.

7.  **Output Generation:** (Matches the drafted response)cw
</think>

The capital of Poland is **Warsaw**, and the capital of Australia is **Canberra**.
```

---

## Usage

### vLLM (recommended for serving)

```bash
vllm serve barryke/Ornith-1.0-9B-FP8-DYNAMIC \
    --served-model-name Ornith-1.0-9B \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.90 \
    --enable-prefix-caching \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    --trust-remote-code
```

The FP8 kernels are picked up automatically — no extra flags needed.

### Hugging Face Transformers

Requires `transformers >= 5.8.1` and `compressed-tensors` installed. The
checkpoint loads via the same multimodal class the base model uses:

```python
import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

model_name = "barryke/Ornith-1.0-9B-FP8-DYNAMIC"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    model_name, dtype=torch.bfloat16, device_map="auto"
)

messages = [{"role": "user", "content": "Write a Python function is_prime(n). Keep it short."}]
templated = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(templated, return_tensors="pt").to(model.device)

out = model.generate(**inputs, max_new_tokens=512, do_sample=True,
                     temperature=0.6, top_p=0.95, top_k=20)
print(tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

### OpenAI-compatible API

Once `vllm serve` is running, any OpenAI SDK works:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="Ornith-1.0-9B",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    temperature=0.6, top_p=0.95,
)
msg = response.choices[0].message
print("reasoning:", getattr(msg, "reasoning_content", None))
print("answer:", msg.content)
```

---

## Hardware & latency notes

- **Inference**: needs an FP8-capable GPU (Ada Lovelace / Hopper / Blackwell,
  i.e. compute capability ≥ 8.9). On older GPUs pick a W4A16 / INT8 quantization instead.
- **KV cache & context**: unchanged from the base model (262k context window).
  The FP8 savings apply to **weights only**; KV cache stays in BF16/FP16.
- **Multimodal**: the vision tower is kept at BF16, so image inputs still work.
  This checkpoint is text+image capable, just like the base.

---

## Limitations

- Same behavior as the base Ornith-1.0-9B (a reasoning model that emits
  `<think>...</think>` before its answer) — quantization introduces only
  negligible output drift, but evaluate on your own workload if exact
  reproduction matters.
- Not calibrated against a specific downstream dataset; dynamic activations
  trade a small amount of peak throughput for zero-shot portability.

---

## Citation

If you use this checkpoint, please cite the **original Ornith model**:

```bibtex
@misc{ornith_9b,
    title = {{Ornith-1.0-9B}: Agentic Coding, Open to All},
    url = {https://deep-reinforce.com/ornith_1_0.html},
    author = {{DeepReinforce Team}},
    year = {2026}
}
```

And reference the quantization toolchain:

```bibtex
@misc{llmcompressor,
    title  = {LLM Compressor},
    author = {Neural Magic},
    url    = {https://github.com/vllm-project/llm-compressor}
}
```
