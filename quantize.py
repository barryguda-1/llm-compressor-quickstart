"""
Quantize a small LLM to FP8 with LLM Compressor.

Defaults to TinyLlama-1.1B-Chat — a tiny model that quantizes in ~30 seconds
on any FP8-capable GPU (Ada Lovelace / Hopper / Blackwell).

Swap MODEL_ID for any Hugging Face causal LM you have access to.
"""

from pathlib import Path

from compressed_tensors.offload import dispatch_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
SCHEME = "FP8_DYNAMIC"
SAVE_DIR = Path("outputs") / f"{MODEL_ID.split('/')[-1]}-FP8-Dynamic"


def main():
    print(f"Loading model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print(f"Applying quantization scheme: {SCHEME}")
    recipe = QuantizationModifier(
        targets="Linear",
        scheme=SCHEME,
        ignore=["lm_head"],
    )
    oneshot(model=model, recipe=recipe)

    print("Running sanity-check generation...")
    dispatch_model(model)
    inputs = tokenizer("Hello, my name is", return_tensors="pt").input_ids.to(
        model.device
    )
    output = model.generate(inputs, max_new_tokens=30)
    print(tokenizer.decode(output[0]))

    print(f"Saving quantized model to: {SAVE_DIR}")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
