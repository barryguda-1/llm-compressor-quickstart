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

# Only transformers + accelerate needed -- we never run a forward pass and
# never import llmcompressor here (layer names come from the HF model class,
# not from llmcompressor's MoE linearization).
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
    cpu=2,                 # CPU-only -- no GPU cost
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

    # Instantiate the model graph WITHOUT weights. This downloads only
    # config.json (a few KB) and builds the module tree in meta tensors.
    print("Instantiating model graph with empty weights (no weight download)...")
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    # Collect every named module -- we care about Linear layers that touch
    # routing (must be ignored) vs. gate_proj/up_proj/down_proj (normal SwiGLU
    # MLP linears that SHOULD be quantized).
    patterns_of_interest = (
        "gate",          # mlp.gate, gate, gate_proj
        "router",        # block_sparse_moe.router, router
        "shared_expert", # mlp.shared_expert_gate, shared_expert
        "MoE",          # any Mixtral/GraniteMoE module class
    )

    matches = []
    all_linear_suffixes = Counter()
    for name, module in model.named_modules():
        cls = type(module).__name__
        if "Linear" in cls:
            parts = name.split(".")
            suffix = ".".join(parts[-2:]) if len(parts) >= 2 else name
            all_linear_suffixes[suffix] += 1
        if any(p.strip().lower() in name.lower() for p in patterns_of_interest):
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

    # Build candidate ignore patterns. Skip gate_proj/up_proj/down_proj --
    # these are normal SwiGLU MLP linears, not routing modules.
    gate_like, router_like, shared_expert_like = [], [], []
    for tmpl, _m in seen_templates.items():
        if tmpl.endswith("gate_proj") or tmpl.endswith("up_proj") or tmpl.endswith("down_proj"):
            continue
        if "router" in tmpl.lower():
            router_like.append(tmpl)
        elif "shared_expert_gate" in tmpl.lower():
            shared_expert_like.append(tmpl)
        elif tmpl.endswith(".gate") or ".gate." in tmpl:
            gate_like.append(tmpl)

    def to_regex(templates: list[str]) -> list[str]:
        out = []
        for t in templates:
            anchor = ".".join(t.split(".")[-2:]).replace("N", r"\d+")
            out.append(f"re:.*{anchor}$")
        return sorted(set(out))

    suggested_ignore = ["lm_head"]
    suggested_ignore += to_regex(gate_like)
    suggested_ignore += to_regex(router_like)
    suggested_ignore += to_regex(shared_expert_like)

    print("\n========== SUGGESTED IGNORE LIST ==========")
    for p in suggested_ignore:
        print(f"  {p!r},")

    hf_cache_vol.commit()

    return {
        "model_id": model_id,
        "model_type": getattr(config, "model_type", "?"),
        "architectures": getattr(config, "architectures", "?"),
        "routing_modules": [
            {"template": t, "class": m["class"]} for t, m in sorted(seen_templates.items())
        ],
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
