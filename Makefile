.DEFAULT_GOAL := help
PYTHON       ?= python
ENV_NAME     ?= llm-compressor-quickstart
VOLUME       ?= llm-compressor-results
MODEL        ?= deepreinforce-ai/Ornith-1.0-9B
OUTPUT_MODEL ?= Ornith-1.0-9B-FP8-DYNAMIC
OUTPUT_MODEL_DIR   ?= /results/$(OUTPUT_MODEL)
SCHEME       ?= FP8_DYNAMIC
HF_REPO      ?= barryke/$(OUTPUT_MODEL)

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

.PHONY: help
help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

.PHONY: setup setup-pip update activate
setup: ## Create the conda env from environment.yml (recommended)
	conda env create -f environment.yml
	@echo ""
	@echo "Done. Activate with:  conda activate $(ENV_NAME)"

setup-pip: ## Create .venv and install via pip (no conda; Windows-first)
	$(PYTHON) -m venv .venv
	@.venv/Scripts/python -m pip install --upgrade pip || .venv/bin/python -m pip install --upgrade pip
	@.venv/Scripts/pip install -r requirements.txt     || .venv/bin/pip install -r requirements.txt
	@echo ""
	@echo "Done. Activate with:  .venv\\Scripts\\activate  (Windows)  or  source .venv/bin/activate  (Linux/macOS)"

update: ## Re-sync an existing conda env after pulling changes
	conda env update -f environment.yml --prune

activate: ## Print the conda activate command
	@echo "conda activate $(ENV_NAME)"

# ---------------------------------------------------------------------------
# Local quantization (needs FP8-capable GPU)
# ---------------------------------------------------------------------------

.PHONY: quantize
quantize: ## Run local FP8 quantization (quantize.py)
	$(PYTHON) quantize.py

# ---------------------------------------------------------------------------
# Modal: quantize on cloud H100
# ---------------------------------------------------------------------------

.PHONY: modal-token quantize-modal quantize-modal-block download-modal
modal-token: ## Authenticate with Modal (one-time)
	modal token new

quantize-modal: ## Quantize on Modal H100: make quantize-modal [MODEL=...] [SCHEME=...]
	modal run modal_quantize.py --model-id $(MODEL) --scheme $(SCHEME)

quantize-modal-block: ## Quantize on Modal H100 with FP8_BLOCK scheme
	modal run modal_quantize.py --model-id $(MODEL) --scheme FP8_BLOCK

download-modal: ## Re-download a checkpoint from the volume (no GPU cost): make download-modal [MODEL=...] [SCHEME=...]
	modal run modal_quantize.py --download-only --model-id $(MODEL) --scheme $(SCHEME)

push-hf-modal: ## Push a quantized checkpoint to the HF Hub: make push-hf-modal REPO=owner/name [MODEL=...] [SCHEME=...]
	modal run modal_push_hf.py --repo-id $(HF_REPO) --model-id $(MODEL) --scheme $(SCHEME)

# ---------------------------------------------------------------------------
# Modal: inference (vLLM runs in the container, no local GPU needed)
# ---------------------------------------------------------------------------

.PHONY: inference-modal inference-modal-prompt serve-modal
inference-modal: ## One-shot generation on Modal GPU
	modal run modal_inference.py --model-dir "$(OUTPUT_MODEL)"

inference-modal-prompt: ## One-shot generation with a custom prompt: make inference-modal-prompt P="..."
	modal run modal_inference.py --prompt "$(P)"

serve-modal: ## Serve a live HTTP endpoint on Modal (billed while up): make serve-modal [OUTPUT_MODEL=...]
	modal run modal_inference.py --serve-mode --model-dir "$(OUTPUT_MODEL)"

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

.PHONY: verify verify-modal
verify: ## Local HF-only check: generation + weight size (no vLLM; needs GPU)
	$(PYTHON) scripts/verify.py

verify-modal: ## Verify on Modal: generation + weight-size vs original
	modal run modal_verify.py --model-name "$(OUTPUT_MODEL)" --model-id "$(MODEL)"
# ---------------------------------------------------------------------------
# Modal volume inspection
# ---------------------------------------------------------------------------

.PHONY: volumes results
volumes: ## List all Modal volumes
	modal volume list

results: ## List files in the results volume (root)
	modal volume ls $(VOLUME) /

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

.PHONY: clean clean-volumes
clean: ## Remove outputs/ and all __pycache__ dirs
	rm -rf outputs
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true

clean-volumes: ## Delete the Modal results volume (irreversible!)
	modal volume delete $(VOLUME) || true
