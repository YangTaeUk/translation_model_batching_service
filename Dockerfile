# =============================================================================
# Gemma 3 Translation Service - Dockerfile
# =============================================================================
# Base: vLLM OpenAI-compatible server
# Model: google/gemma-3-12b-it (with BitsAndBytes 4-bit quantization)
# Target: NVIDIA GPU with 24GB+ VRAM
# =============================================================================

FROM vllm/vllm-openai:latest

# -----------------------------------------------------------------------------
# Metadata
# -----------------------------------------------------------------------------
LABEL maintainer="YangTaeUk"
LABEL version="1.0.0"
LABEL description="Gemma 3 12B Translation Service with vLLM"
LABEL org.opencontainers.image.source="https://github.com/YangTaeUk/translation_model_batching_service"

# -----------------------------------------------------------------------------
# Install dependencies
# -----------------------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        curl \
        wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# -----------------------------------------------------------------------------
# Environment configuration
# -----------------------------------------------------------------------------
WORKDIR /app

# HuggingFace configuration (override at runtime)
ENV HUGGING_FACE_HUB_TOKEN=""
ENV HF_HOME="/root/.cache/huggingface"
ENV HF_HUB_ENABLE_HF_TRANSFER="1"

# vLLM configuration defaults
ENV VLLM_HOST="0.0.0.0"
ENV VLLM_PORT="8000"

# Performance tuning
ENV OMP_NUM_THREADS="1"
ENV TOKENIZERS_PARALLELISM="false"

# -----------------------------------------------------------------------------
# Health check
# -----------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# -----------------------------------------------------------------------------
# Expose port
# -----------------------------------------------------------------------------
EXPOSE 8000

# -----------------------------------------------------------------------------
# Entrypoint: vLLM serve command
# -----------------------------------------------------------------------------
ENTRYPOINT ["vllm", "serve"]

# -----------------------------------------------------------------------------
# Default arguments (can be overridden via docker run or compose)
# Optimized for RTX 3090 (24GB) with BitsAndBytes 4-bit quantization
# -----------------------------------------------------------------------------
CMD ["google/gemma-3-12b-it", \
     "--dtype", "bfloat16", \
     "--tensor-parallel-size", "1", \
     "--gpu-memory-utilization", "0.90", \
     "--max-model-len", "4096", \
     "--trust-remote-code"]
