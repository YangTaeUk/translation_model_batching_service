# Dockerfile
FROM vllm/vllm-openai:latest

# 1. 기본 패키지 설치
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

# 2. 작업 환경 설정
WORKDIR /app
ENV HUGGING_FACE_HUB_TOKEN=""

# 3. 핵심: 컨테이너 실행 시 'vllm serve' 명령어 고정
ENTRYPOINT ["vllm", "serve"]

# 4. 기본 실행 옵션 (CMD)
# 사용자가 별도 옵션 없이 'docker run'만 해도 이 설정으로 돌아갑니다.
# RTX 3090 (24GB) 환경에 최적화된 기본값입니다.
CMD ["google/gemma-3-12b-it", \
     "--dtype", "half", \
     "--tensor-parallel-size", "1", \
     "--gpu-memory-utilization", "0.95", \
     "--max-model-len", "4096", \
     "--trust-remote-code"]