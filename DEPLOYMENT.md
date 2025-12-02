# Deployment Guide - Translation Model Batching Service

이 문서는 Gemma 3 12B 기반 번역 서비스를 외부 클라우드 서버에 배포하는 방법을 설명합니다.

---

## 목차

1. [사전 요구사항](#사전-요구사항)
2. [GitHub Secrets 설정](#github-secrets-설정)
3. [로컬 개발 환경](#로컬-개발-환경)
4. [클라우드 서버 배포](#클라우드-서버-배포)
5. [서비스 관리](#서비스-관리)
6. [모니터링](#모니터링)
7. [문제 해결](#문제-해결)

---

## 사전 요구사항

### GPU 서버 요구사항

| 항목 | 최소 요구사항 | 권장 |
|------|-------------|------|
| GPU | NVIDIA GPU with CUDA | RTX 3090 (24GB) 이상 |
| VRAM | 16GB+ | 24GB+ |
| RAM | 32GB | 64GB |
| 저장공간 | 100GB | 200GB+ (모델 캐시용) |
| OS | Ubuntu 20.04+ | Ubuntu 22.04 |

### 필수 소프트웨어

```bash
# Docker 설치
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Docker Compose 설치
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# NVIDIA Container Toolkit 설치
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 검증

```bash
# Docker 확인
docker --version

# Docker Compose 확인
docker-compose --version

# NVIDIA 런타임 확인
docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu22.04 nvidia-smi
```

---

## GitHub Secrets 설정

GitHub 저장소에서 **Settings > Secrets and variables > Actions**로 이동하여 다음 Secrets를 추가합니다:

| Secret 이름 | 설명 | 예시 |
|------------|------|------|
| `DOCKER_USERNAME` | DockerHub 사용자명 | `yourusername` |
| `DOCKER_PASSWORD` | DockerHub 비밀번호 또는 Access Token | `dckr_pat_xxx` |

> **권장**: DockerHub Access Token 사용 (https://hub.docker.com/settings/security)

---

## 로컬 개발 환경

### 1. 환경 변수 설정

```bash
# .env 파일 생성
cp .env.example .env

# .env 파일 편집
nano .env
```

필수 설정:
```env
HUGGING_FACE_HUB_TOKEN=hf_your_actual_token_here
HF_CACHE_PATH=/path/to/your/hf_cache
```

### 2. 로컬 빌드 및 실행

```bash
# 이미지 빌드 및 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f
```

### 3. 서비스 확인

```bash
# 헬스 체크
curl http://localhost:8000/health

# 모델 정보 확인
curl http://localhost:8000/v1/models

# 테스트 요청
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-3-12b-it",
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "max_tokens": 100
  }'
```

---

## 클라우드 서버 배포

### 방법 1: DockerHub에서 Pull (권장)

GitHub에 푸시하면 자동으로 DockerHub에 이미지가 빌드됩니다.

#### 클라우드 서버에서:

```bash
# 1. 프로젝트 디렉토리 생성
mkdir -p ~/gemma-service
cd ~/gemma-service

# 2. 필요한 파일 다운로드
curl -O https://raw.githubusercontent.com/YangTaeUk/translation_model_batching_service/main/docker-compose.cloud.yml
curl -O https://raw.githubusercontent.com/YangTaeUk/translation_model_batching_service/main/.env.example

# 3. 환경 변수 설정
cp .env.example .env
nano .env
```

`.env` 파일 설정:
```env
HUGGING_FACE_HUB_TOKEN=hf_your_token_here
DOCKERHUB_USERNAME=yangtaeuk
HF_CACHE_PATH=./hf_cache
```

```bash
# 4. 캐시 디렉토리 생성
mkdir -p hf_cache

# 5. 서비스 시작
docker-compose -f docker-compose.cloud.yml up -d

# 6. 로그 확인 (모델 다운로드 진행 상황)
docker-compose -f docker-compose.cloud.yml logs -f
```

### 방법 2: 직접 Pull 및 Run

```bash
# DockerHub에서 이미지 Pull
docker pull yangtaeuk/gemma-3-service:latest

# 컨테이너 실행
docker run -d \
  --name gemma-3-server \
  --runtime=nvidia \
  --gpus all \
  -e HUGGING_FACE_HUB_TOKEN=hf_your_token_here \
  -e HF_HOME=/root/.cache/huggingface \
  -v ./hf_cache:/root/.cache/huggingface \
  -p 8000:8000 \
  --ipc=host \
  yangtaeuk/gemma-3-service:latest \
  google/gemma-3-12b-it \
  --dtype bfloat16 \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --trust-remote-code
```

---

## 서비스 관리

### 기본 명령어

```bash
# 서비스 상태 확인
docker-compose -f docker-compose.cloud.yml ps

# 로그 확인
docker-compose -f docker-compose.cloud.yml logs -f

# 서비스 재시작
docker-compose -f docker-compose.cloud.yml restart

# 서비스 중지
docker-compose -f docker-compose.cloud.yml down

# 이미지 업데이트
docker-compose -f docker-compose.cloud.yml pull
docker-compose -f docker-compose.cloud.yml up -d
```

### 자동 시작 설정 (systemd)

```bash
sudo nano /etc/systemd/system/gemma-service.service
```

```ini
[Unit]
Description=Gemma Translation Service
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/ubuntu/gemma-service
ExecStart=/usr/local/bin/docker-compose -f docker-compose.cloud.yml up -d
ExecStop=/usr/local/bin/docker-compose -f docker-compose.cloud.yml down
TimeoutStartSec=300

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable gemma-service
sudo systemctl start gemma-service
```

---

## 모니터링

### GPU 모니터링

```bash
# 실시간 GPU 상태
watch -n 1 nvidia-smi

# Docker 컨테이너 리소스 사용량
docker stats gemma-3-server
```

### API 엔드포인트

| 엔드포인트 | 설명 |
|-----------|------|
| `GET /health` | 서버 상태 확인 |
| `GET /v1/models` | 로드된 모델 목록 |
| `POST /v1/chat/completions` | 채팅 API (OpenAI 호환) |
| `POST /v1/completions` | 텍스트 완성 API |

### 헬스 체크 스크립트

```bash
#!/bin/bash
# health_check.sh

ENDPOINT="http://localhost:8000/health"
MAX_RETRIES=5
RETRY_INTERVAL=10

for i in $(seq 1 $MAX_RETRIES); do
    if curl -sf "$ENDPOINT" > /dev/null; then
        echo "✅ Service is healthy"
        exit 0
    fi
    echo "⏳ Attempt $i/$MAX_RETRIES - Service not ready, waiting..."
    sleep $RETRY_INTERVAL
done

echo "❌ Service failed health check"
exit 1
```

---

## 문제 해결

### 일반적인 문제

#### 1. GPU를 인식하지 못함

```bash
# NVIDIA 런타임 확인
docker info | grep -i nvidia

# Docker 재시작 후 다시 시도
sudo systemctl restart docker
```

#### 2. 메모리 부족 (CUDA OOM)

```env
# .env 파일에서 메모리 사용률 낮추기
GPU_MEMORY_UTILIZATION=0.85
MAX_MODEL_LEN=2048
```

#### 3. 모델 다운로드 실패

```bash
# HuggingFace 토큰 확인
echo $HUGGING_FACE_HUB_TOKEN

# 캐시 디렉토리 권한 확인
ls -la ./hf_cache

# 수동 로그인 테스트
docker run -it --rm \
  -e HUGGING_FACE_HUB_TOKEN=hf_your_token \
  python:3.10 \
  bash -c "pip install huggingface_hub && huggingface-cli whoami"
```

#### 4. 컨테이너가 시작 후 바로 종료됨

```bash
# 종료 로그 확인
docker logs gemma-3-server

# 리소스 제한 확인
docker inspect gemma-3-server | grep -A 20 "HostConfig"
```

### 로그 레벨 조정

디버깅을 위해 로그 레벨을 높이려면:

```yaml
# docker-compose.cloud.yml command에 추가
--disable-log-requests  # 요청 로깅 비활성화 (성능 향상)
```

---

## 참고 자료

- [vLLM Documentation](https://docs.vllm.ai/)
- [HuggingFace Gemma 3](https://huggingface.co/google/gemma-3-12b-it)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)

---

*마지막 업데이트: 2025-12-02*
