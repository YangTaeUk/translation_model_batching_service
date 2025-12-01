# vLLM 동시처리 성능 테스트 및 GPU 스펙 산정 계획

**작성일**: 2025-11-26
**목적**: RTX 3090에서 vLLM 동시처리 한계 측정 및 비양자화 Gemma 3 12B 50개 동시요청 GPU 스펙 산정

---

## 1. 현재 환경 분석

### 1.1 하드웨어 사양
| 항목 | 사양 |
|------|------|
| GPU | NVIDIA GeForce RTX 3090 |
| VRAM | 24GB GDDR6X |
| Memory Bandwidth | 936 GB/s |
| CUDA Cores | 10,496 |
| Compute Capability | 8.6 |

### 1.2 현재 vLLM 설정
```yaml
Model: google/gemma-3-12b-it
dtype: half (FP16)
tensor-parallel-size: 1
gpu-memory-utilization: 0.95
max-model-len: 4096
```

### 1.3 메모리 이론적 분석

#### Gemma 3 12B 모델 메모리 요구량
| 정밀도 | 모델 가중치 크기 |
|--------|-----------------|
| FP32 | 48GB |
| FP16 (half) | 24GB |
| INT8 | 12GB |
| INT4 | 6GB |

#### KV Cache 계산 (추정)
- Gemma 3 12B 아키텍처: ~36 layers, 16 heads, head_dim 128
- Per-token KV cache (FP16): `2 × 36 × 16 × 128 × 2 = 294,912 bytes ≈ 0.28MB`
- Max context 4096 tokens: `0.28MB × 4096 = 1.15GB per request`

#### RTX 3090 동시처리 이론적 한계
```
사용 가능 VRAM: 24GB × 0.95 = 22.8GB
모델 가중치 (FP16): 24GB → 불가

∴ 현재 설정이 동작하려면:
   - 자동 양자화 적용 (bitsandbytes 등)
   - 또는 AWQ/GPTQ 양자화 모델 사용 중
   - 검증 필요
```

---

## 2. 테스트 계획

### Phase 1: 환경 검증 (사전 분석)

#### 1.1 모델 로드 상태 확인
```bash
# vLLM 서버 실행 후
curl http://localhost:8000/v1/models
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

**확인 항목:**
- [ ] 실제 로드된 모델명
- [ ] 실제 메모리 사용량
- [ ] 양자화 적용 여부

#### 1.2 서버 상태 확인
```bash
curl http://localhost:8000/health
curl http://localhost:8000/metrics  # Prometheus metrics
```

---

### Phase 2: 단일 요청 기준선 측정

#### 2.1 테스트 조건
| 항목 | 값 |
|------|-----|
| Input tokens | 100 |
| Max output tokens | 100 |
| Temperature | 0.7 |
| 반복 횟수 | 10회 |

#### 2.2 측정 메트릭
- **TTFT** (Time to First Token): 첫 토큰 생성까지 시간
- **TPOT** (Time Per Output Token): 평균 토큰 생성 시간
- **Total Latency**: 전체 응답 시간
- **Throughput**: tokens/second

---

### Phase 3: 동시 요청 스트레스 테스트

#### 3.1 테스트 매트릭스

| 테스트 ID | 동시 요청 수 | Input Tokens | Max Output | 목적 |
|-----------|-------------|--------------|------------|------|
| T1 | 1 | 100 | 100 | Baseline |
| T2 | 2 | 100 | 100 | 기본 병렬 |
| T3 | 4 | 100 | 100 | 중간 부하 |
| T4 | 8 | 100 | 100 | 고부하 |
| T5 | 16 | 100 | 100 | 스트레스 |
| T6 | 32 | 100 | 100 | 한계 테스트 |
| T7 | 4 | 500 | 200 | 긴 컨텍스트 |
| T8 | 8 | 500 | 200 | 긴 컨텍스트 고부하 |
| T9 | 16 | 500 | 200 | 긴 컨텍스트 한계 |

#### 3.2 각 테스트 실행 절차
1. GPU 메모리 초기화 확인
2. Warm-up 요청 3회 실행
3. 테스트 실행 (3회 반복)
4. 결과 기록 (latency, throughput, memory)
5. 30초 쿨다운

#### 3.3 성공/실패 기준
| 상태 | 조건 |
|------|------|
| ✅ 성공 | 모든 요청 정상 응답, Latency < 60s |
| ⚠️ 경고 | Latency > 2× baseline |
| ❌ 실패 | OOM, Timeout, Server Error |

---

### Phase 4: 한계점 분석

#### 4.1 수집 데이터
```json
{
  "test_id": "T4",
  "concurrent_requests": 8,
  "input_tokens": 100,
  "max_output_tokens": 100,
  "results": {
    "avg_ttft_ms": 0,
    "avg_tpot_ms": 0,
    "avg_total_latency_ms": 0,
    "throughput_tokens_per_sec": 0,
    "peak_memory_mb": 0,
    "success_rate": 0,
    "errors": []
  }
}
```

#### 4.2 분석 항목
1. **OOM 발생 지점**: 몇 개 동시 요청에서 메모리 부족?
2. **Latency 급증 지점**: 몇 개부터 2배 이상 증가?
3. **Throughput 포화점**: 최대 tokens/sec 달성 지점
4. **권장 동시 요청 수**: 안정적 운영 가능 한계

---

## 3. 비양자화 Gemma 3 12B - 50개 동시요청 GPU 산정

### 3.1 메모리 요구량 계산

#### 공식
```
총 VRAM = 모델 가중치 + KV Cache + 활성화 메모리 + 오버헤드
```

#### 상세 계산 (FP16 비양자화)
| 항목 | 계산식 | 예상값 |
|------|--------|--------|
| 모델 가중치 | 12B × 2 bytes | 24GB |
| KV Cache (50 req × 4096 ctx) | 50 × 1.15GB | 57.5GB |
| 활성화 메모리 | ~10% overhead | 8GB |
| 시스템 오버헤드 | ~5% | 5GB |
| **총합** | | **~94.5GB** |

### 3.2 GPU 후보 분석

| GPU | VRAM | 예상 동시처리 | 비용(참고) | 권장도 |
|-----|------|--------------|-----------|--------|
| RTX 3090 | 24GB | 3-5개 | $1,500 | ❌ |
| RTX 4090 | 24GB | 3-5개 | $2,000 | ❌ |
| A6000 | 48GB | 10-15개 | $4,500 | ⚠️ |
| A100 40GB | 40GB | 8-12개 | $10,000 | ⚠️ |
| A100 80GB | 80GB | 25-35개 | $15,000 | ⚠️ |
| H100 80GB | 80GB | 30-45개 | $30,000 | ⚠️ |
| **2× A100 80GB (TP=2)** | 160GB | **50개+** | $30,000 | ✅ |
| **4× A100 40GB (TP=4)** | 160GB | **50개+** | $40,000 | ✅ |
| **2× H100 80GB (TP=2)** | 160GB | **50개+** | $60,000 | ✅✅ |

### 3.3 권장 사양

#### 최소 사양 (50개 동시요청)
```yaml
Option A: Multi-GPU Setup
  GPUs: 2× NVIDIA A100 80GB
  Total VRAM: 160GB
  Tensor Parallel Size: 2
  예상 비용: ~$30,000

Option B: Cloud Alternative (AWS/GCP)
  Instance: p4d.24xlarge (8× A100 40GB)
  또는: A100 80GB × 2 인스턴스
  시간당 비용: ~$32/hour
```

#### 권장 사양 (50개 동시요청 + 여유)
```yaml
Option A: 고성능 멀티GPU
  GPUs: 2× NVIDIA H100 80GB
  Total VRAM: 160GB
  Memory Bandwidth: 3.35 TB/s × 2
  Tensor Parallel Size: 2
  예상 비용: ~$60,000

Option B: 비용 효율적 선택
  GPUs: 4× NVIDIA A100 40GB
  Total VRAM: 160GB
  Tensor Parallel Size: 4
  예상 비용: ~$40,000
```

---

## 4. 테스트 스크립트 구조

```
scripts/
├── benchmark_concurrent.py    # 동시 요청 벤치마크 메인
├── monitor_gpu.py            # GPU 메모리 실시간 모니터링
├── analyze_results.py        # 결과 분석 및 시각화
├── run_all_tests.sh          # 전체 테스트 실행 스크립트
└── requirements.txt          # Python 의존성

results/
├── baseline_results.json     # 기준선 테스트 결과
├── concurrent_results.json   # 동시 요청 테스트 결과
├── memory_logs/              # GPU 메모리 로그
└── charts/                   # 시각화 그래프
```

---

## 5. 실행 순서

### Step 1: 환경 준비
```bash
# 1. Docker Compose로 vLLM 서버 실행
docker-compose up -d

# 2. 서버 준비 대기 (모델 로드 완료까지)
# 로그 확인: docker logs -f gemma-3-server

# 3. 테스트 스크립트 의존성 설치
pip install -r scripts/requirements.txt
```

### Step 2: 환경 검증
```bash
# 모델 상태 확인
curl http://localhost:8000/v1/models

# GPU 메모리 확인
nvidia-smi
```

### Step 3: 테스트 실행
```bash
# 전체 테스트 실행
./scripts/run_all_tests.sh

# 또는 개별 실행
python scripts/benchmark_concurrent.py --concurrent 1 --runs 10
python scripts/benchmark_concurrent.py --concurrent 8 --runs 3
```

### Step 4: 결과 분석
```bash
python scripts/analyze_results.py
```

---

## 6. 예상 결과 및 활용

### 6.1 RTX 3090 예상 한계
- **안정적 동시처리**: 2-4개 요청
- **최대 동시처리**: 6-8개 (짧은 컨텍스트)
- **권장 운영**: 4개 이하 동시 요청

### 6.2 결과 활용
1. 현재 RTX 3090 환경의 서비스 용량 산정
2. 50개 동시요청 서비스를 위한 GPU 투자 계획
3. 비용 대비 성능 최적화 의사결정

---

## 7. 다음 단계

**계획 승인 후 진행할 작업:**
1. [ ] 테스트 스크립트 구현
2. [ ] vLLM 서버 실행 및 환경 검증
3. [ ] 동시 요청 테스트 실행
4. [ ] 결과 분석 및 문서화
5. [ ] GPU 권장 사양 최종 보고서 작성

---

**문서 버전**: v1.0
**상태**: 계획 완료 - 승인 대기 중
