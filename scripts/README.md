# Scripts

vLLM 서비스 테스트 및 모니터링을 위한 스크립트 모음입니다.

## 빠른 시작

```bash
# 1. 가상환경 설정 (최초 1회)
./run_all_tests.sh setup

# 2. 서비스 상태 확인
./health_check.sh

# 3. 빠른 테스트 (inference 포함)
./health_check.sh --test

# 4. Python 테스트 스크립트
python quick_test.py --stream
```

## 스크립트 목록

### health_check.sh

서비스 상태를 빠르게 확인합니다.

```bash
# 기본 헬스체크 (health, models, GPU)
./health_check.sh

# inference 테스트 포함
./health_check.sh --test

# 환경변수로 URL 지정
VLLM_URL=http://your-server:8000 ./health_check.sh
```

### quick_test.py

Python 기반 빠른 테스트 스크립트입니다.

```bash
# 기본 테스트
python quick_test.py

# 스트리밍 모드 테스트
python quick_test.py --stream

# 커스텀 프롬프트
python quick_test.py --prompt "Translate: Hello world" --max-tokens 50

# 다른 서버 테스트
python quick_test.py --url http://remote-server:8000
```

### run_all_tests.sh

전체 벤치마크 테스트 실행 스크립트입니다.

```bash
# 전체 테스트 (환경설정 → 서버확인 → 벤치마크 → 분석)
./run_all_tests.sh all

# 개별 명령
./run_all_tests.sh setup      # 가상환경 설정만
./run_all_tests.sh check      # 서버/GPU 상태 확인
./run_all_tests.sh benchmark  # 벤치마크 실행
./run_all_tests.sh analyze    # 결과 분석
```

### benchmark_concurrent.py

동시 요청 벤치마크 테스트입니다.

```bash
python benchmark_concurrent.py \
  --url http://localhost:8000 \
  --concurrent 1 2 4 8 16 \
  --max-tokens 100 \
  --runs 3 \
  --output ../results/benchmark.json
```

### benchmark_realistic.py

현실적인 프롬프트로 벤치마크 테스트합니다.

```bash
python benchmark_realistic.py \
  --url http://localhost:8000 \
  --concurrent 1 2 4 8 \
  --max-tokens 512 \
  --runs 2
```

### monitor_gpu.py

GPU 메모리 및 사용률을 모니터링합니다.

```bash
# 실시간 모니터링
python monitor_gpu.py --interval 1

# 60초 동안 모니터링 후 저장
python monitor_gpu.py --duration 60 --output ../results/gpu_metrics.json

# 현재 상태만 조회
python monitor_gpu.py --once
```

### analyze_results.py

벤치마크 결과를 분석하고 보고서를 생성합니다.

```bash
python analyze_results.py \
  --input ../results/concurrent_results.json \
  --output-dir ../results/charts \
  --report ../claudedocs/benchmark_report.md
```

## 의존성

필요한 Python 패키지:

```bash
pip install -r requirements.txt
```

또는 `run_all_tests.sh setup` 명령으로 자동 설치됩니다.

## 결과 파일 위치

```
results/
├── concurrent_results.json      # 벤치마크 결과
├── realistic_benchmark.json     # 현실적 조건 결과
├── gpu_metrics.json            # GPU 모니터링 데이터
└── charts/                     # 차트 이미지

claudedocs/
└── benchmark_report.md         # 분석 보고서
```
