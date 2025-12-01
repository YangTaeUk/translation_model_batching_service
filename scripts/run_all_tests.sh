#!/bin/bash
# =============================================================================
# vLLM Concurrent Benchmark Test Runner
# RTX 3090 동시처리 성능 테스트 실행 스크립트
# =============================================================================

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 프로젝트 루트 디렉토리
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_ROOT/venv"
RESULTS_DIR="$PROJECT_ROOT/results"

echo -e "${CYAN}=======================================${NC}"
echo -e "${CYAN}  vLLM Concurrent Benchmark Runner${NC}"
echo -e "${CYAN}=======================================${NC}"

# =============================================================================
# 1. Python 가상환경 설정
# =============================================================================
setup_venv() {
    echo -e "\n${YELLOW}[1/5] Setting up Python virtual environment...${NC}"

    if [ ! -d "$VENV_DIR" ]; then
        echo "Creating new virtual environment..."
        python3 -m venv "$VENV_DIR"
    else
        echo "Virtual environment already exists."
    fi

    # 가상환경 활성화
    source "$VENV_DIR/bin/activate"

    echo "Installing dependencies..."
    pip install --upgrade pip -q
    pip install -r "$SCRIPT_DIR/requirements.txt" -q

    echo -e "${GREEN}✓ Virtual environment ready${NC}"
}

# =============================================================================
# 2. vLLM 서버 상태 확인
# =============================================================================
check_server() {
    echo -e "\n${YELLOW}[2/5] Checking vLLM server status...${NC}"

    local max_retries=30
    local retry_count=0

    while [ $retry_count -lt $max_retries ]; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo -e "${GREEN}✓ vLLM server is healthy${NC}"

            # 모델 정보 확인
            echo -e "\n${CYAN}Model Info:${NC}"
            curl -s http://localhost:8000/v1/models | python3 -m json.tool 2>/dev/null || echo "Could not parse model info"

            return 0
        fi

        echo "Waiting for server... ($((retry_count + 1))/$max_retries)"
        sleep 5
        ((retry_count++))
    done

    echo -e "${RED}✗ Server is not responding${NC}"
    echo -e "${YELLOW}Please start the server first:${NC}"
    echo "  cd $PROJECT_ROOT && docker-compose up -d"
    exit 1
}

# =============================================================================
# 3. GPU 상태 확인
# =============================================================================
check_gpu() {
    echo -e "\n${YELLOW}[3/5] Checking GPU status...${NC}"

    if ! command -v nvidia-smi &> /dev/null; then
        echo -e "${RED}✗ nvidia-smi not found${NC}"
        exit 1
    fi

    echo -e "${CYAN}GPU Info:${NC}"
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu --format=csv

    echo -e "${GREEN}✓ GPU is available${NC}"
}

# =============================================================================
# 4. 벤치마크 실행
# =============================================================================
run_benchmark() {
    echo -e "\n${YELLOW}[4/5] Running benchmark tests...${NC}"

    # 결과 디렉토리 생성
    mkdir -p "$RESULTS_DIR"
    mkdir -p "$RESULTS_DIR/charts"

    # GPU 모니터링 백그라운드 실행
    echo "Starting GPU monitoring in background..."
    python "$SCRIPT_DIR/monitor_gpu.py" \
        --interval 2 \
        --output "$RESULTS_DIR/gpu_metrics.json" &
    MONITOR_PID=$!

    # 벤치마크 실행
    echo -e "\n${CYAN}Running concurrent request benchmark...${NC}"
    python "$SCRIPT_DIR/benchmark_concurrent.py" \
        --url "http://localhost:8000" \
        --concurrent 1 2 4 8 16 32 \
        --max-tokens 100 \
        --runs 3 \
        --output "$RESULTS_DIR/concurrent_results.json"

    # GPU 모니터링 종료
    echo "Stopping GPU monitor..."
    kill $MONITOR_PID 2>/dev/null || true
    wait $MONITOR_PID 2>/dev/null || true

    echo -e "${GREEN}✓ Benchmark completed${NC}"
}

# =============================================================================
# 5. 결과 분석
# =============================================================================
analyze_results() {
    echo -e "\n${YELLOW}[5/5] Analyzing results...${NC}"

    python "$SCRIPT_DIR/analyze_results.py" \
        --input "$RESULTS_DIR/concurrent_results.json" \
        --output-dir "$RESULTS_DIR/charts" \
        --report "$PROJECT_ROOT/claudedocs/benchmark_report.md" \
        --target-concurrent 50

    echo -e "${GREEN}✓ Analysis completed${NC}"
}

# =============================================================================
# 6. 결과 요약
# =============================================================================
show_summary() {
    echo -e "\n${CYAN}=======================================${NC}"
    echo -e "${CYAN}  Test Completed!${NC}"
    echo -e "${CYAN}=======================================${NC}"
    echo -e ""
    echo -e "Results saved to:"
    echo -e "  - ${GREEN}$RESULTS_DIR/concurrent_results.json${NC}"
    echo -e "  - ${GREEN}$RESULTS_DIR/gpu_metrics.json${NC}"
    echo -e "  - ${GREEN}$RESULTS_DIR/charts/${NC}"
    echo -e "  - ${GREEN}$PROJECT_ROOT/claudedocs/benchmark_report.md${NC}"
    echo -e ""
    echo -e "To view the report:"
    echo -e "  cat $PROJECT_ROOT/claudedocs/benchmark_report.md"
}

# =============================================================================
# 메인 실행
# =============================================================================
main() {
    cd "$PROJECT_ROOT"

    # 명령줄 인자 처리
    case "${1:-all}" in
        setup)
            setup_venv
            ;;
        check)
            setup_venv
            check_server
            check_gpu
            ;;
        benchmark)
            setup_venv
            run_benchmark
            ;;
        analyze)
            setup_venv
            analyze_results
            ;;
        all)
            setup_venv
            check_server
            check_gpu
            run_benchmark
            analyze_results
            show_summary
            ;;
        *)
            echo "Usage: $0 {setup|check|benchmark|analyze|all}"
            echo ""
            echo "Commands:"
            echo "  setup     - Set up Python virtual environment"
            echo "  check     - Check server and GPU status"
            echo "  benchmark - Run benchmark tests only"
            echo "  analyze   - Analyze existing results"
            echo "  all       - Run complete test suite (default)"
            exit 1
            ;;
    esac
}

main "$@"
