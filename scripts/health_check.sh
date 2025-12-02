#!/bin/bash
# =============================================================================
# Quick Health Check Script
# 서비스 상태 빠른 확인용 스크립트
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
BASE_URL="${VLLM_URL:-http://localhost:8000}"
MAX_RETRIES="${MAX_RETRIES:-3}"
RETRY_INTERVAL="${RETRY_INTERVAL:-5}"

echo -e "${CYAN}=======================================${NC}"
echo -e "${CYAN}  vLLM Service Health Check${NC}"
echo -e "${CYAN}=======================================${NC}"
echo -e "Target: ${BASE_URL}"
echo ""

# =============================================================================
# Health Check Functions
# =============================================================================

check_health_endpoint() {
    echo -e "${YELLOW}[1/4] Checking /health endpoint...${NC}"

    for i in $(seq 1 $MAX_RETRIES); do
        if curl -sf "${BASE_URL}/health" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Health endpoint OK${NC}"
            return 0
        fi

        if [ $i -lt $MAX_RETRIES ]; then
            echo "  Retry $i/$MAX_RETRIES..."
            sleep $RETRY_INTERVAL
        fi
    done

    echo -e "${RED}✗ Health endpoint failed${NC}"
    return 1
}

check_models_endpoint() {
    echo -e "\n${YELLOW}[2/4] Checking /v1/models endpoint...${NC}"

    local response=$(curl -sf "${BASE_URL}/v1/models" 2>/dev/null)

    if [ -n "$response" ]; then
        echo -e "${GREEN}✓ Models endpoint OK${NC}"
        echo -e "${CYAN}Available models:${NC}"
        echo "$response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for model in data.get('data', []):
        print(f\"  - {model.get('id', 'unknown')}\")
except:
    print('  Could not parse model list')
" 2>/dev/null || echo "  Could not parse response"
        return 0
    else
        echo -e "${RED}✗ Models endpoint failed${NC}"
        return 1
    fi
}

check_gpu_status() {
    echo -e "\n${YELLOW}[3/4] Checking GPU status...${NC}"

    if ! command -v nvidia-smi &> /dev/null; then
        echo -e "${YELLOW}⚠ nvidia-smi not available${NC}"
        return 0
    fi

    echo -e "${GREEN}✓ GPU available${NC}"
    nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu --format=csv,noheader | while read line; do
        echo -e "  ${CYAN}${line}${NC}"
    done

    return 0
}

test_inference() {
    echo -e "\n${YELLOW}[4/4] Testing inference...${NC}"

    local response=$(curl -sf -X POST "${BASE_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "google/gemma-3-12b-it",
            "messages": [{"role": "user", "content": "Say hello in Korean"}],
            "max_tokens": 20,
            "temperature": 0.7
        }' 2>/dev/null)

    if [ -n "$response" ]; then
        echo -e "${GREEN}✓ Inference OK${NC}"
        echo -e "${CYAN}Response:${NC}"
        echo "$response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    content = data['choices'][0]['message']['content']
    print(f'  {content[:100]}...' if len(content) > 100 else f'  {content}')
except Exception as e:
    print(f'  Could not parse response: {e}')
" 2>/dev/null || echo "  Could not parse response"
        return 0
    else
        echo -e "${RED}✗ Inference failed${NC}"
        return 1
    fi
}

# =============================================================================
# Main
# =============================================================================

main() {
    local errors=0

    check_health_endpoint || ((errors++))
    check_models_endpoint || ((errors++))
    check_gpu_status || ((errors++))

    # Optional: inference test
    if [ "${1:-}" = "--test" ] || [ "${1:-}" = "-t" ]; then
        test_inference || ((errors++))
    fi

    echo ""
    echo -e "${CYAN}=======================================${NC}"
    if [ $errors -eq 0 ]; then
        echo -e "${GREEN}  All checks passed! ✓${NC}"
    else
        echo -e "${RED}  $errors check(s) failed ✗${NC}"
    fi
    echo -e "${CYAN}=======================================${NC}"

    exit $errors
}

# Usage
if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -t, --test    Include inference test"
    echo "  -h, --help    Show this help"
    echo ""
    echo "Environment variables:"
    echo "  VLLM_URL        Server URL (default: http://localhost:8000)"
    echo "  MAX_RETRIES     Max retry attempts (default: 3)"
    echo "  RETRY_INTERVAL  Seconds between retries (default: 5)"
    exit 0
fi

main "$@"
