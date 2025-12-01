#!/usr/bin/env python3
"""
Benchmark Results Analysis Script
테스트 결과 분석, 시각화 및 GPU 스펙 산정
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import seaborn as sns
from rich.console import Console
from rich.table import Table
from tabulate import tabulate

console = Console()


@dataclass
class GPUSpec:
    """GPU 사양"""
    name: str
    vram_gb: int
    memory_bandwidth_gb_s: int
    price_usd: int
    max_concurrent_estimate: int


# GPU 스펙 데이터베이스
GPU_SPECS = {
    "RTX 3090": GPUSpec("RTX 3090", 24, 936, 1500, 0),
    "RTX 4090": GPUSpec("RTX 4090", 24, 1008, 2000, 0),
    "A6000": GPUSpec("A6000", 48, 768, 4500, 0),
    "A100 40GB": GPUSpec("A100 40GB", 40, 1555, 10000, 0),
    "A100 80GB": GPUSpec("A100 80GB", 80, 2039, 15000, 0),
    "H100 80GB": GPUSpec("H100 80GB", 80, 3350, 30000, 0),
}


def load_benchmark_results(file_path: str) -> Dict:
    """벤치마크 결과 로드"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_summary_dataframe(data: Dict) -> pd.DataFrame:
    """결과를 DataFrame으로 변환"""
    records = []
    for result in data.get('results', []):
        records.append({
            'concurrent': result['concurrent_requests'],
            'success_rate': result['success_rate'],
            'avg_ttft_ms': result['avg_ttft_ms'],
            'avg_latency_ms': result['avg_total_latency_ms'],
            'throughput_tok_s': result['total_throughput_tokens_per_sec'],
            'peak_memory_mb': result['peak_memory_mb'],
            'min_latency_ms': result['min_total_latency_ms'],
            'max_latency_ms': result['max_total_latency_ms'],
        })
    return pd.DataFrame(records)


def analyze_limits(df: pd.DataFrame) -> Dict[str, Any]:
    """한계점 분석"""
    analysis = {
        'max_stable_concurrent': 0,
        'latency_doubling_point': None,
        'throughput_saturation_point': None,
        'recommended_concurrent': 0,
        'baseline_latency_ms': 0,
    }

    if df.empty:
        return analysis

    # 기준선 (단일 요청)
    baseline = df[df['concurrent'] == 1]
    if not baseline.empty:
        analysis['baseline_latency_ms'] = baseline.iloc[0]['avg_latency_ms']

    # 95% 이상 성공률 유지되는 최대 동시 요청
    stable_df = df[df['success_rate'] >= 95]
    if not stable_df.empty:
        analysis['max_stable_concurrent'] = stable_df['concurrent'].max()

    # Latency가 2배 이상 증가하는 지점
    if analysis['baseline_latency_ms'] > 0:
        doubled = df[df['avg_latency_ms'] > analysis['baseline_latency_ms'] * 2]
        if not doubled.empty:
            analysis['latency_doubling_point'] = doubled['concurrent'].min()

    # Throughput 포화점 (증가율 10% 미만)
    if len(df) > 1:
        df_sorted = df.sort_values('concurrent')
        throughputs = df_sorted['throughput_tok_s'].tolist()
        for i in range(1, len(throughputs)):
            if throughputs[i-1] > 0:
                increase_rate = (throughputs[i] - throughputs[i-1]) / throughputs[i-1]
                if increase_rate < 0.1:
                    analysis['throughput_saturation_point'] = df_sorted.iloc[i-1]['concurrent']
                    break

    # 권장 동시 요청 수 (성공률 95%+, Latency 2배 미만)
    safe_df = df[(df['success_rate'] >= 95)]
    if analysis['baseline_latency_ms'] > 0:
        safe_df = safe_df[safe_df['avg_latency_ms'] < analysis['baseline_latency_ms'] * 2]

    if not safe_df.empty:
        analysis['recommended_concurrent'] = int(safe_df['concurrent'].max())

    return analysis


def estimate_gpu_requirements(
    current_gpu_vram_gb: int,
    current_max_concurrent: int,
    target_concurrent: int,
    model_size_gb: float = 24.0,  # FP16 Gemma 12B
    kv_cache_per_request_gb: float = 1.15  # 4096 context
) -> Dict[str, Any]:
    """목표 동시 요청을 위한 GPU 요구사양 산정"""

    # 현재 GPU에서 사용된 KV cache 추정
    current_kv_total = current_max_concurrent * kv_cache_per_request_gb

    # 목표를 위한 총 VRAM 요구량 계산
    required_vram = model_size_gb + (target_concurrent * kv_cache_per_request_gb) + 5  # 5GB overhead

    # GPU 추천
    recommendations = []

    for gpu_name, spec in GPU_SPECS.items():
        # 단일 GPU로 가능한 동시 요청 수
        available_for_kv = spec.vram_gb - model_size_gb - 5  # 모델 + 오버헤드 제외
        max_concurrent = int(available_for_kv / kv_cache_per_request_gb) if available_for_kv > 0 else 0

        # 멀티 GPU 옵션
        gpus_needed = 1
        if max_concurrent < target_concurrent:
            gpus_needed = -(-target_concurrent // max_concurrent) if max_concurrent > 0 else 99

        recommendations.append({
            'gpu': gpu_name,
            'vram_gb': spec.vram_gb,
            'single_gpu_max': max_concurrent,
            'gpus_needed': min(gpus_needed, 8),
            'total_cost_usd': spec.price_usd * min(gpus_needed, 8),
            'feasible': max_concurrent >= target_concurrent or gpus_needed <= 4
        })

    return {
        'target_concurrent': target_concurrent,
        'required_vram_gb': required_vram,
        'model_size_gb': model_size_gb,
        'kv_cache_per_request_gb': kv_cache_per_request_gb,
        'recommendations': sorted(recommendations, key=lambda x: x['total_cost_usd'])
    }


def generate_charts(df: pd.DataFrame, output_dir: str):
    """차트 생성"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 스타일 설정
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)

    # 1. Latency vs Concurrent Requests
    fig, ax = plt.subplots()
    ax.plot(df['concurrent'], df['avg_latency_ms'], marker='o', linewidth=2, markersize=8)
    ax.fill_between(df['concurrent'], df['min_latency_ms'], df['max_latency_ms'], alpha=0.3)
    ax.set_xlabel('Concurrent Requests')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Latency vs Concurrent Requests')
    ax.set_xticks(df['concurrent'])
    plt.tight_layout()
    plt.savefig(output_path / 'latency_vs_concurrent.png', dpi=150)
    plt.close()

    # 2. Throughput vs Concurrent Requests
    fig, ax = plt.subplots()
    ax.bar(df['concurrent'].astype(str), df['throughput_tok_s'], color='steelblue')
    ax.set_xlabel('Concurrent Requests')
    ax.set_ylabel('Throughput (tokens/sec)')
    ax.set_title('Throughput vs Concurrent Requests')
    plt.tight_layout()
    plt.savefig(output_path / 'throughput_vs_concurrent.png', dpi=150)
    plt.close()

    # 3. Success Rate
    fig, ax = plt.subplots()
    colors = ['green' if r >= 95 else 'orange' if r >= 80 else 'red' for r in df['success_rate']]
    ax.bar(df['concurrent'].astype(str), df['success_rate'], color=colors)
    ax.axhline(y=95, color='green', linestyle='--', label='95% threshold')
    ax.set_xlabel('Concurrent Requests')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Success Rate vs Concurrent Requests')
    ax.set_ylim(0, 105)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path / 'success_rate.png', dpi=150)
    plt.close()

    # 4. Memory Usage
    if df['peak_memory_mb'].sum() > 0:
        fig, ax = plt.subplots()
        ax.bar(df['concurrent'].astype(str), df['peak_memory_mb'] / 1024, color='purple')
        ax.axhline(y=24, color='red', linestyle='--', label='RTX 3090 VRAM (24GB)')
        ax.set_xlabel('Concurrent Requests')
        ax.set_ylabel('Peak Memory (GB)')
        ax.set_title('GPU Memory Usage vs Concurrent Requests')
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_path / 'memory_usage.png', dpi=150)
        plt.close()

    # 5. Combined Overview
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Latency
    axes[0, 0].plot(df['concurrent'], df['avg_latency_ms'], marker='o')
    axes[0, 0].set_title('Latency')
    axes[0, 0].set_xlabel('Concurrent')
    axes[0, 0].set_ylabel('ms')

    # Throughput
    axes[0, 1].bar(df['concurrent'].astype(str), df['throughput_tok_s'])
    axes[0, 1].set_title('Throughput')
    axes[0, 1].set_xlabel('Concurrent')
    axes[0, 1].set_ylabel('tok/s')

    # Success Rate
    axes[1, 0].bar(df['concurrent'].astype(str), df['success_rate'],
                   color=['green' if r >= 95 else 'red' for r in df['success_rate']])
    axes[1, 0].set_title('Success Rate')
    axes[1, 0].set_xlabel('Concurrent')
    axes[1, 0].set_ylabel('%')
    axes[1, 0].axhline(y=95, color='black', linestyle='--')

    # TTFT
    axes[1, 1].plot(df['concurrent'], df['avg_ttft_ms'], marker='s', color='orange')
    axes[1, 1].set_title('Time to First Token')
    axes[1, 1].set_xlabel('Concurrent')
    axes[1, 1].set_ylabel('ms')

    plt.suptitle('vLLM Concurrent Benchmark Results (RTX 3090)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path / 'overview.png', dpi=150)
    plt.close()

    console.print(f"[green]Charts saved to: {output_path}[/green]")


def generate_report(
    df: pd.DataFrame,
    analysis: Dict,
    gpu_requirements: Dict,
    output_file: str
):
    """최종 보고서 생성"""

    report = f"""# vLLM 동시처리 성능 테스트 결과 보고서

**생성일**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**테스트 환경**: NVIDIA RTX 3090 (24GB)

---

## 1. 테스트 결과 요약

### 1.1 성능 메트릭

| 동시 요청 | 성공률 | 평균 Latency (ms) | Throughput (tok/s) | Peak Memory (MB) |
|----------|--------|-------------------|-------------------|------------------|
"""

    for _, row in df.iterrows():
        report += f"| {int(row['concurrent'])} | {row['success_rate']:.1f}% | {row['avg_latency_ms']:.1f} | {row['throughput_tok_s']:.1f} | {row['peak_memory_mb']:.0f} |\n"

    report += f"""
### 1.2 한계점 분석

| 항목 | 값 |
|------|-----|
| 기준선 Latency (1 req) | {analysis['baseline_latency_ms']:.1f} ms |
| 최대 안정 동시 요청 (95%+ 성공) | {analysis['max_stable_concurrent']} |
| Latency 2배 증가 지점 | {analysis['latency_doubling_point'] or 'N/A'} |
| Throughput 포화점 | {analysis['throughput_saturation_point'] or 'N/A'} |
| **권장 동시 요청 수** | **{analysis['recommended_concurrent']}** |

---

## 2. RTX 3090 운영 권장사항

### 2.1 안정적 운영 (권장)
- **동시 요청 수**: {analysis['recommended_concurrent']}개 이하
- **예상 Throughput**: {df[df['concurrent'] == analysis['recommended_concurrent']]['throughput_tok_s'].values[0] if analysis['recommended_concurrent'] > 0 and analysis['recommended_concurrent'] in df['concurrent'].values else 'N/A':.1f} tokens/sec
- **SLA 목표**: 95% 이상 성공률, Latency < {analysis['baseline_latency_ms'] * 2:.0f}ms

### 2.2 최대 용량 운영 (주의 필요)
- **최대 동시 요청**: {analysis['max_stable_concurrent']}개
- **리스크**: Latency 변동성 증가, 메모리 압박

---

## 3. 비양자화 Gemma 3 12B - 50개 동시요청 GPU 산정

### 3.1 요구사양 계산

| 항목 | 값 |
|------|-----|
| 목표 동시 요청 | {gpu_requirements['target_concurrent']}개 |
| 모델 크기 (FP16) | {gpu_requirements['model_size_gb']:.1f} GB |
| KV Cache/요청 | {gpu_requirements['kv_cache_per_request_gb']:.2f} GB |
| **총 필요 VRAM** | **{gpu_requirements['required_vram_gb']:.1f} GB** |

### 3.2 GPU 옵션 비교

| GPU | VRAM | 단일 GPU 최대 | 필요 GPU 수 | 예상 비용 | 권장 |
|-----|------|--------------|-------------|----------|------|
"""

    for rec in gpu_requirements['recommendations']:
        feasible = "✅" if rec['feasible'] else "❌"
        report += f"| {rec['gpu']} | {rec['vram_gb']}GB | {rec['single_gpu_max']}개 | {rec['gpus_needed']} | ${rec['total_cost_usd']:,} | {feasible} |\n"

    # 권장 옵션 찾기
    feasible_options = [r for r in gpu_requirements['recommendations'] if r['feasible']]
    if feasible_options:
        best_option = feasible_options[0]
        report += f"""
### 3.3 권장 구성

#### 최소 비용 옵션
- **GPU**: {best_option['gpu']} × {best_option['gpus_needed']}
- **총 VRAM**: {best_option['vram_gb'] * best_option['gpus_needed']}GB
- **예상 비용**: ${best_option['total_cost_usd']:,}
- **Tensor Parallel Size**: {best_option['gpus_needed']}

#### 최적 성능 옵션
- **GPU**: 2× H100 80GB
- **총 VRAM**: 160GB
- **예상 비용**: ~$60,000
- **장점**: 높은 메모리 대역폭, 여유 공간 확보
"""

    report += """
---

## 4. 결론

### RTX 3090 환경
- 양자화된 Gemma 3 12B 기준 **{recommended}개** 동시 요청 안정적 처리 가능
- 번역 서비스의 낮은 부하에 적합

### 50개 동시요청 대응
- **최소 요구사양**: ~95GB VRAM (2× A100 80GB 권장)
- **비용 효율적 선택**: 4× A100 40GB ($40,000)
- **클라우드 대안**: AWS p4d.24xlarge (~$32/hour)

---

*이 보고서는 자동 생성되었습니다.*
""".format(recommended=analysis['recommended_concurrent'])

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)

    console.print(f"[green]Report saved to: {output_file}[/green]")


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("--input", default="results/concurrent_results.json", help="Input results file")
    parser.add_argument("--output-dir", default="results/charts", help="Output directory for charts")
    parser.add_argument("--report", default="claudedocs/benchmark_report.md", help="Output report file")
    parser.add_argument("--target-concurrent", type=int, default=50, help="Target concurrent for GPU estimation")

    args = parser.parse_args()

    console.print("[bold]Analyzing Benchmark Results[/bold]\n")

    # 데이터 로드
    try:
        data = load_benchmark_results(args.input)
    except FileNotFoundError:
        console.print(f"[red]Error: Results file not found: {args.input}[/red]")
        console.print("Please run the benchmark first: python benchmark_concurrent.py")
        return

    # DataFrame 생성
    df = create_summary_dataframe(data)

    if df.empty:
        console.print("[red]No results to analyze[/red]")
        return

    # 결과 테이블 출력
    console.print("[bold cyan]Benchmark Results:[/bold cyan]")
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))

    # 한계점 분석
    console.print("\n[bold cyan]Limit Analysis:[/bold cyan]")
    analysis = analyze_limits(df)
    for key, value in analysis.items():
        console.print(f"  {key}: {value}")

    # GPU 요구사양 산정
    console.print(f"\n[bold cyan]GPU Requirements for {args.target_concurrent} concurrent requests:[/bold cyan]")
    gpu_req = estimate_gpu_requirements(
        current_gpu_vram_gb=24,
        current_max_concurrent=analysis['max_stable_concurrent'],
        target_concurrent=args.target_concurrent
    )
    console.print(f"  Required VRAM: {gpu_req['required_vram_gb']:.1f} GB")

    # 차트 생성
    console.print("\n[bold cyan]Generating charts...[/bold cyan]")
    generate_charts(df, args.output_dir)

    # 보고서 생성
    console.print("\n[bold cyan]Generating report...[/bold cyan]")
    generate_report(df, analysis, gpu_req, args.report)

    console.print("\n[bold green]Analysis complete![/bold green]")


if __name__ == "__main__":
    main()
