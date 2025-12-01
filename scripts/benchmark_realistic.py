#!/usr/bin/env python3
"""
vLLM Realistic Benchmark Script
현실적인 번역 서비스 조건에서의 동시 처리 성능 측정
- 입력: 500~1000 토큰 (실제 문서 번역 시나리오)
- 출력: 500~1000 토큰
"""

import asyncio
import aiohttp
import json
import time
import argparse
import statistics
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict, field
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import subprocess

console = Console()

# ============================================================================
# 현실적인 테스트 프롬프트 (500~800 토큰 범위)
# 실제 번역 서비스에서 처리하는 문서 수준의 텍스트
# ============================================================================

REALISTIC_PROMPTS = [
    # 프롬프트 1: 기술 문서 (~600 토큰)
    """You are a professional Korean translator. Translate the following technical document from English to Korean accurately, maintaining the original meaning and technical terminology.

Document to translate:

Artificial Intelligence and Machine Learning: A Comprehensive Overview

Introduction to AI Systems

Artificial intelligence (AI) represents one of the most transformative technologies of the 21st century. At its core, AI refers to computer systems designed to perform tasks that typically require human intelligence. These tasks include visual perception, speech recognition, decision-making, and language translation.

The field of AI can be broadly categorized into two types: narrow AI and general AI. Narrow AI, also known as weak AI, is designed to perform specific tasks. Examples include virtual assistants like Siri and Alexa, recommendation systems used by Netflix and Amazon, and autonomous vehicles. General AI, or strong AI, refers to systems that possess the ability to understand, learn, and apply knowledge across a wide range of tasks, similar to human cognitive abilities.

Machine Learning Fundamentals

Machine learning (ML) is a subset of AI that focuses on developing algorithms that allow computers to learn from and make predictions based on data. Unlike traditional programming, where rules are explicitly coded, machine learning systems identify patterns in data and use these patterns to make decisions.

There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning. In supervised learning, the algorithm is trained on labeled data, learning to map inputs to known outputs. Unsupervised learning involves finding hidden patterns in unlabeled data. Reinforcement learning trains agents to make decisions by rewarding desired behaviors.

Deep Learning and Neural Networks

Deep learning is a specialized form of machine learning that uses artificial neural networks with multiple layers. These deep neural networks can automatically learn hierarchical representations of data, making them particularly effective for complex tasks like image recognition and natural language processing.

The architecture of neural networks consists of interconnected nodes organized in layers. The input layer receives data, hidden layers process information through weighted connections, and the output layer produces the final result. Training involves adjusting these weights through a process called backpropagation to minimize prediction errors.

Please provide a complete Korean translation of this document.""",

    # 프롬프트 2: 비즈니스 문서 (~550 토큰)
    """You are an expert Korean translator specializing in business documents. Translate the following business report from English to Korean with professional terminology.

Document to translate:

Quarterly Business Performance Report - Q3 2024

Executive Summary

This report presents a comprehensive analysis of our company's performance during the third quarter of 2024. Overall, the company has demonstrated strong growth across multiple business segments, with total revenue increasing by 15% compared to the same period last year.

Financial Highlights

Our total revenue for Q3 2024 reached $2.5 billion, representing a significant increase from $2.17 billion in Q3 2023. This growth was primarily driven by our cloud services division, which saw a 25% year-over-year increase. Operating expenses remained controlled at $1.8 billion, resulting in an operating margin of 28%, up from 24% in the previous year.

The company's net income for the quarter was $450 million, reflecting a 20% increase compared to Q3 2023. Earnings per share (EPS) came in at $3.50, exceeding analyst expectations of $3.25. Cash flow from operations remained strong at $600 million, providing ample liquidity for strategic investments.

Market Performance and Strategic Initiatives

Our market share in the enterprise software segment expanded to 18%, up from 15% at the beginning of the year. This growth can be attributed to successful product launches and strategic partnerships. The company announced three major partnerships with Fortune 500 companies during this quarter.

Key strategic initiatives include the expansion of our data center infrastructure in Asia-Pacific region, the acquisition of a leading AI startup for $200 million, and the launch of our next-generation cloud platform. These investments are expected to drive long-term growth and strengthen our competitive position.

Outlook for Q4 2024

Based on current market trends and our sales pipeline, we project revenue growth of 12-15% for Q4 2024. The company remains committed to its strategic priorities of innovation, customer satisfaction, and sustainable growth.

Provide the complete Korean translation.""",

    # 프롬프트 3: 학술 논문 (~650 토큰)
    """As a professional academic translator, translate the following research paper abstract and introduction from English to Korean, preserving academic style and technical accuracy.

Document to translate:

Climate Change Impact on Global Agricultural Systems: A Multi-Regional Assessment Study

Abstract

This study presents a comprehensive analysis of climate change impacts on agricultural productivity across different geographical regions. Using advanced climate models and agricultural simulation tools, we examined crop yield variations under multiple climate scenarios for the period 2025-2100. Our findings indicate significant regional disparities in climate change vulnerability, with tropical and subtropical regions facing the greatest risks to food security.

Introduction

Climate change represents one of the most significant challenges facing global food systems in the 21st century. Rising temperatures, changing precipitation patterns, and increased frequency of extreme weather events are expected to profoundly affect agricultural productivity worldwide. Understanding these impacts is crucial for developing effective adaptation strategies and ensuring food security for a growing global population.

The Intergovernmental Panel on Climate Change (IPCC) has projected global temperature increases ranging from 1.5°C to 4.5°C by 2100, depending on emission scenarios. These temperature changes, combined with altered precipitation patterns, will have varying effects on different crops and regions. While some high-latitude regions may experience extended growing seasons and increased productivity, many tropical and subtropical areas face declining yields due to heat stress and water scarcity.

Previous studies have examined climate change impacts on agriculture at regional or crop-specific levels. However, a comprehensive global assessment that considers interactions between multiple factors remains lacking. This study addresses this gap by integrating climate projections, crop models, and socioeconomic scenarios to provide a holistic view of future agricultural challenges.

Our research methodology combines outputs from five global climate models with the Agricultural Model Intercomparison and Improvement Project (AgMIP) framework. We analyzed yield projections for major crops including wheat, rice, maize, and soybeans across 50 agricultural regions spanning six continents. The analysis considers both direct climate effects and indirect factors such as CO2 fertilization, pest and disease pressure, and water availability.

Please translate this academic document into Korean.""",

    # 프롬프트 4: 법률 문서 (~580 토큰)
    """You are a legal document translator. Translate the following terms and conditions from English to Korean with precise legal terminology.

Document to translate:

Software License Agreement and Terms of Service

Article 1: Definitions and Interpretation

1.1 In this Agreement, unless the context otherwise requires, the following terms shall have the meanings ascribed to them:

"Software" means the computer program, including all updates, upgrades, modifications, and associated documentation provided by the Licensor under this Agreement.

"Licensee" means the individual or entity that has agreed to be bound by the terms of this Agreement and has been granted a license to use the Software.

"Licensor" means the company that owns the intellectual property rights to the Software and grants the license under this Agreement.

"Effective Date" means the date on which this Agreement comes into force, being the earlier of the date the Licensee first uses the Software or the date the Licensee accepts this Agreement.

Article 2: Grant of License

2.1 Subject to the terms and conditions of this Agreement, the Licensor hereby grants to the Licensee a non-exclusive, non-transferable, limited license to use the Software solely for the Licensee's internal business purposes.

2.2 The license granted herein does not include the right to: (a) sublicense, sell, rent, lease, or otherwise transfer the Software to any third party; (b) modify, adapt, translate, reverse engineer, decompile, or disassemble the Software; (c) remove or alter any proprietary notices, labels, or marks on the Software.

Article 3: Intellectual Property Rights

3.1 The Licensee acknowledges that the Software and all copies thereof are proprietary to the Licensor and title thereto remains exclusively with the Licensor. All applicable rights to patents, copyrights, trademarks, and trade secrets in the Software are and shall remain in the Licensor.

3.2 Nothing in this Agreement shall be construed as transferring any aspects of such rights to the Licensee or any third party.

Translate this legal document into Korean with appropriate legal terminology.""",

    # 프롬프트 5: 뉴스 기사 (~520 토큰)
    """Translate the following news article from English to Korean, maintaining journalistic style and accuracy.

Article to translate:

Global Technology Summit 2024: Leaders Discuss Future of Digital Innovation

Seoul, South Korea - The annual Global Technology Summit concluded yesterday with groundbreaking announcements from major technology companies and policy discussions on artificial intelligence regulation. The three-day event, held at the COEX Convention Center, brought together over 5,000 participants from 80 countries.

Opening Keynote Highlights

The summit opened with a keynote address by Dr. Sarah Chen, CEO of TechFuture Corporation, who outlined her vision for the next decade of technological advancement. "We are standing at the threshold of a new era," Dr. Chen stated. "The convergence of artificial intelligence, quantum computing, and biotechnology will fundamentally reshape every aspect of human society."

Dr. Chen announced her company's commitment to investing $10 billion over the next five years in sustainable technology research. The initiative aims to develop carbon-neutral data centers and energy-efficient computing solutions.

AI Governance Panel Discussion

A panel discussion on artificial intelligence governance attracted significant attention from attendees. Representatives from the European Union, United States, China, and Japan debated approaches to regulating AI systems while fostering innovation.

EU Commissioner for Digital Affairs, Marcus Weber, advocated for comprehensive regulatory frameworks. "We must ensure that AI systems are transparent, accountable, and respect fundamental rights," Weber emphasized. Meanwhile, industry representatives argued for self-regulation and flexible guidelines that could adapt to rapidly evolving technology.

Breakthrough Announcements

Several companies unveiled new products and partnerships during the summit. Notable announcements included a collaboration between Samsung Electronics and NVIDIA to develop next-generation AI processors, Google's launch of an enhanced language model capable of real-time translation in 150 languages, and Microsoft's expansion of its cloud infrastructure in Asia.

Translate this news article into Korean.""",
]

# 짧은 프롬프트 (비교용, ~100 토큰)
SHORT_PROMPTS = [
    "Translate the following English text to Korean: 'The quick brown fox jumps over the lazy dog.'",
    "Translate to Korean: 'Machine learning is transforming how we interact with technology.'",
]


@dataclass
class RequestResult:
    """단일 요청 결과"""
    request_id: int
    success: bool
    ttft_ms: float
    total_latency_ms: float
    input_tokens: int
    output_tokens: int
    tokens_per_second: float = 0.0
    error: Optional[str] = None


@dataclass
class BenchmarkResult:
    """벤치마크 결과"""
    test_id: str
    timestamp: str
    concurrent_requests: int
    prompt_type: str  # "realistic" or "short"
    avg_input_tokens: int
    max_output_tokens: int
    runs: int

    # TTFT 통계
    avg_ttft_ms: float
    min_ttft_ms: float
    max_ttft_ms: float
    std_ttft_ms: float
    p50_ttft_ms: float
    p95_ttft_ms: float
    p99_ttft_ms: float

    # Total Latency 통계
    avg_total_latency_ms: float
    min_total_latency_ms: float
    max_total_latency_ms: float
    std_total_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float

    # Throughput 통계
    avg_output_tokens: float
    avg_tokens_per_second: float
    total_throughput_tokens_per_sec: float

    # 성공률 및 메모리
    success_rate: float
    failed_requests: int
    peak_memory_mb: float

    # 상세 결과
    individual_results: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def get_gpu_memory() -> float:
    """현재 GPU 메모리 사용량 (MB) 반환"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def get_gpu_utilization() -> Dict[str, float]:
    """GPU 사용률 반환"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        values = result.stdout.strip().split(', ')
        return {
            'gpu_util': float(values[0]),
            'mem_util': float(values[1]),
            'temperature': float(values[2]),
            'power_draw': float(values[3]) if values[3] != '[N/A]' else 0.0
        }
    except Exception:
        return {'gpu_util': 0, 'mem_util': 0, 'temperature': 0, 'power_draw': 0}


def percentile(data: List[float], p: float) -> float:
    """백분위수 계산"""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


async def count_tokens_estimate(text: str) -> int:
    """토큰 수 추정 (단어 기반, 실제보다 약간 낮음)"""
    # 대략적인 추정: 영어 1단어 ≈ 1.3토큰, 특수문자/숫자 포함
    words = text.split()
    return int(len(words) * 1.3)


async def make_single_request(
    session: aiohttp.ClientSession,
    request_id: int,
    base_url: str,
    prompt: str,
    max_tokens: int,
    timeout: float = 300.0  # 5분 타임아웃 (긴 응답 고려)
) -> RequestResult:
    """단일 API 요청 실행"""

    input_tokens = await count_tokens_estimate(prompt)

    payload = {
        "model": "google/gemma-3-12b-it",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True
    }

    start_time = time.perf_counter()
    ttft = 0.0
    output_tokens = 0
    first_token_received = False

    try:
        async with session.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                return RequestResult(
                    request_id=request_id,
                    success=False,
                    ttft_ms=0,
                    total_latency_ms=(time.perf_counter() - start_time) * 1000,
                    input_tokens=input_tokens,
                    output_tokens=0,
                    error=f"HTTP {response.status}: {error_text[:200]}"
                )

            # 스트리밍 응답 처리
            async for line in response.content:
                if not first_token_received:
                    ttft = (time.perf_counter() - start_time) * 1000
                    first_token_received = True

                line_str = line.decode('utf-8').strip()
                if line_str.startswith('data: ') and line_str != 'data: [DONE]':
                    try:
                        data = json.loads(line_str[6:])
                        if 'choices' in data and len(data['choices']) > 0:
                            delta = data['choices'][0].get('delta', {})
                            if 'content' in delta:
                                output_tokens += 1
                    except json.JSONDecodeError:
                        pass

            total_latency = (time.perf_counter() - start_time) * 1000
            tokens_per_sec = (output_tokens / (total_latency / 1000)) if total_latency > 0 else 0

            return RequestResult(
                request_id=request_id,
                success=True,
                ttft_ms=ttft,
                total_latency_ms=total_latency,
                input_tokens=input_tokens,
                output_tokens=max(output_tokens, 1),
                tokens_per_second=tokens_per_sec
            )

    except asyncio.TimeoutError:
        return RequestResult(
            request_id=request_id,
            success=False,
            ttft_ms=0,
            total_latency_ms=timeout * 1000,
            input_tokens=input_tokens,
            output_tokens=0,
            error="Timeout"
        )
    except Exception as e:
        return RequestResult(
            request_id=request_id,
            success=False,
            ttft_ms=0,
            total_latency_ms=(time.perf_counter() - start_time) * 1000,
            input_tokens=input_tokens,
            output_tokens=0,
            error=str(e)
        )


async def run_concurrent_benchmark(
    base_url: str,
    concurrent: int,
    max_tokens: int,
    prompts: List[str],
    prompt_type: str,
    runs: int = 2
) -> BenchmarkResult:
    """동시 요청 벤치마크 실행"""

    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"[bold cyan]Running: {concurrent} concurrent requests ({prompt_type})[/bold cyan]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]")

    all_results: List[RequestResult] = []
    peak_memory = 0.0

    connector = aiohttp.TCPConnector(limit=concurrent + 10)

    async with aiohttp.ClientSession(connector=connector) as session:
        for run_idx in range(runs):
            console.print(f"\n  [yellow]Run {run_idx + 1}/{runs}[/yellow]")

            # GPU 상태 체크
            gpu_stats = get_gpu_utilization()
            console.print(f"    GPU: {gpu_stats['gpu_util']:.0f}% util, {gpu_stats['temperature']:.0f}°C, {gpu_stats['power_draw']:.0f}W")

            # 프롬프트 준비
            selected_prompts = [prompts[i % len(prompts)] for i in range(concurrent)]

            # 동시 요청 실행
            run_start = time.perf_counter()

            tasks = [
                make_single_request(session, i, base_url, prompt, max_tokens)
                for i, prompt in enumerate(selected_prompts)
            ]

            results = await asyncio.gather(*tasks)
            all_results.extend(results)

            run_elapsed = time.perf_counter() - run_start

            # GPU 메모리 체크
            current_memory = get_gpu_memory()
            peak_memory = max(peak_memory, current_memory)

            # 결과 요약
            success_count = sum(1 for r in results if r.success)
            failed_count = len(results) - success_count

            if success_count > 0:
                successful = [r for r in results if r.success]
                avg_latency = statistics.mean([r.total_latency_ms for r in successful])
                avg_output = statistics.mean([r.output_tokens for r in successful])
                console.print(f"    Results: [green]{success_count}[/green]/[red]{failed_count}[/red] | "
                            f"Avg Latency: {avg_latency:.0f}ms | Avg Output: {avg_output:.0f} tokens | "
                            f"Run Time: {run_elapsed:.1f}s")
            else:
                console.print(f"    Results: [red]ALL FAILED ({failed_count})[/red]")
                for r in results[:3]:
                    if r.error:
                        console.print(f"      Error: {r.error[:100]}")

            # 쿨다운 (메모리 안정화)
            if run_idx < runs - 1:
                console.print("    Cooling down (10s)...")
                await asyncio.sleep(10)

    # 결과 집계
    successful_results = [r for r in all_results if r.success]
    failed_count = len(all_results) - len(successful_results)
    errors = [r.error for r in all_results if r.error]

    if not successful_results:
        return BenchmarkResult(
            test_id=f"T{concurrent}_{prompt_type}",
            timestamp=datetime.now().isoformat(),
            concurrent_requests=concurrent,
            prompt_type=prompt_type,
            avg_input_tokens=0,
            max_output_tokens=max_tokens,
            runs=runs,
            avg_ttft_ms=0, min_ttft_ms=0, max_ttft_ms=0, std_ttft_ms=0,
            p50_ttft_ms=0, p95_ttft_ms=0, p99_ttft_ms=0,
            avg_total_latency_ms=0, min_total_latency_ms=0, max_total_latency_ms=0, std_total_latency_ms=0,
            p50_latency_ms=0, p95_latency_ms=0, p99_latency_ms=0,
            avg_output_tokens=0, avg_tokens_per_second=0, total_throughput_tokens_per_sec=0,
            success_rate=0,
            failed_requests=failed_count,
            peak_memory_mb=peak_memory,
            individual_results=[asdict(r) for r in all_results],
            errors=errors[:10]
        )

    # 통계 계산
    ttfts = [r.ttft_ms for r in successful_results]
    latencies = [r.total_latency_ms for r in successful_results]
    output_tokens_list = [r.output_tokens for r in successful_results]
    input_tokens_list = [r.input_tokens for r in successful_results]
    tps_list = [r.tokens_per_second for r in successful_results]

    total_tokens = sum(output_tokens_list)
    total_time_sec = sum(latencies) / 1000
    throughput = total_tokens / total_time_sec if total_time_sec > 0 else 0

    return BenchmarkResult(
        test_id=f"T{concurrent}_{prompt_type}",
        timestamp=datetime.now().isoformat(),
        concurrent_requests=concurrent,
        prompt_type=prompt_type,
        avg_input_tokens=int(statistics.mean(input_tokens_list)),
        max_output_tokens=max_tokens,
        runs=runs,

        avg_ttft_ms=statistics.mean(ttfts),
        min_ttft_ms=min(ttfts),
        max_ttft_ms=max(ttfts),
        std_ttft_ms=statistics.stdev(ttfts) if len(ttfts) > 1 else 0,
        p50_ttft_ms=percentile(ttfts, 50),
        p95_ttft_ms=percentile(ttfts, 95),
        p99_ttft_ms=percentile(ttfts, 99),

        avg_total_latency_ms=statistics.mean(latencies),
        min_total_latency_ms=min(latencies),
        max_total_latency_ms=max(latencies),
        std_total_latency_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0,
        p50_latency_ms=percentile(latencies, 50),
        p95_latency_ms=percentile(latencies, 95),
        p99_latency_ms=percentile(latencies, 99),

        avg_output_tokens=statistics.mean(output_tokens_list),
        avg_tokens_per_second=statistics.mean(tps_list),
        total_throughput_tokens_per_sec=throughput,

        success_rate=len(successful_results) / len(all_results) * 100,
        failed_requests=failed_count,
        peak_memory_mb=peak_memory,
        individual_results=[asdict(r) for r in all_results],
        errors=errors[:10]
    )


async def check_server_health(base_url: str) -> bool:
    """서버 상태 확인"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/health", timeout=aiohttp.ClientTimeout(total=10)) as resp:
                return resp.status == 200
    except Exception:
        return False


async def warmup(base_url: str, prompts: List[str], max_tokens: int, count: int = 2):
    """웜업 요청 (긴 프롬프트로)"""
    console.print("\n[yellow]Warming up with realistic prompts...[/yellow]")

    async with aiohttp.ClientSession() as session:
        for i in range(count):
            prompt = prompts[i % len(prompts)]
            input_tokens = await count_tokens_estimate(prompt)
            console.print(f"  Warmup {i+1}/{count} (input: ~{input_tokens} tokens)...", end=" ")

            result = await make_single_request(
                session, i, base_url, prompt, max_tokens=max_tokens
            )

            if result.success:
                console.print(f"[green]OK[/green] ({result.total_latency_ms:.0f}ms, {result.output_tokens} tokens)")
            else:
                console.print(f"[red]FAIL: {result.error}[/red]")

            await asyncio.sleep(5)


def print_results_table(results: List[BenchmarkResult]):
    """결과 테이블 출력"""
    table = Table(title="Realistic Benchmark Results Summary")

    table.add_column("Concurrent", justify="right", style="cyan")
    table.add_column("Type", justify="center")
    table.add_column("Input Tok", justify="right")
    table.add_column("Success", justify="right", style="green")
    table.add_column("Avg TTFT", justify="right")
    table.add_column("Avg Latency", justify="right")
    table.add_column("P95 Latency", justify="right", style="yellow")
    table.add_column("Throughput", justify="right", style="magenta")
    table.add_column("Memory", justify="right")

    for r in results:
        success_style = "green" if r.success_rate >= 95 else "yellow" if r.success_rate >= 80 else "red"
        table.add_row(
            str(r.concurrent_requests),
            r.prompt_type[:8],
            str(r.avg_input_tokens),
            f"[{success_style}]{r.success_rate:.1f}%[/{success_style}]",
            f"{r.avg_ttft_ms:.0f}ms",
            f"{r.avg_total_latency_ms/1000:.1f}s",
            f"{r.p95_latency_ms/1000:.1f}s",
            f"{r.total_throughput_tokens_per_sec:.1f}",
            f"{r.peak_memory_mb/1024:.1f}GB"
        )

    console.print(table)


async def main():
    parser = argparse.ArgumentParser(description="vLLM Realistic Benchmark")
    parser.add_argument("--url", default="http://localhost:8000", help="vLLM server URL")
    parser.add_argument("--concurrent", type=int, nargs="+", default=[1, 2, 4, 8],
                        help="Concurrent request counts to test")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max output tokens")
    parser.add_argument("--runs", type=int, default=2, help="Runs per concurrent level")
    parser.add_argument("--output", default="results/realistic_benchmark.json", help="Output file")
    parser.add_argument("--skip-warmup", action="store_true", help="Skip warmup")
    parser.add_argument("--short-only", action="store_true", help="Run short prompts only (comparison)")

    args = parser.parse_args()

    console.print("[bold]=" * 70)
    console.print("[bold]vLLM Realistic Benchmark - GPU HW Selection Analysis[/bold]")
    console.print("[bold]=" * 70)
    console.print(f"\nServer: {args.url}")
    console.print(f"Test levels: {args.concurrent}")
    console.print(f"Max output tokens: {args.max_tokens}")
    console.print(f"Runs per level: {args.runs}")

    # 프롬프트 정보
    sample_prompt = REALISTIC_PROMPTS[0]
    est_tokens = await count_tokens_estimate(sample_prompt)
    console.print(f"\n[cyan]Realistic Prompts:[/cyan]")
    console.print(f"  - Count: {len(REALISTIC_PROMPTS)}")
    console.print(f"  - Avg Input Tokens: ~{est_tokens} (estimated)")
    console.print(f"  - Max Output Tokens: {args.max_tokens}")
    console.print(f"  - Total Tokens/Request: ~{est_tokens + args.max_tokens}")

    # 서버 상태 확인
    console.print("\n[yellow]Checking server health...[/yellow]")
    if not await check_server_health(args.url):
        console.print("[red]Server is not responding. Please start the vLLM server first.[/red]")
        return
    console.print("[green]Server is healthy![/green]")

    # GPU 초기 상태
    console.print("\n[cyan]Initial GPU Status:[/cyan]")
    gpu_stats = get_gpu_utilization()
    gpu_mem = get_gpu_memory()
    console.print(f"  Memory: {gpu_mem/1024:.1f} GB")
    console.print(f"  Utilization: {gpu_stats['gpu_util']:.0f}%")
    console.print(f"  Temperature: {gpu_stats['temperature']:.0f}°C")

    # 웜업
    if not args.skip_warmup:
        await warmup(args.url, REALISTIC_PROMPTS, args.max_tokens)

    # 벤치마크 실행
    all_results: List[BenchmarkResult] = []

    prompts_to_test = [("realistic", REALISTIC_PROMPTS, args.max_tokens)]
    if args.short_only:
        prompts_to_test = [("short", SHORT_PROMPTS, 100)]

    for prompt_type, prompts, max_tok in prompts_to_test:
        console.print(f"\n[bold magenta]{'#'*70}[/bold magenta]")
        console.print(f"[bold magenta]Testing with {prompt_type.upper()} prompts[/bold magenta]")
        console.print(f"[bold magenta]{'#'*70}[/bold magenta]")

        for concurrent in args.concurrent:
            result = await run_concurrent_benchmark(
                args.url,
                concurrent=concurrent,
                max_tokens=max_tok,
                prompts=prompts,
                prompt_type=prompt_type,
                runs=args.runs
            )
            all_results.append(result)

            # 쿨다운
            console.print("\n  [dim]Cooling down (15s)...[/dim]")
            await asyncio.sleep(15)

    # 결과 출력
    console.print("\n")
    print_results_table(all_results)

    # 결과 저장
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "benchmark_info": {
            "timestamp": datetime.now().isoformat(),
            "server_url": args.url,
            "max_output_tokens": args.max_tokens,
            "runs_per_level": args.runs,
            "test_type": "realistic",
            "gpu_model": "RTX 3090",
            "gpu_vram_gb": 24,
            "model": "google/gemma-3-12b-it",
            "quantization": "BitsAndBytes 4-bit"
        },
        "prompt_info": {
            "realistic_prompt_count": len(REALISTIC_PROMPTS),
            "avg_input_tokens_estimate": est_tokens,
            "max_output_tokens": args.max_tokens
        },
        "results": [asdict(r) for r in all_results]
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    console.print(f"\n[green]Results saved to: {output_path}[/green]")

    # 분석 요약
    console.print("\n[bold]=" * 70)
    console.print("[bold]Analysis Summary for GPU HW Selection[/bold]")
    console.print("[bold]=" * 70)

    realistic_results = [r for r in all_results if r.prompt_type == "realistic"]

    if realistic_results:
        # 안정적 동시 요청 수 (95%+ 성공률)
        stable_concurrent = 0
        for r in realistic_results:
            if r.success_rate >= 95:
                stable_concurrent = r.concurrent_requests
            else:
                break

        console.print(f"\n[cyan]Key Findings:[/cyan]")
        console.print(f"  Max Stable Concurrent (≥95% success): {stable_concurrent}")

        # 성능 지표
        if realistic_results:
            best = realistic_results[0]  # 1 concurrent baseline
            console.print(f"\n[cyan]Baseline (1 concurrent):[/cyan]")
            console.print(f"  Avg Input Tokens: {best.avg_input_tokens}")
            console.print(f"  Avg Output Tokens: {best.avg_output_tokens:.0f}")
            console.print(f"  Avg Latency: {best.avg_total_latency_ms/1000:.1f}s")
            console.print(f"  Throughput: {best.total_throughput_tokens_per_sec:.1f} tokens/sec")

        # KV Cache 추정
        console.print(f"\n[cyan]Estimated KV Cache Usage:[/cyan]")
        total_tokens_per_req = est_tokens + args.max_tokens
        console.print(f"  Tokens per request: ~{total_tokens_per_req}")
        console.print(f"  vLLM reported max (4096 tok/req): 8.62 concurrent")
        console.print(f"  Adjusted estimate ({total_tokens_per_req} tok/req): ~{int(8.62 * 4096 / total_tokens_per_req)}")


if __name__ == "__main__":
    asyncio.run(main())
