#!/usr/bin/env python3
"""
vLLM Concurrent Request Benchmark Script
RTX 3090에서 동시 처리 성능 측정
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
from dataclasses import dataclass, asdict
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import subprocess

console = Console()

# 테스트용 프롬프트
TEST_PROMPTS = [
    "Translate the following English text to Korean: 'The quick brown fox jumps over the lazy dog.'",
    "Translate to Korean: 'Machine learning is transforming how we interact with technology.'",
    "Translate this sentence to Korean: 'Artificial intelligence will change the world.'",
    "Please translate to Korean: 'The weather is beautiful today and perfect for a walk.'",
    "Translate into Korean: 'Programming is a valuable skill in the modern economy.'",
]

@dataclass
class RequestResult:
    """단일 요청 결과"""
    request_id: int
    success: bool
    ttft_ms: float  # Time to First Token
    total_latency_ms: float
    input_tokens: int
    output_tokens: int
    error: Optional[str] = None

@dataclass
class BenchmarkResult:
    """벤치마크 결과"""
    test_id: str
    timestamp: str
    concurrent_requests: int
    input_tokens: int
    max_output_tokens: int
    runs: int

    # 집계 결과
    avg_ttft_ms: float
    min_ttft_ms: float
    max_ttft_ms: float
    std_ttft_ms: float

    avg_total_latency_ms: float
    min_total_latency_ms: float
    max_total_latency_ms: float
    std_total_latency_ms: float

    avg_output_tokens: float
    total_throughput_tokens_per_sec: float

    success_rate: float
    peak_memory_mb: float

    individual_results: List[Dict[str, Any]]
    errors: List[str]


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


async def make_single_request(
    session: aiohttp.ClientSession,
    request_id: int,
    base_url: str,
    prompt: str,
    max_tokens: int,
    timeout: float = 120.0
) -> RequestResult:
    """단일 API 요청 실행"""

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
                    input_tokens=len(prompt.split()),
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

            return RequestResult(
                request_id=request_id,
                success=True,
                ttft_ms=ttft,
                total_latency_ms=total_latency,
                input_tokens=len(prompt.split()),
                output_tokens=max(output_tokens, 1)
            )

    except asyncio.TimeoutError:
        return RequestResult(
            request_id=request_id,
            success=False,
            ttft_ms=0,
            total_latency_ms=timeout * 1000,
            input_tokens=len(prompt.split()),
            output_tokens=0,
            error="Timeout"
        )
    except Exception as e:
        return RequestResult(
            request_id=request_id,
            success=False,
            ttft_ms=0,
            total_latency_ms=(time.perf_counter() - start_time) * 1000,
            input_tokens=len(prompt.split()),
            output_tokens=0,
            error=str(e)
        )


async def run_concurrent_benchmark(
    base_url: str,
    concurrent: int,
    max_tokens: int,
    runs: int = 3
) -> BenchmarkResult:
    """동시 요청 벤치마크 실행"""

    console.print(f"\n[bold cyan]Running benchmark: {concurrent} concurrent requests[/bold cyan]")

    all_results: List[RequestResult] = []
    peak_memory = 0.0

    connector = aiohttp.TCPConnector(limit=concurrent + 10)

    async with aiohttp.ClientSession(connector=connector) as session:
        for run_idx in range(runs):
            console.print(f"  Run {run_idx + 1}/{runs}...", end=" ")

            # GPU 메모리 체크
            pre_memory = get_gpu_memory()

            # 프롬프트 준비
            prompts = [TEST_PROMPTS[i % len(TEST_PROMPTS)] for i in range(concurrent)]

            # 동시 요청 실행
            tasks = [
                make_single_request(session, i, base_url, prompt, max_tokens)
                for i, prompt in enumerate(prompts)
            ]

            results = await asyncio.gather(*tasks)
            all_results.extend(results)

            # GPU 메모리 체크
            post_memory = get_gpu_memory()
            peak_memory = max(peak_memory, post_memory)

            # 성공률 표시
            success_count = sum(1 for r in results if r.success)
            console.print(f"[green]{success_count}/{concurrent} succeeded[/green]")

            # 쿨다운
            if run_idx < runs - 1:
                await asyncio.sleep(2)

    # 결과 집계
    successful_results = [r for r in all_results if r.success]
    errors = [r.error for r in all_results if r.error]

    if not successful_results:
        return BenchmarkResult(
            test_id=f"T{concurrent}",
            timestamp=datetime.now().isoformat(),
            concurrent_requests=concurrent,
            input_tokens=len(TEST_PROMPTS[0].split()),
            max_output_tokens=max_tokens,
            runs=runs,
            avg_ttft_ms=0, min_ttft_ms=0, max_ttft_ms=0, std_ttft_ms=0,
            avg_total_latency_ms=0, min_total_latency_ms=0, max_total_latency_ms=0, std_total_latency_ms=0,
            avg_output_tokens=0,
            total_throughput_tokens_per_sec=0,
            success_rate=0,
            peak_memory_mb=peak_memory,
            individual_results=[asdict(r) for r in all_results],
            errors=errors
        )

    ttfts = [r.ttft_ms for r in successful_results]
    latencies = [r.total_latency_ms for r in successful_results]
    output_tokens_list = [r.output_tokens for r in successful_results]

    total_tokens = sum(output_tokens_list)
    total_time_sec = sum(latencies) / 1000
    throughput = total_tokens / total_time_sec if total_time_sec > 0 else 0

    return BenchmarkResult(
        test_id=f"T{concurrent}",
        timestamp=datetime.now().isoformat(),
        concurrent_requests=concurrent,
        input_tokens=len(TEST_PROMPTS[0].split()),
        max_output_tokens=max_tokens,
        runs=runs,
        avg_ttft_ms=statistics.mean(ttfts),
        min_ttft_ms=min(ttfts),
        max_ttft_ms=max(ttfts),
        std_ttft_ms=statistics.stdev(ttfts) if len(ttfts) > 1 else 0,
        avg_total_latency_ms=statistics.mean(latencies),
        min_total_latency_ms=min(latencies),
        max_total_latency_ms=max(latencies),
        std_total_latency_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0,
        avg_output_tokens=statistics.mean(output_tokens_list),
        total_throughput_tokens_per_sec=throughput,
        success_rate=len(successful_results) / len(all_results) * 100,
        peak_memory_mb=peak_memory,
        individual_results=[asdict(r) for r in all_results],
        errors=errors[:10]  # 최대 10개 에러만
    )


async def check_server_health(base_url: str) -> bool:
    """서버 상태 확인"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/health", timeout=aiohttp.ClientTimeout(total=10)) as resp:
                return resp.status == 200
    except Exception:
        return False


async def warmup(base_url: str, count: int = 3):
    """웜업 요청"""
    console.print("\n[yellow]Warming up...[/yellow]")

    async with aiohttp.ClientSession() as session:
        for i in range(count):
            result = await make_single_request(
                session, i, base_url, TEST_PROMPTS[0], max_tokens=50
            )
            status = "[green]OK[/green]" if result.success else f"[red]FAIL: {result.error}[/red]"
            console.print(f"  Warmup {i+1}/{count}: {status}")
            await asyncio.sleep(1)


def print_results_table(results: List[BenchmarkResult]):
    """결과 테이블 출력"""
    table = Table(title="Benchmark Results Summary")

    table.add_column("Concurrent", justify="right", style="cyan")
    table.add_column("Success %", justify="right", style="green")
    table.add_column("Avg TTFT (ms)", justify="right")
    table.add_column("Avg Latency (ms)", justify="right")
    table.add_column("Throughput (tok/s)", justify="right", style="yellow")
    table.add_column("Peak Memory (MB)", justify="right", style="magenta")

    for r in results:
        table.add_row(
            str(r.concurrent_requests),
            f"{r.success_rate:.1f}%",
            f"{r.avg_ttft_ms:.1f}",
            f"{r.avg_total_latency_ms:.1f}",
            f"{r.total_throughput_tokens_per_sec:.1f}",
            f"{r.peak_memory_mb:.0f}"
        )

    console.print(table)


async def main():
    parser = argparse.ArgumentParser(description="vLLM Concurrent Benchmark")
    parser.add_argument("--url", default="http://localhost:8000", help="vLLM server URL")
    parser.add_argument("--concurrent", type=int, nargs="+", default=[1, 2, 4, 8, 16],
                        help="Concurrent request counts to test")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max output tokens")
    parser.add_argument("--runs", type=int, default=3, help="Runs per concurrent level")
    parser.add_argument("--output", default="results/concurrent_results.json", help="Output file")
    parser.add_argument("--skip-warmup", action="store_true", help="Skip warmup")

    args = parser.parse_args()

    console.print("[bold]vLLM Concurrent Request Benchmark[/bold]")
    console.print(f"Server: {args.url}")
    console.print(f"Test levels: {args.concurrent}")
    console.print(f"Max tokens: {args.max_tokens}")
    console.print(f"Runs per level: {args.runs}")

    # 서버 상태 확인
    console.print("\n[yellow]Checking server health...[/yellow]")
    if not await check_server_health(args.url):
        console.print("[red]Server is not responding. Please start the vLLM server first.[/red]")
        return
    console.print("[green]Server is healthy![/green]")

    # 웜업
    if not args.skip_warmup:
        await warmup(args.url)

    # 벤치마크 실행
    all_results: List[BenchmarkResult] = []

    for concurrent in args.concurrent:
        result = await run_concurrent_benchmark(
            args.url,
            concurrent=concurrent,
            max_tokens=args.max_tokens,
            runs=args.runs
        )
        all_results.append(result)

        # 쿨다운
        console.print("  Cooling down...")
        await asyncio.sleep(5)

    # 결과 출력
    print_results_table(all_results)

    # 결과 저장
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "benchmark_info": {
            "timestamp": datetime.now().isoformat(),
            "server_url": args.url,
            "max_tokens": args.max_tokens,
            "runs_per_level": args.runs
        },
        "results": [asdict(r) for r in all_results]
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    console.print(f"\n[green]Results saved to: {output_path}[/green]")

    # 한계점 분석
    console.print("\n[bold]Analysis Summary:[/bold]")

    # 성공률 100% 유지되는 최대 동시 요청 수
    max_stable = 0
    for r in all_results:
        if r.success_rate >= 95:
            max_stable = r.concurrent_requests
        else:
            break

    console.print(f"  Max stable concurrent requests (95%+ success): [cyan]{max_stable}[/cyan]")

    # Latency가 baseline의 2배 이상인 지점
    if len(all_results) > 1:
        baseline_latency = all_results[0].avg_total_latency_ms
        for r in all_results[1:]:
            if r.avg_total_latency_ms > baseline_latency * 2:
                console.print(f"  Latency doubles at: [yellow]{r.concurrent_requests} concurrent[/yellow]")
                break


if __name__ == "__main__":
    asyncio.run(main())
