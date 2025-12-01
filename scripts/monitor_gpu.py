#!/usr/bin/env python3
"""
GPU Memory Monitoring Script
실시간 GPU 메모리 사용량 모니터링 및 로깅
"""

import subprocess
import time
import json
import argparse
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


@dataclass
class GPUMetrics:
    """GPU 메트릭"""
    timestamp: str
    memory_used_mb: float
    memory_total_mb: float
    memory_free_mb: float
    memory_utilization_pct: float
    gpu_utilization_pct: float
    temperature_c: int
    power_draw_w: float


class GPUMonitor:
    """GPU 모니터링 클래스"""

    def __init__(self, gpu_index: int = 0, interval: float = 1.0):
        self.gpu_index = gpu_index
        self.interval = interval
        self.metrics: List[GPUMetrics] = []
        self.running = False
        self.handle = None

        if PYNVML_AVAILABLE:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

    def get_metrics_pynvml(self) -> GPUMetrics:
        """pynvml을 사용한 메트릭 수집"""
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
        temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)

        try:
            power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
        except pynvml.NVMLError:
            power = 0.0

        memory_used = mem_info.used / (1024 * 1024)
        memory_total = mem_info.total / (1024 * 1024)
        memory_free = mem_info.free / (1024 * 1024)

        return GPUMetrics(
            timestamp=datetime.now().isoformat(),
            memory_used_mb=memory_used,
            memory_total_mb=memory_total,
            memory_free_mb=memory_free,
            memory_utilization_pct=(memory_used / memory_total) * 100,
            gpu_utilization_pct=util.gpu,
            temperature_c=temp,
            power_draw_w=power
        )

    def get_metrics_nvidia_smi(self) -> GPUMetrics:
        """nvidia-smi를 사용한 메트릭 수집 (fallback)"""
        try:
            result = subprocess.run(
                [
                    'nvidia-smi',
                    '--query-gpu=memory.used,memory.total,memory.free,utilization.gpu,temperature.gpu,power.draw',
                    '--format=csv,noheader,nounits'
                ],
                capture_output=True, text=True
            )
            values = result.stdout.strip().split(', ')

            memory_used = float(values[0])
            memory_total = float(values[1])
            memory_free = float(values[2])
            gpu_util = float(values[3])
            temp = int(values[4])
            power = float(values[5]) if values[5] != '[N/A]' else 0.0

            return GPUMetrics(
                timestamp=datetime.now().isoformat(),
                memory_used_mb=memory_used,
                memory_total_mb=memory_total,
                memory_free_mb=memory_free,
                memory_utilization_pct=(memory_used / memory_total) * 100,
                gpu_utilization_pct=gpu_util,
                temperature_c=temp,
                power_draw_w=power
            )
        except Exception as e:
            print(f"Error getting metrics: {e}")
            return GPUMetrics(
                timestamp=datetime.now().isoformat(),
                memory_used_mb=0, memory_total_mb=0, memory_free_mb=0,
                memory_utilization_pct=0, gpu_utilization_pct=0,
                temperature_c=0, power_draw_w=0
            )

    def get_metrics(self) -> GPUMetrics:
        """메트릭 수집"""
        if PYNVML_AVAILABLE and self.handle:
            return self.get_metrics_pynvml()
        else:
            return self.get_metrics_nvidia_smi()

    def start_monitoring(self, duration: Optional[float] = None, output_file: Optional[str] = None):
        """모니터링 시작"""
        self.running = True
        start_time = time.time()

        print(f"Starting GPU monitoring (GPU {self.gpu_index}, interval: {self.interval}s)")
        print("Press Ctrl+C to stop\n")

        def signal_handler(sig, frame):
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)

        print(f"{'Time':^12} | {'Used MB':>10} | {'Total MB':>10} | {'Util %':>8} | {'GPU %':>7} | {'Temp':>6} | {'Power W':>8}")
        print("-" * 80)

        while self.running:
            if duration and (time.time() - start_time) >= duration:
                break

            metrics = self.get_metrics()
            self.metrics.append(metrics)

            # 콘솔 출력
            time_str = datetime.now().strftime("%H:%M:%S")
            print(f"{time_str:^12} | {metrics.memory_used_mb:>10.1f} | {metrics.memory_total_mb:>10.1f} | "
                  f"{metrics.memory_utilization_pct:>8.1f} | {metrics.gpu_utilization_pct:>7.0f} | "
                  f"{metrics.temperature_c:>6} | {metrics.power_draw_w:>8.1f}")

            time.sleep(self.interval)

        # 결과 저장
        if output_file:
            self.save_results(output_file)

        self.print_summary()

    def save_results(self, output_file: str):
        """결과 저장"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "monitoring_info": {
                "gpu_index": self.gpu_index,
                "interval_seconds": self.interval,
                "total_samples": len(self.metrics)
            },
            "metrics": [asdict(m) for m in self.metrics]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        print(f"\nResults saved to: {output_file}")

    def print_summary(self):
        """요약 출력"""
        if not self.metrics:
            return

        print("\n" + "=" * 60)
        print("GPU Monitoring Summary")
        print("=" * 60)

        memory_values = [m.memory_used_mb for m in self.metrics]
        gpu_util_values = [m.gpu_utilization_pct for m in self.metrics]

        print(f"Total samples: {len(self.metrics)}")
        print(f"\nMemory Usage (MB):")
        print(f"  Min:  {min(memory_values):.1f}")
        print(f"  Max:  {max(memory_values):.1f}")
        print(f"  Avg:  {sum(memory_values)/len(memory_values):.1f}")

        print(f"\nGPU Utilization (%):")
        print(f"  Min:  {min(gpu_util_values):.1f}")
        print(f"  Max:  {max(gpu_util_values):.1f}")
        print(f"  Avg:  {sum(gpu_util_values)/len(gpu_util_values):.1f}")

    def cleanup(self):
        """리소스 정리"""
        if PYNVML_AVAILABLE:
            pynvml.nvmlShutdown()


def get_current_metrics() -> Dict:
    """현재 GPU 메트릭 반환 (한 번만 조회)"""
    monitor = GPUMonitor()
    metrics = monitor.get_metrics()
    monitor.cleanup()
    return asdict(metrics)


def main():
    parser = argparse.ArgumentParser(description="GPU Memory Monitor")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    parser.add_argument("--interval", type=float, default=1.0, help="Monitoring interval (seconds)")
    parser.add_argument("--duration", type=float, default=None, help="Monitoring duration (seconds)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    parser.add_argument("--once", action="store_true", help="Get metrics once and exit")

    args = parser.parse_args()

    if args.once:
        metrics = get_current_metrics()
        print(json.dumps(metrics, indent=2))
        return

    monitor = GPUMonitor(gpu_index=args.gpu, interval=args.interval)
    try:
        monitor.start_monitoring(duration=args.duration, output_file=args.output)
    finally:
        monitor.cleanup()


if __name__ == "__main__":
    main()
