#!/usr/bin/env python3
"""
Quick Test Script
서비스 빠른 테스트를 위한 스크립트
"""

import argparse
import json
import sys
import time
from typing import Optional

try:
    import httpx
except ImportError:
    print("Error: httpx not installed. Run: pip install httpx")
    sys.exit(1)


def test_health(base_url: str, timeout: float = 10.0) -> bool:
    """Health check"""
    print("Testing /health endpoint...", end=" ")
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(f"{base_url}/health")
            if response.status_code == 200:
                print("✓ OK")
                return True
            else:
                print(f"✗ Status: {response.status_code}")
                return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_models(base_url: str, timeout: float = 10.0) -> Optional[str]:
    """Get available models"""
    print("Testing /v1/models endpoint...", end=" ")
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(f"{base_url}/v1/models")
            if response.status_code == 200:
                data = response.json()
                models = [m["id"] for m in data.get("data", [])]
                print(f"✓ Found {len(models)} model(s)")
                for model in models:
                    print(f"  - {model}")
                return models[0] if models else None
            else:
                print(f"✗ Status: {response.status_code}")
                return None
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def test_inference(
    base_url: str,
    model: str,
    prompt: str = "Translate to Korean: Hello, how are you?",
    max_tokens: int = 100,
    timeout: float = 120.0,
    stream: bool = False
) -> bool:
    """Test inference"""
    print(f"\nTesting inference (stream={stream})...")
    print(f"  Prompt: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
    print(f"  Max tokens: {max_tokens}")

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": stream
    }

    try:
        start_time = time.perf_counter()

        with httpx.Client(timeout=timeout) as client:
            if stream:
                # Streaming request
                with client.stream(
                    "POST",
                    f"{base_url}/v1/chat/completions",
                    json=payload
                ) as response:
                    if response.status_code != 200:
                        print(f"✗ Status: {response.status_code}")
                        return False

                    ttft = None
                    full_content = ""

                    print("  Response: ", end="", flush=True)
                    for line in response.iter_lines():
                        if line.startswith("data: ") and line != "data: [DONE]":
                            if ttft is None:
                                ttft = (time.perf_counter() - start_time) * 1000

                            try:
                                data = json.loads(line[6:])
                                delta = data.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content", "")
                                full_content += content
                                print(content, end="", flush=True)
                            except json.JSONDecodeError:
                                pass

                    print()  # newline after streaming
            else:
                # Non-streaming request
                response = client.post(
                    f"{base_url}/v1/chat/completions",
                    json=payload
                )
                if response.status_code != 200:
                    print(f"✗ Status: {response.status_code}")
                    print(f"  Body: {response.text[:200]}")
                    return False

                ttft = (time.perf_counter() - start_time) * 1000
                data = response.json()
                full_content = data["choices"][0]["message"]["content"]
                print(f"  Response: {full_content[:200]}{'...' if len(full_content) > 200 else ''}")

        total_time = (time.perf_counter() - start_time) * 1000
        print(f"\n  ✓ Inference successful")
        print(f"  TTFT: {ttft:.0f}ms")
        print(f"  Total time: {total_time:.0f}ms")
        print(f"  Output length: {len(full_content)} chars")

        return True

    except httpx.TimeoutException:
        print(f"✗ Timeout after {timeout}s")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Quick vLLM Service Test")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="vLLM server URL"
    )
    parser.add_argument(
        "--prompt",
        default="Translate to Korean: Hello, how are you today?",
        help="Test prompt"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Max output tokens"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Request timeout (seconds)"
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming mode"
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip inference test"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("vLLM Quick Test")
    print("=" * 60)
    print(f"Server: {args.url}")
    print()

    # Run tests
    results = []

    results.append(("Health", test_health(args.url)))
    model = test_models(args.url)
    results.append(("Models", model is not None))

    if not args.skip_inference and model:
        results.append((
            "Inference",
            test_inference(
                args.url,
                model,
                args.prompt,
                args.max_tokens,
                args.timeout,
                args.stream
            )
        ))

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All tests passed! ✓")
        sys.exit(0)
    else:
        print("Some tests failed ✗")
        sys.exit(1)


if __name__ == "__main__":
    main()
