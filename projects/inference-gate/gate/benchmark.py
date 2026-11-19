"""Open-loop HTTP workload: reports every outcome and saves raw evidence."""

import argparse
import asyncio
import hashlib
import json
import math
import platform
import random
import time
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import httpx

from gate.telemetry import sample_gpu


def percentile(values, q):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * q) - 1)] if ordered else None


def package_versions():
    result = {}
    for name in ["fastapi", "httpx", "transformers", "torch", "vllm"]:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


async def benchmark(url, count=40, rate=10, seed=7, deadline=10):
    rng = random.Random(seed)
    trace = [
        {
            "prompt": "Explain why a database transaction matters. "
            * rng.choice([1, 8, 48]),
            "max_new_tokens": rng.choice([16, 32, 64]),
            "deadline_seconds": deadline,
        }
        for _ in range(count)
    ]
    rows = []
    async with httpx.AsyncClient(
        base_url=url,
        timeout=deadline + 15,
        trust_env=False,
        limits=httpx.Limits(max_connections=count),
    ) as client:
        state = (await client.get("/state")).json()
        start = time.monotonic()

        async def send(i, request):
            await asyncio.sleep(max(0, start + i / rate - time.monotonic()))
            sent = time.monotonic()
            try:
                response = await client.post("/generate", json=request)
                body = response.json()
                rows.append(
                    {
                        "index": i,
                        "status": response.status_code,
                        "latency_seconds": time.monotonic() - sent,
                        "schedule_lag_seconds": sent - start - i / rate,
                        "response": body,
                    }
                )
            except httpx.HTTPError as error:
                rows.append(
                    {
                        "index": i,
                        "status": "transport_error",
                        "latency_seconds": time.monotonic() - sent,
                        "error": type(error).__name__,
                    }
                )

        gpu_samples = []
        stop_sampling = asyncio.Event()
        sampling = asyncio.create_task(sample_gpu(stop_sampling, gpu_samples))
        try:
            await asyncio.gather(*(send(i, item) for i, item in enumerate(trace)))
        finally:
            stop_sampling.set()
            await sampling
        elapsed = time.monotonic() - start
        # Wait for timed-out backend work before starting another policy trial.
        for _ in range(240):
            final_state = (await client.get("/state")).json()
            if final_state["active"] == final_state["queued"] == 0:
                break
            await asyncio.sleep(0.5)
        else:
            raise RuntimeError(
                "backend did not drain; comparison cannot safely continue"
            )
    successes = [row for row in rows if row["status"] == 200]
    latencies = [row["latency_seconds"] for row in successes]
    return {
        "backend": state["backend"],
        "configuration": state,
        "final_state": final_state,
        "python": platform.python_version(),
        "packages": package_versions(),
        "platform": platform.platform(),
        "seed": seed,
        "offered_requests_per_second": rate,
        "trace_sha256": hashlib.sha256(
            json.dumps(trace, sort_keys=True).encode()
        ).hexdigest(),
        "summary": {
            "requests": count,
            "outcomes": dict(Counter(str(row["status"]) for row in rows)),
            "successful_requests_per_second": len(successes) / elapsed,
            "successful_output_tokens_per_second": sum(
                row["response"]["output_tokens"] for row in successes
            )
            / elapsed,
            "success_p50_seconds": percentile(latencies, 0.5),
            "success_p95_seconds": percentile(latencies, 0.95),
            "success_p99_seconds": percentile(latencies, 0.99),
            "elapsed_seconds": elapsed,
        },
        "requests": sorted(rows, key=lambda row: row["index"]),
        "gpu_samples": gpu_samples,
        "gpu_measurement_note": "Board-wide samples include engine preallocation; empty means no NVIDIA evidence.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--rate", type=float, default=10)
    parser.add_argument("--deadline", type=float, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="results/benchmark.json")
    args = parser.parse_args()
    if (
        not 1 <= args.count <= 10000
        or args.rate <= 0
        or not 0.01 <= args.deadline <= 120
    ):
        parser.error("invalid workload bounds")
    report = asyncio.run(
        benchmark(args.url, args.count, args.rate, args.seed, args.deadline)
    )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {"backend": report["backend"], "summary": report["summary"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
