"""Run identical open-loop traces against both admission policies."""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

from gate.benchmark import benchmark


def ready(url, process, timeout=180):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("server exited; inspect its log")
        try:
            if httpx.get(url, timeout=1, trust_env=False).is_success:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    raise TimeoutError("server startup timed out")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=["synthetic", "transformers", "vllm"], default="synthetic"
    )
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--rate", type=float, default=10)
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--queue", type=int, default=8)
    parser.add_argument("--deadline", type=float, default=5)
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", default="results/comparison")
    args = parser.parse_args()
    if not 1 <= args.count <= 10000 or args.rate <= 0 or args.repetitions < 1:
        parser.error("invalid experiment bounds")
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    reports = []
    for repeat in range(args.repetitions):
        # Reverse trial order on alternating repetitions to reduce order bias.
        policies = ["fixed", "tokens"] if repeat % 2 == 0 else ["tokens", "fixed"]
        for policy in policies:
            name = f"{repeat}-{policy}"
            environment = os.environ | {
                "BACKEND": args.backend,
                "POLICY": policy,
                "TOKEN_BUDGET": str(args.budget),
                "MAX_CONCURRENCY": str(args.concurrency),
                "QUEUE_LIMIT": str(args.queue),
            }
            with (directory / f"{name}.log").open("w") as log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "uvicorn",
                        "gate.api:app",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(args.port),
                    ],
                    env=environment,
                    stdout=log,
                    stderr=log,
                )
                try:
                    url = f"http://127.0.0.1:{args.port}"
                    ready(url + "/health", process)
                    warm = httpx.post(
                        url + "/generate",
                        json={"prompt": "Hello", "max_new_tokens": 4},
                        timeout=120,
                        trust_env=False,
                    )
                    warm.raise_for_status()
                    report = asyncio.run(
                        benchmark(url, args.count, args.rate, 7 + repeat, args.deadline)
                    )
                    (directory / f"{name}.json").write_text(
                        json.dumps(report, indent=2) + "\n"
                    )
                    reports.append(
                        {
                            "repeat": repeat,
                            "policy": policy,
                            "trace_sha256": report["trace_sha256"],
                            "summary": report["summary"],
                            "state": report["final_state"],
                        }
                    )
                    print(name, json.dumps(report["summary"]), flush=True)
                finally:
                    process.terminate()
                    try:
                        process.wait(timeout=150)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
    (directory / "summary.json").write_text(
        json.dumps({"backend": args.backend, "trials": reports}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
