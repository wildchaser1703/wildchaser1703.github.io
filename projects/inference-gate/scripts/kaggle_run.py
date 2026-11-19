"""Run inside a FREE Kaggle GPU notebook after installing this project and vLLM.

No token, paid API, tunnel, or externally exposed server is required.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from gate.backends import MODEL, REVISION
from gate.experiment import ready


def main():
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "Select a GPU accelerator in the Kaggle notebook settings first."
        )
    if torch.cuda.get_device_capability(0)[0] < 7:
        raise SystemExit(
            "This vLLM experiment requires a T4 or newer GPU; select T4 rather than P100."
        )
    destination = Path("/kaggle/working/inference-gate-results")
    destination.mkdir(parents=True, exist_ok=True)
    inventory = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "model": MODEL,
        "revision": REVISION,
        "gpu_count_used": 1,
        "note": "One GPU only, including on dual-T4 Kaggle sessions.",
    }
    (destination / "hardware.json").write_text(json.dumps(inventory, indent=2))
    packages = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True
    )
    (destination / "packages.txt").write_text(packages)
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        MODEL,
        "--revision",
        REVISION,
        "--tokenizer-revision",
        REVISION,
        "--dtype",
        "half",
        "--max-model-len",
        "2048",
        "--gpu-memory-utilization",
        "0.5",
        "--max-num-seqs",
        "8",
        "--enforce-eager",
        "--host",
        "127.0.0.1",
        "--port",
        "8081",
    ]
    (destination / "engine-command.json").write_text(json.dumps(command, indent=2))
    with (destination / "engine.log").open("w") as log:
        engine = subprocess.Popen(
            command,
            env=os.environ | {"CUDA_VISIBLE_DEVICES": "0"},
            stdout=log,
            stderr=log,
        )
        try:
            ready("http://127.0.0.1:8081/health", engine, timeout=600)
            for rate in [2, 10, 30]:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "gate.experiment",
                        "--backend",
                        "vllm",
                        "--rate",
                        str(rate),
                        "--count",
                        "60",
                        "--budget",
                        "2048",
                        "--concurrency",
                        "8",
                        "--deadline",
                        "10",
                        "--repetitions",
                        "3",
                        "--output",
                        str(destination / f"rate-{rate}"),
                    ],
                    check=True,
                )
        finally:
            engine.terminate()
            try:
                engine.wait(timeout=30)
            except subprocess.TimeoutExpired:
                engine.kill()
                engine.wait()
    print(f"Finished. Download {destination} as the GPU evidence bundle.")


if __name__ == "__main__":
    main()
