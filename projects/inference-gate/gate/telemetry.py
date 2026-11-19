"""Optional board-wide NVIDIA samples; absent hardware is explicitly reported."""

import asyncio
import shutil
from time import monotonic


async def sample_gpu(stop: asyncio.Event, samples: list):
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return
    while not stop.is_set():
        process = await asyncio.create_subprocess_exec(
            executable,
            "--id=0",
            "--query-gpu=memory.used,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=3)
            samples.append(
                {
                    "monotonic_seconds": monotonic(),
                    "board_memory_mib_utilization_percent_power_w": stdout.decode().strip(),
                    "exit_code": process.returncode,
                }
            )
        except TimeoutError:
            process.kill()
            await process.wait()
            samples.append({"error": "nvidia-smi timeout"})
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.5)
        except TimeoutError:
            pass
