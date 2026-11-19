import asyncio
from time import monotonic

import httpx
import pytest

from gate.admission import Admission, KVProfile, Rejected
from gate.api import create_app
from gate.backends import BackendUncertain, Output, SyntheticBackend


async def test_budget_under_contention():
    gate = Admission(100, 5, 100)

    async def run():
        await gate.acquire(30, monotonic() + 3)
        assert gate.used <= 100
        await asyncio.sleep(0.001)
        await gate.release(30)

    await asyncio.gather(*(run() for _ in range(60)))
    assert gate.used == gate.active == 0
    assert gate.peak_used == 90 and gate.peak_active == 3


async def test_fifo_large_request_does_not_starve():
    gate = Admission(100, 4, 4)
    await gate.acquire(60, monotonic() + 3)
    order = []

    async def run(size):
        await gate.acquire(size, monotonic() + 3)
        order.append(size)
        await gate.release(size)

    large = asyncio.create_task(run(80))
    await asyncio.sleep(0)
    small = asyncio.create_task(run(20))
    await asyncio.sleep(0)
    assert order == []
    await gate.release(60)
    await asyncio.gather(large, small)
    assert order == [80, 20]


async def test_queue_full_and_oversize():
    gate = Admission(32, 1, 0)
    with pytest.raises(Rejected, match="request_exceeds_budget"):
        await gate.acquire(33, monotonic() + 1)
    await gate.acquire(16, monotonic() + 1)
    with pytest.raises(Rejected, match="queue_full"):
        await gate.acquire(16, monotonic() + 1)
    await gate.release(16)


async def test_queue_timeout_removes_ticket():
    gate = Admission(32, 1, 2)
    await gate.acquire(16, monotonic() + 1)
    with pytest.raises(Rejected, match="queue_deadline"):
        await gate.acquire(16, monotonic() + 0.01)
    assert not gate.queue
    await gate.release(16)


async def test_cancel_waiter_cleans_queue():
    gate = Admission(32, 1, 2)
    await gate.acquire(16, monotonic() + 1)
    waiter = asyncio.create_task(gate.acquire(16, monotonic() + 30))
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert not gate.queue
    await gate.release(16)


async def test_fixed_baseline_can_exceed_token_budget():
    gate = Admission(32, 2, 0, "fixed")
    await gate.acquire(32, monotonic() + 1)
    await gate.acquire(32, monotonic() + 1)
    assert gate.used == 64
    await gate.release(32)
    await gate.release(32)


def test_gqa_memory_formula_and_block_rounding():
    profile = KVProfile(layers=30, kv_heads=3, head_dim=64)
    assert profile.bytes_per_token == 23040
    assert profile.reservation(17, 16) == 48


class ControlledBackend(SyntheticBackend):
    def __init__(self):
        self.started = asyncio.Event()
        self.finish = asyncio.Event()

    async def generate(self, prompt, limit):
        self.started.set()
        await self.finish.wait()
        return Output("done", 1)


async def test_execution_deadline_retains_reservation():
    backend = ControlledBackend()
    gate = Admission(64, 1, 0)
    app = create_app(backend, gate)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:
            try:
                response = await client.post(
                    "/generate",
                    json={
                        "prompt": "hello",
                        "max_new_tokens": 8,
                        "deadline_seconds": 1,
                    },
                )
                assert response.status_code == 504
                assert response.json()["detail"] == "execution_deadline"
                assert gate.active == 1 and gate.used == 16
                assert (
                    await client.post(
                        "/generate", json={"prompt": "hello", "max_new_tokens": 8}
                    )
                ).status_code == 429
            finally:
                backend.finish.set()
                await asyncio.gather(*app.state.tasks)
            assert gate.active == gate.used == 0


async def test_cancel_running_request_retains_reservation():
    backend = ControlledBackend()
    gate = Admission(64, 1, 0)
    app = create_app(backend, gate)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:
            request = asyncio.create_task(
                client.post("/generate", json={"prompt": "hello", "max_new_tokens": 8})
            )
            await backend.started.wait()
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
            assert gate.active == 1
            backend.finish.set()
            await asyncio.gather(*app.state.tasks)
            assert gate.active == 0


async def test_backend_uncertainty_stops_admission():
    class Broken(SyntheticBackend):
        async def generate(self, *_):
            raise BackendUncertain("lost response")

    gate = Admission(128, 1, 1)
    app = create_app(Broken(), gate)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:
            assert (
                await client.post("/generate", json={"prompt": "hello"})
            ).status_code == 503
            assert (await client.get("/health")).status_code == 503
            assert (
                await client.post("/generate", json={"prompt": "hello"})
            ).status_code == 503


async def test_api_success_context_validation_and_metrics():
    app = create_app(SyntheticBackend())
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:
            result = await client.post(
                "/generate", json={"prompt": "hello world", "max_new_tokens": 2}
            )
            assert result.status_code == 200
            assert result.json()["prompt_tokens"] == 2
            assert result.json()["output_tokens"] == 2
            assert (
                await client.post("/generate", json={"prompt": "a " * 2048})
            ).status_code == 413
            assert (
                await client.post(
                    "/generate", json={"prompt": "a", "max_new_tokens": 0}
                )
            ).status_code == 422
            assert (
                'gate_requests_total{outcome="succeeded"} 1.0'
                in (await client.get("/metrics")).text
            )


async def test_shutdown_wakes_waiters():
    gate = Admission(32, 1, 2)
    await gate.acquire(32, monotonic() + 1)
    task = asyncio.create_task(gate.acquire(16, monotonic() + 30))
    await asyncio.sleep(0)
    await gate.stop()
    with pytest.raises(Rejected, match="backend_unavailable"):
        await task
    await gate.release(32)
