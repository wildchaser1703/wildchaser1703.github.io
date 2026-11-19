from __future__ import annotations

import asyncio
import math
from collections import deque
from dataclasses import dataclass
from time import monotonic


class Rejected(Exception):
    def __init__(self, reason: str, status: int = 429):
        self.reason, self.status = reason, status
        super().__init__(reason)


@dataclass(frozen=True)
class KVProfile:
    layers: int
    kv_heads: int
    head_dim: int
    bytes_per_element: int = 2
    block_tokens: int = 16

    @property
    def bytes_per_token(self) -> int:
        return 2 * self.layers * self.kv_heads * self.head_dim * self.bytes_per_element

    def reservation(self, prompt_tokens: int, output_tokens: int) -> int:
        return (
            math.ceil((prompt_tokens + output_tokens) / self.block_tokens)
            * self.block_tokens
        )


class Admission:
    """FIFO queue with atomic reservations, bounded waiting, and no bypass.

    One instance belongs to one event loop and one backend replica. Multiple
    gateway processes would each have an independent budget and are unsupported.
    """

    def __init__(
        self, budget: int, concurrency: int, queue_limit: int, policy: str = "tokens"
    ):
        if (
            budget <= 0
            or concurrency <= 0
            or queue_limit < 0
            or policy not in {"tokens", "fixed"}
        ):
            raise ValueError("invalid admission configuration")
        self.budget, self.concurrency, self.queue_limit, self.policy = (
            budget,
            concurrency,
            queue_limit,
            policy,
        )
        self.used = 0
        self.active = 0
        self.peak_used = 0
        self.peak_active = 0
        self.queue: deque[object] = deque()
        self.condition = asyncio.Condition()
        self.healthy = True
        self.closed = False

    def fits(self, tokens: int) -> bool:
        return self.active < self.concurrency and (
            self.policy == "fixed" or self.used + tokens <= self.budget
        )

    async def acquire(self, tokens: int, deadline: float) -> None:
        if tokens <= 0:
            raise ValueError("reservation must be positive")
        if self.policy == "tokens" and tokens > self.budget:
            raise Rejected("request_exceeds_budget", 413)
        ticket = object()
        async with self.condition:
            if self.closed or not self.healthy:
                raise Rejected("backend_unavailable", 503)
            if self.queue or not self.fits(tokens):
                if len(self.queue) >= self.queue_limit:
                    raise Rejected("queue_full")
            self.queue.append(ticket)
            try:
                while True:
                    if self.closed or not self.healthy:
                        raise Rejected("backend_unavailable", 503)
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        raise Rejected("queue_deadline", 504)
                    if self.queue[0] is ticket and self.fits(tokens):
                        self.used += tokens
                        self.active += 1
                        self.peak_used = max(self.peak_used, self.used)
                        self.peak_active = max(self.peak_active, self.active)
                        return
                    try:
                        await asyncio.wait_for(self.condition.wait(), remaining)
                    except TimeoutError:
                        raise Rejected("queue_deadline", 504) from None
            finally:
                self.queue.remove(ticket)
                self.condition.notify_all()

    async def release(self, tokens: int) -> None:
        async with self.condition:
            self.used -= tokens
            self.active -= 1
            assert self.used >= 0 and self.active >= 0
            self.condition.notify_all()

    async def stop(self, unhealthy: bool = False) -> None:
        async with self.condition:
            if unhealthy:
                self.healthy = False
            else:
                self.closed = True
            self.condition.notify_all()

    def snapshot(self) -> dict:
        return {
            "policy": self.policy,
            "budget_tokens": self.budget,
            "reserved_tokens": self.used,
            "active": self.active,
            "queued": len(self.queue),
            "peak_reserved_tokens": self.peak_used,
            "peak_active": self.peak_active,
            "healthy": self.healthy,
            "closed": self.closed,
        }
