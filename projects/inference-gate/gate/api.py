from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from time import monotonic
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field
from starlette.responses import Response

from gate.admission import Admission, Rejected
from gate.backends import (
    BackendUncertain,
    SyntheticBackend,
    TransformersBackend,
    VLLMBackend,
)


class Completion(BaseModel):
    prompt: str = Field(min_length=1, max_length=16384)
    max_new_tokens: int = Field(default=64, ge=1, le=1024)
    deadline_seconds: float = Field(default=30, ge=0.01, le=120)


def create_app(backend=None, admission=None, context_limit: int = 2048) -> FastAPI:
    admission = admission or Admission(
        int(os.getenv("TOKEN_BUDGET", "4096")),
        int(os.getenv("MAX_CONCURRENCY", "4")),
        int(os.getenv("QUEUE_LIMIT", "16")),
        os.getenv("POLICY", "tokens"),
    )
    tasks: set[asyncio.Task] = set()
    registry = CollectorRegistry()
    outcomes = Counter(
        "gate_requests_total", "Request outcomes", ["outcome"], registry=registry
    )
    latency = Histogram(
        "gate_latency_seconds", "Client-visible request latency", registry=registry
    )
    queue_time = Histogram(
        "gate_queue_seconds", "Time waiting for admission", registry=registry
    )
    generated = Counter(
        "gate_generated_tokens_total", "Completed backend tokens", registry=registry
    )
    for name, callback in [
        ("reserved_tokens", lambda: admission.used),
        ("active", lambda: admission.active),
        ("queued", lambda: len(admission.queue)),
    ]:
        Gauge("gate_" + name, name, registry=registry).set_function(callback)

    @asynccontextmanager
    async def lifespan(app):
        nonlocal backend
        if backend is None:
            mode = os.getenv("BACKEND", "synthetic")
            if mode == "synthetic":
                backend = SyntheticBackend()
            elif mode == "transformers":
                backend = TransformersBackend()
            elif mode == "vllm":
                backend = VLLMBackend(os.getenv("VLLM_URL", "http://127.0.0.1:8081"))
            else:
                raise ValueError("BACKEND must be synthetic, transformers or vllm")
        app.state.backend = backend
        yield
        await admission.stop()
        # Never release a reservation while a background generation still runs.
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await backend.close()

    app = FastAPI(
        title="Inference Gate", lifespan=lifespan, docs_url=None, redoc_url=None
    )
    app.mount(
        "/docs-assets",
        StaticFiles(directory=Path(__file__).parent / "static"),
        name="docs-assets",
    )

    @app.get("/docs", include_in_schema=False)
    async def documentation():
        return get_swagger_ui_html(
            openapi_url="/openapi.json",
            title="Inference Gate API",
            swagger_js_url="/docs-assets/swagger-ui-bundle.js",
            swagger_css_url="/docs-assets/swagger-ui.css",
            swagger_favicon_url="/docs-assets/favicon-32x32.png",
        )

    app.state.admission = admission
    app.state.tasks = tasks

    @app.get("/health")
    async def health():
        if admission.closed or not admission.healthy:
            raise HTTPException(503, "admission stopped")
        return {"status": "ok", "backend": type(backend).__name__}

    @app.get("/state")
    async def state():
        return admission.snapshot() | {
            "backend": type(backend).__name__,
            "context_limit": context_limit,
            "device": getattr(
                backend,
                "device",
                "remote" if isinstance(backend, VLLMBackend) else "synthetic",
            ),
        }

    @app.get("/metrics")
    async def metrics():
        return Response(
            generate_latest(registry), media_type="text/plain; version=0.0.4"
        )

    async def execute(prompt, limit, tokens):
        try:
            result = await backend.generate(prompt, limit)
            generated.inc(result.tokens)
            return result
        except Exception:
            # A failed request is not proof the remote GPU has stopped working.
            await admission.stop(unhealthy=True)
            raise
        finally:
            await admission.release(tokens)

    @app.post("/generate")
    async def generate(request: Completion):
        start = monotonic()
        deadline = start + request.deadline_seconds
        # Tokenization has a bounded input and does not occupy the event loop.
        count = await asyncio.to_thread(backend.count, request.prompt)
        if count < 1 or count + request.max_new_tokens > context_limit:
            outcomes.labels("context_exceeded").inc()
            raise HTTPException(413, "prompt plus output exceeds context limit")
        # Reserve complete 16-token cache blocks, including maximum output.
        tokens = ((count + request.max_new_tokens + 15) // 16) * 16
        try:
            queued_at = monotonic()
            await admission.acquire(tokens, deadline)
            waited = monotonic() - queued_at
            queue_time.observe(waited)
            task = asyncio.create_task(
                execute(request.prompt, request.max_new_tokens, tokens)
            )
            tasks.add(task)

            def done(completed):
                tasks.discard(completed)
                if not completed.cancelled():
                    completed.exception()  # Retrieve failures after client timeout.

            task.add_done_callback(done)
            try:
                result = await asyncio.wait_for(
                    asyncio.shield(task), max(0, deadline - monotonic())
                )
            except TimeoutError:
                raise Rejected("execution_deadline", 504) from None
            outcomes.labels("succeeded").inc()
            return {
                "id": str(uuid4()),
                "text": result.text,
                "prompt_tokens": count,
                "output_tokens": result.tokens,
                "queue_seconds": waited,
                "latency_seconds": monotonic() - start,
                "reserved_tokens": tokens,
            }
        except Rejected as error:
            outcomes.labels(error.reason).inc()
            raise HTTPException(
                error.status,
                error.reason,
                headers={"Retry-After": "1"} if error.status == 429 else None,
            ) from error
        except BackendUncertain as error:
            outcomes.labels("backend_uncertain").inc()
            raise HTTPException(503, str(error)) from error
        except asyncio.CancelledError:
            outcomes.labels("client_cancelled").inc()
            raise
        finally:
            latency.observe(monotonic() - start)

    return app


app = create_app()
