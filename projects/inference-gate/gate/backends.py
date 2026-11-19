from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

import httpx

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"


@dataclass
class Output:
    text: str
    tokens: int


class BackendUncertain(Exception):
    """The remote engine may still be executing; block new admissions."""


def render_prompt(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


class TransformersBackend:
    def __init__(self, model_id: str = MODEL, revision: str = REVISION):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, trust_remote_code=False
        )
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_id, revision=revision, trust_remote_code=False
            )
            .to(self.device)
            .eval()
        )
        # HF generate is serialized: this adapter proves integration on a laptop.
        # Use vLLM for the GPU concurrency experiment.
        self.lock = asyncio.Lock()

    def count(self, prompt: str) -> int:
        return len(
            self.tokenizer.encode(
                render_prompt(self.tokenizer, prompt), add_special_tokens=False
            )
        )

    def _generate(self, prompt: str, limit: int) -> Output:
        import torch

        inputs = self.tokenizer(
            render_prompt(self.tokenizer, prompt),
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self.device)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=limit,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0, inputs["input_ids"].shape[1] :]
        return Output(
            self.tokenizer.decode(generated, skip_special_tokens=True), len(generated)
        )

    async def generate(self, prompt: str, limit: int) -> Output:
        async with self.lock:
            return await asyncio.to_thread(self._generate, prompt, limit)

    async def close(self) -> None:
        pass


class VLLMBackend:
    def __init__(self, base_url: str, model_id: str = MODEL, revision: str = REVISION):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, trust_remote_code=False
        )
        self.model_id = model_id
        self.client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=120, trust_env=False
        )

    def count(self, prompt: str) -> int:
        return len(
            self.tokenizer.encode(
                render_prompt(self.tokenizer, prompt), add_special_tokens=False
            )
        )

    async def generate(self, prompt: str, limit: int) -> Output:
        try:
            response = await self.client.post(
                "/v1/completions",
                json={
                    "model": self.model_id,
                    "prompt": render_prompt(self.tokenizer, prompt),
                    "max_tokens": limit,
                    "temperature": 0,
                    "stream": False,
                    "add_special_tokens": False,
                },
            )
            response.raise_for_status()
            body = response.json()
            return Output(
                body["choices"][0]["text"], body["usage"]["completion_tokens"]
            )
        except (httpx.HTTPError, ValueError, KeyError, IndexError) as error:
            raise BackendUncertain(
                "upstream outcome unknown; restart gateway after checking engine"
            ) from error

    async def close(self) -> None:
        await self.client.aclose()


class SyntheticBackend:
    """Explicit test fixture; its output and timings are never model evidence."""

    def count(self, prompt: str) -> int:
        return len(prompt.split())

    async def generate(self, prompt: str, limit: int) -> Output:
        await asyncio.sleep(0.02 + limit * 0.002)
        return Output("synthetic " * limit, limit)

    async def close(self) -> None:
        pass
