import json

import httpx
import pytest

from gate.backends import BackendUncertain, VLLMBackend


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return "USER " + messages[0]["content"] + " ASSISTANT"

    def encode(self, prompt, **kwargs):
        return prompt.split()


async def test_vllm_adapter_renders_and_counts_the_same_prompt():
    def respond(request):
        body = json.loads(request.content)
        assert body["prompt"] == "USER hello world ASSISTANT"
        assert body["max_tokens"] == 12
        assert body["add_special_tokens"] is False
        return httpx.Response(
            200,
            json={"choices": [{"text": "hello"}], "usage": {"completion_tokens": 1}},
        )

    backend = VLLMBackend.__new__(VLLMBackend)
    backend.tokenizer, backend.model_id = Tokenizer(), "test-model"
    backend.client = httpx.AsyncClient(
        base_url="http://engine", transport=httpx.MockTransport(respond)
    )
    try:
        assert backend.count("hello world") == 4
        result = await backend.generate("hello world", 12)
        assert result.text == "hello" and result.tokens == 1
    finally:
        await backend.close()


@pytest.mark.parametrize("status,body", [(500, {}), (200, {"choices": []})])
async def test_remote_errors_are_explicitly_uncertain(status, body):
    backend = VLLMBackend.__new__(VLLMBackend)
    backend.tokenizer, backend.model_id = Tokenizer(), "test-model"
    backend.client = httpx.AsyncClient(
        base_url="http://engine",
        transport=httpx.MockTransport(lambda _: httpx.Response(status, json=body)),
    )
    try:
        with pytest.raises(BackendUncertain):
            await backend.generate("hello", 1)
    finally:
        await backend.close()
