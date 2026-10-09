import pytest
from typing import List, Optional
import httpx
from pydantic import BaseModel, Field

from backend.app.services.llm.base import BaseLLMClient
from backend.app.services.llm.ollama_client import OllamaLLMClient, extract_json_payload
from backend.app.services.llm.mock_client import MockLLMClient
from backend.app.services.llm.factory import get_llm_client
from backend.app.core.exceptions import AppException


class SampleNestedItem(BaseModel):
    item_id: int
    name: str


class SampleStructuredOutput(BaseModel):
    title: str = Field(..., description="Post headline")
    bullets: List[str] = Field(default_factory=list, description="Key points")
    score: float = Field(default=0.9)
    items: List[SampleNestedItem] = Field(default_factory=list)


def test_extract_json_payload():
    raw_plain = '{"key": "value"}'
    assert extract_json_payload(raw_plain) == '{"key": "value"}'

    raw_markdown = '```json\n{"key": "value"}\n```'
    assert extract_json_payload(raw_markdown) == '{"key": "value"}'

    raw_preamble = 'Sure! Here is your requested JSON response:\n{"key": "value"}\nHope this helps!'
    assert extract_json_payload(raw_preamble) == '{"key": "value"}'


@pytest.mark.asyncio
async def test_mock_llm_client_generation():
    client = MockLLMClient()
    assert client.model_name == "mock-llama3.2"

    raw_text = await client.generate("Analyze the system architecture")
    assert "Mock completion" in raw_text

    structured = await client.generate_structured(
        prompt="Synthesize executive summary",
        schema=SampleStructuredOutput,
    )
    assert isinstance(structured, SampleStructuredOutput)
    assert len(structured.title) > 0
    assert len(structured.bullets) >= 1
    assert len(structured.items) >= 1
    assert isinstance(structured.items[0], SampleNestedItem)

    # Health check
    assert await client.check_health() is True

    # Empty prompt error
    with pytest.raises(AppException) as exc_info:
        await client.generate("")
    assert exc_info.value.code == "EMPTY_PROMPT_ERROR"


@pytest.mark.asyncio
async def test_ollama_client_mocked_http_success(monkeypatch):
    async def mock_post(self, url, json=None, **kwargs):
        class MockResponse:
            status_code = 200

            def json(self):
                if json and json.get("format") == "json":
                    return {
                        "response": (
                            '{"title": "Ollama Title", "bullets": ["P1", "P2"], '
                            '"score": 0.95, "items": [{"item_id": 1, "name": "N1"}]}'
                        )
                    }
                return {"response": "Ollama text response"}

        return MockResponse()

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    client = OllamaLLMClient(base_url="http://mock-ollama:11434", model_name="test-llama")
    assert client.model_name == "test-llama"

    # Test raw generate
    res = await client.generate("Test prompt")
    assert res == "Ollama text response"

    # Test structured generate
    struct = await client.generate_structured("Test structured prompt", schema=SampleStructuredOutput)
    assert struct.title == "Ollama Title"
    assert struct.bullets == ["P1", "P2"]
    assert struct.items[0].name == "N1"


@pytest.mark.asyncio
async def test_ollama_client_validation_error(monkeypatch):
    async def mock_post(self, url, **kwargs):
        class MockResponse:
            status_code = 200

            def json(self):
                # Return JSON missing required field "title"
                return {"response": '{"bullets": ["P1"]}'}

        return MockResponse()

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    client = OllamaLLMClient(base_url="http://mock-ollama:11434")
    with pytest.raises(AppException) as exc_info:
        await client.generate_structured("Invalid response test", schema=SampleStructuredOutput)

    assert exc_info.value.code == "STRUCTURED_OUTPUT_VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_ollama_client_connection_and_timeout_errors():
    # Unreachable port to verify clean connection error
    client = OllamaLLMClient(base_url="http://127.0.0.1:59998", timeout=1)
    with pytest.raises(AppException) as exc_info:
        await client.generate("Query")
    assert exc_info.value.code in ("OLLAMA_CONNECTION_ERROR", "OLLAMA_TIMEOUT")


@pytest.mark.asyncio
async def test_ollama_client_check_health(monkeypatch):
    async def mock_get_ok(self, url, **kwargs):
        class MockResponse:
            status_code = 200

        return MockResponse()

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get_ok)
    client = OllamaLLMClient()
    assert await client.check_health() is True

    async def mock_get_fail(self, url, **kwargs):
        raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get_fail)
    assert await client.check_health() is False


def test_llm_factory():
    local_client = get_llm_client("local", force_new=True)
    assert isinstance(local_client, MockLLMClient)

    mock_client = get_llm_client("mock", force_new=True)
    assert isinstance(mock_client, MockLLMClient)

    ollama_client = get_llm_client("ollama", force_new=True)
    assert isinstance(ollama_client, OllamaLLMClient)

    with pytest.raises(AppException) as exc_info:
        get_llm_client("unsupported_provider")
    assert exc_info.value.code == "UNKNOWN_LLM_PROVIDER"
