import math
import pytest
import httpx
from backend.app.services.embeddings.mock_provider import DeterministicLocalEmbeddingProvider
from backend.app.services.embeddings.ollama_provider import OllamaEmbeddingProvider
from backend.app.services.embeddings.factory import get_embedding_provider
from backend.app.core.exceptions import AppException


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    norm_a = math.sqrt(sum(a * a for a, b in zip(v1, v2)))
    norm_b = math.sqrt(sum(b * b for a, b in zip(v1, v2)))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


@pytest.mark.asyncio
async def test_deterministic_provider_dimensions_and_norm():
    provider = DeterministicLocalEmbeddingProvider(dimension=384)
    assert provider.dimension == 384
    assert provider.model_name == "local-deterministic-384"

    vector = await provider.embed_query("Content transformation for SIH 2024")
    assert len(vector) == 384
    magnitude = math.sqrt(sum(x * x for x in vector))
    assert pytest.approx(magnitude, 0.01) == 1.0


@pytest.mark.asyncio
async def test_deterministic_provider_batch_consistency():
    provider = DeterministicLocalEmbeddingProvider(dimension=384)
    texts = ["First document chunk", "Second document chunk"]

    batch_vectors = await provider.embed_documents(texts)
    assert len(batch_vectors) == 2

    single_0 = await provider.embed_query(texts[0])
    single_1 = await provider.embed_query(texts[1])

    assert batch_vectors[0] == single_0
    assert batch_vectors[1] == single_1


@pytest.mark.asyncio
async def test_deterministic_provider_semantic_similarity():
    provider = DeterministicLocalEmbeddingProvider(dimension=384)

    v_ai_1 = await provider.embed_query("artificial intelligence and machine learning models")
    v_ai_2 = await provider.embed_query("machine learning AI intelligence systems")
    v_cook = await provider.embed_query("baking sweet chocolate chip cookies in an oven")

    sim_related = sum(a * b for a, b in zip(v_ai_1, v_ai_2))
    sim_unrelated = sum(a * b for a, b in zip(v_ai_1, v_cook))

    # Related texts sharing terminology must have higher cosine similarity than unrelated
    assert sim_related > sim_unrelated


@pytest.mark.asyncio
async def test_ollama_provider_batch_success(monkeypatch):
    mock_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    async def mock_post(self, url, **kwargs):
        class MockResponse:
            status_code = 200

            def json(self):
                return {"embeddings": mock_embeddings}

        return MockResponse()

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    provider = OllamaEmbeddingProvider(base_url="http://mock-ollama:11434", model_name="mock-embed")
    vectors = await provider.embed_documents(["chunk 1", "chunk 2"])

    assert vectors == mock_embeddings
    assert provider.dimension == 3


@pytest.mark.asyncio
async def test_ollama_provider_connection_error():
    # Use non-routable port to verify clean connection / timeout error handling
    provider = OllamaEmbeddingProvider(base_url="http://127.0.0.1:59999", timeout=1)
    with pytest.raises(AppException) as exc_info:
        await provider.embed_query("test query")
    assert exc_info.value.code in ("OLLAMA_CONNECTION_ERROR", "OLLAMA_TIMEOUT", "OLLAMA_NETWORK_ERROR")


def test_embedding_factory():
    local_p = get_embedding_provider("local")
    assert isinstance(local_p, DeterministicLocalEmbeddingProvider)

    mock_p = get_embedding_provider("mock")
    assert isinstance(mock_p, DeterministicLocalEmbeddingProvider)

    ollama_p = get_embedding_provider("ollama")
    assert isinstance(ollama_p, OllamaEmbeddingProvider)

    with pytest.raises(AppException) as exc_info:
        get_embedding_provider("non_existent_provider")
    assert exc_info.value.code == "UNKNOWN_EMBEDDING_PROVIDER"
