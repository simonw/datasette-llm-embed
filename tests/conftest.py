import pytest
from llm.plugins import pm
import llm


class EmbedDemo(llm.EmbeddingModel):
    model_id = "embed-demo"

    def embed_batch(self, texts):
        for text in texts:
            words = text.split()[:16]
            embedding = [len(word) for word in words]
            # Pad with 0 up to 16 words
            embedding += [0] * (16 - len(embedding))
            yield embedding


@pytest.fixture
def embed_demo():
    return EmbedDemo()


@pytest.fixture(autouse=True)
def register_embed_demo_model(embed_demo):
    class MockModelsPlugin:
        __name__ = "MockModelsPlugin"

        @llm.hookimpl
        def register_embedding_models(self, register):
            register(embed_demo)

    pm.register(MockModelsPlugin(), name="undo-mock-models-plugin")
    try:
        yield
    finally:
        pm.unregister(name="undo-mock-models-plugin")
