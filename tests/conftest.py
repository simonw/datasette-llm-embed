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


class EmbedKeyDemo(llm.EmbeddingModel):
    model_id = "embed-key-demo"

    def embed_batch(self, texts):
        for text in texts:
            words = text.split()[:15]
            embedding = [len(word) for word in words]
            # Pad with 0 up to 15 words
            embedding += [0] * (15 - len(embedding))
            # Last word is the length of the key
            embedding.append(len(self.key))
            yield embedding


@pytest.fixture
def embed_demo():
    return EmbedDemo()


@pytest.fixture
def embed_key_demo():
    return EmbedKeyDemo()


@pytest.fixture(autouse=True)
def register_embed_demo_model(embed_demo, embed_key_demo):
    class MockModelsPlugin:
        __name__ = "MockModelsPlugin"

        @llm.hookimpl
        def register_embedding_models(self, register):
            register(embed_demo)
            register(embed_key_demo)

    pm.register(MockModelsPlugin(), name="undo-mock-models-plugin")
    try:
        yield
    finally:
        pm.unregister(name="undo-mock-models-plugin")
