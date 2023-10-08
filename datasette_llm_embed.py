from datasette import hookimpl
import llm


def llm_embed_factory(datasette):
    config = datasette.plugin_config("datasette-llm-embed") or {}
    keys = config.get("keys") or {}

    def llm_embed(model_id, text):
        try:
            model = llm.get_embedding_model(model_id)
            if model.model_id in keys:
                model.key = keys[model.model_id]
            return llm.encode(model.embed(text))
        except Exception as e:
            return str(e)

    return llm_embed


def llm_embed_cosine(a, b):
    try:
        return llm.cosine_similarity(llm.decode(a), llm.decode(b))
    except Exception as e:
        return str(e)


@hookimpl
def prepare_connection(datasette, conn):
    conn.create_function("llm_embed", 2, llm_embed_factory(datasette))
    conn.create_function("llm_embed_cosine", 2, llm_embed_cosine)
    conn.create_aggregate("llm_embed_average", 1, AverageVectorAgg)


class AverageVectorAgg:
    with_scores = False

    def __init__(self):
        self.accumulated = []
        self.vector_size = 0

    def step(self, embedding):
        vector = llm.decode(embedding)
        if len(self.accumulated) == 0:
            self.accumulated = list(vector)
        else:
            for i in range(len(self.accumulated)):
                self.accumulated[i] += vector[i]
        self.vector_size += 1

    def finalize(self):
        vector = [item / self.vector_size for item in self.accumulated]
        return llm.encode(vector)
