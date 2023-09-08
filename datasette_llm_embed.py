from datasette import hookimpl
import llm


def llm_embed(model_id, text):
    try:
        model = llm.get_embedding_model(model_id)
        return llm.encode(model.embed(text))
    except Exception as e:
        return str(e)


@hookimpl
def prepare_connection(conn):
    conn.create_function("llm_embed", 2, llm_embed)
