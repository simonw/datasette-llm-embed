import base64
from datasette.app import Datasette
import llm
import pytest


@pytest.mark.asyncio
async def test_llm_embed():
    ds = Datasette()
    await ds.invoke_startup()
    response = await ds.client.get(
        "/_memory.json",
        params={
            "_shape": "array",
            "sql": "select llm_embed('embed-demo', 'hello world') as e",
        },
    )
    assert response.status_code == 200
    encoded = response.json()[0]["e"]["encoded"]
    assert llm.decode(base64.b64decode(encoded)) == (
        5.0,
        5.0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )


@pytest.mark.asyncio
async def test_llm_embed_cosine():
    ds = Datasette()
    response = await ds.client.get(
        "/_memory.json",
        params={
            "_shape": "array",
            "sql": """
            select llm_embed_cosine(
                llm_embed('embed-demo', 'hello world'),
                llm_embed('embed-demo', 'hello again')
            ) as cosine""",
        },
    )
    assert response.status_code == 200
    pytest.approx(response.json()[0]["cosine"], 0.0001) == 0.9999


@pytest.mark.asyncio
async def test_llm_embed_uses_key():
    ds = Datasette(
        metadata={
            "plugins": {
                "datasette-llm-embed": {
                    "keys": {
                        "embed-key-demo": "default-key",
                        "aliased": "alias",
                    }
                }
            }
        }
    )
    response = await ds.client.get(
        "/_memory.json",
        params={
            "_shape": "array",
            "sql": "select llm_embed('embed-key-demo', 'hello world') as e",
        },
    )
    assert response.status_code == 200
    encoded = response.json()[0]["e"]["encoded"]
    assert llm.decode(base64.b64decode(encoded)) == (
        5.0,
        5.0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        11.0,
    )
