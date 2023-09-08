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
