# datasette-llm-embed

[![PyPI](https://img.shields.io/pypi/v/datasette-llm-embed.svg)](https://pypi.org/project/datasette-llm-embed/)
[![Changelog](https://img.shields.io/github/v/release/simonw/datasette-llm-embed?include_prereleases&label=changelog)](https://github.com/simonw/datasette-llm-embed/releases)
[![Tests](https://github.com/simonw/datasette-llm-embed/workflows/Test/badge.svg)](https://github.com/simonw/datasette-llm-embed/actions?query=workflow%3ATest)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/simonw/datasette-llm-embed/blob/main/LICENSE)

Datasette plugin adding a `llm_embed(model_id, text)` SQL function.

## Installation

```bash
datasette install datasette-llm-embed
```

## Usage

Adds a SQL function that can be called like this:
```sql
select llm_embed('sentence-transformers/all-mpnet-base-v2', 'This is some text')
```
This embeds the provided text using the specified embedding model and returns a binary blob, suitable for use with plugins such as [datasette-faiss](https://datasette.io/plugins/datasette-faiss).

The models need to be installed using [LLM](https://llm.datasette.io/) plugins such as [llm-sentence-transformers](https://github.com/simonw/llm-sentence-transformers).

Use `llm_embed_cosine(a, b)` to calculate cosine similarity between two vector blobs:

```sql
select llm_embed_cosine(
    llm_embed('sentence-transformers/all-mpnet-base-v2', 'This is some text'),
    llm_embed('sentence-transformers/all-mpnet-base-v2', 'This is some other text')
)
```

## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:
```bash
cd datasette-llm-embed
python3 -m venv venv
source venv/bin/activate
```
Now install the dependencies and test dependencies:
```
pip install -e '.[test]'
```
```
To run the tests:
```bash
pytest
```
