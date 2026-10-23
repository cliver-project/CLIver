# cliver-core

The embeddable agent core behind [CLIver](https://github.com/cliver-project/CLIver): a Re-Act loop
with tool calling, per-purpose model bindings, media capabilities, and first-class MCP support.

No CLI, no database, no config files — you pass Python objects in and get messages back.

## Install

```bash
pip install cliver-core          # core library + MCP client (stdio / streamable HTTP)
pip install "cliver-core[local]" # + in-process local embedding / rerank models
```

Python 3.10+.

## Quick start

```python
import asyncio

from cliver_core import AgentCore, ModelBinding, ModelBindings


async def main():
    agent = AgentCore(
        ModelBindings(
            llm=ModelBinding.llm(
                model="deepseek-v4-flash",
                api_key="...",
                base_url="https://api.deepseek.com",
            )
        )
    )
    response = await agent.chat("Summarize the Re-Act pattern in one sentence.")
    print(response.message.content)
    await agent.aclose()


asyncio.run(main())
```

Ready-made bindings for live-verified providers live in `cliver_core.presets`:

```python
from cliver_core import ModelBindings, presets

bindings = ModelBindings(llm=presets.deepseek(api_key="..."))
agent = AgentCore(bindings)
```

## Tools

`cliver_core.tools` ships nine file, shell, todo, and web tools. Register them directly:

```python
from cliver_core.tools import discover_builtin_tools

agent = AgentCore(bindings, builtin_tools=discover_builtin_tools())
```

`ToolRegistry` and the `@tool` decorator build custom tools.

## Capabilities

Every purpose works standalone — no AgentCore required:

```python
from cliver_core import ModelBinding
from cliver_core.capabilities import embed, generate_image

vectors = await embed(
    texts=["hello", "world"],
    binding=ModelBinding.embed(model="text-embedding-3-small", api_key="...", base_url="..."),
)

image = await generate_image(
    prompt="a red panda on a skateboard",
    binding=ModelBinding.image(model="gpt-image-1", api_key="...", base_url="..."),
)
```

Purposes: image, video, text-to-speech, speech-to-text, embed, rerank. Long generations can return
a `MediaJob` future via the `submit_*` variants.

## Local models

Run embedding and rerank models in-process — no endpoint. The backends pull torch, so they are
opt-in:

```bash
pip install "cliver-core[local-st]"     # sentence-transformers embeddings
pip install "cliver-core[local-flag]"   # FlagEmbedding embeddings + rerank
pip install "cliver-core[local]"        # both
```

```python
from cliver_core.capabilities import embed, rerank
from cliver_core.local import (
    LocalEmbeddingProvider,
    LocalRerankProvider,
    local_embed,
    local_rerank,
)

embedder = LocalEmbeddingProvider(
    model="BAAI/bge-base-zh-v1.5",  # Hugging Face id or local directory
    backend="sentence-transformers",  # required; never inferred from the model name
)
reranker = LocalRerankProvider(
    model="BAAI/bge-reranker-v2-m3",
    backend="flag-embedding",
)

vectors = await embed(
    texts=["hello", "world"],
    binding=local_embed(provider=embedder, normalize=True),
)
items = await rerank(
    query="hello",
    documents=["hi", "hey"],
    binding=local_rerank(provider=reranker, top_k=1),
)

await embedder.close()
await reranker.close()
```

A provider owns one loaded model and is meant to be long-lived: create it once, share it across
bindings and calls, and close it at shutdown. There is no process-global model cache.
`ModelBinding.aclose()` never closes a provider you supplied.

The API is deliberately explicit: `model`, `backend`, `device`, and `dtype` are parameters, not
name-based defaults. Call options are generic — `dim` (optional output-size check), `normalize`
(off by default; raw vectors/scores), `batch_size`, and `top_k` (rerank). Query/document prefixes
and instruction templates belong to the application.

`provider.info()` reports the resolved model directory, revision, device, dtype, and load time.
Hugging Face downloads honor `HF_TOKEN`, `cache_dir`, `revision`, and `local_files_only`.

## MCP

```python
from cliver_core.mcp import MCPClient

client = MCPClient.from_discovery()  # ~/.cliver/mcp.json, .cliver/mcp.json, drop-ins
agent = AgentCore(bindings, mcp_client=client)
```

MCP tools are advertised to the model as `mcp__<server>__<tool>`. Prompts and resources are read on
demand — `list_prompts`, `get_prompt`, `list_resources`, `read_resource` — never registered as
tools.

## License

Apache-2.0
