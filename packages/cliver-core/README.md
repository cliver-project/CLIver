# cliver-core

The embeddable agent core behind [CLIver](https://github.com/cliver-project/CLIver): a Re-Act loop
with tool calling, per-purpose model bindings, media capabilities, and optional MCP support.

No CLI, no database, no config files — you pass Python objects in and get messages back.

## Install

```bash
pip install cliver-core          # core library
pip install "cliver-core[mcp]"   # + MCP client (stdio / streamable HTTP)
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
