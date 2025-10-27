import asyncio

from cliver import TaskExecutor
from cliver.config import ModelConfig
from cliver.model_capabilities import ModelCapability

# Configure LLM models
llm_models = {
    "qwen": ModelConfig(
        name="qwen",
        provider="ollama",
        url="http://localhost:11434",
        name_in_provider="qwen2.5:latest",
        capabilities={ModelCapability.TEXT_TO_TEXT, ModelCapability.TOOL_CALLING},
    )
}

executor = TaskExecutor(llm_models=llm_models, mcp_servers={}, default_model="qwen")


# Stream the response
async def stream_query():
    async for chunk in executor.stream_user_input("Write a poem about programming"):
        if hasattr(chunk, "content") and chunk.content:
            print(chunk.content, end="", flush=True)


asyncio.run(stream_query())
