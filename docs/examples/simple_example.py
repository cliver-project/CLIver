from cliver.config import ModelConfig, StdioMCPServerConfig
from cliver import TaskExecutor
from cliver.model_capabilities import ModelCapability

# Example usage
async def run_example():
    # Create configuration for Qwen model
    llm_models = {
        "qwen": ModelConfig(
            name="qwen",
            provider="ollama",
            url="http://localhost:11434",
            name_in_provider="qwen2.5:latest",
            capabilities={ModelCapability.TEXT_TO_TEXT, ModelCapability.TOOL_CALLING}
        )
    }

    # Create MCP server configuration using the proper Pydantic model
    mcp_servers = {
        "time": StdioMCPServerConfig(
            name="time",
            transport="stdio",
            command="uvx",
            args=["mcp-server-time", "--local-timezone=Asia/Shanghai"],
        )
    }

    executor = TaskExecutor(
        llm_models=llm_models,
        mcp_servers={name: server.model_dump() for name, server in mcp_servers.items()},
        default_model="qwen"
    )

    # Process a user query
    user_input = "What time is it now ?"
    result = await executor.process_user_input(user_input)
    print("\n===\n")
    print(result)
    print("\n===\n")

    if hasattr(result, 'content') and result.content:
        print("\nFinal answer:")
        print(str(result.content))
        print("\n===\n")


if __name__ == "__main__":
    import asyncio

    asyncio.run(run_example())