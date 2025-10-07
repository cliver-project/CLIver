from cliver.config import ModelConfig, StdioMCPServerConfig
from cliver import TaskExecutor
from cliver.model_capabilities import ModelCapability

# Example usage
async def run_example():
    # Create configuration for Ollama with capabilities instead of type
    llm_models = {
        "llama3.2": ModelConfig(
            name="llama3.2",
            provider="ollama",
            url="http://localhost:11434",
            name_in_provider="llama3.2:latest",
            api_key="the api key",
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

    executor = TaskExecutor(llm_models, {name: server.model_dump() for name, server in mcp_servers.items()})

    # Process a user query
    user_input = "What time is it now ?"
    result = await executor.process_user_input(user_input)
    print("\n===\n")
    print(result)
    print("\n===\n")

    if result.content:
        print("\nFinal answer:")
        print(str(result.content))
        print("\n===\n")


if __name__ == "__main__":
    import asyncio

    asyncio.run(run_example())