from cliver.config import ModelConfig
from cliver import TaskExecutor

# Example usage
async def run_example():
    # Create configuration for Ollama
    llm_models = {
        "llama3.2": ModelConfig(
            name="llama3.2",
            provider="ollama",
            type="Text-To-Text",
            url="http://localhost:11434",
            name_in_provider="llama3.2:latest",
            api_key="the api key"
        )
    }

    mcp_servers = {
        "time": {
            "transport": "stdio",
            "command": "uvx",
            "args": [
                "mcp-server-time",
                "--local-timezone=Asia/Shanghai"
            ]
        }
    }
    executor = TaskExecutor(llm_models, mcp_servers)

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
