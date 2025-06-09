import os
from langchain.globals import set_debug, set_verbose
from typing import List, Dict, Optional, Union
from abc import ABC, abstractmethod
from cliver.config import ModelConfig
from cliver.mcp import MCPServersCaller
from langchain_core.messages import AIMessage, ToolMessage, SystemMessage, HumanMessage, ChatMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.tools import BaseTool

if os.environ.get("MODE") == 'dev':
    set_debug(True)
    set_verbose(True)
    import logging
    import langchain
    langchain.LANGCHAIN_DEBUG = True
    logging.basicConfig(level=logging.DEBUG)


class LLMInferenceEngine(ABC):
    def __init__(self, config: ModelConfig):
        self.config = config or {}

    # This method focus on the real LLM inference only.
    @abstractmethod
    async def infer(self, messages: List[BaseMessage], tools: Optional[list[BaseTool]]) -> BaseMessage:
        pass


# Ollama inference engine
class OllamaLlamaInferenceEngine(LLMInferenceEngine):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        from langchain_ollama import ChatOllama

        self.llm = ChatOllama(
            base_url=self.config.url,
            model=self.config.name_in_provider,
            **self.config.model_dump()
        )

    async def infer(self, messages: list[BaseMessage], tools: Optional[list[BaseTool]]) -> BaseMessage:
        try:
            _llm = self.llm
            if tools:
                _llm = self.llm.bind_tools(tools)
            response = await _llm.ainvoke(messages)
            return response
        except Exception as e:
            return AIMessage(content=f"Error: {e}", additional_kwargs={"type": "error"})


class TaskExecutor:
    def __init__(self, llm_engine: LLMInferenceEngine, mcp_caller: MCPServersCaller):
        self.llm_engine = llm_engine
        self.mcp_caller = mcp_caller

    async def process_user_input(self, user_input: str, max_iterations: int = 5) -> Union[BaseMessage, str]:
        """Process user input through the LLM, handling tool calls if needed."""

        # Add system message to instruct the LLM about tool usage
        system_message = """
        You are an AI assistant that can use tools to help answer questions.

If you need to call a tools to get more information from client side, always extract MCP tool calls and format them **only** as a function call to `tool_calls`.
- Gather from user: tool_name, arguments.
- When ready, respond only with JSON in the form:
  "tool_calls": [
    {
      "name": "<tool_name>",
      "args": {...},
      "id": "<tool_call_id>",
      "type": "tool_call"
    }
  ]
- Do **not** add any text outside the JSON.

If you have all the information needed to answer directly, provide a text response.
"""

        # TODO we need to make sure some arguments are required so that LLM won't return an empty argument
        mcp_tools = await self.mcp_caller.get_mcp_tools()
        messages = []
        messages.append(SystemMessage(content=system_message))

        # TODO we may need to retrieve some enhanced prompt from some mcp servers before inference

        messages.append(HumanMessage(content=user_input))
        return await self._process_messages(messages, max_iterations, 0, mcp_tools)

    async def _process_messages(self, messages: List[BaseMessage], max_iterations: int, current_iteration: int, mcp_tools: Optional[list[BaseTool]] = None) -> Union[BaseMessage, str]:
        """Handle processing messages recursively with tool calling."""
        if current_iteration >= max_iterations:
            return "Reached maximum number of iterations without a final answer."

        # Get response from LLM
        response = await self.llm_engine.infer(messages, mcp_tools)

        # Handle different response types
        if response.tool_calls:
            try:
                tool_calls: list[Dict] = response.tool_calls
                for tool_call in tool_calls:
                    mcp_server_name = ""
                    tool_name: str = tool_call.get("name")
                    the_tool_name = tool_name
                    if "#" in tool_name:
                        s_array = tool_name.split('#')
                        mcp_server_name = s_array[0]
                        the_tool_name = s_array[1]

                    args = tool_call.get("args")
                    tool_call_id = tool_call.get("id")
                    # TODO ask for confirmation before tool execution ?
                    mcp_tool_result = await self.mcp_caller.call_mcp_server_tool(
                        mcp_server_name, the_tool_name, args)
                    messages.append(ToolMessage(
                        content=mcp_tool_result, tool_call_id=tool_call_id))
                    # TODO improve the messages with errors adding assistant messages ? shall we continue or a way to improve it ?
                    # TODO check the error if it is because of the wrong arguments
                    if any("error" in r for r in mcp_tool_result):
                        messages.append(ChatMessage(role="assistant",
                                                    content=f"Error calling tool {tool_name}: {mcp_tool_result[0].error}"))
                        return await self._process_messages(messages, max_iterations, current_iteration + 1, mcp_tools)
                return await self._process_messages(messages, max_iterations, current_iteration + 1)
            except Exception as e:
                return f"Error processing tool call: {str(e)}"
        else:
            return response


# Example usage
async def run_example():
    # Create configuration for Ollama
    config = ModelConfig(
        name="llama3.2",
        provider="ollama",
        type="Text-To-Text",
        url="http://localhost:11434",
        name_in_provider="llama3.2:latest",
        # Add any other necessary configurations
    )

    # Initialize components
    llm_engine = OllamaLlamaInferenceEngine(config)
    mcp_servers = MCPServersCaller(mcp_servers={
        "time": {
            "transport": "stdio",
            "command": "uvx",
            "args": [
                "mcp-server-time",
                "--local-timezone=Asia/Shanghai"
            ]
        }
    })
    executor = TaskExecutor(llm_engine, mcp_servers)

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
