from typing import List, Dict, Optional, Union

from cliver.config import ModelConfig
from cliver.llm.ollama_engine import OllamaLlamaInferenceEngine
from cliver.mcp import MCPServersCaller
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.tools import BaseTool
from cliver.llm.base import LLMInferenceEngine

def create_llm_engine(model: ModelConfig) -> Optional[LLMInferenceEngine]:
    if model.provider == "ollama":
        return OllamaLlamaInferenceEngine(model)
    return None

class TaskExecutor:
    """
    This is the central place managing the execution of all configured LLM models and MCP servers.
    """
    def __init__(self, llm_models: Dict[str, ModelConfig], mcp_caller: MCPServersCaller, default_model: Optional[ModelConfig] = None):
        self.llm_models = llm_models
        self.default_model = default_model
        self.mcp_caller = mcp_caller
        self.llm_engines: Dict[str, LLMInferenceEngine] = {}

    def _select_llm_engine(self, model: str = None) -> LLMInferenceEngine:
        if model and model in self.llm_models:
            _model = self.llm_models[model]
        else:
            _model = self.default_model
        if not _model:
            models = [v for _, v in self.llm_models.items()]
            _model = models[0] if len(models) > 0 else None
        if not _model:
            raise RuntimeError(f"No model named {model}.")
        if _model.name in self.llm_engines:
            llm_engine = self.llm_engines[_model.name]
        else:
            llm_engine = create_llm_engine(_model)
            self.llm_engines[_model.name] = llm_engine
        return llm_engine

    async def _filter_tools(self, user_input: str, mcp_tools: list[BaseTool]) -> list[BaseTool]:
        # TODO filter based on user input
        return mcp_tools

    async def _enhance_prompt(self, user_input: str) -> list[BaseMessage]:
        # TODO get some enhanced prompts to be added to the final messages
        return []

    async def process_user_input(self, user_input: str, max_iterations: int = 10, model: str = None) -> Union[BaseMessage, str]:
        """
        Process user input through the LLM, handling tool calls if needed.
        Args:
            user_input (str): The user input.
            max_iterations (int): The maximum number of iterations.
            model (str): The model to use, the default one will be used if not specified.
        """

        llm_engine = self._select_llm_engine(model)
        # Add system message to instruct the LLM about tool usage
        system_message = llm_engine.system_message()

        mcp_tools = await self.mcp_caller.get_mcp_tools()
        mcp_tools = await self._filter_tools(user_input, mcp_tools)

        messages: list[BaseMessage] = [SystemMessage(content=system_message), HumanMessage(content=user_input)]
        enhanced_messages = await self._enhance_prompt(user_input)
        if enhanced_messages:
            messages.extend(enhanced_messages)

        return await self._process_messages(llm_engine, messages, max_iterations, 0, mcp_tools)

    async def _process_messages(self, llm_engine: LLMInferenceEngine, messages: List[BaseMessage], max_iterations: int, current_iteration: int, mcp_tools: Optional[list[BaseTool]] = None) -> Union[BaseMessage, str]:
        """Handle processing messages recursively with tool calling."""
        if current_iteration >= max_iterations:
            return "Reached maximum number of iterations without a final answer."

        # Get response from LLM
        response = await llm_engine.infer(messages, mcp_tools)

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
                    # TODO check the error if it is because of the wrong arguments
                    if any("error" in r for r in mcp_tool_result):
                        messages.append(AIMessage(content=f"Error calling tool {tool_name}: {mcp_tool_result[0].get("error")}"))
                        # we need the tool execution again
                        return await self._process_messages(llm_engine, messages, max_iterations, current_iteration + 1, mcp_tools)
                # normally we don't need to send the tools to llm again
                return await self._process_messages(llm_engine, messages, max_iterations, current_iteration + 1)
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
    executor = TaskExecutor({"llama3.2": config}, mcp_servers, config)

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
