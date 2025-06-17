from typing import List, Dict, Optional, Union, Callable, Awaitable, Any, Tuple
import asyncio

from cliver.config import ModelConfig
from cliver.llm.ollama_engine import OllamaLlamaInferenceEngine
from cliver.mcp_server_caller import MCPServersCaller
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
    def __init__(self, llm_models: Dict[str, ModelConfig], mcp_servers: Dict[str, Dict], default_model: Optional[ModelConfig] = None):
        self.llm_models = llm_models
        self.default_model = default_model
        self.mcp_caller = MCPServersCaller(mcp_servers=mcp_servers)
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

    def process_user_input_sync(self, user_input: str,
                                 max_iterations: int = 10,
                                 confirm_tool_exec: bool = False,
                                 model: str = None,
                                 system_message_override: Optional[Callable[[], str]] = None,
                                 filter_tools: Optional[
                                     Callable[[str, list[BaseTool]], Awaitable[list[BaseTool]]]] = None,
                                 enhance_prompt: Optional[Callable[[str], Awaitable[list[BaseMessage]]]] = None,
                                 tool_error_check: Optional[
                                     Callable[[str, list[Dict[str, Any]]], Tuple[bool, str]]] = None,
                                 ) -> Union[BaseMessage, str]:
        return asyncio.run(
            self.process_user_input(user_input, max_iterations, confirm_tool_exec, model, system_message_override,
                                    filter_tools, enhance_prompt, tool_error_check))

    # This is the method that can be used out of box
    async def process_user_input(self, user_input: str,
                                 max_iterations: int = 10,
                                 confirm_tool_exec: bool = False,
                                 model: str = None,
                                 system_message_override: Optional[Callable[[], str]] = None,
                                 filter_tools: Optional[Callable[[str, list[BaseTool]], Awaitable[list[BaseTool]]]] = None,
                                 enhance_prompt: Optional[Callable[[str], Awaitable[list[BaseMessage]]]] = None,
                                 tool_error_check: Optional[Callable[[str, list[Dict[str, Any]]], Tuple[bool, str]]] = None,
                                 ) -> Union[BaseMessage, str]:
        """
        Process user input through the LLM, handling tool calls if needed.
        Args:
            user_input (str): The user input.
            max_iterations (int): The maximum number of iterations.
            confirm_tool_exec(bool): Ask for confirmation on tool execution.
            model (str): The model to use, the default one will be used if not specified.
            system_message_override: The system message override function.
            filter_tools: The function that filters tool calls.
            enhance_prompt: The function that enhances the prompt.
            tool_error_check: The function that checks tool errors. The returned string will be the tool error message
                              sent back to LLM if the first returned value is True.
        """

        llm_engine = self._select_llm_engine(model)
        # Add system message to instruct the LLM about tool usage
        system_message = llm_engine.system_message()
        if system_message_override:
            system_message = system_message_override()

        mcp_tools = await self.mcp_caller.get_mcp_tools()
        if filter_tools:
            mcp_tools = await filter_tools(user_input, mcp_tools)
        if not mcp_tools:
            mcp_tools = []

        messages: list[BaseMessage] = [SystemMessage(content=system_message), HumanMessage(content=user_input)]
        if enhance_prompt:
            enhanced_messages = await enhance_prompt(user_input)
            messages.extend(enhanced_messages)

        return await self._process_messages(llm_engine, messages, max_iterations, 0, mcp_tools, confirm_tool_exec, tool_error_check)

    async def _process_messages(self, llm_engine: LLMInferenceEngine,
                                messages: List[BaseMessage],
                                max_iterations: int,
                                current_iteration: int,
                                mcp_tools: list[BaseTool],
                                confirm_tool_exec: bool,
                                tool_error_check: Optional[Callable[[str, list[Dict[str, Any]]], Tuple[bool, str]]]= None) -> Union[BaseMessage, str]:
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
                    # default we don't care about confirmation and just run the tool
                    proceed = True
                    if confirm_tool_exec:
                        proceed = _confirm_tool_execution(f"This will execute tool: {the_tool_name} from mcp server: {mcp_server_name}")
                    if not proceed:
                        return f"Stopped at tool execution: {tool_call}"
                    mcp_tool_result = await self.mcp_caller.call_mcp_server_tool(
                        mcp_server_name, the_tool_name, args)
                    messages.append(ToolMessage(
                        content=mcp_tool_result, tool_call_id=tool_call_id))
                    if not tool_error_check:
                        tool_error_check = _tool_error_check_internal
                    sent, tool_error_message = tool_error_check(tool_name, mcp_tool_result)
                    if sent:
                        messages.append(AIMessage(content=tool_error_message))
                        return await self._process_messages(llm_engine, messages, max_iterations, current_iteration + 1,
                                                            mcp_tools, confirm_tool_exec, tool_error_check)
                # normally we don't need to send the tools to llm again
                return await self._process_messages(llm_engine, messages, max_iterations, current_iteration + 1, [], confirm_tool_exec, tool_error_check)
            except Exception as e:
                return f"Error processing tool call: {str(e)}"
        else:
            return response

def _tool_error_check_internal(tool_name: str, mcp_tool_result: list[Dict[str, Any]]) -> (bool, str):
    if any("error" in r for r in mcp_tool_result):
        return True, f"Error calling tool {tool_name}: {mcp_tool_result[0].get("error")}, you may need to check the tool arguments and run it again."
    return False, None

def _confirm_tool_execution(prompt="Are you sure? (y/n): ") -> bool:
    while True:
        response = input(prompt).strip().lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
