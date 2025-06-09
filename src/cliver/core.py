# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chat_models import init_chat_model
# from langchain_mcp_adapters.client import MultiServerMCPClient
# from langgraph.prebuilt import create_react_agent
import logging
from cliver.config import ConfigManager

logger = logging.getLogger(__name__)

# system_template = "Translate the following from English into {language}"
#
# prompt_template = ChatPromptTemplate.from_messages(
#     [("system", system_template), ("user", "{text}")]
# )
# model = init_chat_model("gpt-4o-mini", model_provider="openai")
# model.bind_tools([])
# prompt = prompt_template.invoke({"language": "Italian", "text": "hi!"})
# model.invoke(prompt)


class Core:
    def __init__(self, config: ConfigManager = None):
        self.config = config
        self.llm = None
        self.tools = []
        self.system_prompt = ""
        self.mcp_resources = {}

    def init_llm(self, model_config):
        """Initialize the LLM model using langchain and the given ModelConfig."""
        from langchain.llms import OpenAI  # Example, replace with dynamic import if needed
        self.llm = OpenAI(**model_config)
        logger.debug(f"Initialized LLM with config: {model_config}")

    def load_tools_from_mcp(self):
        """Fetch and register available tools from MCP servers."""
        # Example: fetch tool list from MCP and store in self.tools
        # self.tools = self.mcp_client.get_tools()
        logger.debug("Loaded tools from MCP server.")

    def set_system_prompt(self, prompt: str):
        """Set the system prompt for the conversation."""
        self.system_prompt = prompt
        logger.debug(f"System prompt set: {prompt}")

    def fetch_resource_content(self, resource_id):
        """Fetch resource content from MCP server."""
        # content = self.mcp_client.fetch(resource_id)
        content = f"Resource content for {resource_id}"  # Placeholder
        self.mcp_resources[resource_id] = content
        logger.debug(f"Fetched resource {resource_id}: {content}")
        return content

    def construct_prompt(self, user_question, resource_ids=None):
        """Combine user question, system prompt, tools, and resources into a prompt."""
        prompt = self.system_prompt + "\n"
        if resource_ids:
            for rid in resource_ids:
                content = self.mcp_resources.get(
                    rid) or self.fetch_resource_content(rid)
                prompt += f"\n[Resource {rid}]: {content}"
        prompt += f"\n[Tools]: {', '.join([tool['name'] for tool in self.tools])}"
        prompt += f"\n[User]: {user_question}"
        logger.debug(f"Constructed prompt: {prompt}")
        return prompt

    def handle_llm_response(self, response):
        """Process LLM response, call tools or interact with user as needed."""
        # Parse response, determine if tool call or user interaction is needed
        logger.debug(f"Handling LLM response: {response}")
        # Example: if response indicates a tool call, execute it
        # Continue conversation loop as needed

    def start_task_workflow(self, user_question, model_config, resource_ids=None):
        """Main workflow: initialize LLM, load tools, construct prompt, interact with LLM."""
        self.init_llm(model_config)
        self.load_tools_from_mcp()
        prompt = self.construct_prompt(user_question, resource_ids)
        response = self.llm(prompt)
        self.handle_llm_response(response)
        # Continue loop until exit condition is met

    def query_llm(self, prompt, **kwargs):
        logger.debug(f"Querying LLM with prompt: {prompt}")
        # return self.llm_provider.query(prompt, **kwargs)

    def send_to_mcp(self, data):
        logger.debug(f"Sending data to MCP: {data}")
        # return self.mcp_client.send(data)

    def fetch_from_mcp(self, resource_id):
        logger.debug(f"Fetching resource {resource_id} from MCP")
        # return self.mcp_client.fetch(resource_id)
