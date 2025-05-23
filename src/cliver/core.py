import logging
from .config import ConfigManager

logger = logging.getLogger(__name__)


class Core:
    def __init__(self, config: ConfigManager = None):
        self.config = config
        # self.llm_provider = get_llm_provider(
        #     self.config.get('llm_provider'), self.config)
        # self.mcp_client = MCPClient(self.config.get(
        #     'mcp_server_url'), self.config.get('mcp_api_key'))

    def query_llm(self, prompt, **kwargs):
        logger.debug(f"Querying LLM with prompt: {prompt}")
        # return self.llm_provider.query(prompt, **kwargs)

    def send_to_mcp(self, data):
        logger.debug(f"Sending data to MCP: {data}")
        # return self.mcp_client.send(data)

    def fetch_from_mcp(self, resource_id):
        logger.debug(f"Fetching resource {resource_id} from MCP")
        # return self.mcp_client.fetch(resource_id)


# Expose core functions for CLI commands
core_instance = Core()


def query_llm(prompt, **kwargs):
    return core_instance.query_llm(prompt, **kwargs)


def send_to_mcp(data):
    return core_instance.send_to_mcp(data)


def fetch_from_mcp(resource_id):
    return core_instance.fetch_from_mcp(resource_id)
