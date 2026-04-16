import os

from langchain_core.globals import set_debug, set_verbose

if os.environ.get("MODE") == "dev":
    set_debug(True)
    set_verbose(True)
    import logging

    logging.basicConfig(level=logging.DEBUG)

from cliver.llm.llm import AgentCore

# Export for public API
AgentCore = AgentCore
