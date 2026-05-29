import os

if os.environ.get("MODE") == "dev":
    import logging

    logging.basicConfig(level=logging.DEBUG)

from cliver.llm.new_agent import AgentCore

__all__ = ["AgentCore"]
