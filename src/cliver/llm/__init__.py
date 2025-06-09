import os
from langchain.globals import set_debug, set_verbose

if os.environ.get("MODE") == 'dev':
    set_debug(True)
    set_verbose(True)
    import logging
    import langchain
    langchain.LANGCHAIN_DEBUG = True
    logging.basicConfig(level=logging.DEBUG)

from cliver.llm.llm import TaskExecutor