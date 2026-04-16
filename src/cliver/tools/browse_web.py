"""Built-in tool for web scraping via Firecrawl API."""

import logging
import os
from typing import List, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class BrowseWebInput(BaseModel):
    url: str = Field(description="URL to scrape and extract content from")
    formats: List[str] = Field(
        default=["markdown"],
        description="Output formats: markdown, html, links",
    )


class BrowseWebTool(BaseTool):
    name: str = "browse_web"
    description: str = (
        "Fetch a webpage and extract its content as clean markdown. "
        "Much better than web_fetch for reading articles, documentation, "
        "or extracting structured data from URLs. Requires FIRECRAWL_API_KEY."
    )
    args_schema: Type[BaseModel] = BrowseWebInput
    tags: list = ["fetch", "web", "read"]

    def _run(self, url: str, formats: List[str] = None) -> str:
        if formats is None:
            formats = ["markdown"]

        api_key = os.environ.get("FIRECRAWL_API_KEY")
        if not api_key:
            return "Error: FIRECRAWL_API_KEY environment variable not set. Get a free key at https://firecrawl.dev"

        try:
            from firecrawl import FirecrawlApp
        except ImportError:
            return "Error: firecrawl-py not installed. Install with: pip install cliver[browser]"

        try:
            app = FirecrawlApp(api_key=api_key)
            result = app.scrape_url(url, params={"formats": formats})

            if isinstance(result, dict):
                # Return markdown content if available
                if "markdown" in result:
                    return result["markdown"]
                # Fallback to any available format
                for fmt in formats:
                    if fmt in result:
                        return str(result[fmt])
                return str(result)
            return str(result)
        except Exception as e:
            logger.warning(f"Firecrawl error for {url}: {e}")
            return f"Error scraping {url}: {e}"


browse_web = BrowseWebTool()
