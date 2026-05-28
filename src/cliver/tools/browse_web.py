"""Built-in tool for web scraping via Firecrawl API."""

import logging
import os
from typing import Optional

from cliver.tool import tool

logger = logging.getLogger(__name__)


@tool(
    name="Browse",
    description=(
        "Fetch a webpage and extract its content as clean markdown. "
        "Much better than WebFetch for reading articles, documentation, "
        "or extracting structured data from URLs. Requires FIRECRAWL_API_KEY."
    ),
)
def browse_web(url: str, formats: Optional[list[str]] = None) -> list[dict]:
    """Fetch a webpage and extract its content via the Firecrawl API.

    Args:
        url: URL to scrape and extract content from.
        formats: Output formats: markdown, html, links (default: ["markdown"]).
    """
    if formats is None:
        formats = ["markdown"]

    api_key = os.environ.get("FIRECRAWL_API_KEY")
    if not api_key:
        return [{"error": "FIRECRAWL_API_KEY environment variable not set. Get a free key at https://firecrawl.dev"}]

    try:
        from firecrawl import FirecrawlApp
    except ImportError:
        return [{"error": "firecrawl-py not installed. Install with: pip install cliver[browser]"}]

    try:
        app = FirecrawlApp(api_key=api_key)
        result = app.scrape_url(url, params={"formats": formats})

        if isinstance(result, dict):
            # Return markdown content if available
            if "markdown" in result:
                return [{"text": result["markdown"]}]
            # Fallback to any available format
            for fmt in formats:
                if fmt in result:
                    return [{"text": str(result[fmt])}]
            return [{"text": str(result)}]
        return [{"text": str(result)}]
    except Exception as e:
        logger.warning(f"Firecrawl error for {url}: {e}")
        return [{"error": f"Error scraping {url}: {e}"}]
