"""Built-in web_search tool for searching the web."""

import logging
import urllib.parse
import urllib.request
from typing import Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DEFAULT_MAX_RESULTS = 5


class WebSearchInput(BaseModel):
    """Input schema for the web_search tool."""

    query: str = Field(description="The search query to find information on the web.")
    max_results: Optional[int] = Field(
        default=None,
        description=f"Maximum number of results to return. Defaults to {DEFAULT_MAX_RESULTS}.",
    )


class WebSearchTool(BaseTool):
    """Searches the web for up-to-date information."""

    name: str = "web_search"
    description: str = (
        "Searches the web and returns results to inform responses. "
        "Provides up-to-date information for current events and recent data "
        "beyond the training data cutoff. "
        "Returns search results with concise snippets and source links. "
        "Use this tool when you need information that may be outdated or "
        "beyond your knowledge cutoff."
    )
    args_schema: Type[BaseModel] = WebSearchInput
    tags: list = ["search", "web", "fetch"]

    def _run(self, query: str, max_results: Optional[int] = None) -> str:
        num_results = max_results or DEFAULT_MAX_RESULTS

        # Try DuckDuckGo HTML search (no API key required)
        try:
            return self._duckduckgo_search(query, num_results)
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {e}")

        return (
            "Error: Web search is currently unavailable. "
            "Could not reach any search provider. "
            "Please check your network connection."
        )

    def _duckduckgo_search(self, query: str, num_results: int) -> str:
        """Search using DuckDuckGo Lite HTML (no API key needed)."""
        url = "https://lite.duckduckgo.com/lite/"
        data = urllib.parse.urlencode({"q": query}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "User-Agent": "CLIver/1.0",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )

        with urllib.request.urlopen(req, timeout=15) as response:
            html = response.read().decode("utf-8", errors="replace")

        return self._parse_duckduckgo_lite(html, query, num_results)

    def _parse_duckduckgo_lite(self, html: str, query: str, num_results: int) -> str:
        """Parse DuckDuckGo Lite results from HTML."""
        results = []
        # DuckDuckGo Lite returns results in a table format
        # Each result has a link and a snippet
        import re

        # Find result links - they appear as <a> tags with class="result-link"
        # or simply as <a rel="nofollow" href="..."> in the results table
        link_pattern = re.compile(
            r'<a[^>]*rel="nofollow"[^>]*href="([^"]+)"[^>]*>\s*(.*?)\s*</a>',
            re.DOTALL,
        )
        # Find snippets - they appear in <td> tags after the link
        snippet_pattern = re.compile(
            r'<td[^>]*class="result-snippet"[^>]*>(.*?)</td>',
            re.DOTALL,
        )

        links = link_pattern.findall(html)
        snippets = snippet_pattern.findall(html)

        for i, (url, title) in enumerate(links[:num_results]):
            # Clean HTML tags from title and snippet
            clean_title = re.sub(r"<[^>]+>", "", title).strip()
            snippet = ""
            if i < len(snippets):
                snippet = re.sub(r"<[^>]+>", "", snippets[i]).strip()

            if clean_title and url:
                result = f"[{i + 1}] {clean_title}\n    URL: {url}"
                if snippet:
                    result += f"\n    {snippet}"
                results.append(result)

        if not results:
            # Fallback: try a simpler pattern to extract any links from the page
            simple_links = re.findall(r'href="(https?://[^"]+)"[^>]*>([^<]+)</a>', html)
            for i, (url, title) in enumerate(simple_links[:num_results]):
                title = title.strip()
                if title and not url.startswith("https://duckduckgo.com"):
                    results.append(f"[{i + 1}] {title}\n    URL: {url}")

        if not results:
            return f"No search results found for: {query}"

        return f"Search results for: {query}\n\n" + "\n\n".join(results)


web_search = WebSearchTool()
