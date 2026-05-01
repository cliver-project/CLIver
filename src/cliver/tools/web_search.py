"""Built-in web_search tool for searching the web."""

import logging
import urllib.parse
import urllib.request
from typing import Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from cliver.util import BROWSER_USER_AGENT, url_request

logger = logging.getLogger(__name__)

DEFAULT_MAX_RESULTS = 5


SUPPORTED_ENGINES = ["duckduckgo", "bing", "sogou", "google", "baidu"]


class WebSearchInput(BaseModel):
    """Input schema for the web_search tool."""

    query: str = Field(description="The search query to find information on the web.")
    max_results: Optional[int] = Field(
        default=None,
        description=f"Maximum number of results to return. Defaults to {DEFAULT_MAX_RESULTS}.",
    )
    engine: Optional[str] = Field(
        default=None,
        description="Search engine to use: duckduckgo (default), bing, sogou, google, baidu. "
        "If a previous search told you which engine worked, use that one.",
    )


class WebSearchTool(BaseTool):
    """Searches the web for up-to-date information."""

    name: str = "WebSearch"
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

    def _get_configured_engines(self) -> list[str]:
        """Get search engine order from config, or default to [duckduckgo]."""
        try:
            from cliver.agent_profile import get_current_profile

            profile = get_current_profile()
            if profile:
                from cliver.config import ConfigManager

                cm = ConfigManager(profile.config_dir)
                if cm.config.search_engines:
                    valid = [e for e in cm.config.search_engines if e in SUPPORTED_ENGINES]
                    if valid:
                        return valid
        except Exception:
            pass
        return ["duckduckgo"]

    def _run(self, query: str, max_results: Optional[int] = None, engine: Optional[str] = None) -> str:
        num_results = max_results or DEFAULT_MAX_RESULTS

        engines = {
            "duckduckgo": self._duckduckgo_search,
            "bing": self._bing_search,
            "sogou": self._sogou_search,
            "google": self._google_search,
            "baidu": self._baidu_search,
        }

        if engine and engine.lower() in engines:
            order = [engine.lower()]
        else:
            order = self._get_configured_engines()

        errors = []
        for eng_name in order:
            try:
                result = engines[eng_name](query, num_results)
                if result and "No search results found" not in result:
                    prefix = f"[Search engine: {eng_name}]\n\n"
                    return prefix + result
            except Exception as e:
                logger.warning("%s search failed: %s", eng_name, e)
                errors.append(f"{eng_name}: {e}")

        return f"Error: Web search failed. Tried: {', '.join(order)}. Errors: {'; '.join(errors)}"

    def _duckduckgo_search(self, query: str, num_results: int) -> str:
        """Search using DuckDuckGo Lite HTML (no API key needed)."""
        url = "https://lite.duckduckgo.com/lite/"
        data = urllib.parse.urlencode({"q": query}).encode("utf-8")
        req = url_request(
            url,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
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

    def _bing_search(self, query: str, num_results: int) -> str:
        """Search using Bing HTML (no API key needed)."""
        url = f"https://www.bing.com/search?q={urllib.parse.quote(query)}&count={num_results}"
        req = url_request(url)

        with urllib.request.urlopen(req, timeout=15) as response:
            html = response.read().decode("utf-8", errors="replace")

        return self._parse_bing(html, query, num_results)

    def _parse_bing(self, html: str, query: str, num_results: int) -> str:
        """Parse Bing search results from HTML."""
        import re

        results = []
        # Bing results: <li class="b_algo"><h2><a href="...">title</a></h2><p>snippet</p></li>
        block_pattern = re.compile(r'<li class="b_algo">(.*?)</li>', re.DOTALL)
        link_pattern = re.compile(r'<a[^>]*href="(https?://[^"]+)"[^>]*>(.*?)</a>', re.DOTALL)
        snippet_pattern = re.compile(r"<p[^>]*>(.*?)</p>", re.DOTALL)

        for i, block in enumerate(block_pattern.findall(html)[:num_results]):
            link_match = link_pattern.search(block)
            snippet_match = snippet_pattern.search(block)
            if link_match:
                url = link_match.group(1)
                title = re.sub(r"<[^>]+>", "", link_match.group(2)).strip()
                snippet = re.sub(r"<[^>]+>", "", snippet_match.group(1)).strip() if snippet_match else ""
                result = f"[{i + 1}] {title}\n    URL: {url}"
                if snippet:
                    result += f"\n    {snippet}"
                results.append(result)

        if not results:
            return f"No search results found for: {query}"
        return f"Search results for: {query}\n\n" + "\n\n".join(results)

    def _google_search(self, query: str, num_results: int) -> str:
        """Search using Google HTML (no API key needed)."""
        url = f"https://www.google.com/search?q={urllib.parse.quote(query)}&num={num_results}"
        req = url_request(url, user_agent=BROWSER_USER_AGENT)

        with urllib.request.urlopen(req, timeout=15) as response:
            html = response.read().decode("utf-8", errors="replace")

        return self._parse_google(html, query, num_results)

    def _parse_google(self, html: str, query: str, num_results: int) -> str:
        """Parse Google search results from HTML."""
        import re

        results = []
        # Google wraps results in <div class="g"> blocks
        link_pattern = re.compile(r'<a[^>]*href="(https?://(?!www\.google\.)[^"]+)"[^>]*>(.*?)</a>', re.DOTALL)
        snippet_pattern = re.compile(r'<span class="(?:aCOpRe|hgKElc)"[^>]*>(.*?)</span>', re.DOTALL)

        links = link_pattern.findall(html)
        snippets = snippet_pattern.findall(html)

        seen_urls = set()
        for url, title in links:
            if url in seen_urls or "google.com" in url:
                continue
            seen_urls.add(url)
            clean_title = re.sub(r"<[^>]+>", "", title).strip()
            if not clean_title:
                continue

            idx = len(results)
            snippet = ""
            if idx < len(snippets):
                snippet = re.sub(r"<[^>]+>", "", snippets[idx]).strip()

            result = f"[{idx + 1}] {clean_title}\n    URL: {url}"
            if snippet:
                result += f"\n    {snippet}"
            results.append(result)
            if len(results) >= num_results:
                break

        if not results:
            return f"No search results found for: {query}"
        return f"Search results for: {query}\n\n" + "\n\n".join(results)

    def _sogou_search(self, query: str, num_results: int) -> str:
        """Search using Sogou (Chinese search engine, no API key needed)."""
        url = f"https://www.sogou.com/web?query={urllib.parse.quote(query)}"
        req = url_request(url)

        with urllib.request.urlopen(req, timeout=15) as response:
            html = response.read().decode("utf-8", errors="replace")

        return self._parse_sogou(html, query, num_results)

    def _parse_sogou(self, html: str, query: str, num_results: int) -> str:
        """Parse Sogou search results from HTML."""
        import re

        results = []
        link_pattern = re.compile(r'<a[^>]*href="(https?://[^"]+)"[^>]*>(.*?)</a>', re.DOTALL)

        seen_urls = set()
        for url, title in link_pattern.findall(html):
            if url in seen_urls or "sogou.com" in url:
                continue
            seen_urls.add(url)
            clean_title = re.sub(r"<[^>]+>", "", title).strip()
            if not clean_title or len(clean_title) < 5:
                continue

            idx = len(results)
            result = f"[{idx + 1}] {clean_title}\n    URL: {url}"
            results.append(result)
            if len(results) >= num_results:
                break

        if not results:
            return f"No search results found for: {query}"
        return f"Search results for: {query}\n\n" + "\n\n".join(results)

    def _baidu_search(self, query: str, num_results: int) -> str:
        """Search using Baidu (Chinese search engine, no API key needed)."""
        url = f"https://www.baidu.com/s?wd={urllib.parse.quote(query)}&rn={num_results}"
        req = url_request(url, user_agent=BROWSER_USER_AGENT)

        with urllib.request.urlopen(req, timeout=15) as response:
            html = response.read().decode("utf-8", errors="replace")

        return self._parse_baidu(html, query, num_results)

    def _parse_baidu(self, html: str, query: str, num_results: int) -> str:
        """Parse Baidu search results from HTML."""
        import re

        results = []
        # Baidu wraps results in <div class="result"> or <div class="c-container">
        block_pattern = re.compile(r'<div class="[^"]*c-container[^"]*"[^>]*>(.*?)</div>\s*</div>', re.DOTALL)
        link_pattern = re.compile(r'<a[^>]*href="(https?://[^"]+)"[^>]*>(.*?)</a>', re.DOTALL)

        for block in block_pattern.findall(html)[:num_results]:
            link_match = link_pattern.search(block)
            if link_match:
                url = link_match.group(1)
                title = re.sub(r"<[^>]+>", "", link_match.group(2)).strip()
                if title and "baidu.com" not in url:
                    idx = len(results)
                    result = f"[{idx + 1}] {title}\n    URL: {url}"
                    results.append(result)

        # Fallback: extract links directly
        if not results:
            for url, title in link_pattern.findall(html):
                if "baidu.com" in url or not title.strip():
                    continue
                clean_title = re.sub(r"<[^>]+>", "", title).strip()
                if clean_title and len(clean_title) > 5:
                    idx = len(results)
                    results.append(f"[{idx + 1}] {clean_title}\n    URL: {url}")
                    if len(results) >= num_results:
                        break

        if not results:
            return f"No search results found for: {query}"
        return f"Search results for: {query}\n\n" + "\n\n".join(results)


web_search = WebSearchTool()
