"""Built-in web_fetch tool for fetching and extracting content from URLs."""

import logging
import re
import urllib.request
from typing import Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from cliver.util import url_request

logger = logging.getLogger(__name__)

MAX_CONTENT_LENGTH = 100000  # characters


class WebFetchInput(BaseModel):
    """Input schema for the web_fetch tool."""

    url: str = Field(description="The URL to fetch content from. Must be a fully-formed valid URL.")
    prompt: Optional[str] = Field(
        default=None,
        description="Optional: A prompt describing what information to extract from the page. "
        "If not provided, returns the full page content as markdown.",
    )


class WebFetchTool(BaseTool):
    """Fetches content from a URL and converts HTML to readable text."""

    name: str = "WebFetch"
    description: str = (
        "Fetches content from a specified URL and returns it as readable text.\n"
        "- Takes a URL and an optional prompt as input\n"
        "- Fetches the URL content and converts HTML to readable text\n"
        "- Returns the content (optionally filtered by the prompt)\n"
        "- Use this tool when you need to retrieve and analyze web content\n\n"
        "Usage notes:\n"
        "- The URL must be a fully-formed valid URL\n"
        "- This tool is read-only and does not modify any files\n"
        "- Results may be truncated if the content is very large\n"
        "- Supports both public and localhost URLs"
    )
    args_schema: Type[BaseModel] = WebFetchInput
    tags: list = ["fetch", "web", "read"]

    def _run(self, url: str, prompt: Optional[str] = None) -> str:
        try:
            if not url.startswith(("http://", "https://")):
                return "Error: URL must start with http:// or https://"

            req = url_request(
                url,
                headers={
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.7",
                },
            )

            with urllib.request.urlopen(req, timeout=30) as response:
                content_type = response.headers.get("Content-Type", "")

                # Reject binary content types — return metadata instead of garbage
                ct_lower = content_type.lower().split(";")[0].strip()
                if ct_lower.startswith(("image/", "audio/", "video/", "application/octet-stream")):
                    content_length = response.headers.get("Content-Length", "unknown")
                    return (
                        f"This URL points to a binary file ({ct_lower}, {content_length} bytes), "
                        f"not readable text. If you already have this image/media embedded in the "
                        f"conversation, analyze it directly instead of fetching it."
                    )

                raw = response.read()

                # Determine encoding
                encoding = "utf-8"
                if "charset=" in content_type:
                    encoding = content_type.split("charset=")[-1].split(";")[0].strip()

                content = raw.decode(encoding, errors="replace")

            # Convert HTML to readable text
            if "html" in content_type.lower():
                content = self._html_to_text(content)
            elif "json" in content_type.lower():
                # Pretty-format JSON
                import json

                try:
                    parsed = json.loads(content)
                    content = json.dumps(parsed, indent=2)
                except json.JSONDecodeError:
                    pass

            # Truncate if too long
            if len(content) > MAX_CONTENT_LENGTH:
                content = content[:MAX_CONTENT_LENGTH] + "\n\n... (content truncated)"

            if prompt:
                return f"URL: {url}\nPrompt: {prompt}\n\nContent:\n{content}"
            return f"URL: {url}\n\nContent:\n{content}"

        except urllib.error.HTTPError as e:
            return f"Error: HTTP {e.code} - {e.reason} for URL: {url}"
        except urllib.error.URLError as e:
            return f"Error: Could not reach URL: {url} - {e.reason}"
        except Exception as e:
            logger.error(f"Error fetching URL {url}: {e}")
            return f"Error fetching URL: {e}"

    def _html_to_text(self, html: str) -> str:
        """Convert HTML to readable plain text."""
        # Remove script and style elements
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<nav[^>]*>.*?</nav>", "", text, flags=re.DOTALL | re.IGNORECASE)

        # Convert common elements to markdown-like format
        text = re.sub(r"<h1[^>]*>(.*?)</h1>", r"\n# \1\n", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<h2[^>]*>(.*?)</h2>", r"\n## \1\n", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<h3[^>]*>(.*?)</h3>", r"\n### \1\n", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<li[^>]*>(.*?)</li>", r"\n- \1", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"<p[^>]*>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"</p>", "\n", text, flags=re.IGNORECASE)

        # Extract link text with URLs
        text = re.sub(r'<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>', r"\2 (\1)", text, flags=re.DOTALL | re.IGNORECASE)

        # Remove all remaining HTML tags
        text = re.sub(r"<[^>]+>", "", text)

        # Decode HTML entities
        import html

        text = html.unescape(text)

        # Clean up whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)

        return text.strip()


web_fetch = WebFetchTool()
