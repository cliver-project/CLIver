"""Built-in web_fetch tool for fetching and extracting content from URLs."""

import logging
import re
import urllib.request
from pathlib import Path
from typing import Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from cliver.util import url_request

logger = logging.getLogger(__name__)

MAX_CONTENT_LENGTH = 20000  # ~5k tokens — keeps LLM context lean


class WebFetchInput(BaseModel):
    """Input schema for the web_fetch tool."""

    url: str = Field(description="The URL to fetch content from. Must be a fully-formed valid URL.")
    save_to_file: Optional[str] = Field(
        default=None,
        description="Save the full fetched content to this local file path instead of returning it. "
        "Returns a short summary and the file path. Use Grep or ReadFile on the saved file "
        "to search or read specific sections without loading it all into context.",
    )


class WebFetchTool(BaseTool):
    """Fetches content from a URL and converts HTML to readable text."""

    name: str = "WebFetch"
    description: str = (
        "Fetches content from a URL and returns it as readable text.\n\n"
        "For LARGE pages, prefer the save_to_file workflow to avoid flooding the context:\n"
        "  1. WebFetch(url=..., save_to_file='/tmp/page.md') — saves full content to a local file\n"
        "  2. Grep(pattern='...', path='/tmp/page.md') — search the saved file locally\n"
        "  3. ReadFile(path='/tmp/page.md', offset=..., limit=...) — read specific sections\n\n"
        "This keeps large web content out of the conversation and avoids unnecessary round trips.\n\n"
        "Without save_to_file, content is returned directly (truncated to ~20k chars).\n\n"
        "Usage notes:\n"
        "- The URL must be a fully-formed valid URL\n"
        "- This tool is read-only (except when saving to a file)\n"
        "- Supports both public and localhost URLs"
    )
    args_schema: Type[BaseModel] = WebFetchInput
    tags: list = ["fetch", "web", "read"]

    def _run(self, url: str, save_to_file: Optional[str] = None) -> str:
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

                ct_lower = content_type.lower().split(";")[0].strip()
                if ct_lower.startswith(("image/", "audio/", "video/", "application/octet-stream")):
                    content_length = response.headers.get("Content-Length", "unknown")
                    return (
                        f"This URL points to a binary file ({ct_lower}, {content_length} bytes), "
                        f"not readable text. If you already have this image/media embedded in the "
                        f"conversation, analyze it directly instead of fetching it."
                    )

                raw = response.read()

                encoding = "utf-8"
                if "charset=" in content_type:
                    encoding = content_type.split("charset=")[-1].split(";")[0].strip()

                content = raw.decode(encoding, errors="replace")

            if "html" in content_type.lower():
                content = self._html_to_text(content)
            elif "json" in content_type.lower():
                import json

                try:
                    parsed = json.loads(content)
                    content = json.dumps(parsed, indent=2)
                except json.JSONDecodeError:
                    pass

            if save_to_file:
                return self._save_and_summarize(url, content, save_to_file)

            if len(content) > MAX_CONTENT_LENGTH:
                total = len(content)
                content = content[:MAX_CONTENT_LENGTH]
                content += (
                    f"\n\n... (truncated — showing {MAX_CONTENT_LENGTH:,} of {total:,} chars. "
                    f"Use save_to_file to get the full content for local search.)"
                )

            return f"URL: {url}\n\nContent:\n{content}"

        except urllib.error.HTTPError as e:
            return f"Error: HTTP {e.code} - {e.reason} for URL: {url}"
        except urllib.error.URLError as e:
            return f"Error: Could not reach URL: {url} - {e.reason}"
        except Exception as e:
            logger.error(f"Error fetching URL {url}: {e}")
            return f"Error fetching URL: {e}"

    @staticmethod
    def _save_and_summarize(url: str, content: str, file_path: str) -> str:
        """Save full content to a file and return a short summary."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

        lines = content.splitlines()
        line_count = len(lines)
        char_count = len(content)

        preview_lines = lines[:30]
        preview = "\n".join(preview_lines)
        if line_count > 30:
            preview += f"\n... ({line_count - 30} more lines)"

        return (
            f"Saved to: {file_path} ({char_count:,} chars, {line_count:,} lines)\n"
            f"Source: {url}\n\n"
            f"Preview (first 30 lines):\n{preview}\n\n"
            f"Use Grep(pattern='...', path='{file_path}') to search, "
            f"or ReadFile(path='{file_path}', offset=..., limit=...) to read sections."
        )

    def _html_to_text(self, html: str) -> str:
        """Convert HTML to readable plain text."""
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<nav[^>]*>.*?</nav>", "", text, flags=re.DOTALL | re.IGNORECASE)

        text = re.sub(r"<h1[^>]*>(.*?)</h1>", r"\n# \1\n", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<h2[^>]*>(.*?)</h2>", r"\n## \1\n", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<h3[^>]*>(.*?)</h3>", r"\n### \1\n", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<li[^>]*>(.*?)</li>", r"\n- \1", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"<p[^>]*>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"</p>", "\n", text, flags=re.IGNORECASE)

        text = re.sub(r'<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>', r"\2 (\1)", text, flags=re.DOTALL | re.IGNORECASE)

        text = re.sub(r"<[^>]+>", "", text)

        import html

        text = html.unescape(text)

        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)

        return text.strip()


web_fetch = WebFetchTool()
