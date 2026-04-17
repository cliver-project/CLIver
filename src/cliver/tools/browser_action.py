"""Built-in tool for browser automation via Playwright."""

import asyncio
import base64
import logging
from typing import Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Module-level browser session singleton
_browser_session = None


class BrowserSession:
    """Manages a Playwright browser instance for the duration of a task."""

    def __init__(self):
        self._playwright = None
        self._browser = None
        self._page = None

    async def ensure_started(self):
        """Launch browser if not already running. Returns the page."""
        if self._page is not None:
            return self._page

        try:
            from playwright.async_api import async_playwright
        except ImportError as e:
            raise ImportError(
                "playwright is required for browser_action. "
                "Install with: pip install cliver[browser] && playwright install chromium"
            ) from e

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=True)
        self._page = await self._browser.new_page()
        logger.info("Browser session started")
        return self._page

    async def close(self):
        """Close the browser and clean up."""
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
        self._page = None
        logger.info("Browser session closed")

    @property
    def is_active(self) -> bool:
        return self._page is not None


def get_browser_session() -> BrowserSession:
    global _browser_session
    if _browser_session is None:
        _browser_session = BrowserSession()
    return _browser_session


async def close_browser_session():
    global _browser_session
    if _browser_session:
        await _browser_session.close()
        _browser_session = None


class BrowserActionInput(BaseModel):
    action: str = Field(description="Action to perform: navigate, click, fill, screenshot, get_text, evaluate")
    selector: Optional[str] = Field(
        default=None,
        description="CSS selector for the target element (for click, fill, get_text)",
    )
    value: Optional[str] = Field(
        default=None,
        description="URL for navigate, text for fill, JavaScript code for evaluate",
    )


class BrowserActionTool(BaseTool):
    name: str = "Browser"
    description: str = (
        "Control a headless browser — navigate to URLs, click elements, fill forms, "
        "take screenshots, extract text, run JavaScript. Use for web interaction, "
        "testing, and automation that requires a real browser. "
        "The browser session persists across calls within the same task."
    )
    args_schema: Type[BaseModel] = BrowserActionInput
    tags: list = ["browser", "web", "execute"]

    def _run(self, action: str, selector: str = None, value: str = None) -> str:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're inside an async context — create a new thread to run the coroutine
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self._async_run(action, selector, value))
                return future.result()
        else:
            return asyncio.run(self._async_run(action, selector, value))

    async def _async_run(self, action: str, selector: str = None, value: str = None) -> str:
        session = get_browser_session()

        try:
            page = await session.ensure_started()
        except ImportError as e:
            return str(e)

        try:
            if action == "navigate":
                if not value:
                    return "Error: 'value' (URL) is required for navigate action"
                await page.goto(value, wait_until="domcontentloaded", timeout=30000)
                title = await page.title()
                return f"Navigated to {value} — Title: {title}"

            elif action == "click":
                if not selector:
                    return "Error: 'selector' is required for click action"
                await page.click(selector, timeout=10000)
                return f"Clicked: {selector}"

            elif action == "fill":
                if not selector:
                    return "Error: 'selector' is required for fill action"
                if value is None:
                    return "Error: 'value' (text) is required for fill action"
                await page.fill(selector, value, timeout=10000)
                return f"Filled '{selector}' with: {value}"

            elif action == "screenshot":
                screenshot_bytes = await page.screenshot(full_page=False)
                b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
                return f"Screenshot captured ({len(screenshot_bytes)} bytes). Base64: {b64[:100]}..."

            elif action == "get_text":
                if selector:
                    element = await page.query_selector(selector)
                    if element:
                        text = await element.text_content()
                        return text or "(empty)"
                    return f"Element not found: {selector}"
                else:
                    text = await page.text_content("body")
                    # Truncate to avoid massive output
                    if text and len(text) > 5000:
                        return text[:5000] + f"\n\n... (truncated, {len(text)} total chars)"
                    return text or "(empty page)"

            elif action == "evaluate":
                if not value:
                    return "Error: 'value' (JavaScript code) is required for evaluate action"
                result = await page.evaluate(value)
                return str(result) if result is not None else "(no return value)"

            else:
                return f"Unknown action: {action}. Use: navigate, click, fill, screenshot, get_text, evaluate"

        except Exception as e:
            logger.warning(f"Browser action '{action}' failed: {e}")
            return f"Error: {e}"


browser_action = BrowserActionTool()
