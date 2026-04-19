"""Built-in tool for generating images from text descriptions.

Uses whatever configured provider has image_url set, or a model
with TEXT_TO_IMAGE capability. Provider-specific API differences
are handled by generation helpers in llm/media_generation/.
"""

import asyncio
import logging
import threading
from typing import Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from cliver.agent_profile import get_task_executor

logger = logging.getLogger(__name__)


def _run_async(coro):
    """Run an async coroutine from sync context, even inside a running event loop."""
    result = None
    exception = None

    def _thread_target():
        nonlocal result, exception
        try:
            result = asyncio.run(coro)
        except Exception as e:
            exception = e

    thread = threading.Thread(target=_thread_target)
    thread.start()
    thread.join()

    if exception:
        raise exception
    return result


class ImageGenerateInput(BaseModel):
    prompt: str = Field(description="Text description of the image to generate")
    model: Optional[str] = Field(
        default=None,
        description="Model name to use for generation (optional, auto-discovered if not set)",
    )


class ImageGenerateTool(BaseTool):
    name: str = "ImageGenerate"
    description: str = (
        "Generate an image from a text description. Returns the image URL(s). "
        "Use when the user asks to create, draw, or generate an image. "
        "Requires a provider with image_url configured or a model with text_to_image capability."
    )
    args_schema: Type[BaseModel] = ImageGenerateInput

    def _run(self, prompt: str, model: Optional[str] = None) -> str:
        executor = get_task_executor()
        if not executor:
            return "Error: No task executor available for image generation."

        try:
            result = _run_async(executor.generate_image(prompt, model))
            return result.content
        except Exception as e:
            logger.warning("Image generation failed: %s", e)
            return f"Error generating image: {e}"


image_generate = ImageGenerateTool()
