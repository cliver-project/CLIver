"""Built-in tool for generating images from text descriptions.

Uses the provider that has image_url and image_model configured.
Provider-specific API differences are handled by generation
helpers in llm/media_generation/.

Saving behavior:
- Base64 images: always saved to disk (ctx.outputs_dir or ./generated_images/)
- URL images: returned as URLs without downloading
"""

import asyncio
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from cliver.agent_profile import get_agent_core

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


class ImageGenerateTool(BaseTool):
    name: str = "ImageGenerate"
    description: str = (
        "Generate an image from a text description and save it to disk. "
        "Returns the saved file path(s) or URL(s). "
        "Use when the user asks to create, draw, or generate an image. "
    )
    args_schema: Type[BaseModel] = ImageGenerateInput

    def _run(self, prompt: str) -> str:
        executor = get_agent_core()
        if not executor:
            return "Error: No AgentCore available for image generation."

        try:
            from cliver.llm.call_context import CallContext
            from cliver.media import get_file_extension

            ctx = CallContext.get_current() or CallContext()
            media_count_before = len(ctx.generated_media)

            _run_async(executor.generate_image(prompt, ctx=ctx))

            new_media = ctx.generated_media[media_count_before:]
            if not new_media:
                return "Image generation completed but no media was returned."

            save_dir = ctx.outputs_dir or os.path.join(os.getcwd(), "generated_images")

            file_paths = []
            for i, m in enumerate(new_media):
                if m.is_url():
                    file_paths.append(m.data)
                else:
                    save_path = Path(save_dir)
                    save_path.mkdir(parents=True, exist_ok=True)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    ext = get_file_extension(m.mime_type)
                    full_path = save_path / f"image_{ts}_{i}{ext}"
                    m.save(full_path)
                    file_paths.append(str(full_path))
                    logger.info("Saved image to %s", full_path)

            return f"Generated {len(file_paths)} image(s):\n" + "\n".join(file_paths)

        except Exception as e:
            logger.warning("Image generation failed: %s", e)
            return f"Error generating image: {e}"


image_generate = ImageGenerateTool()
