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

from cliver.agent_profile import get_agent_core
from cliver.tool import tool

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


@tool(
    name="ImageGenerate",
    description=(
        "Generate an image from a text description and save it to disk. "
        "Returns the saved file path(s) or URL(s). "
        "Use the `model` parameter to select a specific image-capable model "
        "(see Available Models in the system prompt). Leave empty for auto-detect. "
        "Use `output_dir` to save directly to a user-specified directory "
        "(e.g. '.cliver/images/') instead of the default output directory. "
        "Use when the user asks to create, draw, or generate an image."
    ),
)
def image_generate(
    prompt: str,
    model: str = "",
    output_dir: str = "",
) -> list[dict]:
    """Generate an image from a text description and save it to disk.

    Args:
        prompt: Text description of the image to generate.
        model: Model name to use for image generation. Pick from the Available Models
            list in the system prompt -- choose one with 'image generation' capability.
            Leave empty to auto-select the first available image-capable model.
        output_dir: Directory to save the generated image. If the user specifies a target
            directory, use it here directly (e.g. '.cliver/images/').
            Leave empty to use the default output directory.
    """
    executor = get_agent_core()
    if not executor:
        return [{"error": "No AgentCore available for image generation."}]

    try:
        from cliver.llm.call_context import CallContext
        from cliver.media import get_file_extension

        ctx = CallContext.get_current() or CallContext()
        media_count_before = len(ctx.generated_media)

        model_arg = model.strip() if model else None
        _run_async(executor.generate_image(prompt, model=model_arg, ctx=ctx))

        new_media = ctx.generated_media[media_count_before:]
        if not new_media:
            return [{"text": "Image generation completed but no media was returned."}]

        save_dir = output_dir.strip() or ctx.outputs_dir or os.path.join(os.getcwd(), ".cliver", "generated-images")

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

        return [{"text": f"Generated {len(file_paths)} image(s):\n" + "\n".join(file_paths)}]

    except Exception as e:
        logger.warning("Image generation failed: %s", e)
        return [{"error": f"Error generating image: {e}"}]
