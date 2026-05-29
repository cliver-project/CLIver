"""Built-in tool for generating images from text descriptions.

Uses AgentCore.generate() which delegates to the protocol engine.
"""

import logging
import os

from cliver.agent_profile import get_agent_core
from cliver.tool import tool

logger = logging.getLogger(__name__)


@tool(
    name="ImageGenerate",
    description=(
        "Generate an image from a text description and save it to disk. "
        "Returns the saved file path(s). "
        "Use `model` to select a specific image-capable model. "
        "Use `output_dir` to save to a user-specified directory. "
        "Use when the user asks to create, draw, or generate an image."
    ),
)
def image_generate(
    prompt: str,
    model: str = "",
    output_dir: str = "",
) -> list[dict]:
    """Generate an image from a text description.

    Args:
        prompt: Text description of the image to generate.
        model: Model name for image generation. Leave empty for auto-detect.
        output_dir: Directory to save generated images.
    """
    import asyncio

    agent_core = get_agent_core()
    if not agent_core:
        return [{"error": "No AgentCore available for image generation."}]

    try:
        save_dir = output_dir.strip() or os.path.join(os.getcwd(), ".cliver", "generated-images")
        response = asyncio.run(
            agent_core.generate(
                prompt=prompt,
                media_type="image",
                output_dir=save_dir,
            )
        )

        if not response.media:
            return [{"text": response.message.text or "Image generation completed but no media was returned."}]

        paths = [m.saved_path or m.data for m in response.media]
        return [{"text": f"Generated {len(paths)} image(s):\n" + "\n".join(paths)}]

    except Exception as e:
        logger.warning("Image generation failed: %s", e)
        return [{"error": f"Error generating image: {e}"}]
