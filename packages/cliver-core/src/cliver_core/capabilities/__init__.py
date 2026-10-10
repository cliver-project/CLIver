"""Standalone model-call capabilities — each works out of the box.

Every function here is a self-contained API call: given a
:class:`~cliver_core.bindings.ModelBinding` — the model handle carrying the
model definition, its provider, the endpoint path override, and the default
parameters — it performs the call and returns the result.  No AgentCore
required; AgentCore's embedded methods are thin wrappers that supply the
binding from its ``bindings`` and forward the call-site parameters.

Example::

    from cliver_core.bindings import ModelBinding
    from cliver_core.capabilities import embed, generate_image

    image = await generate_image(
        prompt="a red panda on a skateboard",
        binding=ModelBinding.image(
            model="gpt-image-1",
            api_key="sk-...",
            base_url="https://api.openai.com/v1",
            size="1024x1024",
            n=2,
        ),
    )

    vectors = await embed(
        texts=["hello", "world"],
        binding=ModelBinding.embed(
            model="text-embedding-3-small",
            api_key="sk-...",
            base_url="https://api.openai.com/v1",
        ),
    )

Purposes: :func:`generate_image`, :func:`generate_video`,
:func:`text_to_speech`, :func:`transcribe_audio`, :func:`embed`,
:func:`rerank`.
"""

from cliver_core.capabilities.asr import transcribe_audio
from cliver_core.capabilities.embed import embed
from cliver_core.capabilities.image import generate_image
from cliver_core.capabilities.rerank import rerank
from cliver_core.capabilities.tts import text_to_speech
from cliver_core.capabilities.video import generate_video

__all__ = [
    "embed",
    "generate_image",
    "generate_video",
    "rerank",
    "text_to_speech",
    "transcribe_audio",
]
