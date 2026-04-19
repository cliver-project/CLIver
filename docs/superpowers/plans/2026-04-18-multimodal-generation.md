# Multimodal Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable CLIver to generate images via provider APIs, supporting direct model targeting, LLM tool invocation, and multimodal chat — all through the existing `cliver chat` command.

**Architecture:** Generation helpers (per image API format) are selected by `image_url` pattern, independent of the chat engine. `AgentCore.generate_image()` is the embeddable API entry point. Capability routing in `process_user_input()` detects generation-only models and skips the Re-Act loop. An `ImageGenerate` builtin tool lets chat LLMs call image generation mid-conversation.

**Tech Stack:** Python, Pydantic, httpx (async HTTP), asyncio, Click CLI

**Design Spec:** `docs/superpowers/specs/2026-04-18-multimodal-generation-design.md`

---

### Task 1: Add `image_url` and `audio_url` to ProviderConfig

**Files:**
- Modify: `src/cliver/config.py`
- Modify: `src/cliver/commands/provider.py`
- Modify: `src/cliver/commands/config.py`
- Test: `tests/test_provider_config.py`

- [ ] **Step 1: Write failing tests for new ProviderConfig fields**

Append to `tests/test_provider_config.py`:

```python
class TestProviderConfigMediaUrls:
    def test_image_url(self):
        p = ProviderConfig(
            name="mm", type="openai", api_url="http://x",
            image_url="https://api.minimaxi.com/v1/image_generation",
        )
        assert p.image_url == "https://api.minimaxi.com/v1/image_generation"

    def test_audio_url(self):
        p = ProviderConfig(
            name="mm", type="openai", api_url="http://x",
            audio_url="https://api.minimaxi.com/v1/audio_generation",
        )
        assert p.audio_url == "https://api.minimaxi.com/v1/audio_generation"

    def test_urls_default_none(self):
        p = ProviderConfig(name="mm", type="openai", api_url="http://x")
        assert p.image_url is None
        assert p.audio_url is None

    def test_model_dump_excludes_null_urls(self):
        p = ProviderConfig(name="mm", type="openai", api_url="http://x")
        dumped = p.model_dump()
        assert "image_url" not in dumped
        assert "audio_url" not in dumped

    def test_model_dump_includes_set_urls(self):
        p = ProviderConfig(
            name="mm", type="openai", api_url="http://x",
            image_url="http://img",
        )
        dumped = p.model_dump()
        assert dumped["image_url"] == "http://img"

    def test_load_config_with_image_url(self, tmp_path):
        import yaml
        from cliver.config import ConfigManager

        config_yaml = {
            "providers": {
                "mm": {
                    "type": "openai",
                    "api_url": "http://x",
                    "image_url": "http://img",
                }
            },
        }
        (tmp_path / "config.yaml").write_text(yaml.dump(config_yaml))
        cm = ConfigManager(tmp_path)
        assert cm.config.providers["mm"].image_url == "http://img"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_provider_config.py::TestProviderConfigMediaUrls -v`
Expected: FAIL — `image_url` field does not exist

- [ ] **Step 3: Add fields to ProviderConfig**

In `src/cliver/config.py`, add to `ProviderConfig` after `rate_limit`:

```python
image_url: Optional[str] = Field(default=None, description="Full URL for image generation endpoint")
audio_url: Optional[str] = Field(default=None, description="Full URL for audio generation endpoint")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_provider_config.py -v`
Expected: All PASS

- [ ] **Step 5: Add `--image-url` and `--audio-url` options to provider CLI commands**

In `src/cliver/commands/provider.py`, add options to both `add_provider` and `set_provider` commands:

```python
@click.option("--image-url", type=str, default=None, help="Image generation endpoint URL")
@click.option("--audio-url", type=str, default=None, help="Audio generation endpoint URL")
```

Update `add_or_update_provider()` in `src/cliver/config.py` to accept and pass through `image_url` and `audio_url` parameters.

- [ ] **Step 6: Update `config show` to display image_url/audio_url**

In `src/cliver/commands/config.py`, in the providers section of `show_config`, add an `Image URL` column to the providers table.

In `src/cliver/commands/provider.py`, add an `Image URL` column to the `provider list` table.

- [ ] **Step 7: Run full test suite + lint**

Run: `uv run ruff check && uv run pytest tests/ -x -q`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add src/cliver/config.py src/cliver/commands/provider.py src/cliver/commands/config.py tests/test_provider_config.py
git commit -m "feat: add image_url and audio_url fields to ProviderConfig"
```

---

### Task 2: Image Generation Helper — Base and MiniMax

**Files:**
- Create: `src/cliver/llm/media_generation/__init__.py`
- Create: `src/cliver/llm/media_generation/base.py`
- Create: `src/cliver/llm/media_generation/minimax.py`
- Create: `tests/test_image_generation.py`

- [ ] **Step 1: Write failing tests for MiniMax helper**

```python
# tests/test_image_generation.py
import json
from unittest.mock import AsyncMock, patch

import pytest

from cliver.llm.media_generation import get_image_helper
from cliver.llm.media_generation.base import ImageGenerationHelper
from cliver.llm.media_generation.minimax import MiniMaxImageHelper
from cliver.media import MediaContent, MediaType


class TestHelperRegistry:
    def test_minimax_url_returns_minimax_helper(self):
        helper = get_image_helper("https://api.minimaxi.com/v1/image_generation")
        assert isinstance(helper, MiniMaxImageHelper)

    def test_unknown_url_returns_openai_helper(self):
        from cliver.llm.media_generation.openai_compat import OpenAIImageHelper

        helper = get_image_helper("https://api.openai.com/v1/images/generations")
        assert isinstance(helper, OpenAIImageHelper)


class TestMiniMaxImageHelper:
    @pytest.mark.asyncio
    async def test_generate_returns_media_content(self):
        helper = MiniMaxImageHelper()
        mock_response = {
            "base_resp": {"status_code": 0, "status_msg": "success"},
            "data": {"image_urls": ["https://cdn.minimax.com/img1.png", "https://cdn.minimax.com/img2.png"]},
            "metadata": {"success_count": "2", "failed_count": "0"},
        }

        with patch("cliver.llm.media_generation.minimax.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_resp = AsyncMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = mock_response
            mock_resp.raise_for_status = lambda: None
            mock_client.post.return_value = mock_resp

            result = await helper.generate(
                url="https://api.minimaxi.com/v1/image_generation",
                api_key="sk-test",
                prompt="a sunset over mountains",
            )

            assert len(result) == 2
            assert all(isinstance(m, MediaContent) for m in result)
            assert all(m.type == MediaType.IMAGE for m in result)
            assert result[0].data == "https://cdn.minimax.com/img1.png"
            assert result[0].source == "minimax_image_generation"

            # Verify the POST was called correctly
            call_args = mock_client.post.call_args
            assert call_args[0][0] == "https://api.minimaxi.com/v1/image_generation"
            body = call_args[1]["json"]
            assert body["prompt"] == "a sunset over mountains"
            assert body["model"] == "image-01"

    @pytest.mark.asyncio
    async def test_generate_with_custom_model(self):
        helper = MiniMaxImageHelper()
        mock_response = {
            "base_resp": {"status_code": 0, "status_msg": "success"},
            "data": {"image_urls": ["https://cdn.minimax.com/img.png"]},
            "metadata": {"success_count": "1", "failed_count": "0"},
        }

        with patch("cliver.llm.media_generation.minimax.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_resp = AsyncMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = mock_response
            mock_resp.raise_for_status = lambda: None
            mock_client.post.return_value = mock_resp

            result = await helper.generate(
                url="https://api.minimaxi.com/v1/image_generation",
                api_key="sk-test",
                prompt="a cat",
                model_name="image-02",
            )

            body = mock_client.post.call_args[1]["json"]
            assert body["model"] == "image-02"

    @pytest.mark.asyncio
    async def test_generate_api_error(self):
        helper = MiniMaxImageHelper()
        mock_response = {
            "base_resp": {"status_code": 1001, "status_msg": "invalid prompt"},
            "data": {},
        }

        with patch("cliver.llm.media_generation.minimax.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_resp = AsyncMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = mock_response
            mock_resp.raise_for_status = lambda: None
            mock_client.post.return_value = mock_resp

            with pytest.raises(RuntimeError, match="invalid prompt"):
                await helper.generate(
                    url="https://api.minimaxi.com/v1/image_generation",
                    api_key="sk-test",
                    prompt="bad prompt",
                )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_image_generation.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Create the base module**

```python
# src/cliver/llm/media_generation/__init__.py
"""Media generation helpers — provider-specific image/audio API adapters."""

from cliver.llm.media_generation.base import ImageGenerationHelper
from cliver.llm.media_generation.registry import get_image_helper

__all__ = ["ImageGenerationHelper", "get_image_helper"]
```

```python
# src/cliver/llm/media_generation/base.py
"""Abstract base class for image generation helpers."""

import logging
from abc import ABC, abstractmethod
from typing import Optional

from cliver.media import MediaContent

logger = logging.getLogger(__name__)


class ImageGenerationHelper(ABC):
    """Base class for provider-specific image generation API adapters.

    Each subclass handles one image API format (MiniMax, OpenAI, DashScope, etc.).
    Selected by URL pattern matching, independent of the chat protocol engine.
    """

    @abstractmethod
    async def generate(
        self,
        url: str,
        api_key: str,
        prompt: str,
        model_name: Optional[str] = None,
        **params,
    ) -> list[MediaContent]:
        """Generate images from a text prompt.

        Args:
            url: Full image generation endpoint URL
            api_key: Provider API key
            prompt: Text description of the image
            model_name: Model name override (uses adapter default if None)
            **params: Provider-specific parameters (aspect_ratio, size, n, etc.)

        Returns:
            List of MediaContent with type=IMAGE
        """
        ...
```

```python
# src/cliver/llm/media_generation/registry.py
"""Helper registry — selects the right image generation helper by URL pattern."""

import logging
import re
from typing import Dict, List, Tuple

from cliver.llm.media_generation.base import ImageGenerationHelper

logger = logging.getLogger(__name__)

_REGISTRY: List[Tuple[re.Pattern, type]] = []


def register_helper(url_pattern: str, helper_class: type) -> None:
    _REGISTRY.append((re.compile(url_pattern, re.IGNORECASE), helper_class))


def get_image_helper(image_url: str) -> ImageGenerationHelper:
    for pattern, cls in _REGISTRY:
        if pattern.search(image_url):
            return cls()

    from cliver.llm.media_generation.openai_compat import OpenAIImageHelper

    return OpenAIImageHelper()


def _ensure_registered():
    if _REGISTRY:
        return
    from cliver.llm.media_generation.minimax import MiniMaxImageHelper
    from cliver.llm.media_generation.openai_compat import OpenAIImageHelper

    register_helper(r"minimax", MiniMaxImageHelper)


_ensure_registered()
```

- [ ] **Step 4: Create the MiniMax helper**

```python
# src/cliver/llm/media_generation/minimax.py
"""MiniMax image generation API adapter.

Endpoint: POST {image_url}
Request:  {"model": "image-01", "prompt": "...", "response_format": "url", ...}
Response: {"data": {"image_urls": [...]}, "base_resp": {"status_code": 0, ...}}
"""

import logging
from typing import Optional

import httpx

from cliver.llm.media_generation.base import ImageGenerationHelper
from cliver.media import MediaContent, MediaType

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "image-01"


class MiniMaxImageHelper(ImageGenerationHelper):
    async def generate(
        self,
        url: str,
        api_key: str,
        prompt: str,
        model_name: Optional[str] = None,
        **params,
    ) -> list[MediaContent]:
        body = {
            "model": model_name or DEFAULT_MODEL,
            "prompt": prompt,
            "response_format": "url",
        }
        for key in ("aspect_ratio", "n", "prompt_optimizer", "width", "height"):
            if key in params:
                body[key] = params[key]

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(url, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        base_resp = data.get("base_resp", {})
        if base_resp.get("status_code", 0) != 0:
            raise RuntimeError(
                f"MiniMax image generation failed: {base_resp.get('status_msg', 'unknown error')}"
            )

        image_urls = data.get("data", {}).get("image_urls", [])
        return [
            MediaContent(
                type=MediaType.IMAGE,
                data=img_url,
                mime_type="image/png",
                source="minimax_image_generation",
            )
            for img_url in image_urls
        ]
```

- [ ] **Step 5: Create the OpenAI helper**

```python
# src/cliver/llm/media_generation/openai_compat.py
"""OpenAI-compatible image generation API adapter.

Endpoint: POST {image_url}
Request:  {"model": "dall-e-3", "prompt": "...", "response_format": "url", ...}
Response: {"data": [{"url": "..."}, ...]}
"""

import logging
from typing import Optional

import httpx

from cliver.llm.media_generation.base import ImageGenerationHelper
from cliver.media import MediaContent, MediaType

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "dall-e-3"


class OpenAIImageHelper(ImageGenerationHelper):
    async def generate(
        self,
        url: str,
        api_key: str,
        prompt: str,
        model_name: Optional[str] = None,
        **params,
    ) -> list[MediaContent]:
        body = {
            "model": model_name or DEFAULT_MODEL,
            "prompt": prompt,
            "response_format": "url",
        }
        for key in ("size", "quality", "n", "style"):
            if key in params:
                body[key] = params[key]

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(url, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        images = data.get("data", [])
        return [
            MediaContent(
                type=MediaType.IMAGE,
                data=item.get("url", item.get("b64_json", "")),
                mime_type="image/png",
                source="openai_image_generation",
            )
            for item in images
        ]
```

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/test_image_generation.py -v`
Expected: All PASS

- [ ] **Step 7: Run lint + full test suite**

Run: `uv run ruff check && uv run pytest tests/ -x -q`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add src/cliver/llm/media_generation/ tests/test_image_generation.py
git commit -m "feat: image generation helpers for MiniMax and OpenAI"
```

---

### Task 3: AgentCore.generate_image() Method

**Files:**
- Modify: `src/cliver/llm/llm.py`
- Modify: `tests/test_image_generation.py`

- [ ] **Step 1: Write failing tests for AgentCore.generate_image()**

Append to `tests/test_image_generation.py`:

```python
from unittest.mock import Mock, AsyncMock, patch
from langchain_core.messages import AIMessage

from cliver.config import ModelConfig, ProviderConfig, RateLimitConfig
from cliver.llm.llm import AgentCore
from cliver.model_capabilities import ModelCapability


class TestAgentCoreGenerateImage:
    def _make_provider(self, image_url="https://api.minimaxi.com/v1/image_generation"):
        return ProviderConfig(
            name="mm", type="openai", api_url="http://x",
            api_key="sk-test", image_url=image_url,
        )

    def _make_image_model(self, provider_name="mm"):
        mc = ModelConfig(name="img", provider=provider_name, name_in_provider="image-01")
        mc.capabilities = {ModelCapability.TEXT_TO_IMAGE}
        return mc

    def _make_chat_model(self, provider_name="mm"):
        mc = ModelConfig(name="chat", provider=provider_name, url="http://x")
        mc.capabilities = {ModelCapability.TEXT_TO_TEXT, ModelCapability.TOOL_CALLING}
        return mc

    @pytest.mark.asyncio
    async def test_generate_with_explicit_model(self):
        prov = self._make_provider()
        img_model = self._make_image_model()
        img_model._provider_config = prov

        executor = AgentCore(llm_models={"img": img_model}, mcp_servers={})

        mock_media = [MediaContent(type=MediaType.IMAGE, data="https://img.png", mime_type="image/png", source="test")]
        with patch("cliver.llm.llm.get_image_helper") as mock_get:
            mock_helper = AsyncMock()
            mock_helper.generate.return_value = mock_media
            mock_get.return_value = mock_helper

            result = await executor.generate_image("a sunset", model="img")

            assert isinstance(result, AIMessage)
            assert "media_content" in result.additional_kwargs
            assert len(result.additional_kwargs["media_content"]) == 1
            mock_helper.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_discovers_provider_with_image_url(self):
        prov = self._make_provider()
        chat_model = self._make_chat_model()
        chat_model._provider_config = prov

        executor = AgentCore(llm_models={"chat": chat_model}, mcp_servers={})
        executor.configure_rate_limits({"mm": prov})

        mock_media = [MediaContent(type=MediaType.IMAGE, data="https://img.png", mime_type="image/png", source="test")]
        with patch("cliver.llm.llm.get_image_helper") as mock_get:
            mock_helper = AsyncMock()
            mock_helper.generate.return_value = mock_media
            mock_get.return_value = mock_helper

            result = await executor.generate_image("a cat")

            assert isinstance(result, AIMessage)
            mock_helper.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_no_image_support(self):
        chat_model = ModelConfig(name="chat", provider="openai", url="http://x")
        executor = AgentCore(llm_models={"chat": chat_model}, mcp_servers={})

        result = await executor.generate_image("a cat")
        assert isinstance(result, AIMessage)
        assert "no provider" in result.content.lower() or "no model" in result.content.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_image_generation.py::TestAgentCoreGenerateImage -v`
Expected: FAIL — `generate_image` does not exist

- [ ] **Step 3: Implement generate_image() on AgentCore**

In `src/cliver/llm/llm.py`, add import at top:

```python
from cliver.llm.media_generation import get_image_helper
```

Add these methods to the `AgentCore` class, after `_wait_for_rate_limit`:

```python
async def generate_image(self, prompt: str, model: str = None, **params) -> BaseMessage:
    """Generate images from a text prompt via the provider's image API.

    Resolution order:
    1. If model specified → use its provider's image_url
    2. Scan llm_models for TEXT_TO_IMAGE capability
    3. Scan providers for any with image_url set
    """
    provider_config, model_name = self._resolve_image_provider(model)
    if not provider_config or not provider_config.image_url:
        return AIMessage(content="Error: No model or provider with image generation support configured.")

    # Rate limit
    await self._wait_for_rate_limit(provider_config.name)

    try:
        helper = get_image_helper(provider_config.image_url)
        api_key = provider_config.get_api_key()
        media_list = await helper.generate(
            url=provider_config.image_url,
            api_key=api_key,
            prompt=prompt,
            model_name=model_name,
            **params,
        )
    except Exception as e:
        logger.error("Image generation failed: %s", e)
        return AIMessage(content=f"Error generating image: {e}")

    # Format URLs for display
    urls = [m.data for m in media_list if m.data]
    content = f"Generated {len(media_list)} image(s):\n" + "\n".join(urls) if urls else "Image generated."

    return AIMessage(
        content=content,
        additional_kwargs={"media_content": media_list},
    )

def _resolve_image_provider(self, model: str = None):
    """Resolve which provider and model name to use for image generation.

    Returns (ProviderConfig, model_name) or (None, None).
    """
    from cliver.model_capabilities import ModelCapability

    # 1. Explicit model
    if model:
        mc = self._get_llm_model(model)
        if mc and mc._provider_config and mc._provider_config.image_url:
            return mc._provider_config, mc.name_in_provider or mc.name
        # Model exists but provider has no image_url — check if model itself is usable
        if mc and getattr(mc, "_provider_config", None):
            return mc._provider_config, mc.name_in_provider or mc.name

    # 2. Scan for model with TEXT_TO_IMAGE capability
    for name, mc in self.llm_models.items():
        caps = mc.get_capabilities()
        if ModelCapability.TEXT_TO_IMAGE in caps:
            prov = getattr(mc, "_provider_config", None)
            if prov and prov.image_url:
                return prov, mc.name_in_provider or mc.name

    # 3. Scan providers for any with image_url
    for name, mc in self.llm_models.items():
        prov = getattr(mc, "_provider_config", None)
        if prov and prov.image_url:
            return prov, None

    return None, None
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_image_generation.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/cliver/llm/llm.py tests/test_image_generation.py
git commit -m "feat: AgentCore.generate_image() with provider resolution"
```

---

### Task 4: Capability Routing in process_user_input()

**Files:**
- Modify: `src/cliver/llm/llm.py`
- Modify: `tests/test_image_generation.py`

- [ ] **Step 1: Write failing test for direct routing**

Append to `tests/test_image_generation.py`:

```python
class TestCapabilityRouting:
    @pytest.mark.asyncio
    async def test_text_to_image_only_model_routes_to_generate(self):
        """A model with TEXT_TO_IMAGE but not TEXT_TO_TEXT skips Re-Act loop."""
        prov = ProviderConfig(
            name="mm", type="openai", api_url="http://x",
            api_key="sk-test",
            image_url="https://api.minimaxi.com/v1/image_generation",
        )
        img_model = ModelConfig(name="img", provider="mm", name_in_provider="image-01")
        img_model.capabilities = {ModelCapability.TEXT_TO_IMAGE}
        img_model._provider_config = prov

        executor = AgentCore(llm_models={"img": img_model}, mcp_servers={})

        mock_media = [MediaContent(type=MediaType.IMAGE, data="https://img.png", mime_type="image/png", source="test")]
        with patch("cliver.llm.llm.get_image_helper") as mock_get:
            mock_helper = AsyncMock()
            mock_helper.generate.return_value = mock_media
            mock_get.return_value = mock_helper

            result = await executor.process_user_input("a sunset", model="img")

            assert isinstance(result, AIMessage)
            assert "media_content" in result.additional_kwargs
            mock_helper.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_multimodal_model_uses_react_loop(self):
        """A model with TEXT_TO_TEXT + TEXT_TO_IMAGE goes through normal Re-Act loop."""
        prov = ProviderConfig(
            name="mm", type="openai", api_url="http://x", api_key="sk-test",
        )
        multi_model = ModelConfig(name="multi", provider="mm", name_in_provider="mm-2.7")
        multi_model.capabilities = {ModelCapability.TEXT_TO_TEXT, ModelCapability.TEXT_TO_IMAGE, ModelCapability.TOOL_CALLING}
        multi_model._provider_config = prov

        executor = AgentCore(llm_models={"multi": multi_model}, mcp_servers={})

        # Mock the engine so we can verify it goes through infer(), not generate_image()
        mock_engine = Mock()
        mock_response = AIMessage(content="Here is your image")
        mock_engine.infer = AsyncMock(return_value=mock_response)
        mock_engine.parse_tool_calls.return_value = None
        mock_engine.system_message.return_value = "system"
        mock_engine.supports_capability.return_value = False
        executor.llm_engines["multi"] = mock_engine

        with patch.object(executor, "_prepare_messages_and_tools", new_callable=AsyncMock) as mock_prep:
            mock_prep.return_value = (mock_engine, [], [])
            result = await executor.process_user_input("describe a sunset", model="multi")

            # Should have gone through infer, not generate_image
            mock_engine.infer.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_image_generation.py::TestCapabilityRouting -v`
Expected: FAIL — no routing logic yet

- [ ] **Step 3: Add capability routing to process_user_input()**

In `src/cliver/llm/llm.py`, at the top of `process_user_input()`, BEFORE the `_prepare_messages_and_tools` call, add:

```python
# Route generation-only models directly (skip Re-Act loop)
_model_config = self._get_llm_model(model)
if _model_config:
    _caps = _model_config.get_capabilities()
    if ModelCapability.TEXT_TO_IMAGE in _caps and ModelCapability.TEXT_TO_TEXT not in _caps:
        return await self.generate_image(user_input, model, **(options or {}))
```

Add the same routing at the top of `stream_user_input()` — but since `stream_user_input` is an async generator, yield the result as a chunk:

```python
_model_config = self._get_llm_model(model)
if _model_config:
    _caps = _model_config.get_capabilities()
    if ModelCapability.TEXT_TO_IMAGE in _caps and ModelCapability.TEXT_TO_TEXT not in _caps:
        result = await self.generate_image(user_input, model, **(options or {}))
        yield AIMessageChunk(content=result.content, additional_kwargs=result.additional_kwargs)
        return
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_image_generation.py -v`
Expected: All PASS

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/cliver/llm/llm.py tests/test_image_generation.py
git commit -m "feat: capability routing — generation-only models skip Re-Act loop"
```

---

### Task 5: ImageGenerate Builtin Tool

**Files:**
- Create: `src/cliver/tools/image_generate.py`
- Modify: `src/cliver/tool_registry.py`
- Create: `tests/test_image_generate_tool.py`

- [ ] **Step 1: Write failing test for the tool**

```python
# tests/test_image_generate_tool.py
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage

from cliver.media import MediaContent, MediaType
from cliver.tools.image_generate import ImageGenerateTool


class TestImageGenerateTool:
    def test_tool_metadata(self):
        tool = ImageGenerateTool()
        assert tool.name == "ImageGenerate"
        assert "image" in tool.description.lower()

    def test_run_no_executor(self):
        tool = ImageGenerateTool()
        with patch("cliver.tools.image_generate.get_task_executor", return_value=None):
            result = tool._run(prompt="a cat")
            assert "error" in result.lower()

    def test_run_success(self):
        tool = ImageGenerateTool()
        mock_executor = Mock()
        mock_media = [MediaContent(type=MediaType.IMAGE, data="https://img.png", mime_type="image/png", source="test")]
        mock_result = AIMessage(
            content="Generated 1 image(s):\nhttps://img.png",
            additional_kwargs={"media_content": mock_media},
        )
        mock_executor.generate_image = AsyncMock(return_value=mock_result)

        with patch("cliver.tools.image_generate.get_task_executor", return_value=mock_executor):
            result = tool._run(prompt="a cat")
            assert "https://img.png" in result
            mock_executor.generate_image.assert_called_once_with("a cat", None)

    def test_run_with_model(self):
        tool = ImageGenerateTool()
        mock_executor = Mock()
        mock_result = AIMessage(content="Generated 1 image(s):\nhttps://img.png")
        mock_executor.generate_image = AsyncMock(return_value=mock_result)

        with patch("cliver.tools.image_generate.get_task_executor", return_value=mock_executor):
            result = tool._run(prompt="a dog", model="minimax-image")
            mock_executor.generate_image.assert_called_once_with("a dog", "minimax-image")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_image_generate_tool.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Create the tool**

```python
# src/cliver/tools/image_generate.py
"""Built-in tool for generating images from text descriptions.

Uses whatever configured provider has image_url set, or a model
with TEXT_TO_IMAGE capability. Provider-specific API differences
are handled by generation helpers in llm/media_generation/.
"""

import asyncio
import logging
from typing import Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from cliver.agent_profile import get_task_executor

logger = logging.getLogger(__name__)


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
            result = asyncio.run(executor.generate_image(prompt, model))
            return result.content
        except Exception as e:
            logger.warning("Image generation failed: %s", e)
            return f"Error generating image: {e}"


image_generate = ImageGenerateTool()
```

- [ ] **Step 4: Add to core toolset**

In `src/cliver/tool_registry.py`, add `"ImageGenerate"` to the `"core"` set:

```python
"core": {
    "Read",
    "Write",
    "LS",
    "Grep",
    "Bash",
    "Exec",
    "Transcribe",
    "ImageGenerate",
},
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_image_generate_tool.py -v`
Expected: All PASS

- [ ] **Step 6: Run full test suite + lint**

Run: `uv run ruff check && uv run pytest tests/ -x -q`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/cliver/tools/image_generate.py src/cliver/tool_registry.py tests/test_image_generate_tool.py
git commit -m "feat: ImageGenerate builtin tool for LLM-initiated image generation"
```

---

### Task 6: Add MiniMax Image Model Pattern to Capabilities

**Files:**
- Modify: `src/cliver/model_capabilities.py`

- [ ] **Step 1: Add image-01 model pattern**

In `src/cliver/model_capabilities.py`, in the `_MODEL_CAPABILITIES` dict, add after the `"minimax*"` entry:

```python
"image-01*": {
    ModelCapability.TEXT_TO_IMAGE,
},
```

This means if a user adds a model with `name_in_provider: image-01`, the capability is auto-detected without needing explicit `capabilities: [text_to_image]` in config.

- [ ] **Step 2: Run lint + tests**

Run: `uv run ruff check && uv run pytest tests/ -x -q`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add src/cliver/model_capabilities.py
git commit -m "feat: auto-detect TEXT_TO_IMAGE capability for image-01 models"
```

---

### Task 7: Set image_url on User's Providers

**Files:**
- None (CLI commands only)

- [ ] **Step 1: Set image_url on both minimax providers**

```bash
uv run cliver provider set -n minimax --image-url "https://api.minimaxi.com/v1/image_generation"
uv run cliver provider set -n minimax-anthropic --image-url "https://api.minimaxi.com/v1/image_generation"
```

- [ ] **Step 2: Verify**

```bash
uv run cliver provider list
```

Expected: Both providers show the image URL.

- [ ] **Step 3: Smoke test — tool-based (via chat model)**

```bash
uv run cliver chat --save-media "please generate an image of a sunset over mountains"
```

The LLM should call the `ImageGenerate` tool, which calls MiniMax's image API. The image URL should be displayed and optionally saved.

- [ ] **Step 4: Optional — add dedicated image model and test direct routing**

```bash
uv run cliver model add -n minimax-image -p minimax -N image-01 -c text_to_image
uv run cliver chat -m minimax-image --save-media "a cat sitting on a windowsill"
```

This should route directly to the image API without going through the Re-Act loop.

---

### Task 8: Final Validation

**Files:**
- Various (fix any issues found)

- [ ] **Step 1: Full test suite + lint + format**

```bash
uv run ruff check
uv run ruff format --check
uv run pytest tests/ -x -q
```

Expected: All PASS

- [ ] **Step 2: Verify httpx dependency**

Check that `httpx` is in the project dependencies. If not:

```bash
uv add httpx
```

- [ ] **Step 3: Commit any fixes**

```bash
git add -A
git commit -m "chore: final validation for multimodal generation"
```

---

**Self-review results:**

1. **Spec coverage:** All 10 spec sections mapped to tasks. Section 2 (ProviderConfig) → Task 1. Section 3 (Helpers) → Task 2. Section 4 (AgentCore) → Tasks 3+4. Section 5 (Tool) → Task 5. Section 6 (Dedicated model) → Task 6+7. Section 7 (Response pipeline) → covered by Task 3's AIMessage wrapping. Section 8 (File structure) → all files accounted for. Section 9 (Out of scope) → correctly excluded.

2. **Placeholder scan:** No TBDs, TODOs, or vague instructions found.

3. **Type consistency:** `ImageGenerationHelper.generate()` signature matches across base, minimax, openai_compat. `AgentCore.generate_image()` returns `BaseMessage` consistently. `MediaContent` used throughout. `get_image_helper()` name consistent across registry and llm.py import.
