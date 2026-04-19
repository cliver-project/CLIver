# Multimodal Generation Design Spec

**Goal:** Enable CLIver to generate images (and future audio) via provider APIs, supporting three usage paths: direct model targeting, LLM tool invocation, and multimodal chat models — all through the existing `cliver chat` command.

**Scope:** Design for image + audio generation architecture. Implement image generation with MiniMax adapter first. Audio adapters are future work following the same pattern.

---

## 1. Architecture Overview

Three paths for media generation coexist naturally:

| Path | Trigger | Capabilities | Flow |
|---|---|---|---|
| **Direct** | `cliver chat -m image-model "prompt"` | `TEXT_TO_IMAGE` only | Skip Re-Act loop → generation helper → provider's `image_url` |
| **Tool-based** | LLM calls `ImageGenerate` tool mid-conversation | Chat model + provider has `image_url` | Re-Act loop → tool → generation helper → provider's `image_url` |
| **Multimodal chat** | Chat model generates media inline in response | `TEXT_TO_TEXT` + `TEXT_TO_IMAGE` | Re-Act loop → chat API → `extract_media_from_response()` |

### Separation of concerns

- **LLM engines** (`LLMInferenceEngine` subclasses) handle chat protocol: `infer()`, `stream()`, `extract_media_from_response()`. One per chat protocol (OpenAI, Anthropic, Ollama).
- **Generation helpers** handle dedicated media APIs: `generate_image()`, `generate_audio()`. One per image API format (MiniMax, OpenAI, DashScope). Selected by `image_url`, independent of chat protocol.
- **AgentCore** orchestrates: routes by capability, manages generation helpers, exposes `generate_image()` as embeddable API.

This separation exists because a provider's chat protocol and image API are independent. MiniMax uses OpenAI-compat chat OR Anthropic-compat chat, but the same proprietary image API for both.

---

## 2. Provider Config Changes

Add optional generation endpoint URLs to `ProviderConfig`:

```yaml
providers:
  minimax:
    type: openai
    api_url: https://api.minimaxi.com/v1
    api_key: '{{ keyring("cliver", "minimax.api_key") }}'
    image_url: https://api.minimaxi.com/v1/image_generation
    # audio_url: ...  (future)
    rate_limit:
      requests: 1500
      period: "5h"

  minimax-anthropic:
    type: anthropic
    api_url: https://api.minimaxi.com/anthropic
    api_key: '{{ keyring("cliver", "minimax.api_key") }}'
    image_url: https://api.minimaxi.com/v1/image_generation   # same image API
```

Fields added to `ProviderConfig`:
- `image_url: Optional[str]` — full URL for image generation endpoint
- `audio_url: Optional[str]` — full URL for audio generation endpoint (future)

CLI:
```bash
cliver config provider set -n minimax --image-url "https://api.minimaxi.com/v1/image_generation"
```

Rate limiting applies across all endpoints of the same provider (chat + image share the same pool).

---

## 3. Generation Helpers

Small classes that handle provider-specific image API request/response formats. Located in `src/cliver/llm/media_generation/`.

### Base interface

```python
class ImageGenerationHelper(ABC):
    async def generate(self, url, api_key, prompt, model_name=None, **params) -> list[MediaContent]:
        ...
```

Parameters:
- `url` — the provider's `image_url`
- `api_key` — resolved from provider
- `prompt` — text description
- `model_name` — optional override (from dedicated model's `name_in_provider` or adapter default)
- `**params` — provider-specific options (aspect_ratio, size, n, etc.)

Returns `list[MediaContent]` with `type=IMAGE`.

### MiniMax helper

- POST to `url` with `{"model": model_name or "image-01", "prompt": ..., "response_format": "url"}`
- Parse response `{"data": {"image_urls": [...]}}`
- Supports MiniMax-specific params: `aspect_ratio`, `n`, `prompt_optimizer`

### OpenAI helper

- POST to `url` with `{"model": model_name or "dall-e-3", "prompt": ..., "response_format": "url"}`
- Parse response `{"data": [{"url": "..."}]}`
- Supports OpenAI params: `size`, `quality`, `n`

### Helper selection

Automatic, based on `image_url` pattern matching:

1. URL contains `minimaxi` or `minimax` → `MiniMaxImageHelper`
2. Default fallback → `OpenAIImageHelper` (most common standard)

A registry maps URL patterns to helpers. Extensible — adding a new provider means registering a pattern + helper class.

---

## 4. AgentCore Integration

### New method: `generate_image()`

```python
async def generate_image(self, prompt: str, model: str = None, **params) -> BaseMessage:
```

This is the embeddable API entry point. Returns an `AIMessage` with media content in `additional_kwargs`.

Resolution logic:
1. If `model` is specified → get its provider → use provider's `image_url`
2. If `model` is None → scan `llm_models` for a model with `TEXT_TO_IMAGE` capability
3. If no model found → scan providers for any with `image_url` set
4. Select helper from `image_url` pattern
5. Call helper, wrap `list[MediaContent]` into `AIMessage`
6. Respect rate limiter (same provider pool)

### Capability routing in `process_user_input()`

Add routing at the top of `process_user_input()`:

```python
model_config = self._get_llm_model(model)
if model_config:
    caps = model_config.get_capabilities()
    # Generation-only model: route directly, skip Re-Act loop
    if ModelCapability.TEXT_TO_IMAGE in caps and ModelCapability.TEXT_TO_TEXT not in caps:
        return await self.generate_image(user_input, model, **options_or_empty)
```

This only triggers for **generation-only** models. Models with both `TEXT_TO_TEXT` + `TEXT_TO_IMAGE` (multimodal chat) go through the normal Re-Act loop, and their inline-generated media is extracted via the existing `extract_media_from_response()`.

### Helper registry on AgentCore

```python
def _get_image_helper(self, image_url: str) -> ImageGenerationHelper:
```

Lazily instantiated, cached by URL pattern. Discovers the right helper class from a registry.

---

## 5. ImageGenerate Builtin Tool

New file: `src/cliver/tools/image_generate.py`

```python
class ImageGenerateInput(BaseModel):
    prompt: str = Field(description="Text description of the image to generate")
    model: Optional[str] = Field(default=None, description="Model name to use (optional)")

class ImageGenerateTool(BaseTool):
    name: str = "ImageGenerate"
    description: str = "Generate an image from a text description. ..."
```

The tool:
1. Gets `AgentCore` via `get_task_executor()` (same pattern as `TranscribeAudioTool`)
2. Calls `executor.generate_image(prompt, model)`
3. Returns image URLs/paths as text for the LLM to relay to the user

Added to the `core` toolset in `tool_registry.py` so it's always available.

The LLM's system prompt already describes available tools. When `ImageGenerate` is present AND a provider has `image_url` configured, the LLM can decide to call it during conversation.

---

## 6. Dedicated Image Model (Optional Config)

Users can optionally add a dedicated image model for direct routing:

```yaml
models:
  minimax-image:
    provider: minimax
    name_in_provider: image-01
    capabilities: [text_to_image]
```

This enables `cliver chat -m minimax-image "a sunset"` which routes directly to the image API without the Re-Act loop.

If the user does NOT add this model, image generation still works — the LLM calls the `ImageGenerate` tool during conversation, which discovers the provider's `image_url`.

---

## 7. Response Pipeline

All three paths produce the same output: an `AIMessage` with media in `additional_kwargs`:

```python
AIMessage(
    content="Generated image: a sunset over mountains",
    additional_kwargs={
        "media_content": [MediaContent(type=IMAGE, data="https://...", mime_type="image/png", source="minimax_image_generation")]
    }
)
```

The existing `MultimediaResponseHandler` pipeline handles this:
- `extract_media_from_response()` finds media in `additional_kwargs`
- `--save-media` saves to disk
- `--media-dir` controls output directory
- CLI displays media info

No changes needed to the CLI layer or media handler — they already support this format.

---

## 8. File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/cliver/config.py` | Modify | Add `image_url`, `audio_url` to `ProviderConfig`. Add `--image-url`/`--audio-url` to provider CLI. |
| `src/cliver/llm/media_generation/__init__.py` | Create | Package init, helper registry |
| `src/cliver/llm/media_generation/base.py` | Create | `ImageGenerationHelper` ABC |
| `src/cliver/llm/media_generation/minimax.py` | Create | MiniMax image API adapter |
| `src/cliver/llm/media_generation/openai_compat.py` | Create | OpenAI images API adapter |
| `src/cliver/llm/llm.py` | Modify | Add `generate_image()`, capability routing, helper registry |
| `src/cliver/tools/image_generate.py` | Create | `ImageGenerate` builtin tool |
| `src/cliver/tool_registry.py` | Modify | Add `ImageGenerate` to `core` toolset |
| `src/cliver/commands/provider.py` | Modify | Add `--image-url`/`--audio-url` options |
| `src/cliver/commands/config.py` | Modify | Show image_url/audio_url in `config show` |
| `src/cliver/model_capabilities.py` | Modify | Add MiniMax image model pattern |
| `tests/test_image_generation.py` | Create | Tests for helpers + AgentCore.generate_image() |
| `tests/test_image_generate_tool.py` | Create | Tests for the builtin tool |

---

## 9. What Is NOT In Scope

- Audio generation implementation (designed but not built — `audio_url` field added, helpers follow same pattern later)
- Anthropic `extract_media_from_response()` enhancement (needed when Anthropic-protocol models generate inline media — separate task)
- Image editing / image-to-image (MiniMax supports `subject_reference` but that's a future feature)
- Streaming image generation progress

---

## 10. Summary

The design achieves:
- **Minimal config**: set `image_url` on provider → image generation works via tool
- **Optional dedicated model**: add a `text_to_image` model → direct `cliver chat -m` routing
- **Multimodal chat models**: `TEXT_TO_TEXT` + `TEXT_TO_IMAGE` → existing Re-Act + extraction pipeline
- **Clean separation**: engines handle chat, helpers handle generation, no duplication
- **Provider-agnostic**: helpers selected by URL pattern, not chat protocol type
- **Embeddable API**: `AgentCore.generate_image()` works standalone without CLI
- **Existing CLI honored**: `--save-media`, `--media-dir` work unchanged
- **Extensible**: new providers = register URL pattern + small helper class
