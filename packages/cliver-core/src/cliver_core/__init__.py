"""cliver-core — the embeddable AgentCore library."""

from cliver_core import presets  # noqa: I001 — submodule, imported as a namespace
from cliver_core.agent import AgentCore
from cliver_core.bindings import (
    CapabilityNotSupported,
    ModelBinding,
    ModelBindings,
)
from cliver_core.capabilities import (
    embed,
    generate_image,
    generate_video,
    rerank,
    text_to_speech,
    transcribe_audio,
)
from cliver_core.events import EventHandler
from cliver_core.media import MediaContent
from cliver_core.media_jobs import MediaJob
from cliver_core.messages import (
    CLIverMessage,
    CLIverMessageChunk,
    ToolCall,
)
from cliver_core.skill import Skill, SkillError, SkillManager, SkillNotFoundError
from cliver_core.tool import CLIverTool, ToolRegistry, tool

__all__ = [
    "AgentCore",
    "CapabilityNotSupported",
    "CLIverMessage",
    "CLIverMessageChunk",
    "MediaJob",
    "ModelBinding",
    "ModelBindings",
    "ToolCall",
    "CLIverTool",
    "ToolRegistry",
    "tool",
    "EventHandler",
    "MediaContent",
    # Agent Skills (SKILL.md) loading and run_skill support
    "Skill",
    "SkillError",
    "SkillManager",
    "SkillNotFoundError",
    # Ready-made bindings for well-known providers
    "presets",
    # Standalone capability calls (also importable from cliver_core.capabilities)
    "embed",
    "generate_image",
    "generate_video",
    "rerank",
    "text_to_speech",
    "transcribe_audio",
]
