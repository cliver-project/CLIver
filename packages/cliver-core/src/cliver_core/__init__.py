"""cliver-core — the embeddable AgentCore library."""

from cliver_core import presets  # noqa: I001 — submodule, imported as a namespace
from cliver_core.agent import AgentCore, CandidateSelector, ToolSelector
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
from cliver_core.errors import ProviderError
from cliver_core.events import EventHandler
from cliver_core.media import MediaContent
from cliver_core.media_jobs import MediaJob
from cliver_core.messages import (
    CLIverMessage,
    CLIverMessageChunk,
    ToolCall,
    UsageInfo,
)
from cliver_core.pipeline import (
    FailurePolicy,
    InputBuilder,
    PipelineContext,
    PipelineResult,
    PipelineStep,
    SkillPipeline,
    StepOutput,
    StepResult,
    StepRunResult,
    StepRunner,
)
from cliver_core.policy import (
    SAFE_ENV_VARS,
    Decision,
    DecisionAction,
    EnvSpec,
    ExecutionPolicy,
    PermissionCallback,
    ToolCallContext,
    ToolCategory,
)
from cliver_core.skill import Skill, SkillError, SkillManager, SkillNotFoundError
from cliver_core.skill_tools import SkillSearch, SkillSelector
from cliver_core.tool import CLIverTool, ToolRegistry, tool

__all__ = [
    "AgentCore",
    "CapabilityNotSupported",
    "CLIverMessage",
    "CLIverMessageChunk",
    "MediaJob",
    "ModelBinding",
    "ModelBindings",
    "ProviderError",
    "ToolCall",
    "UsageInfo",
    "CLIverTool",
    "CandidateSelector",
    "ToolRegistry",
    "ToolSelector",
    "tool",
    "EventHandler",
    "MediaContent",
    # Agent Skills (SKILL.md) loading, run_skill, and model-driven dispatch
    "Skill",
    "SkillError",
    "SkillManager",
    "SkillNotFoundError",
    "SkillSearch",
    "SkillSelector",
    # Deterministic skill pipelines (sub-agent style)
    "FailurePolicy",
    "InputBuilder",
    "PipelineContext",
    "PipelineResult",
    "PipelineStep",
    "SkillPipeline",
    "StepOutput",
    "StepResult",
    "StepRunResult",
    "StepRunner",
    # Tool permission + execution policy
    "Decision",
    "DecisionAction",
    "EnvSpec",
    "ExecutionPolicy",
    "PermissionCallback",
    "SAFE_ENV_VARS",
    "ToolCallContext",
    "ToolCategory",
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
