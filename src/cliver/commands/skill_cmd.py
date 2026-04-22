"""
Skill activation command.

/skill <name> or /skill:<name> activates a skill and injects its content into the chat.
"""

import asyncio

import click
from langchain_core.messages import AIMessage, HumanMessage

from cliver.cli import Cliver, pass_cliver
from cliver.skill_manager import SkillManager


def _compress_if_needed(cliver, task_executor, model_config, model_name, new_input):
    """Check and compress conversation history if it exceeds context window budget."""
    from cliver.conversation_compressor import ConversationCompressor, estimate_tokens, get_context_window

    context_window = get_context_window(model_config)
    compressor = ConversationCompressor(context_window)

    if not compressor.needs_compression([], cliver.conversation_messages, new_input):
        return

    before_tokens = estimate_tokens(cliver.conversation_messages)
    llm_engine = task_executor.get_llm_engine(model_name)

    try:
        compressed = asyncio.get_event_loop().run_until_complete(
            compressor.compress(cliver.conversation_messages, llm_engine)
        )
    except RuntimeError:
        # No running event loop -- create one
        compressed = asyncio.run(compressor.compress(cliver.conversation_messages, llm_engine))

    cliver.conversation_messages = compressed
    after_tokens = estimate_tokens(cliver.conversation_messages)
    cliver.output(f"\\[Compressed conversation: ~{before_tokens} → ~{after_tokens} tokens]")


# ---------------------------------------------------------------------------
# Business logic function
# ---------------------------------------------------------------------------


def _activate_skill(cliver: Cliver, name: str, message: str = ""):
    """Activate a skill and inject its content into the conversation."""
    manager = SkillManager()
    skill_obj = manager.get_skill(name)
    if not skill_obj:
        cliver.output(f"[red]Skill '{name}' not found.[/red]")
        names = manager.get_skill_names()
        if names:
            cliver.output(f"Available skills: {', '.join(names)}")
        return

    cliver.output(f"Activating skill '[green]{name}[/green]' ...")

    # Build skill system message — injected only on this call.
    # The conversation history carries the context for follow-up messages.
    skill_content = manager.activate_skill(name)
    skill_system_msg = f"The user has activated the '{name}' skill. Follow these skill instructions:\n\n{skill_content}"

    def skill_appender():
        return skill_system_msg

    user_message = message if message else ""
    if not user_message:
        user_message = (
            f"I want to use the '{name}' skill. Please explain what this skill does and ask me what I'd like to do."
        )

    from cliver.cli_llm_call import LLMCallOptions, llm_call

    session_options = cliver.session_options or {}
    use_model = session_options.get("model", None)
    use_stream = session_options.get("stream", True)
    task_executor = cliver.task_executor

    # Record user turn
    cliver.record_turn("user", user_message)

    # Compress if needed
    model_config = task_executor._get_llm_model(use_model)
    if model_config and cliver.conversation_messages:
        _compress_if_needed(cliver, task_executor, model_config, use_model, user_message)

    conv_history = list(cliver.conversation_messages) if cliver.conversation_messages else None
    cliver.conversation_messages.append(HumanMessage(content=user_message))

    def on_response(text: str):
        cliver.record_turn("assistant", text)
        cliver.conversation_messages.append(AIMessage(content=text))

    llm_call(
        cliver,
        LLMCallOptions(
            user_input=user_message,
            model=use_model,
            stream=use_stream,
            system_message_appender=skill_appender,
            conversation_history=conv_history,
            on_response=on_response,
        ),
    )


# ---------------------------------------------------------------------------
# Dispatch function
# ---------------------------------------------------------------------------


def dispatch(cliver: Cliver, args: str):
    """Activate a skill by name."""
    parts = args.strip().split(None, 1) if args.strip() else []
    if not parts:
        cliver.output("[red]Missing skill name[/red]")
        cliver.output("Usage: /skill <name> [message]")
        return

    skill_name = parts[0]
    message = parts[1] if len(parts) > 1 else ""

    if skill_name in ("--help", "help"):
        cliver.output("Usage: /skill <name> [message]")
        cliver.output("  Activate a skill and optionally provide an initial message")
        cliver.output("  Example: /skill brainstorm design a new feature")
        return

    _activate_skill(cliver, skill_name, message)


# ---------------------------------------------------------------------------
# Click command (thin wrapper)
# ---------------------------------------------------------------------------


@click.command(name="skill", help="Activate a skill by name")
@click.argument("name", type=str)
@click.argument("message", nargs=-1)
@pass_cliver
def skill_cmd(cliver: Cliver, name: str, message: tuple):
    """Activate a skill and inject its content into the conversation.

    /skill <name> [message...] looks up the skill, then routes its instructions
    through chat so the LLM follows the skill guidance. If a message is provided,
    it is sent to the LLM together with the skill content.
    """
    user_message = " ".join(message) if message else ""
    _activate_skill(cliver, name, user_message)
