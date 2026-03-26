"""
Skill activation command.

/skill <name> or /skill:<name> activates a skill and injects its content into the chat.
"""

import click
from langchain_core.messages import AIMessage, HumanMessage

from cliver.cli import Cliver, pass_cliver
from cliver.skill_manager import SkillManager


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

    user_message = " ".join(message) if message else ""
    if not user_message:
        user_message = (
            f"I want to use the '{name}' skill. Please explain what this skill does and ask me what I'd like to do."
        )

    from cliver.cli_llm_call import LLMCallOptions, llm_call
    from cliver.commands.chat import _compress_if_needed

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
