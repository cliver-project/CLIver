"""
Autonomous Skill Learning — post-task review for skill auto-creation.

After a complex task (many tool calls), spawns a background review that
evaluates whether the approach was non-trivial and should be saved as a
reusable SKILL.md file.

Inspired by Hermes's nudge-based skill creation, but simpler:
- Counter tracks tool calls during the Re-Act loop
- After the task completes, if counter >= threshold, trigger review
- Review runs as a separate LLM call (cheap model, low iterations)
- If the review decides to create a skill, it writes SKILL.md to user's skills dir
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Minimum tool calls before considering skill review
DEFAULT_SKILL_NUDGE_THRESHOLD = 10

_SKILL_REVIEW_PROMPT = """You just completed a task that involved {tool_call_count} tool calls.

Review the conversation and determine if a reusable skill should be created.

**Create a skill when:**
- A non-trivial approach was used (not just a simple lookup)
- Errors were overcome through trial-and-error
- A multi-step workflow was discovered
- The approach could benefit future similar tasks

**Do NOT create a skill when:**
- The task was simple (basic file read, simple question)
- The approach is obvious and doesn't need documentation
- A similar skill already exists

If you decide to create a skill:
1. Use the `Write` tool to create a SKILL.md file at: {skills_dir}/{skill_name}/SKILL.md
2. The SKILL.md must have YAML frontmatter with: name, description, keywords
3. The body should document the approach, steps, and key decisions
4. The skill name should be short, descriptive, lowercase with hyphens

If you decide NOT to create a skill, respond with: "No skill needed for this task."

Here is a summary of what was accomplished:
{task_summary}
"""


async def maybe_review_for_skill(
    task_executor,
    tool_call_count: int,
    task_summary: str,
    threshold: int = DEFAULT_SKILL_NUDGE_THRESHOLD,
    skills_dir: Optional[Path] = None,
) -> Optional[str]:
    """Evaluate whether the completed task should generate a skill.

    Args:
        task_executor: The AgentCore instance (for making the review LLM call)
        tool_call_count: Number of tool calls made during the task
        task_summary: Brief summary of what was accomplished
        threshold: Minimum tool calls to trigger review
        skills_dir: Directory to save skills to

    Returns:
        The skill name if created, None otherwise
    """
    if tool_call_count < threshold:
        logger.debug(
            "Skipping skill review: %d tool calls < threshold %d",
            tool_call_count,
            threshold,
        )
        return None

    if not skills_dir:
        from cliver.util import get_config_dir

        skills_dir = get_config_dir() / "skills"

    skills_dir.mkdir(parents=True, exist_ok=True)

    # Generate a suggested skill name from the summary
    skill_name = _suggest_skill_name(task_summary)

    prompt = _SKILL_REVIEW_PROMPT.format(
        tool_call_count=tool_call_count,
        skills_dir=str(skills_dir),
        skill_name=skill_name,
        task_summary=task_summary[:2000],  # truncate to avoid huge prompts
    )

    logger.info(
        "Triggering skill review: %d tool calls, summary: %.100s",
        tool_call_count,
        task_summary,
    )

    try:
        response = await task_executor.process_user_input(
            user_input=prompt,
            max_iterations=8,  # Low cap — just needs to decide + write file
        )

        result = str(response.content) if response and response.content else ""

        if "no skill needed" in result.lower():
            logger.info("Skill review: no skill created")
            return None

        # Check if a skill was actually written
        if skills_dir.is_dir():
            for child in skills_dir.iterdir():
                if child.is_dir() and (child / "SKILL.md").exists():
                    if child.name == skill_name:
                        logger.info("Skill review: created skill '%s'", skill_name)
                        return skill_name

        logger.info("Skill review completed (skill may have been created with a different name)")
        return skill_name if "write_file" in result.lower() else None

    except Exception as e:
        logger.warning("Skill review failed: %s", e)
        return None


def _suggest_skill_name(summary: str) -> str:
    """Suggest a short skill name from a task summary."""
    # Take first few meaningful words, lowercase, hyphenate
    words = summary.lower().split()
    # Filter out common words
    stop = {"the", "a", "an", "to", "for", "and", "or", "in", "on", "is", "was", "how", "what"}
    meaningful = [w for w in words if w not in stop and w.isalnum()]
    name = "-".join(meaningful[:4]) if meaningful else "auto-skill"
    return name[:40]  # cap length
