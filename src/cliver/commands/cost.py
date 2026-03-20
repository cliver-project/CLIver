"""
Token usage cost command — shows token consumption by model and agent.
"""

from datetime import datetime, timezone
from typing import Optional

import click

from cliver.cli import Cliver, pass_cliver
from cliver.token_tracker import TokenUsage, format_tokens


@click.group(name="cost", help="View token usage statistics", invoke_without_command=True)
@pass_cliver
@click.pass_context
def cost(ctx, cliver: Cliver):
    """Show token usage for the current session."""
    if ctx.invoked_subcommand is None:
        _show_session_summary(cliver)


@cost.command(name="session", help="Show token usage for the current session")
@pass_cliver
def cost_session(cliver: Cliver):
    """Show in-memory session totals by model."""
    _show_session_summary(cliver)


@cost.command(name="total", help="Show total token usage from audit logs")
@click.option("--model", "-m", default=None, help="Filter by model name")
@click.option("--agent", "-a", default=None, help="Filter by agent name")
@click.option("--from", "date_from", default=None, help="Start date (YYYY-MM-DD)")
@click.option("--to", "date_to", default=None, help="End date (YYYY-MM-DD)")
@pass_cliver
def cost_total(
    cliver: Cliver,
    model: Optional[str],
    agent: Optional[str],
    date_from: Optional[str],
    date_to: Optional[str],
):
    """Show aggregated token usage from audit logs with optional filters."""
    tracker = getattr(cliver, "token_tracker", None)
    if not tracker:
        cliver.output("Token tracking is not available.")
        return

    # Parse date filters
    start = _parse_date(cliver, date_from) if date_from else None
    end = _parse_date(cliver, date_to) if date_to else None

    results = tracker.query(model=model, agent=agent, start=start, end=end)

    if not results:
        cliver.output("No token usage data found for the given filters.")
        return

    # Build header
    header_parts = ["Token usage"]
    if model:
        header_parts.append(f"model={model}")
    if agent:
        header_parts.append(f"agent={agent}")
    if date_from or date_to:
        date_range = f"{date_from or '...'} to {date_to or '...'}"
        header_parts.append(f"({date_range})")
    cliver.output(" ".join(header_parts) + ":")

    # Display results
    if model:
        # Show per-agent breakdown for a specific model
        _show_per_agent(cliver, results, model)
    else:
        # Show per-model summary
        _show_per_model(cliver, results)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _show_session_summary(cliver: Cliver) -> None:
    """Show in-memory session totals."""
    tracker = getattr(cliver, "token_tracker", None)
    if not tracker:
        cliver.output("Token tracking is not available.")
        return

    summary = tracker.get_session_summary()
    if not summary:
        cliver.output("No token usage in this session.")
        return

    cliver.output("Token usage this session:")
    grand_total = TokenUsage()
    for model_name, usage in sorted(summary.items()):
        _print_usage_line(cliver, model_name, usage)
        grand_total += usage

    if len(summary) > 1:
        cliver.output("  " + "─" * 50)
        _print_usage_line(cliver, "Total", grand_total)


def _show_per_model(cliver: Cliver, results: dict) -> None:
    """Show summary by model (aggregating all agents)."""
    grand_total = TokenUsage()
    for model_name, agents in sorted(results.items()):
        model_total = TokenUsage()
        for usage in agents.values():
            model_total += usage
        _print_usage_line(cliver, model_name, model_total)
        grand_total += model_total

    if len(results) > 1:
        cliver.output("  " + "─" * 50)
        _print_usage_line(cliver, "Total", grand_total)


def _show_per_agent(cliver: Cliver, results: dict, model: str) -> None:
    """Show per-agent breakdown for a specific model."""
    if model not in results:
        cliver.output(f"  No data for model '{model}'.")
        return

    agents = results[model]
    model_total = TokenUsage()
    for agent_name, usage in sorted(agents.items()):
        _print_usage_line(cliver, f"Agent {agent_name}", usage)
        model_total += usage

    if len(agents) > 1:
        cliver.output("  " + "─" * 50)
        _print_usage_line(cliver, "Total", model_total)


def _print_usage_line(cliver: Cliver, label: str, usage: TokenUsage) -> None:
    """Print a formatted usage line."""
    cliver.output(
        f"  {label:20s}  in: {format_tokens(usage.input_tokens):>7s}  "
        f"out: {format_tokens(usage.output_tokens):>7s}  "
        f"total: {format_tokens(usage.total_tokens):>7s}"
    )


def _parse_date(cliver: Cliver, date_str: str) -> Optional[datetime]:
    """Parse a YYYY-MM-DD date string to datetime."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        cliver.output(f"Invalid date format: '{date_str}'. Expected YYYY-MM-DD.")
        return None
