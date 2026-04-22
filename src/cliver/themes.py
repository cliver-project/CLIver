"""
Theme system for CLIver TUI.

Provides built-in themes (dark, light, dracula) and supports custom
color overrides via config.yaml. All UI color references read from
the active theme instance.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class Theme:
    """Color definitions for the CLIver TUI."""

    name: str = "dark"

    # LLM response text (ANSI RGB)
    response_r: int = 220
    response_g: int = 220
    response_b: int = 230

    # User input echo in conversation output
    user_input_fg: str = "bold white"
    user_input_bg: str = "#2d2d44"

    # Input box (prompt_toolkit styles)
    input_area_bg: str = "#1a1a2e"
    input_border: str = "#555555"
    prompt_style: str = "ansigreen bold"
    permission_prompt_style: str = "ansiyellow bold"

    # Toolbar
    toolbar_border: str = "#555555"
    toolbar_cwd: str = "#aaaaaa"
    toolbar_agent: str = "#cc88cc bold"
    toolbar_mode: str = "#88aa88"
    toolbar_rss: str = "#cccc88 italic"
    toolbar_model: str = "#88ccff bold"

    # Separator line
    separator: str = "dim"

    # Tool progress
    tool_name: str = "cyan"
    tool_desc: str = "dim"
    tool_result: str = "dim"
    tool_error: str = "red"
    tool_duration: str = "dim"

    @property
    def response_ansi_start(self) -> str:
        return f"\033[38;2;{self.response_r};{self.response_g};{self.response_b}m"

    @property
    def response_ansi_reset(self) -> str:
        return "\033[0m"

    def prompt_toolkit_styles(self) -> Dict[str, str]:
        """Build prompt_toolkit Style dict from theme colors."""
        return {
            "input-area": f"bg:{self.input_area_bg}",
            "input-border": f"{self.input_border} bg:{self.input_area_bg}",
            "prompt": self.prompt_style,
            "toolbar-border": self.toolbar_border,
            "toolbar-cwd": self.toolbar_cwd,
            "toolbar-agent": self.toolbar_agent,
            "toolbar-mode": self.toolbar_mode,
            "toolbar-rss": self.toolbar_rss,
            "toolbar-model": self.toolbar_model,
            "permission-prompt": self.permission_prompt_style,
        }

    def user_input_markup(self, text: str, width: int) -> str:
        """Format user input echo with background for conversation output."""
        prefix = " ❯ "
        line = f"{prefix}{text}"
        padded = line.ljust(width)
        fg, bg = self.user_input_fg, self.user_input_bg
        return f"\n[{fg} on {bg}]{padded}[/{fg} on {bg}]"


# ---------------------------------------------------------------------------
# Built-in themes
# ---------------------------------------------------------------------------

DARK_THEME = Theme(name="dark")

LIGHT_THEME = Theme(
    name="light",
    response_r=55,
    response_g=65,
    response_b=80,
    user_input_fg="#073642 bold",
    user_input_bg="#eee8d5",
    input_area_bg="#fdf6e3",
    input_border="#93a1a1",
    prompt_style="#859900 bold",
    permission_prompt_style="#cb4b16 bold",
    toolbar_border="#93a1a1",
    toolbar_cwd="#586e75",
    toolbar_agent="#6c71c4 bold",
    toolbar_mode="#2aa198",
    toolbar_rss="#b58900 italic",
    toolbar_model="#268bd2 bold",
    separator="#93a1a1",
    tool_name="#268bd2",
    tool_desc="#657b83",
    tool_result="#657b83",
    tool_error="#dc322f",
    tool_duration="#93a1a1",
)

DRACULA_THEME = Theme(
    name="dracula",
    response_r=248,
    response_g=248,
    response_b=242,
    user_input_fg="bold white",
    user_input_bg="#44475a",
    input_area_bg="#282a36",
    input_border="#6272a4",
    prompt_style="#50fa7b bold",
    permission_prompt_style="#ffb86c bold",
    toolbar_border="#6272a4",
    toolbar_cwd="#f8f8f2",
    toolbar_agent="#ff79c6 bold",
    toolbar_mode="#50fa7b",
    toolbar_rss="#f1fa8c italic",
    toolbar_model="#8be9fd bold",
    separator="dim",
    tool_name="#8be9fd",
    tool_desc="#6272a4",
    tool_result="#6272a4",
    tool_error="#ff5555",
    tool_duration="#6272a4",
)

_BUILTIN_THEMES: Dict[str, Theme] = {
    "dark": DARK_THEME,
    "light": LIGHT_THEME,
    "dracula": DRACULA_THEME,
}

# Active theme — set at startup, read by all UI components
_active_theme: Theme = DARK_THEME


def get_theme() -> Theme:
    """Get the active theme."""
    return _active_theme


def set_theme(theme: Theme) -> None:
    """Set the active theme."""
    global _active_theme
    _active_theme = theme


def load_theme(name: Optional[str] = None, overrides: Optional[dict] = None) -> Theme:
    """Load a theme by name with optional color overrides.

    Args:
        name: Built-in theme name (dark, light, dracula). Defaults to dark.
        overrides: Dict of field_name → value to override specific colors.

    Returns:
        A Theme instance.
    """
    base = _BUILTIN_THEMES.get(name or "dark", DARK_THEME)

    if not overrides:
        return base

    # Create a copy with overrides applied
    fields = {f.name: getattr(base, f.name) for f in base.__dataclass_fields__.values()}
    fields.update(overrides)
    return Theme(**fields)


def list_themes() -> list[str]:
    """List available built-in theme names."""
    return list(_BUILTIN_THEMES.keys())
