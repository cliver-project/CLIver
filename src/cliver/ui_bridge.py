"""UIBridge -- unified protocol for all human interaction."""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable


@dataclass
class FieldSpec:
    key: str
    label: str
    help: str = ""
    required: bool = False
    choices: list[str] | None = None
    default: str | None = None
    secret: bool = False


PERMISSION_CHOICES = {
    "y": "allow",
    "yes": "allow",
    "n": "deny",
    "no": "deny",
    "a": "allow_always",
    "always": "allow_always",
    "d": "deny_always",
    "deny": "deny_always",
}


@runtime_checkable
class UIBridge(Protocol):
    def output(self, text: str, style: str = "") -> None: ...
    def ask_permission(self, tool_name: str, args: dict, meta: dict) -> str: ...
    def ask_input(self, prompt: str, choices: list[str] | None = None) -> str: ...
    def ask_fields(self, fields: list[FieldSpec]) -> dict | None: ...
    def show_tool_event(self, event) -> None: ...


def _mask_secret(value: str) -> str:
    if len(value) <= 8:
        return "****"
    return f"{value[:4]}****{value[-4:]}"


def _ask_fields_impl(bridge: UIBridge, fields: list[FieldSpec]) -> dict | None:
    result = {}
    for i, field in enumerate(fields, 1):
        label_text = f"\n  {i}. {field.label}"
        if field.help:
            label_text += f"\n     {field.help}"
        bridge.output(label_text)

        if field.default:
            if field.secret:
                display = _mask_secret(field.default)
            else:
                display = field.default
            prompt = f"  [{display}] > "
        else:
            prompt = "  > "

        value = bridge.ask_input(prompt, choices=field.choices)

        if not value and field.default:
            value = field.default
        if not value and field.required:
            return None
        if value:
            result[field.key] = value
    return result


class CLIBridge:
    def output(self, text: str, style: str = "") -> None:
        print(text)

    def ask_permission(self, tool_name: str, args: dict, meta: dict) -> str:
        args_summary = ", ".join(f"{k}={v}" for k, v in list(args.items())[:3])
        print(f"  Tool: {tool_name}({args_summary})", file=sys.stderr)
        try:
            response = input("  Allow? [y/n/a/d] > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "deny"
        return PERMISSION_CHOICES.get(response, "deny")

    def ask_input(self, prompt: str, choices: list[str] | None = None) -> str:
        while True:
            try:
                response = input(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                return ""
            if choices is None:
                return response
            if response.lower() in [c.lower() for c in choices]:
                return response
            valid = ", ".join(choices)
            print(f"  Invalid choice. Use: {valid}")

    def ask_fields(self, fields: list[FieldSpec]) -> dict | None:
        return _ask_fields_impl(self, fields)

    def show_tool_event(self, event) -> None:
        pass


class TUIBridge:
    """UIBridge for interactive TUI mode.

    Methods are called from the worker thread and block via threading.Event
    until the main thread (on_accept) provides input via receive_input()
    or try_receive().
    """

    def __init__(self):
        self._pending: Optional[threading.Event] = None
        self._response: Optional[str] = None
        self._valid_choices: Optional[list[str]] = None

    def output(self, text: str, style: str = "") -> None:
        print(text)

    def ask_permission(self, tool_name: str, args: dict, meta: dict) -> str:
        args_summary = ", ".join(f"{k}={v}" for k, v in list(args.items())[:3])
        self.output(f"  Tool: {tool_name}({args_summary})")
        self._valid_choices = list(PERMISSION_CHOICES.keys())
        event = threading.Event()
        self._response = None
        self._pending = event
        event.wait()
        self._pending = None
        raw = self._response or ""
        self._valid_choices = None
        return PERMISSION_CHOICES.get(raw.lower(), "deny")

    def ask_input(self, prompt: str, choices: list[str] | None = None) -> str:
        self.output(prompt)
        self._valid_choices = [c.lower() for c in choices] if choices else None
        event = threading.Event()
        self._response = None
        self._pending = event
        event.wait()
        self._pending = None
        result = self._response or ""
        self._valid_choices = None
        return result

    def ask_fields(self, fields: list[FieldSpec]) -> dict | None:
        return _ask_fields_impl(self, fields)

    def show_tool_event(self, event) -> None:
        pass

    # ── Called from main thread (on_accept) ─────────

    def receive_input(self, text: str) -> None:
        """Unconditionally accept input and signal the waiting worker."""
        self._response = text
        if self._pending:
            self._pending.set()

    def try_receive(self, text: str) -> bool:
        """Try to accept input. Validates against choices if set.
        Returns True if accepted, False if rejected (invalid choice).
        """
        if self._valid_choices:
            if text.lower() not in self._valid_choices:
                return False
        self._response = text
        if self._pending:
            self._pending.set()
        return True

    def cancel_pending(self) -> None:
        """Cancel any pending input prompt (Ctrl+C handler)."""
        self._response = ""
        if self._pending:
            self._pending.set()
