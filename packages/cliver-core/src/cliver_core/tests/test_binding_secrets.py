"""Binding diagnostics must not echo the credential.

``ModelBinding`` is a dataclass, so its default ``repr`` is what ends up in
log lines, debuggers, and exception reports.  The API key is a secret: it is
excluded from ``repr`` (both for a binding and for the whole
``ModelBindings`` container that echoes it).
"""

from __future__ import annotations

from cliver_core.bindings import ModelBinding, ModelBindings

_SECRET = "sk-live-0123456789"


def test_binding_repr_does_not_show_the_api_key():
    binding = ModelBinding.llm(
        model="m",
        api_key=_SECRET,
        base_url="https://example.test/v1",
    )

    assert _SECRET not in repr(binding)


def test_model_bindings_repr_does_not_show_the_api_key():
    bindings = ModelBindings(llm=ModelBinding.llm(model="m", api_key=_SECRET, base_url="https://example.test/v1"))

    assert _SECRET not in repr(bindings)
