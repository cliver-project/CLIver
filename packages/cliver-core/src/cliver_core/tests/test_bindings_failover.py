"""Multiple bindings per purpose — ordered fallbacks behind an unchanged API.

``ModelBindings(llm=binding)`` keeps its exact meaning.  A list mounts its
first element as the primary (``bindings.<purpose>`` stays a single
``ModelBinding``) and keeps the rest as ordered same-purpose fallbacks,
available to AgentCore through ``get_all()``.
"""

from __future__ import annotations

import pytest

from cliver_core.bindings import ModelBinding, ModelBindings
from cliver_core.provider import CLIverRequest, CLIverResponse, Provider


class _ClosableProvider(Provider):
    protocol = "mock"
    brand = "mock-closable"

    def __init__(self) -> None:
        super().__init__(api_key="", base_url="")
        self.closed = False

    async def close(self) -> None:
        self.closed = True

    async def chat(self, request: CLIverRequest) -> CLIverResponse:  # pragma: no cover
        raise NotImplementedError

    async def stream(self, request):  # pragma: no cover
        if False:
            yield None


def _llm(model: str) -> ModelBinding:
    return ModelBinding.llm(model=model)


# ── The single-binding shape is untouched ──────────────────────


def test_single_binding_mount_is_unchanged() -> None:
    primary = _llm("main")
    bindings = ModelBindings(llm=primary)

    assert bindings.llm is primary
    assert bindings.get("llm") is primary
    assert bindings.get_all("llm") == [primary]


def test_size_one_list_is_the_single_binding() -> None:
    primary = _llm("main")
    bindings = ModelBindings(llm=[primary])

    assert bindings.llm is primary
    assert bindings.get("llm") is primary
    assert bindings.get_all("llm") == [primary]


# ── Lists mount a primary + ordered fallbacks ──────────────────


def test_list_keeps_the_first_primary_and_the_rest_in_order() -> None:
    first, second, third = _llm("first"), _llm("second"), _llm("third")
    bindings = ModelBindings(llm=[first, second, third])

    assert bindings.llm is first
    assert bindings.get_all("llm") == [first, second, third]
    assert bindings.bound_purposes() == ["llm"]


def test_each_purpose_accepts_its_own_list() -> None:
    image_a = ModelBinding.image(model="a")
    image_b = ModelBinding.image(model="b")
    embed_a = ModelBinding.embed(model="e")
    bindings = ModelBindings(
        llm=[_llm("main")],
        image=[image_a, image_b],
        embed=[embed_a],
    )

    assert bindings.image is image_a
    assert bindings.get_all("image") == [image_a, image_b]
    assert bindings.get_all("embed") == [embed_a]
    assert bindings.get_all("tts") == []
    assert bindings.bound_purposes() == ["llm", "embed", "image"]


def test_audio_slot_accepts_a_tts_fallback_list() -> None:
    first, second = ModelBinding.tts(model="t1"), ModelBinding.tts(model="t2")
    bindings = ModelBindings(audio=[first, second])

    assert bindings.audio is first
    assert bindings.get_all("audio") == [first, second]


def test_fallback_purposes_are_validated_like_the_primary() -> None:
    bindings = ModelBindings(llm=[_llm("main"), ModelBinding.image(model="img")])

    with pytest.raises(ValueError, match="bindings.llm"):
        bindings.get_all("llm")


# ── Malformed mounts fail at construction ──────────────────────


def test_empty_list_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="at least one binding"):
        ModelBindings(llm=[])


def test_non_binding_entries_are_rejected_at_construction() -> None:
    with pytest.raises(TypeError, match="ModelBindings"):
        ModelBindings(llm=[_llm("main"), "not-a-binding"])  # type: ignore[list-item]


# ── Lifecycle reaches every provider in the group ──────────────


@pytest.mark.asyncio
async def test_aclose_reaches_built_fallback_providers(monkeypatch) -> None:
    built = [_ClosableProvider(), _ClosableProvider()]

    def fake_create_provider(**kwargs):
        return built.pop(0)

    monkeypatch.setattr("cliver_core.bindings.create_provider", fake_create_provider)
    bindings = ModelBindings(
        llm=[
            ModelBinding.llm(model="a", base_url="https://a.example/v1"),
            ModelBinding.llm(model="b", base_url="https://b.example/v1"),
        ]
    )
    providers = [binding.get_provider() for binding in bindings.get_all("llm")]

    await bindings.aclose()

    assert [p.closed for p in providers] == [True, True]
