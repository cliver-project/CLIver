"""Local System One support — provider, binding factory, and the laya backend.

No torch and no laya import: the provider is exercised through a fake
decision backend (the documented extension point), and the laya backend
through a fake ``laya`` module plus patched model resolution.
"""

from __future__ import annotations

import logging
import sys
import types
from typing import Any

import pytest

from cliver_core.agent import AgentCore
from cliver_core.bindings import ModelBinding, ModelBindings
from cliver_core.local import (
    LoadedDecision,
    LoadSpec,
    LocalBackendNotSupportedError,
    LocalConfigurationError,
    LocalDependencyError,
    LocalInferenceError,
    LocalProviderClosedError,
    LocalSystemOneProvider,
    local_system_one,
    resolve_backend,
)
from cliver_core.local.backends.laya import LayaBackend
from cliver_core.provider import CLIverRequest, CLIverResponse, Provider
from cliver_core.system_one import Choice, ChoiceAnswer, Noul, NoulAnswer

# ── Fakes ─────────────────────────────────────────────────────


class FakeDecisionBackend:
    name = "fake-decision"

    def __init__(
        self,
        *,
        noul: float = 0.8,
        payload: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.noul = noul
        self.payload = payload
        self.error = error
        self.load_calls = 0
        self.unload_calls = 0
        self.predict_calls: list[dict[str, Any]] = []

    def load_decision(self, spec: LoadSpec) -> LoadedDecision:
        self.load_calls += 1
        return LoadedDecision(
            handle=object(),
            resolved_model="/fake/decision",
            revision="sha-decision",
            device="cpu",
            dtype="float32",
            max_length=spec.max_length,
        )

    def predict(
        self,
        loaded: LoadedDecision,
        state: Any,
        questions: dict[str, Any],
        *,
        max_length: int | None = None,
    ) -> dict[str, Any]:
        self.predict_calls.append({"state": state, "questions": questions, "max_length": max_length})
        if self.error is not None:
            raise self.error
        if self.payload is not None:
            return self.payload
        return {"answers": {name: {"type": "noul", "noul": self.noul} for name in questions}}

    def unload_decision(self, loaded: LoadedDecision) -> None:
        loaded.handle = None
        self.unload_calls += 1


class _NotADecisionBackend:
    name = "not-a-decision-backend"


class _StubChatProvider(Provider):
    protocol = "mock"
    brand = "mock"

    def __init__(self) -> None:
        super().__init__(api_key="", base_url="")

    async def chat(self, request: CLIverRequest) -> CLIverResponse:  # pragma: no cover
        raise NotImplementedError

    async def stream(self, request: CLIverRequest):  # pragma: no cover
        if False:
            yield None


def _provider(backend: FakeDecisionBackend, **kwargs) -> LocalSystemOneProvider:
    return LocalSystemOneProvider(model="multilingual", backend=backend, **kwargs)


# ── Provider: results ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_system_one_parses_the_local_payload() -> None:
    backend = FakeDecisionBackend(noul=0.75)
    provider = _provider(backend)

    result = await provider.system_one(
        state="I was charged twice",
        questions={"billing": Noul(instructions="Is this about billing?")},
        model="multilingual",
    )

    assert result.answers["billing"] == NoulAnswer(noul=0.75)
    assert backend.predict_calls[0]["state"] == "I was charged twice"
    assert backend.predict_calls[0]["questions"] == {
        "billing": {"type": "noul", "instructions": "Is this about billing?"}
    }
    assert backend.load_calls == 1
    await provider.close()


@pytest.mark.asyncio
async def test_choice_payload_is_parsed() -> None:
    backend = FakeDecisionBackend(
        payload={
            "answers": {
                "department": {
                    "type": "choice",
                    "choice": "billing",
                    "confidence": 0.92,
                    "probabilities": {"billing": 0.92, "support": 0.08},
                }
            },
            "usage": {"input_tokens": 12},
        }
    )
    provider = _provider(backend)

    result = await provider.system_one(
        state="refund please",
        questions={
            "department": Choice(
                instructions="Which department?",
                criteria={"billing": "billing", "support": "support"},
            )
        },
        model="multilingual",
    )

    answer = result.answers["department"]
    assert isinstance(answer, ChoiceAnswer)
    assert answer.choice == "billing"
    assert answer.confidence == pytest.approx(0.92)
    assert result.usage == {"input_tokens": 12}
    await provider.close()


@pytest.mark.asyncio
async def test_empty_questions_skip_loading() -> None:
    backend = FakeDecisionBackend()
    provider = _provider(backend)

    result = await provider.system_one(state="x", questions={}, model="multilingual")

    assert result.answers == {}
    assert backend.load_calls == 0
    assert backend.predict_calls == []


@pytest.mark.asyncio
async def test_binding_model_must_match_the_provider_model() -> None:
    provider = _provider(FakeDecisionBackend())

    with pytest.raises(LocalConfigurationError, match="does not match"):
        await provider.system_one(state="x", questions={"q": Noul(instructions="Yes?")}, model="other")
    await provider.close()


# ── Provider: options ────────────────────────────────────────


@pytest.mark.asyncio
async def test_max_length_defaults_and_call_site_override() -> None:
    backend = FakeDecisionBackend()
    provider = _provider(backend, max_length=512)

    await provider.system_one(state="x", questions={"q": Noul(instructions="Yes?")}, model="multilingual")
    assert backend.predict_calls[-1]["max_length"] == 512

    await provider.system_one(state="x", questions={"q": Noul(instructions="Yes?")}, model="multilingual", max_len=256)
    assert backend.predict_calls[-1]["max_length"] == 256
    await provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"timeout": 5.0}, "unsupported system one option 'timeout'"),
        ({"path": "/v1/systemone"}, "no endpoint path"),
        ({"device": "cpu"}, "load-time"),
        ({"max_len": 0}, "max_len"),
    ],
)
async def test_system_one_rejects_invalid_call_options(options: dict, match: str) -> None:
    provider = _provider(FakeDecisionBackend())

    with pytest.raises(LocalConfigurationError, match=match):
        await provider.system_one(
            state="x", questions={"q": Noul(instructions="Yes?")}, model="multilingual", **options
        )


# ── Provider: lifecycle ─────────────────────────────────────


@pytest.mark.asyncio
async def test_preload_close_and_closed_semantics() -> None:
    backend = FakeDecisionBackend()
    provider = _provider(backend)

    await provider.preload(warmup=True)
    assert backend.predict_calls == [
        {
            "state": "warmup",
            "questions": {"warmup": {"type": "noul", "instructions": "Is this a warmup request?"}},
            "max_length": None,
        }
    ]

    await provider.close()
    assert provider.info().closed is True
    assert backend.unload_calls == 1

    await provider.close()  # idempotent
    assert backend.unload_calls == 1

    with pytest.raises(LocalProviderClosedError, match="closed"):
        await provider.system_one(state="x", questions={"q": Noul(instructions="Yes?")}, model="multilingual")


def test_provider_requires_a_decision_backend() -> None:
    with pytest.raises(LocalBackendNotSupportedError, match="does not implement system one"):
        LocalSystemOneProvider(model="m", backend=_NotADecisionBackend())


# ── Binding factory and AgentCore ───────────────────────────


def test_local_system_one_factory_validates_and_forwards_defaults() -> None:
    provider = _provider(FakeDecisionBackend())
    binding = local_system_one(provider=provider, max_len=512)

    assert binding.model == "multilingual"
    assert binding.protocol == "local"
    assert binding.purpose == "system_one"
    assert binding.provider is provider
    assert binding.call_defaults() == {"max_len": 512}

    assert local_system_one(provider=provider).call_defaults() == {}
    with pytest.raises(TypeError, match="LocalSystemOneProvider"):
        local_system_one(provider=object())
    with pytest.raises(LocalConfigurationError, match="max_len"):
        local_system_one(provider=provider, max_len=0)


@pytest.mark.asyncio
async def test_agent_routes_system_one_to_the_local_provider_without_closing_it() -> None:
    backend = FakeDecisionBackend(noul=0.6)
    provider = _provider(backend, max_length=512)
    agent = AgentCore(
        ModelBindings(
            llm=ModelBinding.llm(model="main", provider=_StubChatProvider()),
            system_one=local_system_one(provider=provider, max_len=256),
        )
    )

    result = await agent.system_one("state", {"q": Noul(instructions="Yes?")})

    assert result.answers["q"] == NoulAnswer(noul=0.6)
    # The binding's default max_len reaches the provider as a call option.
    assert backend.predict_calls[0]["max_length"] == 256
    await agent.aclose()
    assert provider.info().closed is False
    await provider.close()


# ── laya backend ────────────────────────────────────────────


class _FakeRouter:
    instances: list[_FakeRouter] = []

    def __init__(
        self,
        models: dict[str, Any] | None = None,
        device: str | None = None,
        token: str | None = None,
        revision: str | None = None,
        max_loaded: int = 2,
        default: str = "multilingual",
        preload: bool = False,
        **kwargs: Any,
    ) -> None:
        self.init_kwargs = {
            "models": models,
            "device": device,
            "revision": revision,
            "default": default,
            "preload": preload,
        }
        self.predict_calls: list[dict[str, Any]] = []
        self.preload_calls: list[list[str] | None] = []
        self.unload_calls = 0
        self.error: Exception | None = None
        _FakeRouter.instances.append(self)

    def preload(self, names: list[str] | None = None):
        self.preload_calls.append(names)
        return self

    def predict(self, state: Any, questions: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        self.predict_calls.append({"state": state, "questions": questions, **kwargs})
        if self.error is not None:
            raise self.error
        return {
            "answers": {name: {"type": "noul", "noul": 0.5} for name in questions},
            "usage": {"input_tokens": 3},
            "routing": {"model": "multilingual"},
        }

    def unload(self, name: str | None = None) -> None:
        self.unload_calls += 1


def _spec(model: str, **kwargs: Any) -> LoadSpec:
    values: dict[str, Any] = {
        "device": "cpu",
        "dtype": "auto",
        "revision": None,
        "trust_remote_code": False,
        "cache_dir": None,
        "local_files_only": False,
        "max_length": None,
        "logger": logging.getLogger("test-system-one-local"),
    }
    values.update(kwargs)
    return LoadSpec(model=model, **values)


@pytest.fixture
def fake_laya(monkeypatch) -> type[_FakeRouter]:
    module = types.ModuleType("laya")
    module.Router = _FakeRouter
    _FakeRouter.instances = []
    monkeypatch.setitem(sys.modules, "laya", module)
    monkeypatch.setattr("cliver_core.local.backends.laya.resolve_device", lambda requested: "cpu")
    return _FakeRouter


@pytest.fixture
def resolved(monkeypatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def _resolve(model, *, revision, cache_dir, local_files_only, logger):
        calls.append(
            {
                "model": model,
                "revision": revision,
                "cache_dir": cache_dir,
                "local_files_only": local_files_only,
            }
        )
        return f"/fake/models/{model.replace('/', '__')}", f"sha-{len(calls)}"

    monkeypatch.setattr("cliver_core.local.backends.laya.resolve_local_model", _resolve)
    return calls


def test_laya_resolves_a_builtin_checkpoint_before_loading(fake_laya, resolved) -> None:
    backend = LayaBackend()

    loaded = backend.load_decision(_spec("multilingual", revision="v1", cache_dir="/cache", local_files_only=True))

    assert resolved == [
        {
            "model": "convaiinnovations/laya-multilingual",
            "revision": "v1",
            "cache_dir": "/cache",
            "local_files_only": True,
        }
    ]
    router = loaded.handle.router
    assert router.init_kwargs["models"] == {"multilingual": "/fake/models/convaiinnovations__laya-multilingual"}
    assert router.init_kwargs["default"] == "multilingual"
    assert router.init_kwargs["device"] == "cpu"
    assert router.init_kwargs["revision"] is None  # consumed by the resolver
    # Router(preload=True) would also build the bundle's sibling checkpoints.
    assert router.init_kwargs["preload"] is False
    assert router.preload_calls == [["multilingual"]]
    assert loaded.resolved_model == "/fake/models/convaiinnovations__laya-multilingual"
    assert loaded.revision == "sha-1"


@pytest.mark.parametrize("model", ["laya-multilingual", "ml", "Multi"])
def test_laya_aliases_map_to_the_builtin_checkpoint(fake_laya, resolved, model: str) -> None:
    loaded = LayaBackend().load_decision(_spec(model))

    assert resolved[0]["model"] == "convaiinnovations/laya-multilingual"
    assert loaded.handle.checkpoint == "multilingual"


def test_laya_auto_lets_the_router_route_each_request(fake_laya, resolved) -> None:
    backend = LayaBackend()
    loaded = backend.load_decision(_spec("auto", revision="v1", max_length=1024))

    assert resolved == []  # laya manages its own downloads in this mode
    router = loaded.handle.router
    assert router.init_kwargs["models"] is None
    assert router.init_kwargs["revision"] == "v1"
    # Routing mode stays lazy: the first predict loads the routed checkpoint.
    assert router.init_kwargs["preload"] is False
    assert router.preload_calls == []
    assert loaded.handle.checkpoint is None

    backend.predict(loaded, "state", {"q": {"type": "noul", "instructions": "Yes?"}}, max_length=1024)

    assert router.predict_calls[0]["state"] == "state"
    assert router.predict_calls[0]["max_len"] == 1024
    assert "model" not in router.predict_calls[0]


def test_laya_auto_rejects_cache_control_knobs(fake_laya, resolved) -> None:
    for kwargs in ({"cache_dir": "/cache"}, {"local_files_only": True}):
        with pytest.raises(LocalConfigurationError, match="HF_HUB_CACHE"):
            LayaBackend().load_decision(_spec("auto", **kwargs))


def test_laya_registers_a_custom_source_as_a_checkpoint(fake_laya, resolved) -> None:
    backend = LayaBackend()
    loaded = backend.load_decision(_spec("acme/laya-finetune"))

    assert resolved[0]["model"] == "acme/laya-finetune"
    router = loaded.handle.router
    assert router.init_kwargs["models"] == {"custom": "/fake/models/acme__laya-finetune"}
    assert router.init_kwargs["default"] == "custom"
    assert router.preload_calls == [["custom"]]
    assert loaded.handle.checkpoint == "custom"


def test_laya_predict_pins_the_checkpoint(fake_laya, resolved) -> None:
    backend = LayaBackend()
    loaded = backend.load_decision(_spec("typed-decisions"))

    backend.predict(loaded, "state", {"q": {"type": "noul", "instructions": "Yes?"}})

    assert loaded.handle.router.predict_calls[0]["model"] == "typed-decisions"
    assert "max_len" not in loaded.handle.router.predict_calls[0]


def test_laya_predict_wraps_inference_errors(fake_laya, resolved) -> None:
    backend = LayaBackend()
    loaded = backend.load_decision(_spec("multilingual"))
    loaded.handle.router.error = RuntimeError("boom")

    with pytest.raises(LocalInferenceError, match="boom"):
        backend.predict(loaded, "state", {"q": {"type": "noul", "instructions": "Yes?"}})


def test_laya_unload_drops_the_router(fake_laya, resolved) -> None:
    backend = LayaBackend()
    loaded = backend.load_decision(_spec("multilingual"))

    backend.unload_decision(loaded)

    assert loaded.handle is None


def test_laya_rejects_an_explicit_dtype(fake_laya, resolved) -> None:
    with pytest.raises(LocalConfigurationError, match="dtype"):
        LayaBackend().load_decision(_spec("multilingual", dtype="bfloat16"))


def test_laya_reports_a_missing_dependency(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "laya", None)

    with pytest.raises(LocalDependencyError, match="local-laya"):
        LayaBackend().load_decision(_spec("multilingual"))


def test_laya_backend_resolves_from_the_registry_without_importing_laya() -> None:
    assert resolve_backend("laya").name == "laya"
