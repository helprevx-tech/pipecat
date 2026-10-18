#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for service settings and initialization patterns.

Settings objects operate in two modes:

- **Store mode** (``self._settings``): the live state inside a service.
  Every field must hold a real value (``None`` is fine, ``NOT_GIVEN`` is not).
- **Delta mode** (``FooSettings()`` with no args): a sparse update.
  Every field must default to ``NOT_GIVEN`` so ``apply_update()`` skips
  untouched fields and doesn't accidentally overwrite the store.

These tests verify both sides of that contract automatically:

1. **Delta defaults** — Instantiate every ``ServiceSettings`` subclass with
   no arguments and assert that every field is ``NOT_GIVEN``.  Catches the
   bug where a field defaults to ``None`` instead of ``NOT_GIVEN``, which
   would cause partial deltas to silently overwrite unrelated store values.

2. **Store completeness** — Instantiate every concrete service with explicit
   isolated caller inputs and assert that ``_settings`` contains no ``NOT_GIVEN`` values.
   This is the same check that ``validate_complete()`` runs in ``start()``,
   but caught here at unit-test time without needing a running pipeline.
   Catches services that forget to initialize a field in ``default_settings``.

Settings and service classes are auto-discovered. Actual abstract services are
excluded by Python's native abstract-class contract. Known concrete constructors
receive explicit typed unit inputs and owned SDK/discovery/model boundaries; no
provider is contacted, environment credentials read, or missing-input guard
relaxed. An unknown concrete constructor failure is a failure rather than a skip.
This is constructor/settings evidence, never provider authentication acceptance.
"""

import importlib
import inspect
import pkgutil
import socket
import sys
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import aiohttp
import httpx
import pytest
import pytest_asyncio

import pipecat.services
from pipecat.services.ai_service import AIService
from pipecat.services.settings import ServiceSettings
from pipecat.utils.types import is_given

# Modules that define abstract base service classes (not concrete services).
_BASE_MODULES = frozenset(
    {
        "pipecat.services.ai_service",
        "pipecat.services.llm_service",
        "pipecat.services.stt_service",
        "pipecat.services.tts_service",
        "pipecat.services.image_gen_service",
        "pipecat.services.vision_service",
    }
)


# ---------------------------------------------------------------------------
# Auto-discovery
# ---------------------------------------------------------------------------


def _all_subclasses(cls):
    result = set()
    for sub in cls.__subclasses__():
        result.add(sub)
        result.update(_all_subclasses(sub))
    return result


def _import_all_service_modules():
    """Import every module under pipecat.services (skipping missing deps)."""
    package = pipecat.services
    for _importer, modname, _ispkg in pkgutil.walk_packages(
        package.__path__, prefix=package.__name__ + ".", onerror=lambda _name: None
    ):
        try:
            importlib.import_module(modname)
        except Exception:
            continue


_import_all_service_modules()

ALL_SETTINGS_CLASSES = sorted(_all_subclasses(ServiceSettings), key=lambda c: c.__qualname__)
assert ALL_SETTINGS_CLASSES, "No settings classes discovered"


# ---------------------------------------------------------------------------
# Service instantiation helpers
# ---------------------------------------------------------------------------


_UNIT_KEY = "synthetic-constructor-key"
_GATEWAY_CLASSES = {
    "AayiqAgentFluxSTTService",
    "AayiqAgentLLMService",
    "AayiqAgentSTTService",
    "AayiqAgentTTSService",
}
_OPENAI_CLASSES = {
    "BaseOpenAILLMService",
    "BaseWhisperSTTService",
    "OpenAILLMService",
    "OpenAIResponsesHttpLLMService",
    "OpenAIResponsesLLMService",
    "OpenAISTTService",
    "OpenAITTSService",
    "_BaseOpenAIResponsesLLMService",
}
_AZURE_CLASSES = {
    "AzureHttpTTSService",
    "AzureLLMService",
    "AzureRealtimeLLMService",
    "AzureSTTService",
    "AzureTTSService",
    "AzureVoiceLiveLLMService",
}
_GOOGLE_CLASSES = {
    "GeminiLiveVertexLLMService",
    "GeminiTTSService",
    "GoogleHttpTTSService",
    "GoogleSTTService",
    "GoogleTTSService",
    "GoogleVertexLLMService",
}


class ConstructorBoundary:
    """Own only explicit unit identity, SDK resources and outbound tripwires.

    Native constructors/settings/key/voice/URL guards execute unchanged. SDK
    clients keep their real key validation where available; the Google discovery
    seam supplies an explicitly synthetic credential object, never an account.
    This fixture never starts a service, provider request, model or pipeline.
    """

    def __init__(self, monkeypatch, tmp_path):
        self.monkeypatch = monkeypatch
        self.tmp_path = tmp_path
        self.clients = []
        self.egress_attempts = []
        self.google_credentials_available = True
        self.google_sdk_calls = []
        self.google_refreshes = 0
        self.google_identities = []
        self.download_attempts = []
        self.local_model_loads = {}
        self._wrapped = set()
        # Do not fall through to an operator's environment or metadata server.
        for name in (
            "OPENAI_API_KEY",
            "OPENAI_ORG_ID",
            "OPENAI_PROJECT_ID",
            "OPENAI_BASE_URL",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_AD_TOKEN",
            "AZURE_OPENAI_ENDPOINT",
            "GOOGLE_APPLICATION_CREDENTIALS",
            "GOOGLE_API_KEY",
            "GEMINI_API_KEY",
            "GOOGLE_CLOUD_PROJECT",
            "GCLOUD_PROJECT",
            "FAL_KEY",
            "SPEECHMATICS_API_KEY",
        ):
            monkeypatch.delenv(name, raising=False)

        def blocked(*args, **kwargs):
            """Record attempted egress even if a native constructor catches it."""
            self.egress_attempts.append(True)
            raise AssertionError("Constructor attempted external I/O")

        async def blocked_async(*args, **kwargs):
            """Forbid async HTTP before its transport can dispatch or resolve DNS."""
            blocked()

        for owner, name in (
            (socket, "getaddrinfo"),
            (socket, "create_connection"),
            (socket.socket, "connect"),
            (socket.socket, "connect_ex"),
            (httpx.Client, "send"),
        ):
            monkeypatch.setattr(owner, name, blocked)
        monkeypatch.setattr(httpx.AsyncClient, "send", blocked_async)
        monkeypatch.setattr(aiohttp.ClientSession, "_request", blocked_async)

        # Retain real OpenAI/Azure SDK construction and credential checks. Track
        # clients at the imported factory seam so partial construction also has
        # a cleanup owner. No service constructor or _settings is replaced.
        for module_name, names in (
            ("pipecat.services.openai.base_llm", ("AsyncOpenAI", "DefaultAsyncHttpxClient")),
            ("pipecat.services.whisper.base_stt", ("AsyncOpenAI",)),
            ("pipecat.services.openai.tts", ("AsyncOpenAI",)),
            ("pipecat.services.openai.responses.llm", ("AsyncOpenAI", "DefaultAsyncHttpxClient")),
            ("pipecat.services.azure.llm", ("AsyncOpenAI", "AsyncAzureOpenAI")),
        ):
            module = sys.modules.get(module_name)
            if module is not None:
                for name in names:
                    if hasattr(module, name):
                        self.wrap_client(
                            module, name, "aclose" if name == "DefaultAsyncHttpxClient" else "close"
                        )

    def wrap_client(self, module, name, close_method):
        """Track a real SDK client without bypassing its input validation."""
        key = (id(module), name)
        if key in self._wrapped:
            return
        self._wrapped.add(key)
        original = getattr(module, name)

        def construct(*args, **kwargs):
            """Construct only the original SDK and retain its close ownership."""
            client = original(*args, **kwargs)
            self.clients.append((client, close_method))
            return client

        self.monkeypatch.setattr(module, name, construct)

    def google(self):
        """Isolate credential discovery/refresh and SDK transports, not settings.

        This test identity cannot authorize a real request. Missing credentials
        still travel through native GoogleAuthError-to-ValueError branches. SDK
        constructors are inert like the existing Google HTTP TTS fixture and
        every requested credential/project is observable before closure.
        """
        from google.auth.credentials import Credentials
        from google.auth.exceptions import GoogleAuthError

        boundary = self

        class UnitCredentials(Credentials):
            """Explicit native credential interface with no refresh transport."""

            def __init__(self):
                super().__init__()
                self.token = "synthetic-unit-token"

            def refresh(self, request):
                """Exercise the native refresh caller without network/account I/O."""
                boundary.google_refreshes += 1
                self.token = "synthetic-unit-token"

        credentials = UnitCredentials()
        self.google_identities.append(credentials)

        def discover(*args, **kwargs):
            """Return only this known unit identity or genuine discovery refusal."""
            if not boundary.google_credentials_available:
                raise GoogleAuthError("Synthetic credentials unavailable")
            return credentials, "synthetic-project"

        def sdk_client(*args, **kwargs):
            """Own a close-aware inert SDK boundary; never pretend it authenticated."""
            boundary.google_sdk_calls.append((args, kwargs))
            client = MagicMock()
            client.close = AsyncMock()
            client.aio = MagicMock()
            client.aio.aclose = AsyncMock()
            boundary.clients.append((client, "close"))
            return client

        for module_name in (
            "pipecat.services.google.tts",
            "pipecat.services.google.stt",
            "pipecat.services.google.vertex.llm",
            "pipecat.services.google.gemini_live.vertex.llm",
        ):
            module = sys.modules.get(module_name)
            if module is None:
                continue
            if hasattr(module, "default"):
                self.monkeypatch.setattr(module, "default", discover)
            if hasattr(module, "Request"):
                self.monkeypatch.setattr(module, "Request", lambda: Mock())
            if hasattr(module, "Client"):
                self.monkeypatch.setattr(module, "Client", sdk_client)
            if hasattr(module, "texttospeech_v1"):
                self.monkeypatch.setattr(
                    module.texttospeech_v1, "TextToSpeechAsyncClient", sdk_client
                )
            if hasattr(module, "speech_v2"):
                self.monkeypatch.setattr(module.speech_v2, "SpeechAsyncClient", sdk_client)
        return credentials

    def kwargs(self, cls):
        """Supply valid typed caller inputs only where the real source requires them."""
        kwargs = {}
        for name, param in inspect.signature(cls.__init__).parameters.items():
            if name == "self" or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            if param.default is param.empty:
                kwargs[name] = MagicMock() if name == "aiohttp_session" else "test"
        name = cls.__name__
        if name in _GATEWAY_CLASSES:
            kwargs.update(
                gateway_base_url="https://gateway.example.test/api/v1",
                invocation_token="synthetic-invocation",
                request_id="synthetic-original",
                model="synthetic-model",
            )
            if name == "AayiqAgentTTSService":
                kwargs["voice"] = "synthetic-voice"
        if name in _OPENAI_CLASSES | _AZURE_CLASSES | {"FalSTTService", "SpeechmaticsSTTService"}:
            kwargs["api_key"] = _UNIT_KEY
        if name in {"AzureSTTService", "AzureTTSService", "AzureHttpTTSService"}:
            kwargs["region"] = "eastus"
        if name in {"AzureLLMService", "AzureVoiceLiveLLMService"}:
            kwargs["endpoint"] = "https://synthetic-resource.openai.azure.com"
        if name == "AzureRealtimeLLMService":
            kwargs["base_url"] = "wss://synthetic-resource.openai.azure.com/openai/v1/realtime"
        if name in _GOOGLE_CLASSES:
            self.google()
            if name in {"GoogleVertexLLMService", "GeminiLiveVertexLLMService"}:
                kwargs["project_id"] = "synthetic-project"
                kwargs["location"] = "us-central1"
        if name == "XTTSService":
            kwargs["base_url"] = "http://127.0.0.1:1"
            kwargs["settings"] = cls.Settings(voice="synthetic-voice")
        if name == "PiperTTSService":
            module = sys.modules[cls.__module__]
            voice = "en_US-synthetic-medium"
            cache = self.tmp_path / "piper"
            cache.mkdir()
            (cache / f"{voice}.onnx").write_bytes(b"synthetic-model-placeholder")

            def download(*args, **kwargs):
                """A constructor must not download; even caught attempts fail teardown."""
                self.download_attempts.append(True)
                raise AssertionError("Constructor attempted model download")

            self.monkeypatch.setattr(module, "download_voice", download)
            self.monkeypatch.setattr(module.PiperVoice, "load", Mock(return_value=object()))
            kwargs.update(download_dir=cache, settings=cls.Settings(voice=voice))
        if name in {
            "KokoroTTSService",
            "MoondreamService",
            "PocketTTSService",
            "WhisperSTTService",
        }:
            # These are constructor/settings tests, not model-weight acceptance.
            # Keep the real service and native settings/voice guards; replace only
            # its exact external weight-loader boundary and observe that call.
            module = sys.modules[cls.__module__]
            if name == "KokoroTTSService":
                model, voices = self.tmp_path / "kokoro.onnx", self.tmp_path / "voices.bin"
                model.write_bytes(b"synthetic-weight-placeholder")
                voices.write_bytes(b"synthetic-voice-placeholder")
                loader = Mock(return_value=SimpleNamespace(voices=["synthetic-voice"]))
                self.monkeypatch.setattr(module, "Kokoro", loader)
                kwargs.update(
                    model_path=str(model),
                    voices_path=str(voices),
                    settings=cls.Settings(voice="synthetic-voice"),
                )
            elif name == "MoondreamService":
                loader = Mock(return_value=object())
                self.monkeypatch.setattr(module, "load_model", loader)
                kwargs["use_cpu"] = True
            elif name == "PocketTTSService":
                model = SimpleNamespace(
                    sample_rate=24000,
                    get_state_for_audio_prompt=Mock(return_value={"synthetic": True}),
                )
                loader = Mock(return_value=model)
                self.monkeypatch.setattr(module.TTSModel, "load_model", loader)
            else:
                loader = Mock(return_value=SimpleNamespace(supported_languages=["en"]))
                self.monkeypatch.setattr(module, "WhisperModel", loader)
                kwargs.update(device="cpu", compute_type="float32")
            self.local_model_loads[name] = loader
        return kwargs

    async def close(self):
        """Close only SDK clients created by this fixture, even on assertion failure."""
        failures = []
        for client, close_method in reversed(self.clients):
            try:
                closed = getattr(client, "is_closed", False)
                if callable(closed):
                    closed = closed()
                if closed is True:
                    continue
                result = getattr(client, close_method)()
                if inspect.isawaitable(result):
                    await result
                # GenAI owns a separate async interface; only our inert GenAI
                # fixture exposes both close methods here.
                if isinstance(client, MagicMock):
                    await client.aio.aclose()
                    client.close.assert_awaited_once()
                    client.aio.aclose.assert_awaited_once()
            except Exception as error:
                failures.append(type(error).__name__)
        assert not failures, "Owned constructor SDK cleanup failed: " + ", ".join(failures)
        assert not self.egress_attempts, "Constructor attempted external I/O"
        assert not self.download_attempts, "Constructor attempted model download"


@pytest_asyncio.fixture
async def constructor_boundary(monkeypatch, tmp_path):
    """Restore environment/SDK patches and close all owned clients per service."""
    boundary = ConstructorBoundary(monkeypatch, tmp_path)
    try:
        yield boundary
    finally:
        await boundary.close()


def _try_instantiate(cls, boundary):
    """Run the real constructor with explicit unit inputs; errors are not skipped."""
    return cls(**boundary.kwargs(cls))


def _concrete_service_classes():
    """Return concrete (non-abstract) service classes under pipecat.services.

    Pure class-hierarchy walk with no instantiation, so collection stays fast.
    Construction runs inside an isolated constructor fixture. True abstract
    classes are excluded by Python, not caught as credential-like failures.
    """
    return [
        cls
        for cls in sorted(_all_subclasses(AIService), key=lambda c: c.__qualname__)
        # Services only: subclasses defined elsewhere, such as the stand-ins
        # other test modules build, are not what this checks.
        if cls.__module__.startswith("pipecat.services.")
        # Skip abstract base classes defined in framework modules.
        and cls.__module__ not in _BASE_MODULES
        and not inspect.isabstract(cls)
    ]


ALL_SERVICE_CLASSES = _concrete_service_classes()
assert ALL_SERVICE_CLASSES, "No service classes discovered"


# ---------------------------------------------------------------------------
# 1. Settings defaults: delta-mode safety
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("settings_cls", ALL_SETTINGS_CLASSES, ids=lambda c: c.__qualname__)
def test_delta_defaults_are_not_given(settings_cls):
    """Every field must default to NOT_GIVEN so empty deltas are no-ops.

    A field that defaults to None instead of NOT_GIVEN will cause
    apply_update() to overwrite the corresponding store value whenever
    a partial delta is applied.
    """
    instance = settings_cls()
    for f in fields(instance):
        if f.name == "extra":
            continue
        val = getattr(instance, f.name)
        assert not is_given(val), (
            f"{settings_cls.__qualname__}.{f.name} defaults to {val!r}, expected NOT_GIVEN"
        )


# ---------------------------------------------------------------------------
# 2. Service construction: store-mode completeness
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("service_cls", ALL_SERVICE_CLASSES, ids=lambda c: c.__qualname__)
async def test_service_settings_complete(service_cls, constructor_boundary):
    """After construction, _settings must have no NOT_GIVEN values.

    This is what validate_complete() checks in start().  Catching it
    here means we don't need a running pipeline to find missing defaults.
    """
    svc = _try_instantiate(service_cls, constructor_boundary)
    assert hasattr(svc, "_settings"), f"{service_cls.__qualname__} has no settings store"
    for f in fields(svc._settings):
        if f.name == "extra":
            continue
        val = getattr(svc._settings, f.name)
        assert is_given(val), (
            f"{service_cls.__qualname__}._settings.{f.name} is NOT_GIVEN after construction"
        )
    if service_cls.__name__ in _GOOGLE_CLASSES:
        assert constructor_boundary.google_sdk_calls
        assert all(
            call[1].get("credentials") in constructor_boundary.google_identities
            for call in constructor_boundary.google_sdk_calls
        )
        if "Vertex" in service_cls.__name__:
            assert constructor_boundary.google_refreshes == 1
            assert all(
                call[1].get("project") == "synthetic-project"
                for call in constructor_boundary.google_sdk_calls
            )
    if service_cls.__name__ in constructor_boundary.local_model_loads:
        constructor_boundary.local_model_loads[service_cls.__name__].assert_called_once()
        if service_cls.__name__ == "KokoroTTSService":
            assert svc._settings.voice in svc._kokoro.voices
        if service_cls.__name__ == "PocketTTSService":
            svc._model.get_state_for_audio_prompt.assert_called_once_with(svc._settings.voice)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "service_cls",
    [
        cls
        for cls in ALL_SERVICE_CLASSES
        if cls.__name__
        in (
            _GATEWAY_CLASSES
            | _OPENAI_CLASSES
            | _AZURE_CLASSES
            | _GOOGLE_CLASSES
            | {"FalSTTService", "SpeechmaticsSTTService"}
        )
    ],
    ids=lambda cls: cls.__qualname__,
)
async def test_constructor_fixture_does_not_authorize_missing_credentials(
    service_cls, constructor_boundary
):
    """Native missing credentials still refuse; no fabricated fixture grant or SDK I/O.

    Google discovery emits its genuine known authentication exception. The real
    service turns it into its native refusal, before the inert SDK factory. All
    other services retain native/real SDK key validation with no environment key.
    """
    from openai import OpenAIError

    kwargs = constructor_boundary.kwargs(service_cls)
    if service_cls.__name__ in _GATEWAY_CLASSES:
        kwargs["invocation_token"] = None
    elif service_cls.__name__ in _GOOGLE_CLASSES:
        constructor_boundary.google_credentials_available = False
    else:
        kwargs["api_key"] = None
    with pytest.raises((ValueError, OpenAIError)):
        service_cls(**kwargs)
    if service_cls.__name__ in _GOOGLE_CLASSES:
        assert not constructor_boundary.google_sdk_calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "service_cls",
    [
        cls
        for cls in ALL_SERVICE_CLASSES
        if cls.__name__
        in {
            "KokoroTTSService",
            "MoondreamService",
            "PocketTTSService",
            "WhisperSTTService",
        }
    ],
    ids=lambda cls: cls.__qualname__,
)
async def test_local_weight_failure_remains_a_constructor_failure(
    service_cls, constructor_boundary
):
    """An unavailable weight loader must fail the real constructor without inference, fallback or download."""
    kwargs = constructor_boundary.kwargs(service_cls)
    constructor_boundary.local_model_loads[service_cls.__name__].side_effect = RuntimeError(
        "Synthetic model weights unavailable"
    )
    with pytest.raises(RuntimeError, match="Synthetic model weights unavailable"):
        service_cls(**kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "service_cls", [cls for cls in ALL_SERVICE_CLASSES if cls.__name__ == "KokoroTTSService"]
)
async def test_local_kokoro_fixture_preserves_native_voice_catalog_guard(
    service_cls, constructor_boundary
):
    """An inert loader cannot authorize a voice absent from its returned catalog."""
    kwargs = constructor_boundary.kwargs(service_cls)
    constructor_boundary.local_model_loads[service_cls.__name__].return_value.voices = []
    with pytest.raises(ValueError, match="not in the voices file"):
        service_cls(**kwargs)


def test_true_abstract_services_are_not_constructor_cases():
    """Abstract registration is not skipped concrete-service completeness evidence."""
    assert ALL_SERVICE_CLASSES
    assert all(not inspect.isabstract(cls) for cls in ALL_SERVICE_CLASSES)


@pytest.mark.asyncio
async def test_unexpected_constructor_failure_is_not_hidden_by_a_generic_skip(constructor_boundary):
    """A new failure must remain visible instead of being labeled credentials-needed."""

    class BrokenConstructor:
        def __init__(self):
            raise RuntimeError("Synthetic constructor failure")

    with pytest.raises(RuntimeError, match="Synthetic constructor failure"):
        _try_instantiate(BrokenConstructor, constructor_boundary)


@pytest.mark.asyncio
async def test_constructor_egress_tripwire_keeps_caught_attempts_visible(monkeypatch, tmp_path):
    """Even a swallowed network error leaves a failing original fixture receipt."""
    boundary = ConstructorBoundary(monkeypatch, tmp_path)
    with pytest.raises(AssertionError, match="Constructor attempted external I/O"):
        socket.getaddrinfo("provider.example.test", 443)
    with pytest.raises(AssertionError, match="Constructor attempted external I/O"):
        await boundary.close()
    assert boundary.egress_attempts == [True]


@pytest.mark.asyncio
async def test_constructor_cleanup_preserves_failure_and_closes_remaining_owned_clients(
    monkeypatch, tmp_path
):
    """One SDK close failure cannot strand the remaining owned clients or leak text."""
    boundary = ConstructorBoundary(monkeypatch, tmp_path)
    observed = []

    class ClosingClient:
        """Minimal owned resource for testing the fixture's failure-atomic cleanup."""

        def __init__(self, name, fail=False):
            self.name, self.fail = name, fail

        async def close(self):
            """Record actual close attempts before the controlled synthetic failure."""
            observed.append(self.name)
            if self.fail:
                raise RuntimeError("never-expose-unit-private-detail")

    boundary.clients.extend(
        [(ClosingClient("remaining"), "close"), (ClosingClient("failure", True), "close")]
    )
    with pytest.raises(AssertionError, match="Owned constructor SDK cleanup failed") as failure:
        await boundary.close()
    assert observed == ["failure", "remaining"]
    assert "RuntimeError" in str(failure.value)
    assert "never-expose" not in str(failure.value)
