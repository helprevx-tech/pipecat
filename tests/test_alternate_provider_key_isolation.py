#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Keep alternate-provider constructors from inheriting an OpenAI environment key.

The actual five public adapters and their shared base constructor paths run with
inert native SDK/HTTP factories. No provider, socket, inference, tenant, billing,
or credential service is contacted. None/empty rejection precedes construction;
explicit keys/endpoints and omitted Speaches local placeholders retain their
original identity. A cold subprocess blocks optional vendor SDK imports while
requiring the existing core OpenAI dependency; missing dependencies never skip
these assertions. Constructor-only tests create no started workers/tasks/clients.
"""

import importlib
import os
import subprocess
import sys
from dataclasses import dataclass
from types import ModuleType
from unittest.mock import Mock

import httpx
import pytest

_ENV_KEY = "synthetic-openai-environment-marker"
_EXPLICIT_KEY = "synthetic-alternate-provider-key"


@dataclass(frozen=True)
class _Provider:
    """Name one actual adapter, native constructor and unchanged default endpoint.

    Values are test metadata only; no registry, model capability or endpoint
    authorization is introduced by this matrix.
    """

    module: str
    name: str
    role: str
    default_endpoint: str


_PROVIDERS = (
    _Provider(
        "pipecat.services.huggingface.llm",
        "HuggingFaceLLMService",
        "llm",
        "https://router.huggingface.co/v1",
    ),
    _Provider(
        "pipecat.services.minimax.llm",
        "MiniMaxLLMService",
        "llm",
        "https://api.minimax.io/v1",
    ),
    _Provider(
        "pipecat.services.speaches.llm",
        "SpeachesLLMService",
        "llm",
        "http://localhost:11434/v1",
    ),
    _Provider(
        "pipecat.services.speaches.stt",
        "SpeachesSTTService",
        "stt",
        "http://localhost:8000/v1",
    ),
    _Provider(
        "pipecat.services.speaches.tts",
        "SpeachesTTSService",
        "tts",
        "http://localhost:8000/v1",
    ),
)


@dataclass
class _NativeBoundary:
    """Observe inert native constructor calls and prohibit actual HTTP work.

    Each case owns fresh mocks; fixture teardown asserts no egress and pytest's
    monkeypatch owner restores every dependency even after an assertion failure.
    """

    factories: dict[str, Mock]
    http_factory: Mock
    egress: tuple[Mock, Mock]


def _load_service(provider: _Provider):
    """Resolve an actual maintained adapter; missing core dependencies fail tests.

    There is no importorskip, substitute class, fake service or conditional
    compatibility path that could hide an unavailable implementation.
    """
    return getattr(importlib.import_module(provider.module), provider.name)


def _module(name: str) -> ModuleType:
    """Load one existing native boundary module without changing import authority."""
    return importlib.import_module(name)


@pytest.fixture
def native_boundary(monkeypatch: pytest.MonkeyPatch):
    """Replace only SDK/HTTP construction and requests; leave adapter logic real.

    The LLM base unconditionally builds DefaultAsyncHttpxClient, so that factory
    is inert too. STT/TTS receive their unchanged optional HTTP-client argument.
    No real client is allocated and no async cleanup owner is required.
    """
    factories = {
        "llm": Mock(name="inert_llm_sdk", return_value=object()),
        "stt": Mock(name="inert_stt_sdk", return_value=object()),
        "tts": Mock(name="inert_tts_sdk", return_value=object()),
    }
    http_factory = Mock(name="inert_default_http_client", return_value=object())
    llm_module = _module("pipecat.services.openai.base_llm")
    stt_module = _module("pipecat.services.whisper.base_stt")
    tts_module = _module("pipecat.services.openai.tts")
    monkeypatch.setattr(llm_module, "AsyncOpenAI", factories["llm"])
    monkeypatch.setattr(llm_module, "DefaultAsyncHttpxClient", http_factory)
    monkeypatch.setattr(stt_module, "AsyncOpenAI", factories["stt"])
    monkeypatch.setattr(tts_module, "AsyncOpenAI", factories["tts"])
    egress = (
        Mock(side_effect=AssertionError("Constructor attempted async HTTP egress")),
        Mock(side_effect=AssertionError("Constructor attempted sync HTTP egress")),
    )
    monkeypatch.setattr(httpx.AsyncClient, "request", egress[0])
    monkeypatch.setattr(httpx.Client, "request", egress[1])
    yield _NativeBoundary(factories, http_factory, egress)
    for request in egress:
        request.assert_not_called()


def _assert_no_native_client(boundary: _NativeBoundary) -> None:
    """Require refusal before any provider SDK or owned HTTP resource allocation."""
    for factory in boundary.factories.values():
        factory.assert_not_called()
    boundary.http_factory.assert_not_called()


@pytest.mark.parametrize("provider", _PROVIDERS, ids=[p.name for p in _PROVIDERS])
@pytest.mark.parametrize("api_key", [None, ""], ids=["explicit-none", "empty"])
@pytest.mark.parametrize("environment_present", [False, True], ids=["no-env", "env-set"])
def test_explicit_missing_keys_refuse_before_any_native_client(
    provider: _Provider,
    api_key,
    environment_present: bool,
    native_boundary: _NativeBoundary,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
):
    """None/empty never becomes absence/default, SDK fallback or error disclosure.

    Both environment states exercise the real adapter guard. A synthetic
    configured global key must remain outside the native boundary and errors.
    """
    if environment_present:
        monkeypatch.setenv("OPENAI_API_KEY", _ENV_KEY)
    else:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    service_type = _load_service(provider)
    with pytest.raises(ValueError, match="requires an explicit non-empty api_key") as error:
        service_type(api_key=api_key)
    _assert_no_native_client(native_boundary)
    assert _ENV_KEY not in str(error.value)
    captured = capfd.readouterr()
    assert _ENV_KEY not in captured.out + captured.err


@pytest.mark.parametrize("provider", _PROVIDERS, ids=[p.name for p in _PROVIDERS])
@pytest.mark.parametrize("custom_endpoint", [False, True], ids=["default", "custom"])
def test_explicit_provider_key_and_endpoint_reach_only_their_native_owner(
    provider: _Provider,
    custom_endpoint: bool,
    native_boundary: _NativeBoundary,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
):
    """Preserve exact explicit credentials/endpoint and untouched sibling clients.

    This is constructor wiring evidence, not SDK authentication or live-provider
    acceptance. Opaque inert clients have no network/resource lifecycle.
    """
    monkeypatch.setenv("OPENAI_API_KEY", _ENV_KEY)
    options = {"api_key": _EXPLICIT_KEY}
    if custom_endpoint:
        options["base_url"] = "https://synthetic-provider.example.invalid/v1"
    service = _load_service(provider)(**options)
    factory = native_boundary.factories[provider.role]
    factory.assert_called_once()
    arguments = factory.call_args.kwargs
    assert arguments["api_key"] == _EXPLICIT_KEY
    assert arguments["api_key"] != _ENV_KEY
    assert arguments["base_url"] == options.get("base_url", provider.default_endpoint)
    assert service._client is factory.return_value
    for role, sibling in native_boundary.factories.items():
        if role != provider.role:
            sibling.assert_not_called()
    if provider.role == "llm":
        native_boundary.http_factory.assert_called_once()
        assert arguments["http_client"] is native_boundary.http_factory.return_value
    else:
        native_boundary.http_factory.assert_not_called()
        assert arguments["http_client"] is None
    captured = capfd.readouterr()
    assert _ENV_KEY not in captured.out + captured.err
    assert _EXPLICIT_KEY not in captured.out + captured.err


@pytest.mark.parametrize("provider", _PROVIDERS[2:], ids=[p.name for p in _PROVIDERS[2:]])
def test_omitted_speaches_key_preserves_the_explicit_local_placeholder(
    provider: _Provider,
    native_boundary: _NativeBoundary,
    monkeypatch: pytest.MonkeyPatch,
):
    """Keep existing keyless-local callers usable without inheriting global keys.

    Omission retains the existing literal 'none'; it is not converted into a
    provider credential or a new permission for arbitrary remote endpoints.
    """
    monkeypatch.setenv("OPENAI_API_KEY", _ENV_KEY)
    service = _load_service(provider)()
    factory = native_boundary.factories[provider.role]
    factory.assert_called_once()
    assert factory.call_args.kwargs["api_key"] == "none"
    assert factory.call_args.kwargs["base_url"] == provider.default_endpoint
    assert service._client is factory.return_value


@pytest.mark.parametrize("provider", _PROVIDERS[:2], ids=[p.name for p in _PROVIDERS[:2]])
def test_hosted_provider_still_requires_the_api_key_keyword(
    provider: _Provider,
    native_boundary: _NativeBoundary,
    monkeypatch: pytest.MonkeyPatch,
):
    """Python's existing missing-keyword contract refuses before client creation."""
    monkeypatch.setenv("OPENAI_API_KEY", _ENV_KEY)
    with pytest.raises(TypeError, match="api_key") as error:
        _load_service(provider)()
    _assert_no_native_client(native_boundary)
    assert _ENV_KEY not in str(error.value)


@pytest.mark.parametrize("provider", _PROVIDERS, ids=[p.name for p in _PROVIDERS])
def test_explicit_valid_keys_preserve_native_construction_failure(
    provider: _Provider,
    native_boundary: _NativeBoundary,
    monkeypatch: pytest.MonkeyPatch,
):
    """An unavailable native dependency remains an error, never another client.

    Guards do not wrap genuine constructor failures, fall back to the OpenAI
    endpoint, silently skip a provider or retry a possibly allocated resource.
    """
    monkeypatch.setenv("OPENAI_API_KEY", _ENV_KEY)
    factory = native_boundary.factories[provider.role]
    factory.side_effect = ImportError("Synthetic native SDK unavailable")
    with pytest.raises(ImportError, match="Synthetic native SDK unavailable"):
        _load_service(provider)(api_key=_EXPLICIT_KEY)
    factory.assert_called_once()
    assert factory.call_args.kwargs["api_key"] == _EXPLICIT_KEY
    assert factory.call_args.kwargs["base_url"] == provider.default_endpoint
    for role, sibling in native_boundary.factories.items():
        if role != provider.role:
            sibling.assert_not_called()


def test_adapters_import_with_no_optional_vendor_sdk_in_a_cold_process():
    """Prove core-only imports without cached vendor modules or silent skips.

    The child receives only source-path discovery and a synthetic environment
    key, performs imports only, and blocks optional vendor SDK module families.
    subprocess.run kills and reaps a child that exceeds this test-only bound;
    no worker/model deadline, account or service is involved.
    """
    child_source = '''
import importlib
import importlib.abc
import sys

class RejectOptionalVendorSDK(importlib.abc.MetaPathFinder):
    """Reject optional vendor SDKs; importing a core adapter must not require them."""

    def find_spec(self, fullname, path=None, target=None):
        """Fail before a prohibited SDK loads; all other imports retain normal lookup."""
        if fullname.split(".")[0] in {"huggingface_hub", "minimax", "speaches"}:
            raise ImportError("Optional vendor SDK intentionally unavailable")
        return None

sys.meta_path.insert(0, RejectOptionalVendorSDK())
for module in (
    "pipecat.services.huggingface.llm",
    "pipecat.services.minimax.llm",
    "pipecat.services.speaches.llm",
    "pipecat.services.speaches.stt",
    "pipecat.services.speaches.tts",
):
    importlib.import_module(module)
print("five_core_only_adapters")
'''
    child_environment = {"OPENAI_API_KEY": _ENV_KEY}
    if os.environ.get("PYTHONPATH"):
        child_environment["PYTHONPATH"] = os.environ["PYTHONPATH"]
    result = subprocess.run(
        [sys.executable, "-c", child_source],
        env=child_environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "five_core_only_adapters"
    assert _ENV_KEY not in result.stdout + result.stderr
