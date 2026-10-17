#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Sarvam capabilities, language wire serialization, switching and text gating.

Local fake HTTP/WebSocket peers record native payloads without credentials or
provider requests. Legacy language settings remain stable while current JSON
uses language_code; these tests do not establish live synthesis acceptance.
"""

import json
import unittest
from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlparse

import pytest
from websockets.protocol import State

from pipecat.services.sarvam.tts import (
    SarvamHttpTTSService,
    SarvamTTSService,
    _SpeakableTextFilter,
)

V4_FLASH = "bulbul:v4-flash"


class _FakeWebsocket:
    def __init__(self, *, state=State.OPEN):
        self.state = state
        self.sent = []

    async def send(self, message):
        self.sent.append(message)

    async def close(self):
        self.state = State.CLOSED


class _FakeResponse:
    status = 200

    async def json(self):
        return {"audios": [""], "request_id": "test"}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class _FakeSession:
    """Records the JSON body of each POST."""

    def __init__(self):
        self.payloads = []

    def post(self, url, json=None, headers=None):
        self.payloads.append(json)
        return _FakeResponse()


def _service(**settings_kwargs) -> SarvamTTSService:
    settings_kwargs.setdefault("model", V4_FLASH)
    return SarvamTTSService(
        api_key="test-key", settings=SarvamTTSService.Settings(**settings_kwargs)
    )


def _http_service(session: _FakeSession, **settings_kwargs) -> SarvamHttpTTSService:
    settings_kwargs.setdefault("model", V4_FLASH)
    service = SarvamHttpTTSService(
        api_key="test-key",
        aiohttp_session=session,  # type: ignore[arg-type]
        settings=SarvamHttpTTSService.Settings(**settings_kwargs),
    )
    service._sample_rate = service._init_sample_rate
    return service


def _sent_config(service: SarvamTTSService) -> dict:
    """Read back the config the service put on the wire."""
    messages = [json.loads(m) for m in service._websocket.sent]
    configs = [m["data"] for m in messages if m["type"] == "config"]
    assert configs, "no config message was sent"
    return configs[-1]


def _url_model(service: SarvamTTSService) -> str:
    return parse_qs(urlparse(service._websocket_url).query)["model"][0]


async def _synthesize(service: SarvamHttpTTSService) -> dict:
    async for _ in service.run_tts("hello", "ctx"):
        pass
    return service._session.payloads[-1]


def test_v4_flash_defaults():
    """Resolve the framework's explicit app speaker and output rate, independently of omitted API defaults."""
    service = _service()

    assert service._settings.voice == "shubh_en_narration_gentle"
    assert service._init_sample_rate == 24000
    assert service._settings.enable_preprocessing is True


@pytest.mark.asyncio
async def test_v4_flash_sends_pitch_and_loudness():
    """v4-flash applies both; the v3 models ignore them."""
    service = _service(pitch=0.3, loudness=1.5, pace=1.2)
    service._websocket = _FakeWebsocket()

    await service._send_config()
    config = _sent_config(service)

    assert config["pitch"] == 0.3
    assert config["loudness"] == 1.5
    assert config["pace"] == 1.2


@pytest.mark.asyncio
@pytest.mark.parametrize("temperature", [0.6, 0.9])
async def test_v4_flash_drops_temperature(temperature):
    """The API pins it to 0.6, so sending any value would only mislead."""
    service = _service(temperature=temperature)
    service._websocket = _FakeWebsocket()

    await service._send_config()

    assert "temperature" not in _sent_config(service)


@pytest.mark.asyncio
async def test_v4_flash_drops_a_runtime_temperature_update():
    service = _service()
    service._websocket = _FakeWebsocket()

    await service._update_settings(SarvamTTSService.Settings(temperature=0.9))

    assert "temperature" not in _sent_config(service)


def test_v4_flash_clamps_pace_to_its_own_range():
    service = _service(pace=2.5)

    assert service._settings.pace == 2.0


@pytest.mark.asyncio
async def test_switching_to_v4_flash_reconnects_with_its_capabilities(monkeypatch):
    """The model is pinned on the URL, so the switch needs a new connection."""
    service = _service(model="bulbul:v3", voice=None)
    disconnect, connect = AsyncMock(), AsyncMock()
    monkeypatch.setattr(service, "_disconnect", disconnect)
    monkeypatch.setattr(service, "_connect", connect)

    await service._update_settings(SarvamTTSService.Settings(model=V4_FLASH))

    assert _url_model(service) == V4_FLASH
    assert service._config.supports_pitch is True
    # bulbul:v4-flash rejects the v3 default, so the voice moves with the model.
    assert service._settings.voice == "shubh_en_narration_gentle"
    disconnect.assert_awaited_once()
    connect.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_voice_switched_with_the_model_is_kept(monkeypatch):
    service = _service(model="bulbul:v3", voice=None)
    monkeypatch.setattr(service, "_disconnect", AsyncMock())
    monkeypatch.setattr(service, "_connect", AsyncMock())

    await service._update_settings(
        SarvamTTSService.Settings(model=V4_FLASH, voice="ritu_hi_edtech")
    )

    assert service._settings.voice == "ritu_hi_edtech"


@pytest.mark.asyncio
async def test_an_unsupported_model_update_is_ignored(monkeypatch):
    service = _service(model="bulbul:v3", voice=None)
    service._websocket = _FakeWebsocket()
    connect = AsyncMock()
    monkeypatch.setattr(service, "_connect", connect)

    changed = await service._update_settings(SarvamTTSService.Settings(model="bulbul:v9"))

    assert service._settings.model == "bulbul:v3"
    assert _url_model(service) == "bulbul:v3"
    assert "model" not in changed
    connect.assert_not_awaited()


@pytest.mark.asyncio
async def test_http_payload_carries_only_the_parameters_v4_flash_supports():
    session = _FakeSession()
    service = _http_service(session, pitch=0.3, loudness=1.5, temperature=0.9)

    payload = await _synthesize(service)

    assert payload["model"] == V4_FLASH
    assert payload["speaker"] == "shubh_en_narration_gentle"
    assert payload["pitch"] == 0.3
    assert payload["loudness"] == 1.5
    assert "temperature" not in payload


@pytest.mark.asyncio
async def test_switching_the_http_service_to_v4_flash_takes_its_capabilities():
    """`run_tts` builds the payload from the model config, so the switch has to swap it."""
    session = _FakeSession()
    service = _http_service(session, model="bulbul:v3", voice=None)

    await service._update_settings(
        SarvamHttpTTSService.Settings(model=V4_FLASH, pitch=0.3, loudness=1.5)
    )
    payload = await _synthesize(service)

    assert payload["model"] == V4_FLASH
    assert payload["speaker"] == "shubh_en_narration_gentle"
    assert payload["pitch"] == 0.3
    assert "temperature" not in payload


class TestSpeakableTextFilter(unittest.IsolatedAsyncioTestCase):
    """Sarvam rejects requests holding no target-language character."""

    async def test_punctuation_only_text_is_emptied(self):
        filter = _SpeakableTextFilter()

        # A doubled full stop is split by sentence aggregation, stranding this.
        self.assertEqual(await filter.filter("."), "")
        self.assertEqual(await filter.filter(".."), "")
        self.assertEqual(await filter.filter("…"), "")
        self.assertEqual(await filter.filter(" -- "), "")
        self.assertEqual(await filter.filter("🙂"), "")

    async def test_speakable_text_passes_through_unchanged(self):
        filter = _SpeakableTextFilter()

        for text in (
            "hamare teeno models alag-alag power requirements ke liye bane hain.",
            "दूसरी तरफ, इलेक्ट्रिक मोटर में सीधे बैटरी से ताकत मिलती है।",
            "X45",
            "123",
        ):
            self.assertEqual(await filter.filter(text), text)

    async def test_script_is_not_assumed_to_be_latin(self):
        """Devanagari, Han and Cyrillic are speakable exactly as Latin is."""
        filter = _SpeakableTextFilter()

        for text in ("हाँ", "你好", "да"):
            self.assertEqual(await filter.filter(text), text)


@pytest.mark.asyncio
@pytest.mark.parametrize("use_http", [False, True], ids=["websocket", "http"])
@pytest.mark.parametrize("pace,expected", [(0.3, 0.5), (2.8, 2.0)])
async def test_model_switch_reapplies_pace_and_required_preprocessing(
    monkeypatch, use_http, pace, expected
):
    """A valid v2 setting must not become an invalid v4 request after a model-only change."""
    service = (
        _http_service(_FakeSession(), model="bulbul:v2", pace=pace, enable_preprocessing=False)
        if use_http
        else _service(model="bulbul:v2", pace=pace, enable_preprocessing=False)
    )
    if not use_http:
        monkeypatch.setattr(service, "_disconnect", AsyncMock())
        monkeypatch.setattr(service, "_connect", AsyncMock())
    await service._update_settings(service.Settings(model=V4_FLASH, enable_preprocessing=False))
    assert service._settings.pace == expected
    assert service._settings.enable_preprocessing is True
    assert service._settings.voice == "shubh_en_narration_gentle"


@pytest.mark.asyncio
@pytest.mark.parametrize("use_http", [False, True], ids=["websocket", "http"])
async def test_model_switch_keeps_explicit_voice_even_when_value_is_unchanged(
    monkeypatch, use_http
):
    """Explicit caller intent is not equivalent to membership in the changed-value dictionary.

    A custom ID deliberately exercises provider-owned voice validation; this test
    makes no claim that a real vendor accepts it for either model.
    """
    voice = "custom-persona-kept-by-caller"
    service = (
        _http_service(_FakeSession(), model="bulbul:v2", voice=voice)
        if use_http
        else _service(model="bulbul:v2", voice=voice)
    )
    if not use_http:
        monkeypatch.setattr(service, "_disconnect", AsyncMock())
        monkeypatch.setattr(service, "_connect", AsyncMock())
    await service._update_settings(service.Settings(model=V4_FLASH, voice=voice))
    assert service._settings.voice == voice


@pytest.mark.asyncio
@pytest.mark.parametrize("use_http", [False, True], ids=["websocket", "http"])
async def test_explicit_null_voice_resolves_new_model_default(monkeypatch, use_http):
    """Null means resolve the selected model's speaker, never send null or the legacy default."""
    service = (
        _http_service(_FakeSession(), model="bulbul:v3")
        if use_http
        else _service(model="bulbul:v3")
    )
    if not use_http:
        monkeypatch.setattr(service, "_disconnect", AsyncMock())
        monkeypatch.setattr(service, "_connect", AsyncMock())
    await service._update_settings(service.Settings(model=V4_FLASH, voice=None))
    assert service._settings.voice == "shubh_en_narration_gentle"


@pytest.mark.asyncio
@pytest.mark.parametrize("use_http", [False, True], ids=["websocket", "http"])
@pytest.mark.parametrize("model", [None, "", "bulbul:v400", "模型/unknown"])
async def test_invalid_model_update_preserves_current_model_capabilities(
    monkeypatch, use_http, model
):
    """Unsupported/missing model values cannot replace an accepted model or open a new connection."""
    service = _http_service(_FakeSession()) if use_http else _service()
    connect = AsyncMock()
    if not use_http:
        monkeypatch.setattr(service, "_disconnect", AsyncMock())
        monkeypatch.setattr(service, "_connect", connect)
    config = service._config
    changed = await service._update_settings(service.Settings(model=model))
    assert service._settings.model == V4_FLASH
    assert service._config is config
    assert "model" not in changed
    connect.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("use_http", [False, True], ids=["websocket", "http"])
async def test_identical_model_and_voice_update_has_no_transport_effect(monkeypatch, use_http):
    """Repeating already accepted settings is a local no-op, with no duplicate reconnect/config request."""
    service = _http_service(_FakeSession()) if use_http else _service()
    connect, config = AsyncMock(), AsyncMock()
    if not use_http:
        monkeypatch.setattr(service, "_disconnect", AsyncMock())
        monkeypatch.setattr(service, "_connect", connect)
        monkeypatch.setattr(service, "_send_config", config)
    changed = await service._update_settings(
        service.Settings(model=V4_FLASH, voice=service._settings.voice)
    )
    assert changed == {}
    connect.assert_not_awaited()
    config.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("use_http", [False, True], ids=["websocket", "http"])
@pytest.mark.parametrize(
    "model,pitch_loudness,temperature",
    [("bulbul:v2", True, False), ("bulbul:v3", False, True), (V4_FLASH, True, False)],
)
@pytest.mark.parametrize("language,wire_language", [("en", "en-IN"), ("hi", "hi-IN")])
async def test_language_wire_field_matches_current_provider_contract(
    use_http, model, pitch_loudness, temperature, language, wire_language
):
    """Both native transports send language_code while retaining model-specific controls.

    Official REST/AsyncAPI and the released Sarvam SDK agree on this JSON key.
    Fake peers record payloads without provider requests. V2 remains a synthetic
    compatibility case; this test does not establish current provider acceptance.
    """
    from pipecat.transcriptions.language import Language

    settings = dict(
        model=model, language=Language(language), pitch=0.25, loudness=1.2, temperature=0.8
    )
    if use_http:
        service = _http_service(_FakeSession(), **settings)
        config = await _synthesize(service)
    else:
        service = _service(**settings)
        service._websocket = _FakeWebsocket()
        await service._send_config()
        config = _sent_config(service)
    assert config["language_code"] == wire_language
    assert "target_language_code" not in config
    assert config["model"] == model
    assert ("pitch" in config) is pitch_loudness
    assert ("loudness" in config) is pitch_loudness
    assert ("temperature" in config) is temperature


@pytest.mark.asyncio
async def test_legacy_language_alias_stays_public_but_never_becomes_a_wire_key():
    """The existing mapping alias still configures initial/runtime language without leaking to JSON."""
    from pipecat.transcriptions.language import Language

    settings = SarvamTTSService.Settings.from_mapping(
        {"model": V4_FLASH, "target_language_code": Language.HI}
    )
    assert settings.language == Language.HI
    assert "target_language_code" not in settings.extra
    service = SarvamTTSService(api_key="test-key", settings=settings)
    service._websocket = _FakeWebsocket()
    await service._send_config()
    initial = _sent_config(service)
    assert initial["language_code"] == "hi-IN"
    assert "target_language_code" not in initial

    changed = await service._update_settings(
        service.Settings.from_mapping({"target_language_code": Language.MR})
    )
    updated = _sent_config(service)
    assert changed == {"language": "hi-IN"}
    assert updated == {**initial, "language_code": "mr-IN"}
    assert service._settings.language == "mr-IN"


@pytest.mark.asyncio
@pytest.mark.parametrize("use_http", [False, True], ids=["websocket", "http"])
@pytest.mark.parametrize(
    "original,selected",
    [("bulbul:v2", V4_FLASH), ("bulbul:v3", V4_FLASH), (V4_FLASH, "bulbul:v3")],
)
async def test_model_switch_preserves_language_and_uses_only_current_wire_key(
    monkeypatch, use_http, original, selected
):
    """A model switch preserves configured language and emits the new model's exact payload shape."""
    from pipecat.transcriptions.language import Language

    if use_http:
        service = _http_service(_FakeSession(), model=original, language=Language.TA)
    else:
        service = _service(model=original, language=Language.TA)
        service._websocket = _FakeWebsocket()

        async def reconnect():
            """Replace only the fake connection and execute the real configuration serializer."""
            service._websocket = _FakeWebsocket()
            await service._send_config()

        monkeypatch.setattr(service, "_disconnect", AsyncMock())
        monkeypatch.setattr(service, "_connect", AsyncMock(side_effect=reconnect))
    await service._update_settings(service.Settings(model=selected))
    if use_http:
        config = await _synthesize(service)
    else:
        config = _sent_config(service)
        assert _url_model(service) == selected
        service._disconnect.assert_awaited_once()
        service._connect.assert_awaited_once()
    assert config["language_code"] == "ta-IN"
    assert "target_language_code" not in config
    assert config["model"] == selected
    assert config["speaker"] == ("shubh_en_narration_gentle" if selected == V4_FLASH else "shubh")
