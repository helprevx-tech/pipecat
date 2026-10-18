#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for AzureTTSService cross-thread audio delivery.

Azure's Speech SDK fires its synthesis callbacks from native (non-event-loop)
threads. Those callbacks must deliver to the awaiting ``run_tts`` getter even when
the event loop is otherwise idle — e.g. a headless pipeline with no output transport
pumping audio. A bare ``asyncio.Queue.put_nowait()`` from another thread does NOT
wake an idle selector, so the audio sits unread; the callbacks marshal onto the loop
with ``asyncio.run_coroutine_threadsafe(queue.put(...), self.get_event_loop())``.

Constructor regressions also require a real nonempty subscription-key string
before either transport retains Azure configuration or reaches SDK setup. SDK
spies wrap their actual native constructors; no authentication guard is mocked
and no provider connection or setup is performed.

These are deterministic regression tests: the getter is parked on an idle loop
*first*, then the callback is fired from a real thread. With the fix the getter wakes
in ~ms; without it the loop stays blocked in ``select()`` until the ``wait_for`` timer
(seconds), so the elapsed assertion fails (or ``wait_for`` raises ``TimeoutError``).
"""

import asyncio
import threading
import time
from unittest.mock import Mock

import pytest

pytest.importorskip("azure.cognitiveservices.speech")

from azure.cognitiveservices.speech import CancellationReason  # noqa: E402

import pipecat.services.azure.tts as azure_tts  # noqa: E402
from pipecat.services.azure.tts import AzureHttpTTSService, AzureTTSService  # noqa: E402

SSML_SERVICE_CLASSES = (AzureTTSService, AzureHttpTTSService)

# The thread fires after this delay, by which point the awaiting getter has parked
# the loop in select() — so only the callback itself can wake it.
_FIRE_DELAY = 0.1
# Generous wait so the buggy path blocks on the timer; tight bound so the fix (~ms
# after _FIRE_DELAY) passes while the bug (~_WAIT) fails.
_WAIT = 5.0
_MAX_DELIVERY = 1.0


def _make_service() -> AzureTTSService:
    svc = AzureTTSService(api_key="test-key", region="eastus")
    # The SDK callbacks call get_event_loop(); without a started pipeline there is no
    # task manager, so point it at the running test loop.
    loop = asyncio.get_running_loop()
    svc.get_event_loop = lambda: loop
    return svc


async def _assert_idle_loop_wakeup(get_coro, fire):
    """Park ``get_coro`` on an idle loop, then ``fire()`` the callback from a thread.

    Returns the value the getter received; asserts it arrived promptly (i.e. the
    cross-thread put woke the idle loop rather than waiting for the ``wait_for`` timer).
    """
    loop = asyncio.get_running_loop()
    task = asyncio.ensure_future(get_coro)
    t0 = loop.time()
    threading.Thread(target=lambda: (time.sleep(_FIRE_DELAY), fire())).start()
    result = await asyncio.wait_for(task, timeout=_WAIT)
    assert loop.time() - t0 < _MAX_DELIVERY
    return result


@pytest.mark.asyncio
async def test_synthesizing_audio_wakes_idle_loop():
    """Audio pushed from an SDK thread reaches a parked getter on an idle loop."""
    svc = _make_service()

    audio = b"\x00\x01" * 256
    evt = Mock()
    evt.result.audio_data = audio

    data = await _assert_idle_loop_wakeup(
        svc._audio_queue.get(), lambda: svc._handle_synthesizing(evt)
    )
    assert data == audio


@pytest.mark.asyncio
async def test_canceled_error_wakes_idle_loop():
    """A non-user cancellation delivers its error to a parked getter on an idle loop."""
    svc = _make_service()

    evt = Mock()
    evt.result.cancellation_details.reason = CancellationReason.Error
    evt.result.cancellation_details.error_details = "boom"

    item = await _assert_idle_loop_wakeup(svc._audio_queue.get(), lambda: svc._handle_canceled(evt))
    assert isinstance(item, Exception)
    assert "boom" in str(item)


@pytest.mark.asyncio
async def test_completion_sentinel_wakes_idle_loop():
    """The completion sentinel reaches the word-boundary getter on an idle loop.

    ``_handle_completed`` routes completion through the word-boundary queue, whose
    getter is the (loop-side) word-processor task; that handoff is also a cross-thread
    put and must wake an idle loop.
    """
    svc = _make_service()

    evt = Mock()
    evt.result.audio_duration = None  # skip duration bookkeeping

    item = await _assert_idle_loop_wakeup(
        svc._word_boundary_queue.get(), lambda: svc._handle_completed(evt)
    )
    assert item is None


@pytest.mark.parametrize("service_class", SSML_SERVICE_CLASSES)
def test_construct_ssml_default_has_no_lang_element(service_class):
    """Default settings emit no <lang> element anywhere in the output."""
    service = service_class(api_key="test-key", region="eastus")

    ssml = service._construct_ssml("Hello there.")

    assert "<lang" not in ssml
    assert ssml == (
        "<speak version='1.0' xml:lang='en-US' "
        "xmlns='http://www.w3.org/2001/10/synthesis' "
        "xmlns:mstts='http://www.w3.org/2001/mstts'>"
        "<voice name='en-US-SaraNeural'>"
        "<mstts:silence type='Sentenceboundary' value='20ms' />"
        "Hello there."
        "</voice></speak>"
    )


@pytest.mark.parametrize("service_class", SSML_SERVICE_CLASSES)
def test_construct_ssml_force_locale_false_matches_default(service_class):
    """force_locale=False explicitly must match the unset default byte-for-byte."""
    default_service = service_class(api_key="test-key", region="eastus")
    explicit_false_service = service_class(
        api_key="test-key",
        region="eastus",
        settings=service_class.Settings(force_locale=False),
    )

    assert default_service._construct_ssml(
        "Hello there."
    ) == explicit_false_service._construct_ssml("Hello there.")


@pytest.mark.parametrize("service_class", SSML_SERVICE_CLASSES)
def test_construct_ssml_force_locale_wraps_text_in_lang_element(service_class):
    """force_locale=True wraps the text in <lang xml:lang> inside <voice>."""
    service = service_class(
        api_key="test-key",
        region="eastus",
        settings=service_class.Settings(language="en-GB", force_locale=True),
    )

    ssml = service._construct_ssml("Hello there.")

    assert ssml == (
        "<speak version='1.0' xml:lang='en-GB' "
        "xmlns='http://www.w3.org/2001/10/synthesis' "
        "xmlns:mstts='http://www.w3.org/2001/mstts'>"
        "<voice name='en-US-SaraNeural'>"
        "<mstts:silence type='Sentenceboundary' value='20ms' />"
        "<lang xml:lang='en-GB'>"
        "Hello there."
        "</lang>"
        "</voice></speak>"
    )


@pytest.mark.parametrize("service_class", SSML_SERVICE_CLASSES)
def test_construct_ssml_force_locale_wraps_nested_style_and_prosody(service_class):
    """force_locale=True wraps the entire nested style/prosody/emphasis block,
    not just the raw text.
    """
    service = service_class(
        api_key="test-key",
        region="eastus",
        settings=service_class.Settings(
            language="en-GB",
            force_locale=True,
            style="cheerful",
            rate="slow",
            emphasis="strong",
        ),
    )

    ssml = service._construct_ssml("Hello there.")

    assert ssml == (
        "<speak version='1.0' xml:lang='en-GB' "
        "xmlns='http://www.w3.org/2001/10/synthesis' "
        "xmlns:mstts='http://www.w3.org/2001/mstts'>"
        "<voice name='en-US-SaraNeural'>"
        "<mstts:silence type='Sentenceboundary' value='20ms' />"
        "<lang xml:lang='en-GB'>"
        "<mstts:express-as style='cheerful'>"
        "<prosody rate='slow'>"
        "<emphasis level='strong'>"
        "Hello there."
        "</emphasis>"
        "</prosody>"
        "</mstts:express-as>"
        "</lang>"
        "</voice></speak>"
    )


@pytest.fixture
def sdk_constructor_spies(monkeypatch):
    """Observe real SDK constructor seams, never replace a native auth validator.

    Both actual service constructors must remain lazy. If they unexpectedly
    construct an SDK object, its real constructor still runs and the exact call
    assertion fails; no successful provider authentication is implied.
    """
    observed = {}
    for name in ("SpeechConfig", "SpeechSynthesizer"):
        observed[name] = Mock(wraps=getattr(azure_tts, name))
        monkeypatch.setattr(azure_tts, name, observed[name])
    return observed


@pytest.mark.parametrize("service_class", SSML_SERVICE_CLASSES)
@pytest.mark.parametrize(
    "api_key",
    [None, "", " ", "\t\n", True, False, 0, 1, 1.5, b"synthetic-private-byte-key", [], {}],
    ids=[
        "none",
        "empty",
        "space",
        "whitespace",
        "true",
        "false",
        "zero",
        "integer",
        "float",
        "bytes",
        "list",
        "object",
    ],
)
def test_both_tts_transports_reject_invalid_subscription_key_before_sdk_setup(
    service_class, api_key, sdk_constructor_spies
):
    """Invalid credentials have a fixed private error and no native SDK effect."""
    with pytest.raises(ValueError) as failure:
        service_class(api_key=api_key, region="eastus")
    assert str(failure.value) == "Azure TTS requires a nonempty subscription key"
    assert "synthetic-private" not in str(failure.value)
    for spy in sdk_constructor_spies.values():
        spy.assert_not_called()


@pytest.mark.parametrize("service_class", SSML_SERVICE_CLASSES)
def test_missing_required_subscription_key_keeps_the_native_signature_contract(
    service_class, sdk_constructor_spies
):
    """Omitting the required argument cannot become an environment/keyless fallback."""
    with pytest.raises(TypeError):
        service_class(region="eastus")
    for spy in sdk_constructor_spies.values():
        spy.assert_not_called()


@pytest.mark.parametrize("service_class", SSML_SERVICE_CLASSES)
@pytest.mark.parametrize("api_key", ["synthetic-valid-key", "  synthetic-preserve-raw-key  "])
@pytest.mark.parametrize(
    "endpoint",
    [{"region": "eastus"}, {"private_endpoint": "https://speech.example.test/"}],
    ids=["region", "private-endpoint"],
)
def test_valid_subscription_key_preserves_region_endpoint_and_settings_without_sdk_setup(
    service_class, api_key, endpoint, sdk_constructor_spies
):
    """Valid raw keys stay unchanged; exact settings and lazy SDK ownership remain."""
    service = service_class(
        api_key=api_key,
        **endpoint,
        settings=service_class.Settings(voice="en-US-SaraNeural", style="cheerful"),
    )
    assert service._api_key == api_key
    assert service._region == endpoint.get("region")
    assert service._private_endpoint == endpoint.get("private_endpoint")
    assert service._settings.voice == "en-US-SaraNeural"
    assert service._settings.style == "cheerful"
    assert service._speech_config is None
    assert service._speech_synthesizer is None
    for spy in sdk_constructor_spies.values():
        spy.assert_not_called()


@pytest.mark.parametrize("service_class", SSML_SERVICE_CLASSES)
def test_valid_key_does_not_bypass_existing_region_or_endpoint_requirement(
    service_class, sdk_constructor_spies
):
    """A subscription key alone cannot remove the separate native endpoint guard."""
    with pytest.raises(ValueError, match="Either 'region' or 'private_endpoint' must be provided"):
        service_class(api_key="synthetic-valid-key")
    for spy in sdk_constructor_spies.values():
        spy.assert_not_called()


@pytest.mark.parametrize("service_class", SSML_SERVICE_CLASSES)
def test_invalid_shared_azure_reinitialization_does_not_overwrite_retained_configuration(
    service_class, sdk_constructor_spies
):
    """The shared guard rejects before changing even an already-retained source."""
    service = service_class(api_key="synthetic-original-key", region="eastus")
    before = (
        service._api_key,
        service._region,
        service._private_endpoint,
        service._speech_synthesizer,
    )
    with pytest.raises(ValueError, match="Azure TTS requires a nonempty subscription key"):
        service._init_azure_base(api_key=None, region="westus")
    assert (
        service._api_key,
        service._region,
        service._private_endpoint,
        service._speech_synthesizer,
    ) == before
    for spy in sdk_constructor_spies.values():
        spy.assert_not_called()
