#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Qualify output-tail hooks for every serializer used by current carrier factories.

Real stream resamplers and existing carrier wire encoders run locally. Automatic
hangup is disabled and no lifecycle command or provider REST request is issued.
Cloudonix is tested through its inherited Twilio implementation; Telnyx exercises
both supported output codecs. These checks establish local boundary preservation,
not carrier acceptance or remote playback/cancellation.
"""

import audioop
import base64
import json
import struct
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from starlette.websockets import WebSocketState

from pipecat.frames.frames import EndFrame, OutputAudioRawFrame, TTSStoppedFrame
from pipecat.serializers.asterisk import AsteriskFrameSerializer
from pipecat.serializers.cloudonix import CloudonixFrameSerializer
from pipecat.serializers.exotel import ExotelFrameSerializer
from pipecat.serializers.plivo import PlivoFrameSerializer
from pipecat.serializers.telnyx import TelnyxFrameSerializer
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.serializers.vobiz import VobizFrameSerializer
from pipecat.serializers.vonage import VonageFrameSerializer
from pipecat.transports.base_output import AudioOutputPolicy
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport

_SERIALIZERS = (
    "asterisk",
    "cloudonix",
    "exotel",
    "plivo",
    "telnyx-pcmu",
    "telnyx-pcma",
    "twilio",
    "vobiz",
    "vonage",
)


def _serializer(name):
    """Construct the actual maintained codec without secrets or automatic lifecycle I/O."""
    options = {"auto_hang_up": False, "resampler_clear_after_secs": None}
    if name == "asterisk":
        return AsteriskFrameSerializer(
            channel_id="synthetic-channel",
            ari_endpoint="http://invalid.example",
            app_name="synthetic-app",
            app_password="synthetic-placeholder",
            params=AsteriskFrameSerializer.InputParams(**options),
        )
    if name == "cloudonix":
        return CloudonixFrameSerializer(
            call_id="synthetic-call",
            stream_sid="synthetic-stream",
            params=TwilioFrameSerializer.InputParams(**options),
        )
    if name == "exotel":
        return ExotelFrameSerializer(
            stream_sid="synthetic-stream",
            params=ExotelFrameSerializer.InputParams(resampler_clear_after_secs=None),
        )
    if name == "plivo":
        return PlivoFrameSerializer(
            stream_id="synthetic-stream", params=PlivoFrameSerializer.InputParams(**options)
        )
    if name.startswith("telnyx-"):
        encoding = "PCMA" if name.endswith("pcma") else "PCMU"
        return TelnyxFrameSerializer(
            stream_id="synthetic-stream",
            inbound_encoding=encoding,
            outbound_encoding=encoding,
            params=TelnyxFrameSerializer.InputParams(**options),
        )
    if name == "twilio":
        return TwilioFrameSerializer(
            stream_sid="synthetic-stream", params=TwilioFrameSerializer.InputParams(**options)
        )
    if name == "vobiz":
        return VobizFrameSerializer(
            stream_id="synthetic-stream", params=VobizFrameSerializer.InputParams(**options)
        )
    if name == "vonage":
        return VonageFrameSerializer(
            call_uuid="synthetic-call", params=VonageFrameSerializer.InputParams(**options)
        )
    raise AssertionError("Unknown synthetic serializer case")


def _pcm(value, count=2048):
    """Produce deterministic mono signed PCM at the actual frame's declared source rate."""
    return value.to_bytes(2, "little", signed=True) * count


def _wire_audio(payload):
    """Read only audio from each real protocol, excluding metadata and wrapper differences."""
    if not payload:
        return b""
    if isinstance(payload, bytes):
        return payload
    return base64.b64decode(json.loads(payload)["media"]["payload"])


def _capture_output_pcm(serializer):
    """Observe maintained real resample/flush results without changing DSP or returned bytes.

    Output-tail hooks feed their flushed target-rate PCM through the existing
    serializer audio branch. Its passthrough resample is therefore recorded once
    among encoded PCM outputs; flush results are recorded separately, not counted
    twice. Wrappers live only on this test-owned serializer/resampler instance.
    """
    results = {"output": [], "flush": []}
    resample = serializer._output_resampler.resample
    flush = serializer._output_resampler.flush

    async def observe_resample(audio, in_rate, out_rate):
        """Return actual real-resampler PCM unchanged and retain its exact encoder input."""
        data = await resample(audio, in_rate, out_rate)
        results["output"].append(data)
        return data

    async def observe_flush():
        """Record real filter tail separately, without replacing or duplicating its encoding."""
        data = await flush()
        results["flush"].append(data)
        return data

    serializer._output_resampler.resample = observe_resample
    serializer._output_resampler.flush = observe_flush
    return results


def _assert_pcm_equal(actual, expected):
    """Require exact raw sample counts and only the proven independent int16 dither bound.

    python-soxr v1.0.0 uses libsoxr's default
    TPDF integer dither (<1 sample before rounding), so two equivalent int16
    streams can differ by at most two LSB. Sample counts remain exact and every
    signed sample is checked before companding. Negative controls prove this
    narrow bound rejects an actual unreset old tail for every factory codec.

    See https://github.com/dofuuz/python-soxr/blob/v1.0.0/src/soxr_ext.cpp#L76
    and https://github.com/chirlu/soxr/blob/master/src/rint-clip.h#L4.
    """
    assert len(actual) == len(expected)
    assert len(actual) % 2 == 0
    left = struct.iter_unpack("<h", actual)
    right = struct.iter_unpack("<h", expected)
    max_delta = max((abs(a[0] - b[0]) for a, b in zip(left, right, strict=True)), default=0)
    assert max_delta <= 2


def _assert_wire_matches_pcm(name, wire, pcm):
    """Require byte-exact wire encoding of each stream's own actually produced PCM.

    A tiny legitimate int16 dither difference can cross a G.711 quantization
    threshold, so an independently dithered stream is not an exact wire oracle.
    Here the oracle uses unchanged observed PCM and the maintained codec; no
    decoded-signal or encoded-byte tolerance is used. The separate raw comparison
    above validates sample preservation independently of this wire assertion.
    """
    if name in {"exotel", "vonage"}:
        expected = pcm
    elif name == "telnyx-pcma":
        expected = audioop.lin2alaw(pcm, 2)
    else:
        expected = audioop.lin2ulaw(pcm, 2)
    assert wire == expected


def _real_output(serializer):
    """Connect a real codec and native output/client to recorded in-memory socket sends."""
    payloads = []

    async def record(payload):
        """Record actual native socket-call acceptance without network traffic."""
        payloads.append(payload)

    socket = SimpleNamespace(
        client_state=WebSocketState.CONNECTED,
        application_state=WebSocketState.CONNECTED,
        send_bytes=AsyncMock(side_effect=record),
        send_text=AsyncMock(side_effect=record),
    )
    output = FastAPIWebsocketTransport(
        socket,
        FastAPIWebsocketParams(audio_out_enabled=True, serializer=serializer, allowed_origins=[]),
    ).output()
    allowed = {"old", "new"}

    def capture(frame):
        """Read only the original explicitly issued synthetic source."""
        return getattr(frame, "test_source", None)

    def copy(context, frame):
        """Retain original owner through actual native output frame rebuilding."""
        frame.test_source = context

    def current(context):
        """Model local eligibility without repairing an absent source."""
        return context in allowed

    def native(kind, destination):
        """This direct-write test issues no native ambient/lifecycle audio source."""
        return None

    output.set_audio_output_policy(AudioOutputPolicy(capture, copy, current, native))
    output._sample_rate = 24000
    output._write_audio_sleep = AsyncMock()
    return output, payloads, allowed


@pytest.mark.asyncio
@pytest.mark.parametrize("name", _SERIALIZERS)
async def test_real_output_tail_is_encoded_once_and_never_resets_inbound_state(name):
    """Every factory codec finishes real output tail exactly once while preserving input history."""
    serializer = _serializer(name)
    captured = _capture_output_pcm(serializer)
    assert serializer.supports_output_audio_boundary is True
    inbound = serializer._input_resampler
    await inbound.resample(_pcm(1300), 24000, 8000)
    first = await serializer.serialize(
        OutputAudioRawFrame(audio=_pcm(2100), sample_rate=24000, num_channels=1)
    )
    tail = await serializer.flush_output_audio()
    assert _wire_audio(first) + _wire_audio(tail)
    _assert_wire_matches_pcm(
        name, _wire_audio(first) + _wire_audio(tail), b"".join(captured["output"])
    )
    assert captured["flush"] and captured["flush"][0]
    assert _wire_audio(tail)
    assert await serializer.flush_output_audio() is None
    assert serializer._input_resampler is inbound
    assert await inbound.flush()
    if isinstance(tail, str):
        assert "_pipecat_audio_output_context" not in tail


@pytest.mark.asyncio
@pytest.mark.parametrize("name", _SERIALIZERS)
async def test_reset_real_old_output_matches_a_fresh_serializer_without_old_tail(name):
    """Discarded old filter samples cannot affect the next owner's real codec output."""
    serializer, fresh = _serializer(name), _serializer(name)
    actual_pcm, expected_pcm = _capture_output_pcm(serializer), _capture_output_pcm(fresh)
    await serializer.serialize(
        OutputAudioRawFrame(audio=_pcm(3000), sample_rate=24000, num_channels=1)
    )
    await serializer.reset_output_audio()
    actual_pcm["output"].clear()
    next_frame = OutputAudioRawFrame(audio=_pcm(-3000), sample_rate=24000, num_channels=1)
    actual = _wire_audio(await serializer.serialize(next_frame))
    actual += _wire_audio(await serializer.flush_output_audio())
    expected = _wire_audio(await fresh.serialize(next_frame))
    expected += _wire_audio(await fresh.flush_output_audio())
    _assert_pcm_equal(b"".join(actual_pcm["output"]), b"".join(expected_pcm["output"]))
    _assert_wire_matches_pcm(name, actual, b"".join(actual_pcm["output"]))
    _assert_wire_matches_pcm(name, expected, b"".join(expected_pcm["output"]))
    assert actual


@pytest.mark.asyncio
@pytest.mark.parametrize("name", _SERIALIZERS)
async def test_real_codec_and_websocket_drop_old_filter_tail_before_new_owner(name):
    """Actual output/client/codec integration cannot leak old resampler tail into new sends."""
    serializer, fresh = _serializer(name), _serializer(name)
    actual_pcm, expected_pcm = _capture_output_pcm(serializer), _capture_output_pcm(fresh)
    output, payloads, allowed = _real_output(serializer)
    old = OutputAudioRawFrame(audio=_pcm(7000, 96), sample_rate=24000, num_channels=1)
    old.test_source = "old"
    assert await output.write_audio_frame(old) is True
    accepted_before_change = len(payloads)
    actual_pcm["output"].clear()
    allowed.remove("old")
    new = OutputAudioRawFrame(audio=_pcm(-3000), sample_rate=24000, num_channels=1)
    new.test_source = "new"
    assert await output.write_audio_frame(new) is True
    assert await output._write_frame(EndFrame()) is True
    actual = b"".join(_wire_audio(payload) for payload in payloads[accepted_before_change:])
    expected = _wire_audio(await fresh.serialize(new))
    expected += _wire_audio(await fresh.flush_output_audio())
    _assert_pcm_equal(b"".join(actual_pcm["output"]), b"".join(expected_pcm["output"]))
    _assert_wire_matches_pcm(name, actual, b"".join(actual_pcm["output"]))
    _assert_wire_matches_pcm(name, expected, b"".join(expected_pcm["output"]))
    assert actual


@pytest.mark.asyncio
@pytest.mark.parametrize("name", _SERIALIZERS)
async def test_small_real_speech_tail_finishes_at_its_exact_tts_stop(name):
    """A short current utterance reaches the actual socket before its ordered TTS completion."""
    serializer, fresh = _serializer(name), _serializer(name)
    actual_pcm, expected_pcm = _capture_output_pcm(serializer), _capture_output_pcm(fresh)
    output, payloads, _ = _real_output(serializer)
    speech = OutputAudioRawFrame(audio=_pcm(2300, 96), sample_rate=24000, num_channels=1)
    speech.test_source = "old"
    speech.context_id = "synthetic-short-response"
    assert await output.write_audio_frame(speech) is True
    stop = TTSStoppedFrame(context_id="synthetic-short-response")
    stop.test_source = "old"
    assert await output.write_transport_frame(stop) is True
    actual = b"".join(_wire_audio(payload) for payload in payloads)
    expected = _wire_audio(await fresh.serialize(speech))
    expected += _wire_audio(await fresh.flush_output_audio())
    _assert_pcm_equal(b"".join(actual_pcm["output"]), b"".join(expected_pcm["output"]))
    _assert_wire_matches_pcm(name, actual, b"".join(actual_pcm["output"]))
    _assert_wire_matches_pcm(name, expected, b"".join(expected_pcm["output"]))
    assert actual_pcm["flush"] and actual_pcm["flush"][0]
    assert actual
    assert await serializer.flush_output_audio() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("name", _SERIALIZERS)
async def test_pcm_dither_bound_rejects_real_unreset_previous_owner_contamination(name):
    """The narrow PCM bound cannot hide old samples from a genuine unreset SOXR stream."""
    contaminated, fresh = _serializer(name), _serializer(name)
    leaked_pcm, expected_pcm = _capture_output_pcm(contaminated), _capture_output_pcm(fresh)
    await contaminated.serialize(
        OutputAudioRawFrame(audio=_pcm(7000), sample_rate=24000, num_channels=1)
    )
    leaked_pcm["output"].clear()
    new = OutputAudioRawFrame(audio=_pcm(-3000), sample_rate=24000, num_channels=1)
    leaked = _wire_audio(await contaminated.serialize(new))
    leaked += _wire_audio(await contaminated.flush_output_audio())
    expected = _wire_audio(await fresh.serialize(new))
    expected += _wire_audio(await fresh.flush_output_audio())
    actual_raw, expected_raw = b"".join(leaked_pcm["output"]), b"".join(expected_pcm["output"])
    _assert_wire_matches_pcm(name, leaked, actual_raw)
    _assert_wire_matches_pcm(name, expected, expected_raw)
    assert len(actual_raw) >= len(expected_raw)
    # Compare equal counts deliberately: rejection must detect actual stale
    # signal values, rather than succeeding solely because the tail is longer.
    with pytest.raises(AssertionError):
        _assert_pcm_equal(actual_raw[: len(expected_raw)], expected_raw)


@pytest.mark.asyncio
async def test_asterisk_guarded_packet_padding_uses_real_ulaw_silence():
    """G.711 padding matches the existing codec's zero-PCM encoding, rather than loud zero bytes."""
    serializer = _serializer("asterisk")
    wire = await serializer.serialize(
        OutputAudioRawFrame(audio=b"\x00\x00", sample_rate=8000, num_channels=1)
    )
    assert wire == serializer.output_audio_packet_padding == b"\xff"
