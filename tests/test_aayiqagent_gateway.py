# Copyright (c) 2024-2026, Daily
# SPDX-License-Identifier: BSD 2-Clause License

"""Owned gateway contract tests: all servers/providers are local and deterministic."""

import asyncio
import base64
import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from aiohttp import web
from websockets.asyncio.server import serve

from pipecat.frames.frames import InputAudioRawFrame, TranscriptionFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.aayiqagent.flux.stt import AayiqAgentFluxSTTService
from pipecat.services.aayiqagent.llm import AayiqAgentLLMService
from pipecat.services.aayiqagent.protocol import GatewayError, GatewayIdentity
from pipecat.services.aayiqagent.stt import AayiqAgentSTTService
from pipecat.services.aayiqagent.tts import AayiqAgentTTSService
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.workers.runner import WorkerRunner

KW = dict(
    invocation_token="local-test-invocation",
    request_id="run-trace",
    model="test-model",
    allow_insecure=True,
)


@asynccontextmanager
async def websocket_gateway(handler):
    async with serve(handler, "127.0.0.1", 0) as server:
        yield f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"


@asynccontextmanager
async def http_gateway(handler):
    app = web.Application()
    app.router.add_post("/api/v1/llm/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    try:
        yield f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
    finally:
        await runner.cleanup()


@pytest.mark.parametrize(
    "url",
    [
        "",
        "ftp://example.test",
        "https://user:pass@example.test",
        "https://example.test/?token=secret",
        "https://example.test/#token",
        "http://example.test",
        "https://example.test/arbitrary",
    ],
)
def test_endpoint_validation_rejects_implicit_or_unsafe_targets(url):
    with pytest.raises(ValueError):
        GatewayIdentity(url, "test", "request")


def test_credentials_cannot_be_empty_or_injected_and_route_join_is_exact():
    for token in ["", "  ", "abc\r\nInjected: value"]:
        with pytest.raises(ValueError):
            GatewayIdentity("https://gateway.test", token, "request")
    identity = GatewayIdentity("https://gateway.test/api/v1/", "secret", "request")
    assert identity.url("stt/stream", websocket=True) == "wss://gateway.test/api/v1/stt/stream"
    assert "secret" not in repr(identity)


@pytest.mark.asyncio
async def test_stt_readiness_gates_audio_and_auth_is_only_in_headers():
    configured, release, audio_received = asyncio.Event(), asyncio.Event(), asyncio.Event()
    observations = {}

    async def handler(ws):
        observations["headers"] = ws.request.headers
        observations["path"] = ws.request.path
        observations["config"] = json.loads(await ws.recv())
        configured.set()
        await release.wait()
        await ws.send(json.dumps({"type": "ready"}))
        observations["audio"] = await ws.recv()
        audio_received.set()
        await ws.wait_closed()

    async with websocket_gateway(handler) as url:
        service = AayiqAgentSTTService(gateway_base_url=url, **KW)
        service._sample_rate = 16000
        connection = asyncio.create_task(service._connect_websocket())
        await configured.wait()
        with pytest.raises(GatewayError):
            await anext(service.run_stt(b"\x00\x00"))
        assert not connection.done()
        release.set()
        await connection
        await anext(service.run_stt(b"\x00\x00"))
        await audio_received.wait()
        await service._disconnect_websocket()
    assert observations["audio"] == b"\x00\x00"
    assert observations["headers"]["Authorization"] == "Bearer local-test-invocation"
    assert observations["path"] == "/api/v1/stt/stream"
    assert "invocation_token" not in observations["config"]
    request_id = observations["config"]["request_id"]
    assert UUID(request_id)
    assert observations["headers"]["X-AayiqAgent-Request-Id"] == request_id


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["quota", "timeout", "malformed"])
async def test_rejected_or_unready_connection_is_closed_and_emits_once(failure):
    async def handler(ws):
        await ws.recv()
        if failure == "quota":
            await ws.send(
                json.dumps(
                    {
                        "type": "error",
                        "code": "quota_exceeded",
                        "retryable": True,
                        "message": "sensitive-provider-detail",
                    }
                )
            )
        elif failure == "malformed":
            await ws.send("[1,2,3]")
        await ws.wait_closed()

    async with websocket_gateway(handler) as url:
        service = AayiqAgentSTTService(gateway_base_url=url, ready_timeout=0.03, **KW)
        service.push_error = AsyncMock()
        with pytest.raises(GatewayError):
            await service._connect_websocket()
        assert service._websocket is None
        assert service._gateway_ready is False
        service.push_error.assert_awaited_once()
        assert service.push_error.await_args.kwargs["force_treat_as_permanent"] is True
        assert "sensitive" not in str(service.push_error.await_args)


@pytest.mark.asyncio
async def test_empty_finalize_ack_does_not_mark_next_utterance_finalized():
    service = AayiqAgentSTTService(gateway_base_url="https://gateway.test", **KW)
    service.push_frame = AsyncMock()
    service.request_finalize()
    await service._handle_transcription(
        {"type": "transcription", "text": "", "is_final": True, "from_finalize": True}
    )
    assert not service._finalize_requested
    assert not service._finalize_pending
    service.push_frame.assert_not_awaited()


@pytest.mark.asyncio
async def test_real_stt_pipeline_delivers_transcript_and_cleans_tasks():
    async def handler(ws):
        config = json.loads(await ws.recv())
        assert config["sample_rate"] == 16000
        await ws.send(json.dumps({"type": "ready"}))
        async for message in ws:
            if isinstance(message, bytes):
                await ws.send(
                    json.dumps(
                        {
                            "type": "transcription",
                            "text": "hello",
                            "is_final": True,
                            "language": "en",
                        }
                    )
                )

    async with websocket_gateway(handler) as url:
        service = AayiqAgentSTTService(gateway_base_url=url, **KW)
        down, _ = await asyncio.wait_for(
            run_test(
                service,
                frames_to_send=[
                    InputAudioRawFrame(b"\x00\x00" * 1600, 16000, 1),
                    SleepFrame(sleep=0.05),
                ],
            ),
            timeout=5,
        )
    assert any(isinstance(frame, TranscriptionFrame) and frame.text == "hello" for frame in down)
    assert service._receive_task is None
    assert service._protocol_keepalive_task is None
    assert service._websocket is None


@pytest.mark.asyncio
async def test_real_worker_cancels_on_gateway_authorization_failure():
    async def handler(ws):
        await ws.recv()
        await ws.send(json.dumps({"type": "error", "code": "unauthorized", "retryable": True}))
        await ws.wait_closed()

    async with websocket_gateway(handler) as url:
        service = AayiqAgentSTTService(gateway_base_url=url, **KW)
        worker = PipelineWorker(
            Pipeline([service]),
            enable_rtvi=False,
            cancel_on_idle_timeout=False,
            processor_unusable_policy=ProcessorUnusablePolicy.CANCEL,
        )
        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(worker)
        await asyncio.wait_for(runner.run(), timeout=5)
    assert not service.is_usable
    assert service._websocket is None


@pytest.mark.asyncio
async def test_llm_stream_and_one_shot_have_distinct_admissions_and_schema():
    requests = []

    async def handler(request):
        body = await request.json()
        requests.append((body, request.headers["X-AayiqAgent-Request-Id"]))
        assert request.headers["Authorization"] == "Bearer local-test-invocation"
        if body["stream"]:
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            chunk = {
                "id": "test",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "test-model",
                "choices": [{"index": 0, "delta": {"content": "hello"}, "finish_reason": None}],
            }
            await response.write(("data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n").encode())
            await response.write_eof()
            return response
        return web.json_response(
            {
                "id": "test",
                "object": "chat.completion",
                "created": 1,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": '{"ok":true}'},
                        "finish_reason": "stop",
                    }
                ],
            }
        )

    async with http_gateway(handler) as url:
        service = AayiqAgentLLMService(gateway_base_url=url, **KW)
        context = LLMContext([{"role": "user", "content": "hello"}])
        stream = await service.get_chat_completions(context)
        assert [chunk.choices[0].delta.content async for chunk in stream] == ["hello"]
        answer = await service.run_inference(
            context,
            response_schema={
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": False,
            },
        )
        assert answer == '{"ok":true}'
        assert service._client.max_retries == 0
        await service._client.close()
    assert requests[0][1] != requests[1][1]
    for body, request_id in requests:
        assert UUID(request_id)
        assert body["metadata"]["request_id"] == request_id
    assert requests[1][0]["response_format"]["type"] == "json_schema"


@pytest.mark.asyncio
async def test_llm_quota_error_is_sanitized_and_single_report_for_each_path():
    async def handler(request):
        return web.json_response(
            {"error": {"code": "quota_exceeded", "retryable": True, "message": "secret detail"}},
            status=403,
        )

    async with http_gateway(handler) as url:
        service = AayiqAgentLLMService(gateway_base_url=url, **KW)
        service.push_error = AsyncMock()
        context = LLMContext([{"role": "user", "content": "hello"}])
        stream = await service.get_chat_completions(context)
        assert [chunk async for chunk in stream] == []
        service.push_error.assert_awaited_once()
        with pytest.raises(GatewayError, match="quota_exceeded"):
            await service.run_inference(context)
        assert service.push_error.await_count == 2
        assert all(
            call.kwargs["force_treat_as_permanent"] for call in service.push_error.await_args_list
        )
        assert "secret detail" not in str(service.push_error.await_args_list)
        await service._client.close()


@pytest.mark.asyncio
async def test_tts_context_alignment_terminal_and_late_audio_are_isolated():
    service = AayiqAgentTTSService(gateway_base_url="https://gateway.test", voice="voice", **KW)
    service._remote_contexts = {"one": False, "two": False}
    service.audio_context_available = lambda context_id: True
    service.add_word_timestamps = AsyncMock()
    service.append_to_audio_context = AsyncMock()
    service.remove_audio_context = AsyncMock()
    for context_id, times in [("one", [100, 300]), ("two", [0, 40]), ("one", [700])]:
        await service._handle_gateway_message(
            {
                "type": "alignment",
                "context_id": context_id,
                "data": {"words": [{"word": "x", "start": n} for n in times]},
            }
        )
    assert service.add_word_timestamps.await_args_list[-1].args == ([("x", 0.7)], "one")
    assert service.add_word_timestamps.await_args_list[1].args == ([("x", 0), ("x", 0.04)], "two")
    await service._handle_gateway_message({"type": "final", "context_id": "one"})
    service.remove_audio_context.assert_awaited_once_with("one")
    assert "one" not in service._alignment_times
    await service._handle_gateway_message({"type": "audio", "context_id": "one", "audio": "AAA="})
    service.append_to_audio_context.assert_not_awaited()
    with pytest.raises(GatewayError):
        await service._handle_gateway_message({"type": "audio", "audio": "AAA="})


@pytest.mark.asyncio
async def test_tts_protocol_context_cancel_flush_and_audio():
    messages = []
    cancelled = asyncio.Event()

    async def handler(ws):
        messages.append(json.loads(await ws.recv()))
        await ws.send(json.dumps({"type": "ready"}))
        async for raw in ws:
            message = json.loads(raw)
            messages.append(message)
            if message["type"] == "cancel":
                cancelled.set()
            if message["type"] == "synthesize":
                await ws.send(
                    json.dumps(
                        {
                            "type": "audio",
                            "context_id": message["context_id"],
                            "audio": base64.b64encode(b"\x00\x00").decode(),
                        }
                    )
                )

    async with websocket_gateway(handler) as url:
        service = AayiqAgentTTSService(gateway_base_url=url, voice="voice", **KW)
        service._sample_rate = 24000
        # Keep the real wire and contexts; isolate playback task setup for this transport test.
        service.audio_context_available = lambda context_id: True
        await service._connect_websocket()
        assert [frame async for frame in service.run_tts("hello", "one")]
        await service.flush_audio("one")
        with pytest.raises(GatewayError):
            await anext(service.run_tts("too late", "one"))
        await service._finish_context("one")  # idempotent after terminal flush
        await service.on_audio_context_interrupted("one")
        assert not service._remote_contexts
        await asyncio.wait_for(cancelled.wait(), 1)
        await service._disconnect_websocket()
    assert [m["type"] for m in messages] == [
        "config",
        "create_context",
        "synthesize",
        "flush",
        "cancel",
    ]


@pytest.mark.asyncio
async def test_flux_native_connected_query_and_configure():
    observed = {}

    async def handler(ws):
        observed["path"] = ws.request.path
        observed["headers"] = ws.request.headers
        await ws.send(json.dumps({"type": "Connected", "request_id": "provider-request"}))
        observed["configure"] = json.loads(await ws.recv())
        await ws.send(json.dumps({"type": "ConfigureSuccess"}))
        await ws.wait_closed()

    async with websocket_gateway(handler) as url:
        service = AayiqAgentFluxSTTService(gateway_base_url=url, **KW)
        service._sample_rate = 16000
        await service._connect_websocket()
        assert service._connection_established_event.is_set()
        await service._transport_send_json({"type": "Configure", "keyterms": ["AayiqAgent"]})
        assert json.loads(await service._websocket.recv())["type"] == "ConfigureSuccess"
        await service._disconnect_websocket()
    assert observed["configure"]["keyterms"] == ["AayiqAgent"]
    assert "request_id" not in observed["path"]
    assert "invocation" not in observed["path"]
    assert "encoding=linear16" in observed["path"]
    assert UUID(observed["headers"]["X-AayiqAgent-Request-Id"])


@pytest.mark.asyncio
@pytest.mark.parametrize("split_sample", [False, True])
async def test_real_tts_pipeline_emits_audio_alignment_and_terminal_cleanup(split_sample):
    from pipecat.frames.frames import (
        BotStoppedSpeakingFrame,
        TTSAudioRawFrame,
        TTSSpeakFrame,
        TTSTextFrame,
    )

    contexts = set()

    async def handler(ws):
        await ws.recv()
        await ws.send(json.dumps({"type": "ready"}))
        async for raw in ws:
            message = json.loads(raw)
            context_id = message.get("context_id")
            if message["type"] == "create_context":
                contexts.add(context_id)
            elif message["type"] == "synthesize":
                assert context_id in contexts
                await ws.send(
                    json.dumps(
                        {
                            "type": "alignment",
                            "context_id": context_id,
                            "data": {"words": [{"word": "hello", "start": 0}]},
                        }
                    )
                )
                audio = b"\x00\x00" * 480
                chunks = [audio[:1], audio[1:]] if split_sample else [audio]
                for chunk in chunks:
                    await ws.send(
                        json.dumps(
                            {
                                "type": "audio",
                                "context_id": context_id,
                                "audio": base64.b64encode(chunk).decode(),
                            }
                        )
                    )
            elif message["type"] in {"flush", "close_context"}:
                contexts.discard(context_id)
                await ws.send(json.dumps({"type": "final", "context_id": context_id}))

    async with websocket_gateway(handler) as url:
        service = AayiqAgentTTSService(gateway_base_url=url, voice="voice", **KW)
        down, _ = await asyncio.wait_for(
            run_test(
                service,
                frames_to_send=[
                    TTSSpeakFrame("hello"),
                    SleepFrame(sleep=0.15),
                    BotStoppedSpeakingFrame(),
                ],
            ),
            timeout=5,
        )
    assert any(isinstance(frame, TTSAudioRawFrame) and len(frame.audio) == 960 for frame in down)
    assert any(isinstance(frame, TTSTextFrame) and "hello" in frame.text for frame in down)
    assert not contexts
    assert not service._remote_contexts
    assert not service._alignment_times
    assert service._receive_task is None
    assert service._protocol_keepalive_task is None
    assert service._websocket is None


@pytest.mark.asyncio
async def test_real_flux_pipeline_preserves_partial_and_final_transcriptions():
    from pipecat.frames.frames import InterimTranscriptionFrame

    async def handler(ws):
        await ws.send(json.dumps({"type": "Connected"}))
        async for raw in ws:
            if isinstance(raw, bytes):
                for event in ["StartOfTurn", "Update", "EndOfTurn"]:
                    await ws.send(
                        json.dumps(
                            {
                                "type": "TurnInfo",
                                "event": event,
                                "transcript": "hello there",
                                "turn_index": 0,
                                "audio_window_start": 0,
                                "audio_window_end": 0.1,
                                "end_of_turn_confidence": 0.99,
                            }
                        )
                    )

    async with websocket_gateway(handler) as url:
        service = AayiqAgentFluxSTTService(gateway_base_url=url, **KW)
        down, _ = await asyncio.wait_for(
            run_test(
                service,
                frames_to_send=[
                    InputAudioRawFrame(b"\x00\x00" * 1600, 16000, 1),
                    SleepFrame(sleep=0.1),
                ],
            ),
            timeout=5,
        )
    assert any(isinstance(frame, InterimTranscriptionFrame) for frame in down)
    assert any(
        isinstance(frame, TranscriptionFrame) and frame.text == "hello there" for frame in down
    )
    assert service._watchdog_task is None
    assert service._receive_task is None
    assert service._websocket is None


@pytest.mark.asyncio
async def test_llm_stream_error_after_partial_output_does_not_retry_or_leak_details():
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        chunk = {
            "id": "test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "test-model",
            "choices": [{"index": 0, "delta": {"content": "partial"}, "finish_reason": None}],
        }
        error = {
            "error": {
                "code": "provider_unavailable",
                "message": "private diagnostic",
                "retryable": True,
            }
        }
        await response.write(
            ("data: " + json.dumps(chunk) + "\n\ndata: " + json.dumps(error) + "\n\n").encode()
        )
        await response.write_eof()
        return response

    async with http_gateway(handler) as url:
        service = AayiqAgentLLMService(gateway_base_url=url, **KW)
        service.push_error = AsyncMock()
        stream = await service.get_chat_completions(
            LLMContext([{"role": "user", "content": "hello"}])
        )
        assert [chunk.choices[0].delta.content async for chunk in stream] == ["partial"]
        service.push_error.assert_awaited_once()
        assert "private diagnostic" not in str(service.push_error.await_args)
        assert service.push_error.await_args.kwargs["force_treat_as_permanent"] is False
        assert calls == 1
        await service.cleanup()
        assert service._client.is_closed()


@pytest.mark.parametrize("code", [{"bad": "shape"}, ["quota_exceeded"], None, "unknown"])
def test_unknown_error_shape_fails_closed_without_echoing_data(code):
    error = GatewayError.from_message({"code": code, "retryable": True, "message": "secret"})
    assert error.code == "protocol_error"
    assert error.permanent
    assert not error.retryable
    assert "secret" not in str(error)


@pytest.mark.parametrize(
    "settings",
    [
        AayiqAgentTTSService.Settings(language="en"),
        AayiqAgentTTSService.Settings(speed=1.7),
        AayiqAgentTTSService.Settings(extra={"pitch": 0.2}),
    ],
)
def test_unsupported_tts_settings_are_rejected_before_connect(settings):
    with pytest.raises(ValueError):
        AayiqAgentTTSService(
            gateway_base_url="https://gateway.test", voice="voice", settings=settings, **KW
        )


@pytest.mark.asyncio
async def test_large_pcm_frame_is_split_without_losing_or_repeating_samples():
    payload = b"\x00\x01" * 40000
    received = []
    complete = asyncio.Event()

    async def handler(ws):
        await ws.recv()
        await ws.send(json.dumps({"type": "ready"}))
        async for frame in ws:
            if isinstance(frame, bytes):
                received.append(frame)
                if sum(map(len, received)) == len(payload):
                    complete.set()

    async with websocket_gateway(handler) as url:
        service = AayiqAgentSTTService(gateway_base_url=url, **KW)
        service._sample_rate = 16000
        await service._connect_websocket()
        await anext(service.run_stt(payload))
        await asyncio.wait_for(complete.wait(), 1)
        await service._disconnect_websocket()
    assert b"".join(received) == payload
    assert all(len(frame) <= 65536 and len(frame) % 2 == 0 for frame in received)
