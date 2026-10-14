# AayiqAgent model gateway clients

These clients require an operator-owned gateway implementing protocol version 1.
They do not provision inference, supply provider credentials, or route to a hosted
vendor by default. The server authenticates each invocation, selects providers,
enforces scope/budgets, and records authoritative usage. Framework metrics remain
observability, not authorization or billable proof.

All constructors require `gateway_base_url`, `invocation_token`, `request_id`,
and `model`. TTS also requires `voice`. Authorization must exist before setup:
Flux connects during `setup`, while generic STT/TTS connect after `StartFrame`.
Use HTTPS/WSS; `allow_insecure=True` explicitly permits an operator-controlled
local/private HTTP/WS endpoint. The URL must be a root URL or end in `/api/v1`.
The credential is sent only in a Bearer header. Each HTTP request and WebSocket
connection gets a fresh UUID admission ID, shared by its header and metadata or
config. Constructor `request_id` identifies the caller's invocation context;
it is not an authorization decision or a reusable metered-request ID.

| Client | Route | Supported settings |
| --- | --- | --- |
| `AayiqAgentLLMService` | `/api/v1/llm/chat/completions` | OpenAI-compatible model settings; gateway catalogue must allow the selected fields/model. Streaming, tools and one-shot JSON schema use the same authenticated path. |
| `AayiqAgentSTTService` | `/api/v1/stt/stream` | Model/language (updates reconnect); interim results, VAD proposals and keyterms are connection options. PCM16 mono. |
| `AayiqAgentFluxSTTService` | `/api/v1/stt/flux` | Native Flux thresholds, keyterms, numerals, profanity filter, redaction, confidence and multilingual language hints. Inherited Configure serialization/watchdog/turn handling is preserved. |
| `AayiqAgentTTSService` | `/api/v1/tts/stream` | Model/voice; optional catalogue-supported speed 0.6–1.5. PCM16 mono; language, pitch and volume are rejected. Settings cannot change while contexts are active. |

STT/TTS require `ready`; Flux requires native `Connected` before audio. Receive,
keepalive and watchdog tasks use the framework task manager. Incoming JSON and PCM buffers are limited to 2 MiB; outgoing control messages
and individual PCM wire frames are limited to 64 KiB. Larger PCM inputs are split
without replay. The WebSocket receive queue is bounded. A failed speech
connection becomes unusable, allowing `ProcessorUnusablePolicy.CANCEL` to end the
call. It never transparently replays partly processed audio. Replacing a failed
service requires a fresh, server-authorized attempt. HTTP SDK and timeout retries
are also disabled to prevent duplicate inference charges.

TTS alignment uses absolute context-relative start times in milliseconds on the
wire and seconds in Pipecat. Context IDs are mandatory. Final/cancel removes live
state, suppressing late audio and alignment even while queued audio drains. The
Cartesia-backed `flush` is terminal; further text needs a new context. `close_context`
is idempotent after flush, and `cancel` aborts provider work. Unsupported settings
must be rejected by the gateway rather than silently ignored.

`tests/test_aayiqagent_gateway.py` uses local HTTP/WebSocket fake providers and
real PipelineWorker execution. This validates client protocol/lifecycle behavior;
it does not certify deployed infrastructure, provider credentials, invoices,
real-provider voice quality, or end-to-end product billing.
