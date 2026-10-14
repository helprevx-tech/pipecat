# Copyright (c) 2024-2026, Daily
# SPDX-License-Identifier: BSD 2-Clause License

"""OpenAI-compatible completions through an owned invocation gateway."""

from uuid import uuid4

from openai import APIError, APIStatusError

from pipecat.services.aayiqagent.protocol import GatewayError, GatewayIdentity, required_text
from pipecat.services.openai.llm import OpenAILLMService


class AayiqAgentLLMService(OpenAILLMService):
    """Streaming and one-shot completions using scoped Bearer authorization.

    The operator gateway decides which model settings and schemas are supported.
    Automatic SDK and timeout retries are disabled: another attempt needs its own
    admission and cost attribution. The client never resubmits partial output.
    """

    supports_developer_role = False

    def __init__(
        self,
        *,
        gateway_base_url: str,
        invocation_token: str,
        request_id: str,
        model: str,
        allow_insecure: bool = False,
        settings=None,
        **kwargs,
    ):
        """Configure the explicit endpoint, authorized model and invocation credential."""
        self._gateway = GatewayIdentity(
            gateway_base_url, invocation_token, request_id, allow_insecure
        )
        required_text(model, "model")
        if settings is not None and settings.given_fields().get("model", model) != model:
            raise ValueError("settings.model must match the authorized model")
        for key in ("api_key", "base_url", "default_headers", "retry_on_timeout"):
            if key in kwargs:
                raise ValueError(f"{key} is controlled by the invocation gateway")
        effective_settings = self.Settings(model=model)
        if settings is not None:
            effective_settings.apply_update(settings)
        super().__init__(
            api_key=invocation_token,
            base_url=self._gateway.url("llm"),
            settings=effective_settings,
            retry_on_timeout=False,
            default_headers=self._gateway.headers(),
            **kwargs,
        )
        self._client.max_retries = 0

    def build_chat_completion_params(self, params_from_context: dict) -> dict:
        """Allocate a distinct metered request ID for each completion attempt."""
        params = super().build_chat_completion_params(params_from_context)
        request_id = str(uuid4())
        params["extra_headers"] = {
            **self._gateway.headers(),
            "X-AayiqAgent-Request-Id": request_id,
            "X-AayiqAgent-Trace-Id": self._gateway.request_id,
        }
        params["metadata"] = {
            **(params.get("metadata") or {}),
            "request_id": request_id,
            "protocol_version": "1",
        }
        # Avoid custom extras overriding the authentication-bound attribution.
        extra_body = dict(params.get("extra_body") or {})
        extra_body.pop("metadata", None)
        if extra_body:
            params["extra_body"] = extra_body
        elif "extra_body" in params:
            del params["extra_body"]
        return params

    def _classify_error(self, exception):
        if isinstance(exception, GatewayError):
            return exception.category
        return super()._classify_error(exception)

    async def _gateway_http_error(self, exception: APIError) -> GatewayError:
        body = exception.body if isinstance(exception.body, dict) else {}
        body = body.get("error", body)
        error = (
            GatewayError.from_message(body)
            if isinstance(body, dict)
            else GatewayError("protocol_error")
        )
        status = getattr(exception, "status_code", None)
        if error.code == "protocol_error":
            code = {
                401: "unauthorized",
                403: "forbidden",
                402: "budget_exceeded",
                400: "invalid_request",
                404: "invalid_configuration",
                409: "invalid_request",
                422: "invalid_configuration",
                429: "rate_limited",
            }.get(status)
            if code is None and status is not None and status >= 500:
                code = "provider_unavailable"
            if code:
                error = GatewayError(
                    code, retryable=code in {"rate_limited", "provider_unavailable"}
                )
        await self.push_error(
            error_msg=str(error), exception=error, force_treat_as_permanent=error.permanent
        )
        return error

    async def get_chat_completions(self, context):
        """Stream completions with sanitized errors and deterministic socket cleanup."""
        try:
            stream = await super().get_chat_completions(context)

            async def guarded_stream():
                try:
                    async for chunk in stream:
                        yield chunk
                except APIError as exception:
                    await self._gateway_http_error(exception)
                finally:
                    await stream.close()

            return guarded_stream()
        except APIStatusError as exception:
            await self._gateway_http_error(exception)

            async def empty_stream():
                if False:
                    yield None

            return empty_stream()

    async def run_inference(
        self, context, max_tokens=None, system_instruction=None, response_schema=None
    ) -> str | None:
        """Run a one-shot completion, preserving optional strict JSON-schema output."""
        try:
            return await super().run_inference(
                context,
                max_tokens=max_tokens,
                system_instruction=system_instruction,
                response_schema=response_schema,
            )
        except APIStatusError as exception:
            error = await self._gateway_http_error(exception)
            raise error from None

    async def cleanup(self):
        """Release the invocation-scoped HTTP client after pipeline teardown."""
        try:
            await super().cleanup()
        finally:
            await self._client.close()
