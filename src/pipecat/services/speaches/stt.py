"""Speaches STT Service — uses the OpenAI-compatible /v1/audio/transcriptions endpoint.

Alternate endpoint credentials stay explicit: the unchanged local placeholder
is retained on omission, while None/empty refuses before native SDK creation.
"""

from dataclasses import dataclass

from pipecat.services.openai.stt import OpenAISTTService, OpenAISTTSettings


@dataclass
class SpeachesSTTSettings(OpenAISTTSettings):
    """Settings for Speaches STT service."""

    pass


class SpeachesSTTService(OpenAISTTService):
    """Speaches STT service using the OpenAI-compatible transcription endpoint.

    Speaches exposes ``/v1/audio/transcriptions`` with an OpenAI-compatible
    multipart request format, so we can reuse the segmented HTTP STT adapter
    instead of a custom websocket protocol.
    """

    _requires_api_key = True

    Settings = SpeachesSTTSettings

    def __init__(
        self,
        *,
        api_key: str = "none",
        base_url: str = "http://localhost:8000/v1",
        settings: SpeachesSTTSettings | None = None,
        **kwargs,
    ):
        """Initialize the Speaches STT service.

        Args:
            api_key: Explicit endpoint key. Omission keeps the existing "none"
                local placeholder; explicit None/empty cannot inherit an OpenAI key.
            base_url: Base URL for the Speaches API.
            settings: Optional STT settings.
            **kwargs: Additional arguments passed to parent.

        Raises:
            ValueError: An explicit None/empty key is rejected before SDK/client
                construction. No connection, retry or fallback is attempted.
        """
        # Never let this alternate endpoint inherit OPENAI_API_KEY through
        # the SDK. Other values keep the existing native validation contract.
        if api_key is None or api_key == "":
            raise ValueError("SpeachesSTTService requires an explicit non-empty api_key.")

        super().__init__(
            api_key=api_key,
            base_url=base_url,
            settings=settings,
            **kwargs,
        )
