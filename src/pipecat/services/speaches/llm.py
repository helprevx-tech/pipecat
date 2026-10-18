"""Speaches LLM Service — uses an OpenAI-compatible /v1/chat/completions endpoint.

Alternate endpoint credentials stay explicit: the unchanged local placeholder
is retained on omission, while None/empty refuses before native SDK creation.
"""

from dataclasses import dataclass

from pipecat.services.openai.base_llm import OpenAILLMSettings
from pipecat.services.openai.llm import OpenAILLMService


@dataclass
class SpeachesLLMSettings(OpenAILLMSettings):
    """Settings for Speaches LLM service."""

    pass


class SpeachesLLMService(OpenAILLMService):
    """Speaches LLM service using an OpenAI-compatible chat completions endpoint."""

    Settings = SpeachesLLMSettings

    def __init__(
        self,
        *,
        api_key: str = "none",
        base_url: str = "http://localhost:11434/v1",
        settings: SpeachesLLMSettings | None = None,
        **kwargs,
    ):
        """Initialize the Speaches LLM service.

        Args:
            api_key: Explicit endpoint key. Omission keeps the existing "none"
                local placeholder; explicit None/empty cannot inherit an OpenAI key.
            base_url: Base URL of the Speaches-compatible endpoint.
            settings: Optional service settings.
            **kwargs: Additional arguments passed to parent.

        Raises:
            ValueError: An explicit None/empty key is rejected before SDK/client
                construction. No connection, retry or fallback is attempted.
        """
        # Never let this alternate endpoint inherit OPENAI_API_KEY through
        # the SDK. Other values keep the existing native validation contract.
        if api_key is None or api_key == "":
            raise ValueError("SpeachesLLMService requires an explicit non-empty api_key.")

        super().__init__(
            api_key=api_key,
            base_url=base_url,
            settings=settings,
            **kwargs,
        )
