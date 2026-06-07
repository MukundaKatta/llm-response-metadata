"""Extract usage, model, and stop-reason metadata from LLM API responses.

:class:`ResponseMetadata` normalises the fields that differ between providers
(Anthropic, OpenAI, Gemini) into one flat dataclass.  Pass a raw response dict
from the API or SDK and get back structured metadata without taking on any SDK
dependency.

Example::

    from llm_response_metadata import ResponseMetadata

    # From an Anthropic messages.create() response (as dict)
    meta = ResponseMetadata.from_anthropic({
        "id": "msg_01abc",
        "model": "claude-sonnet-4-5",
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 1024,
            "output_tokens": 256,
            "cache_creation_input_tokens": 512,
            "cache_read_input_tokens": 128,
        },
    })
    print(meta.total_tokens)  # 1280  (input + output only)
    print(meta.cache_savings_tokens)  # 128

    # From an OpenAI chat.completions.create() response (as dict)
    meta2 = ResponseMetadata.from_openai({
        "id": "chatcmpl-xyz",
        "model": "gpt-5.4",
        "choices": [{"finish_reason": "stop"}],
        "usage": {"prompt_tokens": 800, "completion_tokens": 200, "total_tokens": 1000},
    })
    print(meta2.stop_reason)  # 'stop'
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Provider(str, Enum):
    """Known LLM provider."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"
    UNKNOWN = "unknown"


@dataclass
class TokenUsage:
    """Token usage breakdown.

    Attributes:
        input_tokens:                 Tokens in the prompt / input.
        output_tokens:                Tokens in the completion / output.
        cache_creation_input_tokens:  Tokens written to Anthropic prompt cache.
        cache_read_input_tokens:      Tokens read from Anthropic prompt cache.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Sum of input and output tokens."""
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-serialisable dict."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TokenUsage:
        """Reconstruct from a plain dict."""
        return cls(
            input_tokens=int(data.get("input_tokens", 0)),
            output_tokens=int(data.get("output_tokens", 0)),
            cache_creation_input_tokens=int(data.get("cache_creation_input_tokens", 0)),
            cache_read_input_tokens=int(data.get("cache_read_input_tokens", 0)),
        )

    def __repr__(self) -> str:
        return (
            f"TokenUsage(input={self.input_tokens}, output={self.output_tokens},"
            f" cache_create={self.cache_creation_input_tokens},"
            f" cache_read={self.cache_read_input_tokens})"
        )


@dataclass
class ResponseMetadata:
    """Normalised metadata extracted from an LLM API response.

    Attributes:
        provider:    Which provider produced the response.
        model:       Model identifier string.
        response_id: Provider-assigned response ID (empty string if absent).
        stop_reason: Why the generation stopped (e.g. ``"end_turn"``, ``"stop"``).
        usage:       Token usage breakdown.
        latency_ms:  Optional round-trip latency in milliseconds.
        raw:         The original response dict, preserved for inspection.
    """

    provider: Provider = Provider.UNKNOWN
    model: str = ""
    response_id: str = ""
    stop_reason: str = ""
    usage: TokenUsage = field(default_factory=TokenUsage)
    latency_ms: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_tokens(self) -> int:
        """Sum of input and output tokens."""
        return self.usage.total_tokens

    @property
    def cache_savings_tokens(self) -> int:
        """Tokens served from the Anthropic prompt cache."""
        return self.usage.cache_read_input_tokens

    @property
    def is_complete(self) -> bool:
        """``True`` if the response indicates a normal (non-truncated) stop."""
        return self.stop_reason in {"end_turn", "stop", "stop_sequence"}

    @property
    def was_truncated(self) -> bool:
        """``True`` if the response was cut short by a length limit."""
        return self.stop_reason in {"max_tokens", "length"}

    # ------------------------------------------------------------------
    # Factories — provider-specific
    # ------------------------------------------------------------------

    @classmethod
    def from_anthropic(
        cls,
        response: dict[str, Any],
        *,
        latency_ms: float | None = None,
    ) -> ResponseMetadata:
        """Build from an Anthropic ``messages.create()`` response dict.

        Supports both direct API dicts and SDK object ``.__dict__`` dumps
        that have been JSON-serialised.
        """
        usage_raw = response.get("usage") or {}
        usage = TokenUsage(
            input_tokens=int(usage_raw.get("input_tokens", 0)),
            output_tokens=int(usage_raw.get("output_tokens", 0)),
            cache_creation_input_tokens=int(
                usage_raw.get("cache_creation_input_tokens", 0)
            ),
            cache_read_input_tokens=int(usage_raw.get("cache_read_input_tokens", 0)),
        )
        return cls(
            provider=Provider.ANTHROPIC,
            model=str(response.get("model", "")),
            response_id=str(response.get("id", "")),
            stop_reason=str(response.get("stop_reason", "")),
            usage=usage,
            latency_ms=latency_ms,
            raw=response,
        )

    @classmethod
    def from_openai(
        cls,
        response: dict[str, Any],
        *,
        latency_ms: float | None = None,
    ) -> ResponseMetadata:
        """Build from an OpenAI ``chat.completions.create()`` response dict."""
        usage_raw = response.get("usage") or {}
        # OpenAI uses prompt_tokens / completion_tokens
        input_tokens = int(
            usage_raw.get("prompt_tokens", usage_raw.get("input_tokens", 0))
        )
        output_tokens = int(
            usage_raw.get("completion_tokens", usage_raw.get("output_tokens", 0))
        )
        usage = TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)

        # stop_reason lives in choices[0].finish_reason
        choices = response.get("choices") or []
        stop_reason = ""
        if choices:
            first = choices[0]
            if isinstance(first, dict):
                stop_reason = str(first.get("finish_reason", ""))
            else:
                stop_reason = str(getattr(first, "finish_reason", ""))

        return cls(
            provider=Provider.OPENAI,
            model=str(response.get("model", "")),
            response_id=str(response.get("id", "")),
            stop_reason=stop_reason,
            usage=usage,
            latency_ms=latency_ms,
            raw=response,
        )

    @classmethod
    def from_gemini(
        cls,
        response: dict[str, Any],
        *,
        latency_ms: float | None = None,
    ) -> ResponseMetadata:
        """Build from a Gemini ``generate_content()`` response dict.

        Supports the ``usageMetadata`` shape from the REST API.
        """
        usage_raw = response.get("usageMetadata") or {}
        input_tokens = int(
            usage_raw.get("promptTokenCount", usage_raw.get("input_tokens", 0))
        )
        output_tokens = int(
            usage_raw.get("candidatesTokenCount", usage_raw.get("output_tokens", 0))
        )
        usage = TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)

        # model may be in modelVersion or model_version
        model = str(response.get("modelVersion", response.get("model_version", "")))

        # finish reason lives in candidates[0].finishReason
        candidates = response.get("candidates") or []
        stop_reason = ""
        if candidates:
            first = candidates[0]
            if isinstance(first, dict):
                stop_reason = str(first.get("finishReason", "")).lower()

        return cls(
            provider=Provider.GEMINI,
            model=model,
            response_id=str(response.get("responseId", "")),
            stop_reason=stop_reason,
            usage=usage,
            latency_ms=latency_ms,
            raw=response,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResponseMetadata:
        """Reconstruct a :class:`ResponseMetadata` from a plain dict."""
        try:
            provider = Provider(data.get("provider", Provider.UNKNOWN.value))
        except ValueError:
            provider = Provider.UNKNOWN
        return cls(
            provider=provider,
            model=str(data.get("model", "")),
            response_id=str(data.get("response_id", "")),
            stop_reason=str(data.get("stop_reason", "")),
            usage=TokenUsage.from_dict(data.get("usage", {})),
            latency_ms=data.get("latency_ms"),
            raw=data.get("raw", {}),
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict (excludes *raw*)."""
        d: dict[str, Any] = {
            "provider": self.provider.value,
            "model": self.model,
            "response_id": self.response_id,
            "stop_reason": self.stop_reason,
            "usage": self.usage.to_dict(),
        }
        if self.latency_ms is not None:
            d["latency_ms"] = self.latency_ms
        return d

    def __repr__(self) -> str:
        return (
            f"ResponseMetadata(provider={self.provider.value!r},"
            f" model={self.model!r}, stop_reason={self.stop_reason!r},"
            f" total_tokens={self.total_tokens})"
        )
