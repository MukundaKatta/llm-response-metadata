"""Tests for llm-response-metadata."""

from __future__ import annotations

import pytest

from llm_response_metadata import Provider, ResponseMetadata, TokenUsage

# ---------------------------------------------------------------------------
# TokenUsage
# ---------------------------------------------------------------------------


def test_token_usage_defaults():
    u = TokenUsage()
    assert u.input_tokens == 0
    assert u.output_tokens == 0
    assert u.cache_creation_input_tokens == 0
    assert u.cache_read_input_tokens == 0


def test_token_usage_total():
    u = TokenUsage(input_tokens=1000, output_tokens=250)
    assert u.total_tokens == 1250


def test_token_usage_total_excludes_cache():
    # cache tokens do NOT add to total_tokens
    u = TokenUsage(
        input_tokens=100,
        output_tokens=50,
        cache_creation_input_tokens=200,
        cache_read_input_tokens=300,
    )
    assert u.total_tokens == 150


def test_token_usage_to_dict():
    u = TokenUsage(
        input_tokens=10,
        output_tokens=5,
        cache_creation_input_tokens=2,
        cache_read_input_tokens=3,
    )
    d = u.to_dict()
    assert d["input_tokens"] == 10
    assert d["output_tokens"] == 5
    assert d["cache_creation_input_tokens"] == 2
    assert d["cache_read_input_tokens"] == 3


def test_token_usage_from_dict_round_trip():
    u = TokenUsage(input_tokens=100, output_tokens=50, cache_read_input_tokens=25)
    restored = TokenUsage.from_dict(u.to_dict())
    assert restored.input_tokens == u.input_tokens
    assert restored.output_tokens == u.output_tokens
    assert restored.cache_read_input_tokens == u.cache_read_input_tokens


def test_token_usage_from_dict_empty():
    u = TokenUsage.from_dict({})
    assert u.total_tokens == 0


def test_token_usage_repr():
    u = TokenUsage(input_tokens=100, output_tokens=50)
    r = repr(u)
    assert "100" in r
    assert "50" in r


# ---------------------------------------------------------------------------
# Provider enum
# ---------------------------------------------------------------------------


def test_provider_values():
    assert Provider.ANTHROPIC.value == "anthropic"
    assert Provider.OPENAI.value == "openai"
    assert Provider.GEMINI.value == "gemini"
    assert Provider.UNKNOWN.value == "unknown"


def test_provider_is_str():
    assert isinstance(Provider.ANTHROPIC, str)


# ---------------------------------------------------------------------------
# ResponseMetadata — from_anthropic
# ---------------------------------------------------------------------------


_ANTHROPIC_RESPONSE = {
    "id": "msg_01abc",
    "model": "claude-sonnet-4-5",
    "stop_reason": "end_turn",
    "usage": {
        "input_tokens": 1024,
        "output_tokens": 256,
        "cache_creation_input_tokens": 512,
        "cache_read_input_tokens": 128,
    },
}


def test_from_anthropic_provider():
    meta = ResponseMetadata.from_anthropic(_ANTHROPIC_RESPONSE)
    assert meta.provider == Provider.ANTHROPIC


def test_from_anthropic_model():
    meta = ResponseMetadata.from_anthropic(_ANTHROPIC_RESPONSE)
    assert meta.model == "claude-sonnet-4-5"


def test_from_anthropic_response_id():
    meta = ResponseMetadata.from_anthropic(_ANTHROPIC_RESPONSE)
    assert meta.response_id == "msg_01abc"


def test_from_anthropic_stop_reason():
    meta = ResponseMetadata.from_anthropic(_ANTHROPIC_RESPONSE)
    assert meta.stop_reason == "end_turn"


def test_from_anthropic_usage():
    meta = ResponseMetadata.from_anthropic(_ANTHROPIC_RESPONSE)
    assert meta.usage.input_tokens == 1024
    assert meta.usage.output_tokens == 256
    assert meta.usage.cache_creation_input_tokens == 512
    assert meta.usage.cache_read_input_tokens == 128


def test_from_anthropic_total_tokens():
    meta = ResponseMetadata.from_anthropic(_ANTHROPIC_RESPONSE)
    assert meta.total_tokens == 1024 + 256


def test_from_anthropic_cache_savings():
    meta = ResponseMetadata.from_anthropic(_ANTHROPIC_RESPONSE)
    assert meta.cache_savings_tokens == 128


def test_from_anthropic_latency():
    meta = ResponseMetadata.from_anthropic(_ANTHROPIC_RESPONSE, latency_ms=250.5)
    assert meta.latency_ms == pytest.approx(250.5)


def test_from_anthropic_no_latency():
    meta = ResponseMetadata.from_anthropic(_ANTHROPIC_RESPONSE)
    assert meta.latency_ms is None


def test_from_anthropic_is_complete():
    meta = ResponseMetadata.from_anthropic(_ANTHROPIC_RESPONSE)
    assert meta.is_complete


def test_from_anthropic_max_tokens_truncated():
    resp = {**_ANTHROPIC_RESPONSE, "stop_reason": "max_tokens"}
    meta = ResponseMetadata.from_anthropic(resp)
    assert meta.was_truncated
    assert not meta.is_complete


def test_from_anthropic_missing_usage():
    resp = {"model": "claude-haiku-3-5", "stop_reason": "end_turn"}
    meta = ResponseMetadata.from_anthropic(resp)
    assert meta.usage.total_tokens == 0


def test_from_anthropic_stop_sequence_is_complete():
    resp = {**_ANTHROPIC_RESPONSE, "stop_reason": "stop_sequence"}
    meta = ResponseMetadata.from_anthropic(resp)
    assert meta.is_complete
    assert not meta.was_truncated


def test_from_anthropic_tool_use_not_complete_or_truncated():
    resp = {**_ANTHROPIC_RESPONSE, "stop_reason": "tool_use"}
    meta = ResponseMetadata.from_anthropic(resp)
    assert not meta.is_complete
    assert not meta.was_truncated


# ---------------------------------------------------------------------------
# ResponseMetadata — from_openai
# ---------------------------------------------------------------------------


_OPENAI_RESPONSE = {
    "id": "chatcmpl-xyz",
    "model": "gpt-5.4",
    "choices": [{"finish_reason": "stop", "message": {"content": "hi"}}],
    "usage": {"prompt_tokens": 800, "completion_tokens": 200, "total_tokens": 1000},
}


def test_from_openai_provider():
    meta = ResponseMetadata.from_openai(_OPENAI_RESPONSE)
    assert meta.provider == Provider.OPENAI


def test_from_openai_model():
    meta = ResponseMetadata.from_openai(_OPENAI_RESPONSE)
    assert meta.model == "gpt-5.4"


def test_from_openai_stop_reason():
    meta = ResponseMetadata.from_openai(_OPENAI_RESPONSE)
    assert meta.stop_reason == "stop"


def test_from_openai_usage():
    meta = ResponseMetadata.from_openai(_OPENAI_RESPONSE)
    assert meta.usage.input_tokens == 800
    assert meta.usage.output_tokens == 200


def test_from_openai_total_tokens():
    meta = ResponseMetadata.from_openai(_OPENAI_RESPONSE)
    assert meta.total_tokens == 1000


def test_from_openai_is_complete():
    meta = ResponseMetadata.from_openai(_OPENAI_RESPONSE)
    assert meta.is_complete


def test_from_openai_length_truncated():
    resp = {**_OPENAI_RESPONSE, "choices": [{"finish_reason": "length"}]}
    meta = ResponseMetadata.from_openai(resp)
    assert meta.was_truncated


def test_from_openai_no_choices():
    resp = {**_OPENAI_RESPONSE, "choices": []}
    meta = ResponseMetadata.from_openai(resp)
    assert meta.stop_reason == ""


def test_from_openai_total_tokens_without_total_field():
    # total_tokens is derived from input + output, not the API total field
    resp = {
        "id": "chatcmpl-2",
        "model": "gpt-5.4",
        "choices": [{"finish_reason": "stop"}],
        "usage": {"prompt_tokens": 800, "completion_tokens": 200},
    }
    meta = ResponseMetadata.from_openai(resp)
    assert meta.total_tokens == 1000


def test_from_openai_tool_calls_not_complete_or_truncated():
    resp = {**_OPENAI_RESPONSE, "choices": [{"finish_reason": "tool_calls"}]}
    meta = ResponseMetadata.from_openai(resp)
    assert not meta.is_complete
    assert not meta.was_truncated


def test_from_openai_latency():
    meta = ResponseMetadata.from_openai(_OPENAI_RESPONSE, latency_ms=123.0)
    assert meta.latency_ms == pytest.approx(123.0)


# ---------------------------------------------------------------------------
# ResponseMetadata — from_gemini
# ---------------------------------------------------------------------------


_GEMINI_RESPONSE = {
    "modelVersion": "gemini-2.0-flash",
    "responseId": "resp-abc",
    "usageMetadata": {
        "promptTokenCount": 500,
        "candidatesTokenCount": 100,
    },
    "candidates": [{"finishReason": "STOP", "content": {}}],
}


def test_from_gemini_provider():
    meta = ResponseMetadata.from_gemini(_GEMINI_RESPONSE)
    assert meta.provider == Provider.GEMINI


def test_from_gemini_model():
    meta = ResponseMetadata.from_gemini(_GEMINI_RESPONSE)
    assert meta.model == "gemini-2.0-flash"


def test_from_gemini_stop_reason():
    meta = ResponseMetadata.from_gemini(_GEMINI_RESPONSE)
    assert meta.stop_reason == "stop"


def test_from_gemini_usage():
    meta = ResponseMetadata.from_gemini(_GEMINI_RESPONSE)
    assert meta.usage.input_tokens == 500
    assert meta.usage.output_tokens == 100


def test_from_gemini_response_id():
    meta = ResponseMetadata.from_gemini(_GEMINI_RESPONSE)
    assert meta.response_id == "resp-abc"


def test_from_gemini_no_candidates():
    resp = {**_GEMINI_RESPONSE, "candidates": []}
    meta = ResponseMetadata.from_gemini(resp)
    assert meta.stop_reason == ""


def test_from_gemini_is_complete():
    meta = ResponseMetadata.from_gemini(_GEMINI_RESPONSE)
    assert meta.is_complete


def test_from_gemini_max_tokens_truncated():
    resp = {**_GEMINI_RESPONSE, "candidates": [{"finishReason": "MAX_TOKENS"}]}
    meta = ResponseMetadata.from_gemini(resp)
    assert meta.was_truncated
    assert not meta.is_complete


def test_from_gemini_total_tokens():
    meta = ResponseMetadata.from_gemini(_GEMINI_RESPONSE)
    assert meta.total_tokens == 600


# ---------------------------------------------------------------------------
# ResponseMetadata — serialisation round-trip
# ---------------------------------------------------------------------------


def test_to_dict_from_dict_round_trip():
    meta = ResponseMetadata.from_anthropic(_ANTHROPIC_RESPONSE, latency_ms=99.0)
    d = meta.to_dict()
    restored = ResponseMetadata.from_dict(d)
    assert restored.provider == meta.provider
    assert restored.model == meta.model
    assert restored.stop_reason == meta.stop_reason
    assert restored.total_tokens == meta.total_tokens
    assert restored.latency_ms == pytest.approx(99.0)


def test_to_dict_excludes_raw_key_by_default():
    meta = ResponseMetadata.from_anthropic(_ANTHROPIC_RESPONSE)
    d = meta.to_dict()
    # raw is preserved on the object but not serialised by to_dict
    assert "raw" not in d


def test_to_dict_includes_latency_when_set():
    meta = ResponseMetadata.from_anthropic(_ANTHROPIC_RESPONSE, latency_ms=42.0)
    d = meta.to_dict()
    assert d["latency_ms"] == pytest.approx(42.0)


def test_to_dict_omits_latency_when_none():
    meta = ResponseMetadata.from_anthropic(_ANTHROPIC_RESPONSE)
    d = meta.to_dict()
    assert "latency_ms" not in d


def test_from_dict_unknown_provider():
    meta = ResponseMetadata.from_dict({"provider": "mystery_llm", "model": "xyz"})
    assert meta.provider == Provider.UNKNOWN


def test_from_dict_no_latency():
    meta = ResponseMetadata.from_dict({"model": "m"})
    assert meta.latency_ms is None


# ---------------------------------------------------------------------------
# ResponseMetadata — repr
# ---------------------------------------------------------------------------


def test_repr():
    meta = ResponseMetadata.from_anthropic(_ANTHROPIC_RESPONSE)
    r = repr(meta)
    assert "ResponseMetadata" in r
    assert "anthropic" in r
    assert "end_turn" in r
