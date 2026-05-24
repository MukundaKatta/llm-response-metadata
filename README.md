# llm-response-metadata

Extract usage, model, and stop-reason metadata from LLM API responses.

Normalises the fields that differ between Anthropic, OpenAI, and Gemini into one flat dataclass — no SDK dependency, just raw response dicts.

## Install

```bash
pip install llm-response-metadata
```

## Quick start

```python
from llm_response_metadata import ResponseMetadata

# Anthropic
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
print(meta.total_tokens)        # 1280
print(meta.cache_savings_tokens) # 128
print(meta.is_complete)          # True

# OpenAI
meta2 = ResponseMetadata.from_openai({
    "id": "chatcmpl-xyz",
    "model": "gpt-5.4",
    "choices": [{"finish_reason": "stop"}],
    "usage": {"prompt_tokens": 800, "completion_tokens": 200},
})
print(meta2.total_tokens)  # 1000
print(meta2.stop_reason)   # 'stop'

# Gemini
meta3 = ResponseMetadata.from_gemini({
    "modelVersion": "gemini-2.0-flash",
    "usageMetadata": {"promptTokenCount": 500, "candidatesTokenCount": 100},
    "candidates": [{"finishReason": "STOP"}],
})
print(meta3.total_tokens)  # 600
```

## API

### `ResponseMetadata`

| Factory | Description |
|---|---|
| `ResponseMetadata.from_anthropic(resp, *, latency_ms)` | Parse Anthropic response dict |
| `ResponseMetadata.from_openai(resp, *, latency_ms)` | Parse OpenAI response dict |
| `ResponseMetadata.from_gemini(resp, *, latency_ms)` | Parse Gemini response dict |
| `ResponseMetadata.from_dict(data)` | Restore from serialised dict |

| Attribute / Property | Type | Description |
|---|---|---|
| `provider` | `Provider` | `ANTHROPIC`, `OPENAI`, `GEMINI`, or `UNKNOWN` |
| `model` | `str` | Model identifier |
| `response_id` | `str` | Provider-assigned ID |
| `stop_reason` | `str` | Why generation stopped |
| `usage` | `TokenUsage` | Token breakdown |
| `latency_ms` | `float \| None` | Round-trip time if supplied |
| `total_tokens` | `int` | `input + output` tokens |
| `cache_savings_tokens` | `int` | Anthropic cache-read tokens |
| `is_complete` | `bool` | Normal stop (`end_turn` / `stop`) |
| `was_truncated` | `bool` | Cut short by length limit |
| `to_dict()` | `dict` | Serialise (excludes `raw`) |

### `TokenUsage`

| Field | Description |
|---|---|
| `input_tokens` | Prompt tokens |
| `output_tokens` | Completion tokens |
| `cache_creation_input_tokens` | Tokens written to Anthropic cache |
| `cache_read_input_tokens` | Tokens read from Anthropic cache |
| `total_tokens` | `input + output` (cache not included) |

## License

MIT
