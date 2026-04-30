# Anthropic prompt caching in graphiti

This doc describes the prompt-cache configuration available on `AnthropicClient`,
and explains why graphiti's default is no longer to enable caching unconditionally.

## TL;DR

- **Default:** `cache_mode='disabled'`. No `cache_control` is sent.
- The previous default (`cache_control={'type':'ephemeral'}` as a top-level
  kwarg) wrote ~218K tokens of cache per 128 calls in our demo trace and got
  ~1.1% read share back — net cost is the +25% cache-write surcharge for
  effectively zero benefit on graphiti's per-episode workload.
- To get real caching wins, prompts need to be **restructured** so static
  content lives before per-call dynamic content, AND the cacheable prefix has
  to clear the per-model threshold (~2048 tokens for Sonnet 4.x, ~4500 for
  Haiku 4.x). Graphiti's natural prompts don't clear that threshold even after
  restructure.

## Configuration

```python
from graphiti_core.llm_client.config import LLMConfig

config = LLMConfig(
    api_key=...,
    model='claude-sonnet-4-5-latest',
    cache_mode='system-block',                # 'disabled' (default), 'top-level', 'system-block'
    cache_ttl='1h',                           # '5m' (default), '1h'
    cache_padding_text='<your stable text>',  # optional substantive padding
    cache_padding_tokens=1500,                # used only if cache_padding_text is None
)
```

### `cache_mode`

| value | behavior |
|---|---|
| `'disabled'` (default) | No `cache_control` is sent. Safe choice. |
| `'top-level'` | Legacy. Pass `cache_control={'type':'ephemeral'}` as a kwarg on `messages.create()` — Anthropic auto-places the marker on the last cacheable block. Useful only if you re-issue identical prompts. |
| `'system-block'` | Wrap the system message in a single text block with `cache_control: {'type':'ephemeral'}`. Use this together with restructured prompts (static content in system, dynamic content in user) to get cache reads on subsequent calls. |

### `cache_ttl`

`'5m'` is the default Anthropic ephemeral TTL. Set `'1h'` for batch ingestion
runs that span > 5 minutes between bursts. The 1h write surcharge is higher
but amortizes cleanly across hundreds of reads.

### `cache_padding_text` / `cache_padding_tokens`

The cacheable system block must clear the per-model minimum to engage caching.
If your restructured system prompt is below threshold, you can pad it:

- **Preferred**: provide `cache_padding_text` with substantive, neutral domain
  content — extended entity-type definitions, additional few-shot examples,
  detailed format specifications. Anything that doesn't bias the model toward
  particular outputs.
- **Last resort / measurement**: set `cache_padding_tokens=N`. The client
  appends N tokens of license-style boilerplate that the model is trained to
  ignore. Empirically this preserves extraction output (within LLM
  nondeterminism), but is wasted bytes that you pay for at the cache-read
  rate. Don't use this in production unless you're certain it doesn't move
  your evaluation metrics.

**Beware task-shaped padding.** Initial implementations used "be precise and
conservative" filler text and observed measurable bias in entity counts on
multiple chunks. Generic instructions interact with the task. The default
filler now uses opaque boilerplate to minimize this.

### Per-model threshold notes

Empirically probed (April 2026):

| model | minimum cacheable prefix |
|---|---|
| `claude-sonnet-4-5-*` (4.6) | between 2042 and 2316 tokens |
| `claude-haiku-4-5` | between 4033 and 5010 tokens |

Caching only "engages" — produces non-zero `cache_creation_input_tokens` on
the first call, then `cache_read_input_tokens` on subsequent calls — when the
cached prefix exceeds these thresholds. Below threshold, the API silently
no-ops the cache_control marker.

## Cost arithmetic

Sonnet 4.6 input rates (USD per million tokens):

| metric | rate |
|---|---|
| normal input | $3.00 |
| cache write (5m TTL) | $3.75 (1.25× input) |
| cache write (1h TTL) | $6.00 (2× input) |
| cache read | $0.30 (0.10× input) |

Cache writes pay back after ~3 cache reads. On Haiku 4.5 (input $1/MTok), the
gap between input and cache-write is smaller in absolute terms, and the higher
threshold means more padding is needed; the math doesn't favor caching for
small Haiku-routed prompts in graphiti.

## Recommended configuration recipes

### Sonnet extraction with restructured prompts and a substantive padding

```python
LLMConfig(
    cache_mode='system-block',
    cache_padding_text=open('extra_entity_definitions.md').read(),  # ~1500-2000 tokens
)
```

### Long-running batch ingestion (1h TTL)

```python
LLMConfig(
    cache_mode='system-block',
    cache_ttl='1h',
    cache_padding_text=...,
)
```

### Short-lived dev / eval runs

Default `cache_mode='disabled'` — don't pay for caching infrastructure if
you're not amortizing across many calls.
