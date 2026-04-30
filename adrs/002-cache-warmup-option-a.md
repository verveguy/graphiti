# ADR 002 — Cache warmup: Option A (explicit warmup method) over Option B (staged concurrency)

**Date:** 2026-04-30
**Status:** Accepted

## Context

When `add_episode_bulk` cold-starts with a high concurrency limit (e.g.,
`SEMAPHORE_LIMIT=20`), all 20 concurrent LLM calls race to write the same
Anthropic prompt-cache prefix. Only one write is needed; the other 19 each pay
the +25% cache-write surcharge for content that ends up duplicated. This is a
thundering-herd problem specific to `system-block` caching mode.

Two implementation shapes were considered:

**Option A — Explicit `warmup` hook on `LLMClient`:**
Issue one minimal API call (max_tokens=1) per distinct prompt type before the
bulk fan-out. The call seeds the server-side cache; the result is discarded.
`AnthropicClient` overrides with a real call; the base class default is a no-op.

**Option B — Staged concurrency ramp:**
Process the first episode at `SEMAPHORE_LIMIT=1`, then raise to the configured
value. The first episode's real API calls naturally seed the cache before the
fan-out.

## Decision

**Option A was chosen.**

## Rationale

### Option A is targeted and low-cost

Option A sends `max_tokens=1` calls with no meaningful output. The only purpose
is to write the cache entry server-side. This costs roughly $0.003–0.01 per
warmup batch (4–6 calls × ≥2048 tokens × cache-write rate) — orders of
magnitude less than a full episode.

### Option B processes real work serially

Option B must wait for a full episode to be processed at concurrency=1 before
the fan-out begins. This:
- Adds the latency of one full episode before the batch starts.
- Bills for real extraction work (node extraction, edge extraction, dedup) rather
  than just a cache seed.
- Changes the observable timing semantics of `add_episode_bulk` — callers who
  depend on parallel processing from the start may see unexpected behavior.

### `system-block` mode is the only mode that benefits

- **`disabled`:** No `cache_control` is sent. There is no server-side cache to
  prime. Warmup is a no-op.
- **`top-level`:** The cache key includes the last user message. Each episode has
  a unique user message (the episode content), so a static warmup with a trivial
  user message (`.`) seeds an entry that real calls will never hit. Warmup would
  waste money without benefit.
- **`system-block`:** `cache_control` is pinned to the system block. The cache
  key is `system_message + cache_padding_text`, identical across all calls for
  the same prompt type. A warmup with any user message seeds this entry
  correctly.

### Concurrent warmup

All 4–6 warmup calls are issued via `asyncio.gather`. This caps the warmup
overhead to approximately one Anthropic API round-trip rather than N sequential
round-trips.

### Best-effort semantics

A warmup failure is non-fatal. The bulk operation proceeds and pays the
cache-write surcharge as before. This is appropriate for a P2 cost optimization:
correctness is unaffected.

### Provider-agnostic call site

Because `LLMClient.warmup` is a no-op, the `add_episode_bulk` call site needs
no `isinstance` check. Provider-specific logic is entirely inside
`AnthropicClient.warmup`.

## Consequences

- Every `add_episode_bulk` call at `cache_mode='system-block'` pays for 4–6
  `max_tokens=1` API calls. Callers at `cache_mode='disabled'` (the default)
  see zero overhead.
- After a successful warmup, a run with `SEMAPHORE_LIMIT=20` should produce
  approximately 1 `cache_creation_input_tokens` event per distinct prompt type
  on the first episode batch, rather than ~20.
- Adding warmup support for a new LLM provider requires overriding `warmup` in
  that provider's client class — no changes to `add_episode_bulk` or
  `_warmup_prompt_cache` are needed.
