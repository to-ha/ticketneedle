"""Minimal OpenAI-compatible chat-completions client.

Forked from alexziskind1/codeneedle (PR #3 by to-ha). Honors HTTP 429 with
bounded retry: parses `Retry-After` (or falls back to exponential backoff)
and waits before re-issuing the request. `omit_temperature` for Claude
Opus 4.7+ which rejects requests that include the field.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import httpx


DEFAULT_RETRY_AFTER_SECONDS = 60.0
MAX_RETRIES_ON_429 = 3


@dataclass
class ClientConfig:
    base_url: str
    model: str
    api_key: str = "not-needed"
    temperature: float = 0.0
    max_tokens: int = 6000
    timeout: float = 600.0
    reasoning_effort: str | None = None
    prefill_no_think: bool = False
    stop: list[str] | None = None
    use_max_completion_tokens: bool = False
    omit_temperature: bool = False
    # Anthropic prompt caching. Auto-detected from base_url when None.
    # Set explicitly to True/False to force the behavior (debugging,
    # cost A/B comparisons, or future API-format changes).
    anthropic_cache: bool | None = None


def _is_anthropic(base_url: str) -> bool:
    return "anthropic.com" in base_url.lower()


_anthropic_cache_logged: set[str] = set()


def _parse_retry_after(headers: httpx.Headers, attempt: int) -> float:
    header_value = headers.get("retry-after")
    if header_value:
        try:
            return max(float(header_value), 1.0)
        except ValueError:
            pass
    return DEFAULT_RETRY_AFTER_SECONDS * (2 ** attempt)


def chat_complete(cfg: ClientConfig, system: str | None, user: str) -> str:
    # Decide once per process whether to use Anthropic-style cache markup.
    use_anthropic_cache = (
        cfg.anthropic_cache
        if cfg.anthropic_cache is not None
        else _is_anthropic(cfg.base_url)
    )

    if use_anthropic_cache and cfg.base_url not in _anthropic_cache_logged:
        print(
            f"  Anthropic prompt cache: enabled (base_url={cfg.base_url})",
            file=sys.stderr,
        )
        _anthropic_cache_logged.add(cfg.base_url)
    elif (
        not use_anthropic_cache
        and _is_anthropic(cfg.base_url)
        and cfg.base_url not in _anthropic_cache_logged
    ):
        print(
            f"  Anthropic prompt cache: disabled (explicit override)",
            file=sys.stderr,
        )
        _anthropic_cache_logged.add(cfg.base_url)

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})

    if use_anthropic_cache:
        # Anthropic's OpenAI-compatible shim accepts the same
        # cache_control={"type": "ephemeral", "ttl": "1h"} markup as the
        # native API. The 1-hour TTL avoids cache-misses when 16-call
        # benchmark runs with 70 s pacing exceed the 5-min default TTL
        # (4 misses + 12 hits at 5-min vs 1 miss + 15 hits at 1-hour).
        # Cache-write premium 2.0× (1h) vs 1.25× (5-min) is more than
        # offset by avoiding the extra writes.
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user,
                        "cache_control": {"type": "ephemeral", "ttl": "1h"},
                    }
                ],
            }
        )
    else:
        messages.append({"role": "user", "content": user})

    if cfg.prefill_no_think:
        messages.append({"role": "assistant", "content": "<think>\n</think>\n\n"})

    payload = {
        "model": cfg.model,
        "messages": messages,
        "stream": False,
    }
    if not cfg.omit_temperature:
        payload["temperature"] = cfg.temperature
    if cfg.use_max_completion_tokens:
        payload["max_completion_tokens"] = cfg.max_tokens
    else:
        payload["max_tokens"] = cfg.max_tokens
    if cfg.reasoning_effort is not None:
        payload["reasoning_effort"] = cfg.reasoning_effort
    if cfg.stop:
        payload["stop"] = cfg.stop

    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    if use_anthropic_cache:
        # Required to opt into 1-hour cache TTL on Anthropic
        headers["anthropic-beta"] = "extended-cache-ttl-2025-04-11"
    url = f"{cfg.base_url.rstrip('/')}/v1/chat/completions"

    attempt = 0
    while True:
        with httpx.Client(timeout=cfg.timeout) as client:
            r = client.post(url, json=payload, headers=headers)
        if r.status_code == 429 and attempt < MAX_RETRIES_ON_429:
            wait_s = _parse_retry_after(r.headers, attempt)
            attempt += 1
            print(
                f"  pausing {wait_s:.1f}s on HTTP 429 (retry {attempt}/{MAX_RETRIES_ON_429})",
                file=sys.stderr,
            )
            time.sleep(wait_s)
            continue
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
        data = r.json()
        return data["choices"][0]["message"]["content"]
