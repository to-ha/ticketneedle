"""Minimal Ollama HTTP client for local generation.

Targets Ollama's /api/chat endpoint with a system+user message. Supports
deterministic generation via the `seed` option. Streams the response and
returns the full assistant text.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import httpx


DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_TIMEOUT_S = 600  # llama3.3:70b on M5 Max can take minutes per ticket


@dataclass
class OllamaOptions:
    temperature: float = 0.7
    seed: int | None = None
    num_ctx: int = 16384
    num_predict: int = -1
    top_p: float = 0.9
    repeat_penalty: float = 1.05
    stop: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d: dict = {
            "temperature": self.temperature,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
            "top_p": self.top_p,
            "repeat_penalty": self.repeat_penalty,
        }
        if self.seed is not None:
            d["seed"] = self.seed
        if self.stop:
            d["stop"] = self.stop
        return d


@dataclass
class OllamaClient:
    model: str
    base_url: str = DEFAULT_BASE_URL
    timeout_s: float = DEFAULT_TIMEOUT_S

    def chat(
        self,
        *,
        system: str,
        user: str,
        options: OllamaOptions,
        format_json: bool = False,
    ) -> str:
        """Send a chat request and return the full assistant content."""
        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": True,
            "options": options.to_dict(),
        }
        if format_json:
            payload["format"] = "json"

        chunks: list[str] = []
        with httpx.stream(
            "POST",
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout_s,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                event = json.loads(line)
                if "message" in event and "content" in event["message"]:
                    chunks.append(event["message"]["content"])
                if event.get("done"):
                    break
        return "".join(chunks)
