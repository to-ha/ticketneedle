"""Load TOML corpus + model configs from configs/."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .client import ClientConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"


@dataclass
class CorpusConfig:
    name: str
    directory: Path     # corpus directory containing index.json + INC-*.md
    k: int = 16         # how many tickets to sample per run (use 0 for ALL)
    seed: int = 42
    pacing_seconds: float = 0.0  # for rate-limited cloud endpoints


@dataclass
class ModelConfig:
    name: str
    client: ClientConfig
    suppress_thinking: bool = True


def _resolve_path(name_or_path: str, kind: str) -> Path:
    p = Path(name_or_path)
    if p.exists() and p.is_file():
        return p
    candidate = CONFIGS_DIR / kind / f"{name_or_path}.toml"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"could not find {kind} config '{name_or_path}' "
        f"(tried as path and as configs/{kind}/{name_or_path}.toml)"
    )


def load_corpus(name_or_path: str) -> CorpusConfig:
    path = _resolve_path(name_or_path, "corpora")
    raw = tomllib.loads(path.read_text())
    name = raw.get("name") or path.stem
    directory_raw = raw["directory"]
    directory = Path(directory_raw)
    if not directory.is_absolute():
        directory = (REPO_ROOT / directory).resolve()
    sample = raw.get("sample", {})
    return CorpusConfig(
        name=name,
        directory=directory,
        k=int(sample.get("k", 16)),
        seed=int(sample.get("seed", 42)),
        pacing_seconds=float(sample.get("pacing_seconds", 0.0)),
    )


def load_model(name_or_path: str) -> tuple[ModelConfig, bool]:
    """Returns (config, found_file). If no file matched, falls back to raw
    model identifier with sane local-Ollama defaults.
    """
    try:
        path = _resolve_path(name_or_path, "models")
    except FileNotFoundError:
        # Treat the input as a raw model identifier and assume local Ollama.
        cfg = ClientConfig(
            base_url="http://localhost:11434",
            model=name_or_path,
            api_key="not-needed",
            temperature=0.0,
            max_tokens=6000,
        )
        return ModelConfig(name=name_or_path, client=cfg), False

    raw = tomllib.loads(path.read_text())
    name = raw.get("name") or path.stem
    client_raw = raw.get("client", {})
    cfg = ClientConfig(
        base_url=client_raw["base_url"],
        model=client_raw["model"],
        api_key=_resolve_api_key(client_raw.get("api_key", "not-needed")),
        temperature=float(client_raw.get("temperature", 0.0)),
        max_tokens=int(client_raw.get("max_tokens", 6000)),
        timeout=float(client_raw.get("timeout", 600.0)),
        reasoning_effort=client_raw.get("reasoning_effort"),
        prefill_no_think=bool(client_raw.get("prefill_no_think", False)),
        stop=client_raw.get("stop"),
        use_max_completion_tokens=bool(client_raw.get("use_max_completion_tokens", False)),
        omit_temperature=bool(client_raw.get("omit_temperature", False)),
        anthropic_cache=client_raw.get("anthropic_cache"),  # None = auto-detect
    )
    return (
        ModelConfig(
            name=name,
            client=cfg,
            suppress_thinking=bool(raw.get("suppress_thinking", True)),
        ),
        True,
    )


def _resolve_api_key(value: str) -> str:
    """Allow `file:.secrets/anthropic.key` or `env:ANTHROPIC_API_KEY` indirection."""
    if not value:
        return "not-needed"
    if value.startswith("file:"):
        path = REPO_ROOT / value[len("file:") :]
        return path.read_text().strip()
    if value.startswith("env:"):
        import os
        return os.environ.get(value[len("env:") :], "")
    return value


def auto_dump_path(corpus_name: str, model_name: str, results_dir: Path) -> Path:
    safe_model = model_name.replace("/", "_").replace(":", "_")
    return results_dir / f"{corpus_name}__{safe_model}.json"
