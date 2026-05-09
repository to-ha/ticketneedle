"""Load template TOML / JSON files into typed dataclasses."""
from __future__ import annotations

import tomllib
from pathlib import Path

from .types import Cluster, ClusterVariant, Domain

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = REPO_ROOT / "templates"


def load_domains() -> dict[str, Domain]:
    with open(TEMPLATES_DIR / "domains.toml", "rb") as f:
        raw = tomllib.load(f)
    return {
        key: Domain(
            key=key,
            display_name=d["display_name"],
            weight_large=d["weight_large"],
            priority_skew=list(d["priority_skew"]),
            components=list(d["components"]),
            tools=list(d["tools"]),
            typical_symptoms=list(d["typical_symptoms"]),
        )
        for key, d in raw.items()
    }


def load_clusters() -> dict[str, Cluster]:
    with open(TEMPLATES_DIR / "halluc_clusters.toml", "rb") as f:
        raw = tomllib.load(f)["clusters"]
    return {
        name: Cluster(
            name=name,
            domain=c["domain"],
            shared_symptom=c["shared_symptom"],
            counts={
                "small": c["count_small"],
                "medium": c["count_medium"],
                "large": c["count_large"],
            },
            variants=[
                ClusterVariant(
                    root_cause_hint=v["root_cause_hint"],
                    resolution_hint=v["resolution_hint"],
                )
                for v in c["variants"]
            ],
        )
        for name, c in raw.items()
    }


def load_personas() -> dict:
    with open(TEMPLATES_DIR / "personas.toml", "rb") as f:
        return tomllib.load(f)


def load_prompt_system() -> str:
    return (TEMPLATES_DIR / "prompts" / "system.md").read_text()


def load_prompt_user_template() -> str:
    return (TEMPLATES_DIR / "prompts" / "user.j2").read_text()
