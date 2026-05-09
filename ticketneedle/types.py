"""Shared dataclasses used across the generator pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class Domain:
    key: str
    display_name: str
    weight_large: int
    priority_skew: list[str]
    components: list[str]
    tools: list[str]
    typical_symptoms: list[str]


@dataclass(frozen=True)
class ClusterVariant:
    root_cause_hint: str
    resolution_hint: str


@dataclass(frozen=True)
class Cluster:
    name: str
    domain: str
    shared_symptom: str
    counts: dict[str, int]
    variants: list[ClusterVariant]


@dataclass(frozen=True)
class LengthBucket:
    label: str
    target_words: int


@dataclass(frozen=True)
class PrimaryNeedles:
    resolution_steps: list[str]


@dataclass(frozen=True)
class BonusNeedles:
    root_cause: str
    key_command: str
    escalation_path: list[str]
    incident_timestamp: str


@dataclass(frozen=True)
class Needles:
    primary: PrimaryNeedles
    bonus: BonusNeedles


@dataclass(frozen=True)
class TicketFocus:
    """Per-ticket focus picks: one component, one symptom, a few tools.
    Pre-picked seeded so each ticket has ONE clear theme rather than the
    whole domain vocabulary slopped into the body.
    """
    component: str
    symptom: str
    tools: list[str]


@dataclass(frozen=True)
class TicketSpec:
    ticket_id: str
    domain_key: str
    length_bucket: str
    priority: str
    cluster_name: str | None
    cluster_variant: int | None


@dataclass
class TicketRecord:
    spec: TicketSpec
    needles: Needles
    markdown_path: str


SizeLabel = Literal["small", "medium", "large"]

CORPUS_SIZES: dict[str, int] = {"small": 30, "medium": 80, "large": 150}

# Per-domain floor counts so domain-stratified analyses retain power
# at smaller corpus sizes.
DOMAIN_FLOOR: dict[str, int] = {"small": 3, "medium": 6, "large": 18}

LENGTH_BUCKETS: dict[str, LengthBucket] = {
    "short": LengthBucket("short", 400),
    "medium": LengthBucket("medium", 750),
    "long": LengthBucket("long", 1500),
}

LENGTH_DISTRIBUTION: list[tuple[str, float]] = [
    ("short", 0.50),
    ("medium", 0.30),
    ("long", 0.20),
]
