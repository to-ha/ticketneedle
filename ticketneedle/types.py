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
    structure_hint: str  # explicit per-section size guidance for the LLM


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
    "short": LengthBucket(
        "short",
        400,
        structure_hint=(
            "Word budget per section (treat as targets, not maximums): "
            "Symptom ~120 words / 2-3 paragraphs. "
            "Escalation timeline: include every entry from the escalation_path above (do not skip any), one short sentence each. "
            "Diagnosis steps and hypotheses ~120 words / 1-2 hypotheses with the evidence that ruled them out (or in). "
            "Resolution steps: as given (verbatim). "
            "Post-mortem note ~60 words / 1 paragraph. "
            "TOTAL TARGET: 400 words."
        ),
    ),
    "medium": LengthBucket(
        "medium",
        750,
        structure_hint=(
            "Word budget per section (treat as targets, not maximums): "
            "Symptom ~240 words / 3-4 paragraphs covering observations and impact. "
            "Escalation timeline: include every entry from the escalation_path above (do not skip any), one or two sentences each describing what that person did. "
            "Diagnosis steps and hypotheses ~300 words / 2-3 hypotheses, each with the evidence that ruled it out. "
            "Resolution steps: as given (verbatim). "
            "Post-mortem note ~120 words / 1-2 paragraphs. "
            "TOTAL TARGET: 750 words."
        ),
    ),
    "long": LengthBucket(
        "long",
        1500,
        structure_hint=(
            "Word budget per section (treat as targets, not maximums — long tickets must actually be long): "
            "Symptom ~400 words / 4-6 paragraphs including how symptoms first appeared, how they evolved, the user/business impact, and what monitoring caught vs. missed. "
            "Escalation timeline: include every entry from the escalation_path above (do not skip any), two to three sentences each describing what that person did and what they found. "
            "Diagnosis steps and hypotheses ~700 words — this is the LARGEST section by a wide margin. "
            "It MUST contain 4-6 distinct ruled-out hypotheses. For each: 100-150 words covering "
            "(a) what the team suspected and why, (b) what diagnostic step was taken, (c) what evidence ruled it out. "
            "Then a final 100-150 word paragraph describing how the actual root cause emerged. "
            "Resolution steps: as given (verbatim). "
            "Post-mortem note ~250 words / 2-3 paragraphs covering the root cause sentence, why it was missed initially, and follow-up actions. "
            "TOTAL TARGET: 1500 words."
        ),
    ),
}

LENGTH_DISTRIBUTION: list[tuple[str, float]] = [
    ("short", 0.50),
    ("medium", 0.30),
    ("long", 0.20),
]
