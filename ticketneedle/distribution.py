"""Compute the corpus composition (per-domain counts, length bucket assignment, cluster slots)."""
from __future__ import annotations

import random
from collections import Counter

from .types import (
    CORPUS_SIZES,
    DOMAIN_FLOOR,
    LENGTH_DISTRIBUTION,
    Cluster,
    Domain,
    TicketSpec,
)


def compute_domain_counts(
    domains: dict[str, Domain], total: int, floor: int
) -> dict[str, int]:
    """Scale weight_large proportionally to `total` with a per-domain floor.

    The result is guaranteed to sum to exactly `total`.
    """
    weight_total = sum(d.weight_large for d in domains.values())
    raw = {
        k: max(floor, round(d.weight_large * total / weight_total))
        for k, d in domains.items()
    }
    # Trim from largest (above floor) until we are at total.
    while sum(raw.values()) > total:
        ks = sorted(raw, key=lambda k: raw[k], reverse=True)
        for k in ks:
            if raw[k] > floor:
                raw[k] -= 1
                break
        else:
            raise ValueError(
                f"Cannot reduce domain counts below floor={floor} to reach total={total}"
            )
    while sum(raw.values()) < total:
        ks = sorted(raw, key=lambda k: raw[k], reverse=True)
        raw[ks[0]] += 1
    return raw


def _pick_length(rng: random.Random) -> str:
    r = rng.random()
    cum = 0.0
    for label, p in LENGTH_DISTRIBUTION:
        cum += p
        if r <= cum:
            return label
    return LENGTH_DISTRIBUTION[-1][0]


def build_specs(
    *,
    size: str,
    domains: dict[str, Domain],
    clusters: dict[str, Cluster],
    seed: int,
) -> list[TicketSpec]:
    """Build the list of TicketSpecs for a given corpus size."""
    total = CORPUS_SIZES[size]
    floor = DOMAIN_FLOOR[size]
    rng = random.Random(seed)

    dom_counts = compute_domain_counts(domains, total, floor)

    cluster_assignments: list[tuple[str, int]] = []
    cluster_per_domain: Counter = Counter()
    for cname, cluster in clusters.items():
        n = cluster.counts.get(size, 0)
        if n == 0:
            continue
        variant_indices = list(range(len(cluster.variants)))
        if n <= len(variant_indices):
            picked = rng.sample(variant_indices, n)
        else:
            picked = variant_indices + [
                rng.choice(variant_indices) for _ in range(n - len(variant_indices))
            ]
        for vi in picked:
            cluster_assignments.append((cname, vi))
            cluster_per_domain[cluster.domain] += 1

    for dom_key, taken in cluster_per_domain.items():
        if taken > dom_counts[dom_key]:
            raise ValueError(
                f"Cluster slots for domain {dom_key} ({taken}) exceed quota ({dom_counts[dom_key]})"
            )

    raw_specs: list[TicketSpec] = []
    for cname, vi in cluster_assignments:
        cluster = clusters[cname]
        raw_specs.append(
            TicketSpec(
                ticket_id="",
                domain_key=cluster.domain,
                length_bucket=_pick_length(rng),
                priority=rng.choice(domains[cluster.domain].priority_skew),
                cluster_name=cname,
                cluster_variant=vi,
            )
        )

    for dom_key, dom in domains.items():
        remaining = dom_counts[dom_key] - cluster_per_domain.get(dom_key, 0)
        for _ in range(remaining):
            raw_specs.append(
                TicketSpec(
                    ticket_id="",
                    domain_key=dom_key,
                    length_bucket=_pick_length(rng),
                    priority=rng.choice(dom.priority_skew),
                    cluster_name=None,
                    cluster_variant=None,
                )
            )

    rng.shuffle(raw_specs)

    return [
        TicketSpec(
            ticket_id=f"INC-2026-{i:04d}",
            domain_key=s.domain_key,
            length_bucket=s.length_bucket,
            priority=s.priority,
            cluster_name=s.cluster_name,
            cluster_variant=s.cluster_variant,
        )
        for i, s in enumerate(raw_specs, start=1)
    ]
