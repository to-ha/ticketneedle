"""Load a ticket corpus + sample targets for a benchmark run."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TicketTarget:
    """One ticket from the corpus, with the planted needles for scoring."""
    ticket_id: str
    domain: str
    length_bucket: str
    priority: str
    cluster: str | None
    primary: list[str]   # resolution_steps lines
    bonus: dict          # root_cause, key_command, escalation_path, incident_timestamp
    file_path: Path

    @property
    def primary_lines(self) -> list[str]:
        return self.primary


@dataclass
class TicketCorpus:
    """A whole corpus directory: full text fed to the model + per-ticket targets."""
    corpus_dir: Path
    text: str
    targets: list[TicketTarget] = field(default_factory=list)

    @property
    def display_name(self) -> str:
        return self.corpus_dir.name


def load_corpus(corpus_dir: Path) -> TicketCorpus:
    """Read index.json + all per-ticket Markdown files, return a TicketCorpus.

    Tickets are concatenated in index order with a blank line between them.
    Each ticket already starts with `# Major Incident — INC-...` so the
    boundaries are self-explanatory to the model.
    """
    index_path = corpus_dir / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"index.json not found in {corpus_dir}")
    index = json.loads(index_path.read_text())

    parts: list[str] = []
    targets: list[TicketTarget] = []
    for entry in index:
        path = corpus_dir / entry["file"]
        md = path.read_text()
        parts.append(md)
        if not md.endswith("\n"):
            parts.append("\n")
        parts.append("\n")  # blank line between tickets

        meta = entry["metadata"]
        targets.append(
            TicketTarget(
                ticket_id=entry["ticket_id"],
                domain=meta["domain"],
                length_bucket=meta["length_bucket"],
                priority=meta["priority"],
                cluster=meta.get("cluster"),
                primary=list(entry["primary"]["resolution_steps"]),
                bonus=dict(entry["bonus"]),
                file_path=path,
            )
        )

    return TicketCorpus(
        corpus_dir=corpus_dir,
        text="".join(parts),
        targets=targets,
    )


def stratified_sample(
    targets: list[TicketTarget],
    k: int,
    seed: int = 42,
) -> list[TicketTarget]:
    """Pick k tickets with light stratification.

    Goals: cover multiple domains; ensure at least one cluster ticket
    (hallucination-trap) is present whenever possible; otherwise random.
    Returns all targets if k >= len(targets).
    """
    if k >= len(targets):
        return list(targets)

    rng = random.Random(seed)
    chosen: list[TicketTarget] = []

    cluster_targets = [t for t in targets if t.cluster]
    if cluster_targets and k >= 4:
        n_cluster = min(max(1, k // 8), len(cluster_targets), 3)
        chosen.extend(rng.sample(cluster_targets, n_cluster))

    pool = [t for t in targets if t not in chosen]
    rng.shuffle(pool)

    # Round-robin by domain so the sample doesn't pile up on one area.
    by_domain: dict[str, list[TicketTarget]] = {}
    for t in pool:
        by_domain.setdefault(t.domain, []).append(t)
    domains = list(by_domain.keys())
    rng.shuffle(domains)

    while len(chosen) < k and any(by_domain.values()):
        for d in domains:
            if not by_domain[d]:
                continue
            chosen.append(by_domain[d].pop())
            if len(chosen) >= k:
                break

    return chosen[:k]
