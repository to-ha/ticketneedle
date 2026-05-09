"""Render the Phase-2 user prompt by feeding needles + domain into the Jinja2 template."""
from __future__ import annotations

from jinja2 import Environment, StrictUndefined

from .loader import load_prompt_user_template
from .types import Cluster, Domain, LENGTH_BUCKETS, Needles, TicketFocus, TicketSpec


_env = Environment(undefined=StrictUndefined, trim_blocks=False, lstrip_blocks=False)


def render_phase2_user(
    *,
    spec: TicketSpec,
    domain: Domain,
    cluster: Cluster | None,
    needles: Needles,
    focus: TicketFocus,
    name_pool: list[str],
) -> str:
    template = _env.from_string(load_prompt_user_template())
    # Note: cluster.name is intentionally omitted — leaking the cluster
    # name into the ticket body would trivialize the hallucination trap.
    cluster_ctx = (
        {"shared_symptom": cluster.shared_symptom} if cluster is not None else None
    )
    return template.render(
        ticket_id=spec.ticket_id,
        domain=domain,
        priority=spec.priority,
        length_bucket=LENGTH_BUCKETS[spec.length_bucket],
        needles=needles,
        cluster=cluster_ctx,
        focus=focus,
        name_pool=name_pool,
    )
