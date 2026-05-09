"""Console rendering of TicketScore results."""
from __future__ import annotations

from .scorer import LineTag, TicketScore


def render_ticket(sc: TicketScore) -> str:
    cluster_tag = f" [{sc.cluster}]" if sc.cluster else ""
    if sc.error:
        head = (
            f"  ERROR  {sc.ticket_id}{cluster_tag}  ({sc.domain}) — {sc.error}"
        )
        return head

    p_pct = 100.0 * sc.primary_matched / max(1, sc.primary_total)
    p_status = "PASS" if sc.primary_passed else "FAIL"
    bonus_status = (
        f"bonus {sc.bonus_matched}/{sc.bonus_total} "
        f"(rc={'Y' if sc.root_cause_match else 'N'} "
        f"kc={'Y' if sc.key_command_match else 'N'} "
        f"esc={sc.escalation_path_score:.0%}{'P' if sc.escalation_path_passed else 'f'} "
        f"ts={'Y' if sc.incident_timestamp_match else 'N'})"
    )
    halluc = f"halluc={sc.primary_hallucinated}"
    overall = "PASS" if sc.passed else "FAIL"
    return (
        f"  {overall:4s}  {sc.ticket_id}{cluster_tag}  ({sc.domain})  "
        f"primary {sc.primary_matched}/{sc.primary_total} ({p_pct:.0f}% {p_status})  "
        f"{halluc}  {bonus_status}"
    )


def render_summary(scores: list[TicketScore]) -> str:
    if not scores:
        return "\n(no scored tickets)"
    n = len(scores)
    n_err = sum(1 for s in scores if s.error)
    n_real = n - n_err

    p_matched = sum(s.primary_matched for s in scores if not s.error)
    p_total = sum(s.primary_total for s in scores if not s.error)
    p_hall = sum(s.primary_hallucinated for s in scores if not s.error)
    p_pass = sum(1 for s in scores if not s.error and s.primary_passed)

    rc = sum(1 for s in scores if not s.error and s.root_cause_match)
    kc = sum(1 for s in scores if not s.error and s.key_command_match)
    esc_p = sum(1 for s in scores if not s.error and s.escalation_path_passed)
    ts = sum(1 for s in scores if not s.error and s.incident_timestamp_match)
    overall_pass = sum(1 for s in scores if not s.error and s.passed)

    lines = [
        "",
        f"=== Summary  ({n_real} scored, {n_err} errored) ===",
        f"Primary recall:    {p_matched}/{p_total}  ({100.0 * p_matched / max(1, p_total):.1f} %)",
        f"Primary halluc:    {p_hall}",
        f"Primary pass:      {p_pass}/{n_real}",
        f"Bonus root_cause:  {rc}/{n_real}",
        f"Bonus key_command: {kc}/{n_real}",
        f"Bonus escalation:  {esc_p}/{n_real}  (≥80% entries verbatim)",
        f"Bonus timestamp:   {ts}/{n_real}",
        f"Overall pass:      {overall_pass}/{n_real}",
    ]

    # Cluster discrimination breakdown
    cluster_scores = [s for s in scores if not s.error and s.cluster]
    if cluster_scores:
        cluster_pass = sum(1 for s in cluster_scores if s.primary_passed)
        lines.append(
            f"Cluster (trap):    {cluster_pass}/{len(cluster_scores)} primary-passed"
        )

    return "\n".join(lines)
