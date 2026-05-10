"""Score model output against a ticket's planted needles.

The benchmark prompt asks for five labeled sections:
  ## resolution_steps   — primary needle, multi-line block
  ## root_cause         — bonus, one sentence
  ## key_command        — bonus, exact string
  ## escalation_path    — bonus, ordered list
  ## incident_timestamp — bonus, exact string

Each slot is scored against the body of its own section (parsed out of
the response). Hallucinations are counted only inside the
`## resolution_steps` section, so emitting other sections does not inflate
the hallucination count for the primary metric. If a section heading is
missing, the matcher falls back to the full response — graceful for
models that ignore the structure.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum


# A ticket "passes" when at least half of its resolution_steps are recalled.
# Resolution lists are short (3-7 steps), so a stricter threshold than
# codeneedle's 8/20 is appropriate.
PRIMARY_PASS_FRACTION = 0.5
BONUS_PASS_THRESHOLD = 2  # of 4 bonus slots
ESCALATION_PASS_FRACTION = 0.8


class LineTag(str, Enum):
    MATCHED = "matched"
    MISSING = "missing"
    HALLUCINATED = "hallucinated"


@dataclass
class LineResult:
    tag: LineTag
    text: str


@dataclass
class TicketScore:
    ticket_id: str
    domain: str
    cluster: str | None
    # Primary (resolution_steps)
    primary_matched: int
    primary_total: int
    primary_hallucinated: int
    primary_passed: bool
    # Bonus
    root_cause_match: bool
    key_command_match: bool
    escalation_path_score: float
    escalation_path_passed: bool
    incident_timestamp_match: bool
    bonus_matched: int
    bonus_total: int
    # Aggregate
    passed: bool
    error: str | None = None
    # For per-line rendering
    expected_tagged: list[LineResult] | None = None
    predicted_tagged: list[LineResult] | None = None


def score(
    *,
    ticket,
    response: str,
    relax_indent: bool = False,
) -> TicketScore:
    primary = ticket.primary
    bonus = ticket.bonus

    primary_section = _extract_section(response, "resolution_steps")
    pred_lines = _clean_output(primary_section)
    norm = _norm_relaxed if relax_indent else _norm

    exp_primary = [norm(l) for l in primary]
    pred = [norm(l) for l in pred_lines]
    while pred and pred[-1] == "":
        pred.pop()

    sm = SequenceMatcher(a=exp_primary, b=pred, autojunk=False)
    matched_exp = [False] * len(exp_primary)
    pred_kind = [-1] * len(pred)  # -1 hallucinated, 0 matched primary
    for block in sm.get_matching_blocks():
        if block.size == 0:
            continue
        for i in range(block.size):
            matched_exp[block.a + i] = True
            pred_kind[block.b + i] = 0

    primary_matched = sum(1 for x in matched_exp if x)
    hallucinated = sum(1 for k in pred_kind if k == -1)
    # Don't penalize blank-line spacing the model adds.
    hallucinated -= sum(
        1 for i, k in enumerate(pred_kind) if k == -1 and pred[i].strip() == ""
    )
    # Don't penalize numbering markers ("1." etc.) emitted as their own "line"
    # by overly literal cleanup. These are model output, but they're never in
    # primary because primary stores the step text, not the number.
    hallucinated -= sum(
        1
        for i, k in enumerate(pred_kind)
        if k == -1 and _is_numbering_artifact(pred[i])
    )
    hallucinated = max(0, hallucinated)

    # Bonus slot scoring — each in its own section, with full-response fallback.
    rc_text = bonus["root_cause"]
    kc_text = bonus["key_command"]
    ts_text = bonus["incident_timestamp"]
    esc_entries = bonus["escalation_path"]

    rc_section = _extract_section(response, "root_cause")
    kc_section = _extract_section(response, "key_command")
    ts_section = _extract_section(response, "incident_timestamp")
    esc_section = _extract_section(response, "escalation_path")

    rc_match = rc_text in rc_section or _root_cause_paraphrase_match(rc_text, rc_section)
    kc_match = kc_text in kc_section
    ts_match = ts_text in ts_section
    esc_present = sum(1 for e in esc_entries if e in esc_section)
    esc_score = esc_present / max(1, len(esc_entries))
    esc_passed = esc_score >= ESCALATION_PASS_FRACTION

    bonus_matched = sum([rc_match, kc_match, ts_match, esc_passed])
    bonus_total = 4

    primary_passed = primary_matched >= max(2, int(len(exp_primary) * PRIMARY_PASS_FRACTION))

    expected_display = [l.rstrip() for l in primary]
    pred_display = [l.rstrip() for l in _clean_output(primary_section)]
    while pred_display and pred_display[-1] == "":
        pred_display.pop()
    if len(pred_display) != len(pred):
        pred_display = pred_display[: len(pred)] + [""] * max(
            0, len(pred) - len(pred_display)
        )

    expected_tagged = [
        LineResult(
            LineTag.MATCHED if matched_exp[i] else LineTag.MISSING,
            expected_display[i],
        )
        for i in range(len(exp_primary))
    ]
    predicted_tagged = [
        LineResult(
            LineTag.MATCHED if pred_kind[i] == 0 else LineTag.HALLUCINATED,
            pred_display[i],
        )
        for i in range(len(pred))
    ]

    return TicketScore(
        ticket_id=ticket.ticket_id,
        domain=ticket.domain,
        cluster=ticket.cluster,
        primary_matched=primary_matched,
        primary_total=len(exp_primary),
        primary_hallucinated=hallucinated,
        primary_passed=primary_passed,
        root_cause_match=rc_match,
        key_command_match=kc_match,
        escalation_path_score=esc_score,
        escalation_path_passed=esc_passed,
        incident_timestamp_match=ts_match,
        bonus_matched=bonus_matched,
        bonus_total=bonus_total,
        passed=primary_passed and bonus_matched >= BONUS_PASS_THRESHOLD,
        expected_tagged=expected_tagged,
        predicted_tagged=predicted_tagged,
    )


_SECTION_RE_CACHE: dict[str, re.Pattern[str]] = {}

# The five known section names — used to find both the matching section
# and the start of the *next* section (for cross-stack robustness against
# format-drift, e.g. NVIDIA-CUDA-Ollama-runner output that omits the `## `
# Markdown header but keeps the section names as plain text on their own
# line).
_KNOWN_SECTIONS = (
    "resolution_steps",
    "root_cause",
    "key_command",
    "escalation_path",
    "incident_timestamp",
)


def _section_header_pattern(name: str) -> re.Pattern[str]:
    """Pattern that matches a section heading line, with or without `## ` prefix.

    Accepts:
      `## resolution_steps`
      `## **resolution_steps**`
      `resolution_steps`
      `**resolution_steps**`
      `resolution_steps:`
    All anchored to a full line (^ ... $) to avoid false-positive matches
    on the section name appearing inside body text.
    """
    pat = _SECTION_RE_CACHE.get(name)
    if pat is None:
        pat = re.compile(
            rf"^\s*(?:##\s*)?\**\s*{re.escape(name)}\s*\**\s*:?\s*$",
            re.MULTILINE | re.IGNORECASE,
        )
        _SECTION_RE_CACHE[name] = pat
    return pat


def _extract_section(response: str, section_name: str) -> str:
    """Return the body of a `<section_name>` heading.

    Falls back to the full response when no heading is present (graceful
    for primary-only outputs that have no labeled sections at all).
    """
    pat = _section_header_pattern(section_name)
    m = pat.search(response)
    if not m:
        return response
    body_start = m.end()

    # The body ends at the next *known* section heading — not just any
    # `##`. This is what gives us robustness against runners that emit
    # plain-text section names without `## `: the body still terminates
    # cleanly at the next section name we recognize.
    next_starts = []
    for other in _KNOWN_SECTIONS:
        if other == section_name:
            continue
        nm = _section_header_pattern(other).search(response, body_start)
        if nm:
            next_starts.append(nm.start())
    body_end = min(next_starts) if next_starts else len(response)
    return response[body_start:body_end].strip()


def _norm(s: str) -> str:
    return s.rstrip()


def _norm_relaxed(s: str) -> str:
    # Drop the leading numbering marker "1.", "2.", etc. so a step "Restart X"
    # matches whether or not the model included the number prefix.
    s = s.strip()
    return _strip_leading_number(s)


def _strip_leading_number(s: str) -> str:
    # Match "1. ", "12) ", "1: " etc.
    i = 0
    while i < len(s) and s[i].isdigit():
        i += 1
    if i > 0 and i < len(s) and s[i] in (".", ")", ":"):
        rest = s[i + 1:].lstrip()
        return rest
    return s


def _is_numbering_artifact(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    return s.rstrip(".)") .isdigit()


def _root_cause_paraphrase_match(needle: str, response: str) -> bool:
    """Tolerate the LLM ending root_cause with a comma-extension instead of a period.

    The most common drift seen during corpus generation: LLMs write
    "...silently, leading to ..." instead of "...silently." — substring
    match fails on the period. Strip trailing punctuation from both
    sides and retry.
    """
    n = needle.rstrip(".!?;,")
    if n in response:
        return True
    return False


def _clean_output(text: str) -> list[str]:
    """Strip markdown fences and surrounding blank lines."""
    lines = text.splitlines()
    fence_idxs = [i for i, l in enumerate(lines) if l.lstrip().startswith("```")]
    if len(fence_idxs) >= 2:
        lines = lines[fence_idxs[0] + 1 : fence_idxs[-1]]
    else:
        lines = [l for l in lines if not l.lstrip().startswith("```")]
    while lines and lines[0].strip() == "":
        lines.pop(0)
    while lines and lines[-1].strip() == "":
        lines.pop()
    return lines
