"""Post-generation validation: section parsing, vendor regex, needle presence."""
from __future__ import annotations

from .needles import VENDOR_PATTERNS
from .types import Needles


REQUIRED_H2 = [
    "Symptom",
    "Escalation timeline",
    "Diagnosis steps and hypotheses",
    "Resolution steps",
    "Post-mortem note",
]


def parse_sections(markdown: str) -> dict[str, str]:
    """Parse a Markdown doc into a dict of {heading_text: body_text}.

    Both H1 and H2 headings produce keys. Body text is the lines between
    this heading and the next heading of any level.
    """
    sections: dict[str, str] = {}
    cur_key: str | None = None
    cur_body: list[str] = []

    def flush() -> None:
        if cur_key is not None:
            sections[cur_key] = "\n".join(cur_body).strip()

    for line in markdown.splitlines():
        if line.startswith("# "):
            flush()
            cur_key = line[2:].strip()
            cur_body = []
        elif line.startswith("## "):
            flush()
            cur_key = line[3:].strip()
            cur_body = []
        else:
            cur_body.append(line)
    flush()
    return sections


def validate_ticket(markdown: str, needles: Needles) -> list[str]:
    """Return a list of validation error messages (empty list = pass)."""
    errors: list[str] = []
    sections = parse_sections(markdown)

    # H1 must contain "Major Incident"
    h1_keys = [k for k in sections if k.startswith("Major Incident")]
    if not h1_keys:
        errors.append("Missing H1 starting with 'Major Incident'")

    for required in REQUIRED_H2:
        if required not in sections:
            errors.append(f"Missing H2 section: {required!r}")

    vendor_match = VENDOR_PATTERNS.search(markdown)
    if vendor_match:
        errors.append(f"Vendor name detected: {vendor_match.group(0)!r}")

    for i, step in enumerate(needles.primary.resolution_steps):
        if step not in markdown:
            errors.append(f"resolution_steps[{i}] not verbatim: {step[:80]!r}")

    if needles.bonus.root_cause not in markdown:
        errors.append(f"root_cause not verbatim: {needles.bonus.root_cause[:80]!r}")
    if needles.bonus.key_command not in markdown:
        errors.append(f"key_command not verbatim: {needles.bonus.key_command!r}")
    if needles.bonus.incident_timestamp not in markdown:
        errors.append(
            f"incident_timestamp not verbatim: {needles.bonus.incident_timestamp!r}"
        )
    for entry in needles.bonus.escalation_path:
        if entry not in markdown:
            errors.append(f"escalation_path entry not verbatim: {entry!r}")

    return errors
