"""Phase-1 needle generation (LLM JSON call) plus deterministic helpers
for escalation paths and incident timestamps.
"""
from __future__ import annotations

import json
import random
import re
from datetime import datetime, timedelta, timezone

from .ollama_client import OllamaClient, OllamaOptions
from .types import (
    BonusNeedles,
    Cluster,
    ClusterVariant,
    Domain,
    Needles,
    PrimaryNeedles,
    TicketFocus,
    TicketSpec,
)

PHASE1_SYSTEM = """\
You generate IT-operations incident-ticket needles as a single JSON object.
Output ONLY the JSON, no markdown fence, no commentary, no surrounding text.
Never mention real vendor or product names. Use generic open-source tooling
only (kubectl, psql, dig, tcpdump, openssl, curl, etc.).
"""

# Conservative vendor / company-name filter. False positives just trigger a
# retry with a bumped seed, so we err on the side of catching too much.
VENDOR_PATTERNS = re.compile(
    r"\b("
    r"cisco|juniper|arista|fortinet|netscaler|broadcom|"
    r"palo\s*alto|check\s*point|f5\s+networks|"
    r"oracle|microsoft|azure|aws|amazon|gcp|google\s+cloud|alibaba\s*cloud|"
    r"vodafone|deutsche\s*telekom|t-mobile|verizon|orange\s+s\.?a\.?|bt\s*group|"
    r"splunk|servicenow|datadog|dynatrace|new\s*relic|grafana\s+cloud|"
    r"mongodb\s+atlas|elastic\s+cloud|confluent\s+cloud|databricks|snowflake|"
    r"vmware|red\s*hat|suse|canonical|"
    r"sap\s+se|salesforce|adobe|nutanix|veeam|netapp|pure\s+storage|emc\s+isilon"
    r")\b",
    flags=re.IGNORECASE,
)


def contains_vendor(text: str) -> str | None:
    """Return the first vendor name detected, or None if clean."""
    m = VENDOR_PATTERNS.search(text)
    return m.group(0) if m else None


def _build_phase1_user(
    domain: Domain,
    cluster: Cluster | None,
    variant: ClusterVariant | None,
    length_bucket: str,
    focus_component: str,
    focus_tools: list[str],
) -> str:
    parts: list[str] = [
        f"Generate needles for a {domain.display_name} major-incident ticket.",
        f"Length bucket: {length_bucket} (hints at how detailed the resolution may be).",
        "",
        f"This SPECIFIC incident is centered on: {focus_component}.",
        "Do NOT mix in unrelated components from the domain — keep the focus tight.",
        "",
        f"Tools the engineer most likely reached for: {', '.join(focus_tools)}.",
    ]
    if cluster is not None and variant is not None:
        parts += [
            "",
            "This ticket shares the following observable symptom with related tickets in the corpus:",
            f"  {cluster.shared_symptom}",
            "",
            "But this SPECIFIC ticket has its own root cause and resolution, distinct from the others. Use these hints (expand them, do not just copy):",
            f"- root_cause_hint: {variant.root_cause_hint}",
            f"- resolution_hint: {variant.resolution_hint}",
            "",
            "The resolution_steps you produce MUST reflect the resolution_hint.",
            "The root_cause you produce MUST reflect the root_cause_hint.",
            "Do NOT produce a generic resolution that could apply to the other tickets sharing this symptom.",
        ]
    parts += [
        "",
        "Output schema (strict JSON, exactly these keys, no others):",
        "{",
        '  "resolution_steps": ["step 1", "step 2", "step 3", ...],',
        '  "root_cause": "one-sentence explanation of what went wrong",',
        '  "key_command": "one specific diagnostic or fix command"',
        "}",
        "",
        "Rules:",
        "- 3 to 7 resolution_steps; each 10-180 characters; imperative voice ('Restart X', 'Set Y to Z').",
        "- Each step should reference at least one concrete value (a number, a path, a flag, a config key).",
        "- All resolution_steps must address the SAME root_cause and form a coherent",
        "  fix sequence. Do NOT mix unrelated fixes (e.g. don't combine a BGP timer fix",
        "  with a firewall rule change unless both are caused by the same underlying issue).",
        "  A reader should be able to see why each step is needed given the root_cause.",
        "- root_cause: one sentence in past tense, 30-200 chars, ends with a period.",
        "- key_command: a realistic command using generic tooling, 5-100 chars,",
        "  RELATED to diagnosing the root_cause above (not unrelated).",
        "- Use no markdown formatting in any value.",
        "- Use no real vendor or product names anywhere.",
    ]
    return "\n".join(parts)


def _validate_phase1_payload(data: object) -> tuple[bool, str]:
    if not isinstance(data, dict):
        return False, "payload is not an object"
    required = {"resolution_steps", "root_cause", "key_command"}
    extra = set(data.keys()) - required
    missing = required - set(data.keys())
    if missing:
        return False, f"missing keys: {sorted(missing)}"
    if extra:
        return False, f"unexpected extra keys: {sorted(extra)}"

    rs = data["resolution_steps"]
    if not isinstance(rs, list):
        return False, "resolution_steps is not a list"
    if not (3 <= len(rs) <= 7):
        return False, f"resolution_steps must have 3-7 entries (got {len(rs)})"
    for i, s in enumerate(rs):
        if not isinstance(s, str):
            return False, f"resolution_steps[{i}] is not a string"
        if not (10 <= len(s) <= 200):
            return False, f"resolution_steps[{i}] length {len(s)} outside 10-200"
        if "\n" in s:
            return False, f"resolution_steps[{i}] contains a newline"

    rc = data["root_cause"]
    if not isinstance(rc, str) or not (20 <= len(rc) <= 250):
        return False, "root_cause must be a 20-250 char string"
    if "\n" in rc:
        return False, "root_cause contains a newline"

    kc = data["key_command"]
    if not isinstance(kc, str) or not (3 <= len(kc) <= 150):
        return False, "key_command must be a 3-150 char string"
    if "\n" in kc:
        return False, "key_command contains a newline"

    return True, ""


def generate_phase1(
    *,
    client: OllamaClient,
    domain: Domain,
    cluster: Cluster | None,
    variant: ClusterVariant | None,
    length_bucket: str,
    focus_component: str,
    focus_tools: list[str],
    seed: int,
    max_retries: int = 3,
) -> tuple[list[str], str, str]:
    """Run the Phase-1 needle-generation LLM call. Returns
    (resolution_steps, root_cause, key_command).
    """
    user = _build_phase1_user(
        domain, cluster, variant, length_bucket, focus_component, focus_tools,
    )
    last_error = ""
    for attempt in range(max_retries):
        opts = OllamaOptions(
            temperature=0.7,
            seed=seed + attempt * 1000,
            num_ctx=8192,
            top_p=0.9,
        )
        raw = client.chat(
            system=PHASE1_SYSTEM,
            user=user,
            options=opts,
            format_json=True,
        )
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            last_error = f"JSON parse failed: {e}; raw[:200]={raw[:200]!r}"
            continue

        ok, why = _validate_phase1_payload(data)
        if not ok:
            last_error = f"schema check failed: {why}"
            continue

        vendor = contains_vendor(json.dumps(data))
        if vendor:
            last_error = f"vendor name detected in needles: {vendor!r}"
            continue

        return data["resolution_steps"], data["root_cause"], data["key_command"]

    raise RuntimeError(
        f"Phase-1 needle generation failed after {max_retries} attempts: {last_error}"
    )


_INCIDENT_WINDOW_START = datetime(2026, 1, 1, tzinfo=timezone.utc)
_INCIDENT_WINDOW_DAYS = 127  # through 2026-05-07


def deterministic_timestamp(rng: random.Random) -> str:
    """Pick a timestamp in 2026-01-01..2026-05-07, formatted '%Y-%m-%d %H:%M UTC'."""
    days = rng.randint(0, _INCIDENT_WINDOW_DAYS)
    hour = rng.randint(0, 23)
    minute = rng.randint(0, 59)
    dt = _INCIDENT_WINDOW_START + timedelta(days=days, hours=hour, minutes=minute)
    return dt.strftime("%Y-%m-%d %H:%M UTC")


_DOMAIN_ROLE_KEYWORD = {
    "network": "Network",
    "kubernetes": "Platform",
    "database": "Database",
    "auth": "Security",
    "storage": "Platform",
    "monitoring": "SRE",
    "platform": "Platform",
}


def deterministic_escalation_path(
    rng: random.Random,
    personas: dict,
    domain_key: str,
) -> list[str]:
    """Build a chronological escalation path: 2-4 entries 'Role: FirstName'."""
    depth = rng.randint(2, 4)
    layers = ["L1", "L2", "L3", "mgmt"][:depth]
    role_pool = personas["roles"]
    name_pool = list(personas["first_names"])
    rng.shuffle(name_pool)
    domain_kw = _DOMAIN_ROLE_KEYWORD.get(domain_key, "")

    entries: list[str] = []
    for i, layer in enumerate(layers):
        options = role_pool[layer]
        matching = [r for r in options if domain_kw and domain_kw in r]
        role = rng.choice(matching) if matching else rng.choice(options)
        name = name_pool[i % len(name_pool)]
        entries.append(f"{role}: {name}")
    return entries


def build_needles(
    *,
    spec: TicketSpec,
    domain: Domain,
    cluster: Cluster | None,
    client: OllamaClient,
    personas: dict,
    base_seed: int,
    ticket_index: int,
) -> tuple[Needles, TicketFocus]:
    """End-to-end needle assembly for one ticket.

    Pre-picks ONE focus component, ONE focus symptom, and a few tools
    (seeded), then runs Phase 1 with that focus. Returns the needles
    plus the focus picks so Phase 2 can keep the same theme.

    Phase 1 (LLM-generated): resolution_steps, root_cause, key_command.
    Deterministic: incident_timestamp, escalation_path, focus picks.
    """
    seed = base_seed + ticket_index
    rng = random.Random(seed)

    focus_component = rng.choice(domain.components)
    focus_symptom = rng.choice(domain.typical_symptoms)
    focus_tools = rng.sample(domain.tools, k=min(3, len(domain.tools)))
    focus = TicketFocus(
        component=focus_component, symptom=focus_symptom, tools=focus_tools,
    )

    variant = (
        cluster.variants[spec.cluster_variant]
        if (cluster is not None and spec.cluster_variant is not None)
        else None
    )
    res_steps, root_cause, key_command = generate_phase1(
        client=client,
        domain=domain,
        cluster=cluster,
        variant=variant,
        length_bucket=spec.length_bucket,
        focus_component=focus_component,
        focus_tools=focus_tools,
        seed=seed,
    )
    needles = Needles(
        primary=PrimaryNeedles(resolution_steps=res_steps),
        bonus=BonusNeedles(
            root_cause=root_cause,
            key_command=key_command,
            escalation_path=deterministic_escalation_path(rng, personas, spec.domain_key),
            incident_timestamp=deterministic_timestamp(rng),
        ),
    )
    return needles, focus
