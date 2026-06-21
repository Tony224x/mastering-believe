"""Day 15 — CAPSTONE: Agent Governance Toolkit (end-to-end, stdlib only).

What this script demonstrates
-----------------------------
A single, self-contained, reusable toolkit that wires the whole domain into one
pipeline run on an example agent fleet:

    fleet (dict)
      -> (1) INGEST   : load into a live registry, validate the 4 pillars,
                        flag orphan agents (no named owner)            [J2, J3]
      -> (2) ENFORCE  : run a PDP/PEP policy engine over attempted
                        actions, with an MCP-style consent gate        [J8, J14]
      -> (3) LOG      : write every decision into a hash-chained,
                        tamper-evident audit trail                     [J9]
      -> (4) SCORE    : likelihood x impact + agentic modulators,
                        TREAT/MONITOR/ACCEPT                           [J4]
      -> (5) MAP      : crosswalk controls -> {EU AI Act, NIST RMF,
                        ISO/IEC 42001}, isolate MANDATORY gaps         [J7]
      -> (6) REPORT   : board-ready output in BOTH markdown and JSON,
                        ending on an actionable verdict

WHY a single file: a governance toolkit is a deliverable you hand to a client.
It must run on its own, with no external deps and no other day imported. Each
brick is re-implemented here in miniature; the real-world tool it mirrors is
named in a comment (OPA/Rego, MCP, NIST AI RMF, OTel GenAI semconv, EU AI Act).

Run:
    python domains/gouvernance-ia/02-code/15-capstone-governance-toolkit.py

# requires: stdlib only
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum, IntEnum
from typing import Callable


def _utc_now_iso() -> str:
    # WHY: governance evidence must be timezone-explicit and sortable across
    # systems; we pin UTC ISO-8601 so timestamps are unambiguous.
    return datetime.now(timezone.utc).isoformat()


# ===========================================================================
# CORE MODEL — the fil-rouge agent (id / owner / scopes / risk_tier).
# Same shape used end to end so the pipeline stages can hand it along.   [J2]
# ===========================================================================

@dataclass
class GovernedAgent:
    """The unit we govern. Echoes the 4 pillars: identity, owner, permissions,
    auditability (the last is provided by the shared audit trail, below)."""
    agent_id: str                       # pillar 1: unique identity
    owner: str | None                   # pillar 2: a NAMED accountable human
    scopes: tuple[str, ...]             # pillar 3: least-privilege grants  [J8]
    risk_tier: str                      # "low" | "medium" | "high"
    # agentic context that drives risk modulation (J4)
    autonomous: bool = False            # acts with no human in the loop
    handles_irreversible: bool = False  # can take actions that cannot be undone

    def is_fully_governed(self) -> tuple[bool, list[str]]:
        # WHY: completeness is the smell-test of an ungoverned agent. We return
        # the list of missing pillars so the report can name the gap, not just
        # assert a boolean.
        missing: list[str] = []
        if not self.agent_id:
            missing.append("identity")
        if not self.owner:
            missing.append("owner")            # orphan / shadow agent
        if not self.scopes:
            missing.append("permissions")
        return (len(missing) == 0, missing)


# ===========================================================================
# (1) INGEST — load a raw fleet into a live registry (control, not a spreadsheet).
#     Real-world: Microsoft Entra Agent ID / Google A2A Agent Card (emergent). [J3]
# ===========================================================================

class Registry:
    """In-memory live registry. Source of truth for who-owns-what queries."""

    def __init__(self) -> None:
        self._agents: dict[str, GovernedAgent] = {}

    def ingest(self, raw_fleet: list[dict]) -> None:
        # WHY tolerate missing keys: a real export is imperfect. We do NOT drop
        # incomplete agents — we admit them so their gaps become visible/counted.
        for raw in raw_fleet:
            agent = GovernedAgent(
                agent_id=raw.get("agent_id", ""),
                owner=raw.get("owner") or None,
                scopes=tuple(raw.get("scopes", ())),
                risk_tier=raw.get("risk_tier", "low"),
                autonomous=bool(raw.get("autonomous", False)),
                handles_irreversible=bool(raw.get("handles_irreversible", False)),
            )
            if not agent.agent_id:
                continue  # an agent with no id at all cannot be a registry key
            self._agents[agent.agent_id] = agent

    def all(self) -> list[GovernedAgent]:
        return list(self._agents.values())

    def get(self, agent_id: str) -> GovernedAgent | None:
        return self._agents.get(agent_id)

    def orphans(self) -> list[GovernedAgent]:
        # The founding question: which agents have NO named owner?  [Microsoft, 2026]
        return [a for a in self._agents.values() if not a.owner]

    def coverage(self) -> tuple[int, int, float]:
        # Governance coverage = fully-governed agents / total. The headline KPI.
        total = len(self._agents)
        governed = sum(1 for a in self._agents.values() if a.is_fully_governed()[0])
        pct = (governed / total * 100.0) if total else 0.0
        return governed, total, pct


# ===========================================================================
# (2) ENFORCE — policy engine: PDP (decide) + PEP (apply) + MCP consent gate.
#     Real-world: Open Policy Agent / Rego (CNCF) + MCP Security & Trust.   [J14]
# ===========================================================================

class Verdict(IntEnum):
    """Ordered by SAFETY precedence: higher wins on conflict (deny > oblige > allow),
    mirroring OPA's `default allow = false`."""
    ALLOW = 0
    OBLIGE = 1   # action suspended until an obligation (e.g. human approval) is met
    DENY = 2


@dataclass(frozen=True)
class Action:
    """An action an agent wants to take: a tool call + the scope it needs."""
    tool: str
    params: dict
    required_scope: str


@dataclass
class Decision:
    verdict: Verdict
    rule: str
    reason: str
    obligation: str | None = None


# A rule is a pure function; None means "this rule does not apply".
Rule = Callable[[Action, GovernedAgent, dict], "Decision | None"]


def rule_scope(action: Action, agent: GovernedAgent, ctx: dict) -> Decision | None:
    """Least privilege (J8): required scope must be granted. Missing scope is a
    hard authorization failure -> DENY (never papered over by approval)."""
    if action.required_scope not in agent.scopes:
        return Decision(Verdict.DENY, "scope",
                        f"agent {agent.agent_id} lacks scope '{action.required_scope}'")
    return None


def rule_budget(action: Action, agent: GovernedAgent, ctx: dict) -> Decision | None:
    """Budget (J10): a monetary action above the auto-limit needs human approval.
    OBLIGE, not DENY: the action is legitimate, it just needs a second human."""
    limit = ctx.get("auto_amount_limit", 1000)
    amount = action.params.get("amount", 0)
    if amount > limit:
        return Decision(Verdict.OBLIGE, "budget",
                        f"amount {amount} exceeds auto-limit {limit}",
                        obligation="human_approval")
    return None


def rule_high_risk_irreversible(action: Action, agent: GovernedAgent, ctx: dict) -> Decision | None:
    """Autonomy (J4/J10): a high-tier agent doing an irreversible action escalates."""
    if agent.risk_tier == "high" and action.tool in ctx.get("irreversible_tools", set()):
        return Decision(Verdict.OBLIGE, "high_risk_irreversible",
                        f"high-tier agent attempting irreversible '{action.tool}'",
                        obligation="human_approval")
    return None


def rule_pii_egress(action: Action, agent: GovernedAgent, ctx: dict) -> Decision | None:
    """Data / RGPD (J6): block exporting personal data outside the perimeter.
    Hard DENY: there is no acceptable auto-exfiltration."""
    if action.params.get("contains_pii") and action.params.get("destination") == "external":
        return Decision(Verdict.DENY, "pii_egress",
                        "action would export personal data to an external destination")
    return None


class PolicyEngine:
    """PDP + PEP + MCP consent gate, collapsed into one governable choke point.

    The whole point: no path to a tool may bypass `enforce`. Every attempted
    action is decided, applied, and (by the caller) logged."""

    def __init__(self, rules: list[Rule], version: str, consent_required: set[str]) -> None:
        self.rules = rules
        self.version = version
        self.consent_required = consent_required

    def decide(self, action: Action, agent: GovernedAgent, ctx: dict) -> Decision:
        # Collect every rule that fired, then collapse by safety precedence.
        fired = [d for r in self.rules if (d := r(action, agent, ctx)) is not None]
        if not fired:
            return Decision(Verdict.ALLOW, "default", "no rule objected")
        return max(fired, key=lambda d: d.verdict)  # IntEnum order = severity

    def enforce(self, action: Action, agent: GovernedAgent, ctx: dict,
                user_consented: bool = False, human_approves: bool = False) -> tuple[Decision, bool]:
        # MCP principle: explicit user consent before invoking a sensitive tool.
        if action.tool in self.consent_required and not user_consented:
            return (Decision(Verdict.DENY, "mcp_consent",
                             f"tool '{action.tool}' requires explicit consent (not given)"), False)
        decision = self.decide(action, agent, ctx)
        if decision.verdict == Verdict.ALLOW:
            return decision, True
        if decision.verdict == Verdict.OBLIGE and decision.obligation == "human_approval" and human_approves:
            return decision, True           # obligation satisfied -> may run
        return decision, False              # DENY or unsatisfied OBLIGE -> blocked


# ===========================================================================
# (3) LOG — append-only, hash-chained, tamper-evident audit trail.
#     Field names echo OTel GenAI semconv (experimental, H1 2026).          [J9]
# ===========================================================================

GENESIS = "GENESIS"


def _canonical(payload: dict) -> str:
    # WHY: the hash must be reproducible byte-for-byte by any verifier ->
    # deterministic serialization (sorted keys, compact separators).
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _hash_entry(prev_hash: str, payload: dict) -> str:
    # WHY chaining: one silent edit ripples through every following hash ->
    # tamper-EVIDENT (we detect it). Same idea as a Git commit / Merkle chain.
    return hashlib.sha256((prev_hash + _canonical(payload)).encode("utf-8")).hexdigest()


class AuditTrail:
    """Append-only hash chain. Records the who/what/authorization/outcome of
    every governance decision so any action can be reconstructed and proven."""

    def __init__(self) -> None:
        self._chain: list[dict] = []

    @property
    def head_hash(self) -> str:
        return self._chain[-1]["entry_hash"] if self._chain else GENESIS

    def record(self, *, agent: GovernedAgent, action: Action, decision: Decision,
               executed: bool, policy_version: str) -> None:
        payload = {
            "ts": _utc_now_iso(),
            "agent_id": agent.agent_id,
            "owner": agent.owner,
            "tool": action.tool,
            "params": action.params,
            "required_scope": action.required_scope,
            "verdict": decision.verdict.name,
            "rule": decision.rule,
            "executed": executed,
            "policy_version": policy_version,
        }
        prev = self.head_hash
        self._chain.append({
            "index": len(self._chain),
            "payload": payload,
            "prev_hash": prev,
            "entry_hash": _hash_entry(prev, payload),
        })

    def verify(self) -> tuple[bool, int | None]:
        # Re-walk and recompute every hash. Return (False, index) at the FIRST
        # broken position (the exact locus of tampering), else (True, None).
        prev = GENESIS
        for rec in self._chain:
            if rec["prev_hash"] != prev:
                return False, rec["index"]
            if _hash_entry(prev, rec["payload"]) != rec["entry_hash"]:
                return False, rec["index"]
            prev = rec["entry_hash"]
        return True, None

    def entries(self) -> list[dict]:
        return [rec["payload"] for rec in self._chain]


# ===========================================================================
# (4) SCORE — risk scorer with agentic modulators (NIST AI RMF Measure->Manage). [J4]
# ===========================================================================

LIKELIHOOD_BY_TIER = {"low": 2, "medium": 3, "high": 4}   # anchored seed by risk tier
IMPACT_BY_TIER = {"low": 2, "medium": 3, "high": 4}
TREAT_THRESHOLD = 12   # criticality >= 12 -> TREAT
MONITOR_THRESHOLD = 6  # 6..11 -> MONITOR ; < 6 -> ACCEPT


@dataclass
class RiskScore:
    agent_id: str
    eff_likelihood: int
    eff_impact: int
    criticality: int
    decision: str          # TREAT | MONITOR | ACCEPT
    rationale: list[str] = field(default_factory=list)


def score_agent(agent: GovernedAgent) -> RiskScore:
    """WHY modulators: the SAME logical risk is worse on an autonomous +
    irreversible agent. Irreversible -> impact +1 (nothing to roll back);
    full autonomy -> likelihood +1 (no human net). Caps at 5 keep the scale sane."""
    rationale: list[str] = []
    likelihood = LIKELIHOOD_BY_TIER.get(agent.risk_tier, 2)
    impact = IMPACT_BY_TIER.get(agent.risk_tier, 2)
    if agent.handles_irreversible:
        impact = min(5, impact + 1)
        rationale.append("irreversible action -> impact +1")
    if agent.autonomous:
        likelihood = min(5, likelihood + 1)
        rationale.append("no human in the loop -> likelihood +1")
    crit = likelihood * impact
    decision = ("TREAT" if crit >= TREAT_THRESHOLD
                else "MONITOR" if crit >= MONITOR_THRESHOLD
                else "ACCEPT")
    return RiskScore(agent.agent_id, likelihood, impact, crit, decision, rationale)


# ===========================================================================
# (5) MAP — compliance crosswalk: one control -> N frameworks; isolate MANDATORY
#     gaps. Real-world frameworks: EU AI Act (2024/1689), NIST AI RMF, ISO 42001. [J7]
# ===========================================================================

@dataclass(frozen=True)
class Requirement:
    framework: str
    ref: str
    label: str
    mandatory: bool   # True = legal obligation (EU AI Act) ; False = voluntary standard


# Pedagogical subset; refs/dates verified against the domain references.
REQUIREMENTS: list[Requirement] = [
    Requirement("EU AI Act", "Art. 9", "Risk management system", mandatory=True),
    Requirement("EU AI Act", "Art. 12", "Record-keeping / logging", mandatory=True),
    Requirement("EU AI Act", "Art. 14", "Human oversight", mandatory=True),
    Requirement("NIST AI RMF", "Govern", "Roles & responsibilities", mandatory=False),
    Requirement("NIST AI RMF", "Map", "Risk mapping", mandatory=False),
    Requirement("NIST AI RMF", "Measure", "Measurement & traceability", mandatory=False),
    Requirement("NIST AI RMF", "Manage", "Risk treatment", mandatory=False),
    Requirement("ISO/IEC 42001", "6.1", "AI risk assessment", mandatory=False),
    Requirement("ISO/IEC 42001", "8.2", "Operational logging", mandatory=False),
    Requirement("ISO/IEC 42001", "5.3", "AIMS roles & responsibilities", mandatory=False),
]


def _req_key(r: Requirement) -> str:
    return f"{r.framework}::{r.ref}"


_REQ_BY_KEY = {_req_key(r): r for r in REQUIREMENTS}


@dataclass
class Control:
    control_id: str
    label: str
    covers: set[str] = field(default_factory=set)   # set of req keys


def default_controls(audit_ok: bool, has_named_owner: bool) -> list[Control]:
    """Controls implemented by the toolkit ITSELF, gated on live evidence.

    WHY evidence-gated: a crosswalk that always claims coverage is theatre. The
    audit-trail control only counts if the chain actually verifies; the human-
    oversight control (Art. 14) only counts if every agent has a named owner.
    This is how a missing owner becomes a MANDATORY legal gap in the report."""
    controls = [
        Control("CTRL-RISK", "Per-agent risk assessment", covers={
            "EU AI Act::Art. 9", "NIST AI RMF::Map", "NIST AI RMF::Measure",
            "ISO/IEC 42001::6.1",
        }),
    ]
    if audit_ok:
        controls.append(Control("CTRL-AUDIT", "Tamper-evident audit trail", covers={
            "EU AI Act::Art. 12", "NIST AI RMF::Manage", "ISO/IEC 42001::8.2",
        }))
    if has_named_owner:
        controls.append(Control("CTRL-OWNER", "Named human owner per agent", covers={
            "EU AI Act::Art. 14", "NIST AI RMF::Govern", "ISO/IEC 42001::5.3",
        }))
    return controls


def crosswalk(controls: list[Control]) -> dict:
    """Compute coverage per framework + the list of gaps (mandatory flagged)."""
    covered: set[str] = set()
    for c in controls:
        covered |= c.covers
    by_fw: dict[str, list[int]] = {}
    for r in REQUIREMENTS:
        stat = by_fw.setdefault(r.framework, [0, 0])
        stat[1] += 1
        if _req_key(r) in covered:
            stat[0] += 1
    gaps = [r for r in REQUIREMENTS if _req_key(r) not in covered]
    return {
        "coverage": {fw: (c, t) for fw, (c, t) in by_fw.items()},
        "gaps": gaps,
        "mandatory_gaps": [r for r in gaps if r.mandatory],
    }


# ===========================================================================
# (6) REPORT — board-ready output in BOTH markdown and JSON, ending on a verdict.
# ===========================================================================

@dataclass
class GovernanceResult:
    """Everything the report needs, gathered by the pipeline run."""
    org: str
    generated_at: str
    coverage: tuple[int, int, float]
    orphans: list[str]
    audit_entries: int
    audit_ok: bool
    audit_head: str
    enforcement: dict          # {"attempts": int, "blocked": int}
    scores: list[RiskScore]
    crosswalk: dict


def render_markdown(res: GovernanceResult) -> str:
    g, t, pct = res.coverage
    lines: list[str] = []
    lines.append("=" * 66)
    lines.append(f"AGENT GOVERNANCE REPORT — {res.org} — {res.generated_at[:10]}")
    lines.append("=" * 66)
    lines.append(f"Fleet: {t} agents | Governance coverage: {pct:.0f}% ({g}/{t} fully governed)")
    lines.append(f"Orphan agents (no named owner): {len(res.orphans)}"
                 + (f"  -> {', '.join(res.orphans)}" if res.orphans else ""))
    lines.append(f"Audit trail: {res.audit_entries} entries, integrity = "
                 f"{'VERIFIED' if res.audit_ok else 'BROKEN'} (head {res.audit_head[:8]}..)")
    lines.append("")
    lines.append("Risk posture (NIST AI RMF):")
    for s in sorted(res.scores, key=lambda x: x.criticality, reverse=True):
        why = ("; ".join(s.rationale)) if s.rationale else "-"
        lines.append(f"  {s.decision:<8} {s.agent_id:<18} crit={s.criticality:<3} "
                     f"(L{s.eff_likelihood} x I{s.eff_impact})  {why}")
    lines.append("")
    en = res.enforcement
    lines.append("Policy enforcement (runtime):")
    lines.append(f"  attempts={en['attempts']}  blocked={en['blocked']}  -> "
                 f"{en['blocked']} risky action(s) stopped before execution")
    lines.append("")
    lines.append("Compliance crosswalk:")
    for fw, (cov, tot) in sorted(res.crosswalk["coverage"].items()):
        pctfw = (cov / tot * 100) if tot else 0
        gap_note = ""
        fw_mand_gaps = [r for r in res.crosswalk["mandatory_gaps"] if r.framework == fw]
        if fw_mand_gaps:
            g0 = fw_mand_gaps[0]
            gap_note = f"   ! gap {g0.ref} ({g0.label}) — MANDATORY"
        lines.append(f"  {fw:<14} {cov}/{tot} ({pctfw:.0f}%){gap_note}")
    lines.append("")
    # WHY end on a verdict: a board decides, it does not read raw tables.
    n_mand = len(res.crosswalk["mandatory_gaps"])
    if not res.audit_ok:
        lines.append("VERDICT: audit integrity BROKEN -> evidence not trustworthy, halt scale-up.")
    elif n_mand:
        lines.append(f"VERDICT: {n_mand} mandatory legal gap(s) -> remediation required before scale-up.")
    elif res.orphans:
        lines.append("VERDICT: no legal gap, but orphan agents remain -> assign owners before scale-up.")
    else:
        lines.append("VERDICT: fleet governed, no mandatory gap -> cleared to scale (re-run monthly).")
    lines.append("=" * 66)
    return "\n".join(lines)


def render_json(res: GovernanceResult) -> str:
    g, t, pct = res.coverage
    payload = {
        "org": res.org,
        "generated_at": res.generated_at,
        "coverage": {"governed": g, "total": t, "pct": round(pct, 1)},
        "orphans": res.orphans,
        "audit": {"entries": res.audit_entries, "verified": res.audit_ok,
                  "head_hash": res.audit_head},
        "enforcement": res.enforcement,
        "risk": [{"agent_id": s.agent_id, "criticality": s.criticality,
                  "decision": s.decision} for s in res.scores],
        "compliance": {
            "coverage": {fw: {"covered": c, "total": t2}
                         for fw, (c, t2) in res.crosswalk["coverage"].items()},
            "mandatory_gaps": [f"{r.framework} {r.ref}"
                               for r in res.crosswalk["mandatory_gaps"]],
        },
    }
    # JSON is the machine-replayable, diffable, archivable form of the same report.
    return json.dumps(payload, indent=2, sort_keys=True)


# ===========================================================================
# THE PIPELINE — ingest -> enforce -> log -> score -> map -> report.
# ===========================================================================

def run_governance(org: str, fleet: list[dict],
                   attempts: list[dict],
                   ctx: dict) -> tuple[GovernanceResult, AuditTrail]:
    """Run the full toolkit end-to-end and return the gathered result + trail."""
    # (1) INGEST
    registry = Registry()
    registry.ingest(fleet)

    # (2) ENFORCE + (3) LOG (indissociable: every decision is recorded)
    engine = PolicyEngine(
        rules=[rule_scope, rule_budget, rule_high_risk_irreversible, rule_pii_egress],
        version="policy-v1.0.0",
        consent_required=ctx.get("consent_required", set()),
    )
    trail = AuditTrail()
    blocked = 0
    for att in attempts:
        agent = registry.get(att["agent_id"])
        if agent is None:
            continue  # cannot enforce on an unknown agent
        action = Action(att["tool"], att.get("params", {}), att["required_scope"])
        decision, executed = engine.enforce(
            action, agent, ctx,
            user_consented=att.get("user_consented", False),
            human_approves=att.get("human_approves", False),
        )
        trail.record(agent=agent, action=action, decision=decision,
                     executed=executed, policy_version=engine.version)
        if not executed:
            blocked += 1
    audit_ok, _ = trail.verify()

    # (4) SCORE
    scores = [score_agent(a) for a in registry.all()]

    # (5) MAP (controls gated on live evidence: audit integrity + every owner named)
    has_named_owner = len(registry.orphans()) == 0
    controls = default_controls(audit_ok=audit_ok, has_named_owner=has_named_owner)
    cw = crosswalk(controls)

    # (6) gather everything the REPORT needs
    res = GovernanceResult(
        org=org,
        generated_at=_utc_now_iso(),
        coverage=registry.coverage(),
        orphans=[a.agent_id for a in registry.orphans()],
        audit_entries=len(trail.entries()),
        audit_ok=audit_ok,
        audit_head=trail.head_hash,
        enforcement={"attempts": len(trail.entries()), "blocked": blocked},
        scores=scores,
        crosswalk=cw,
    )
    return res, trail


# ===========================================================================
# EXAMPLE FLEET + DEMO
# ===========================================================================

def _example_fleet() -> list[dict]:
    return [
        {"agent_id": "agent-finance-01", "owner": "a.dupont",
         "scopes": ["payments:execute", "ledger:read"], "risk_tier": "high",
         "autonomous": True, "handles_irreversible": True},
        {"agent_id": "agent-support-02", "owner": "m.martin",
         "scopes": ["ticket:read", "refund:execute"], "risk_tier": "medium",
         "autonomous": False, "handles_irreversible": False},
        {"agent_id": "agent-report-03", "owner": "c.bernard",
         "scopes": ["data:read"], "risk_tier": "low",
         "autonomous": False, "handles_irreversible": False},
        # Orphan: no owner -> shadow agent. The toolkit must SURFACE it.
        {"agent_id": "agent-scraper-07", "owner": None,
         "scopes": ["web:fetch"], "risk_tier": "medium"},
    ]


def _example_attempts() -> list[dict]:
    return [
        # finance agent, in scope, small amount -> ALLOW
        {"agent_id": "agent-finance-01", "tool": "bank_transfer",
         "params": {"amount": 800, "currency": "EUR"},
         "required_scope": "payments:execute"},
        # finance agent, big amount -> OBLIGE; no approval -> blocked
        {"agent_id": "agent-finance-01", "tool": "bank_transfer",
         "params": {"amount": 40000, "currency": "EUR"},
         "required_scope": "payments:execute", "human_approves": False},
        # same big amount, human approves -> executes
        {"agent_id": "agent-finance-01", "tool": "bank_transfer",
         "params": {"amount": 40000, "currency": "EUR"},
         "required_scope": "payments:execute", "human_approves": True},
        # support agent issuing a refund: sensitive tool needs MCP consent (missing) -> DENY
        {"agent_id": "agent-support-02", "tool": "issue_refund",
         "params": {"amount": 50}, "required_scope": "refund:execute"},
        # report agent tries an action whose scope it lacks -> DENY
        {"agent_id": "agent-report-03", "tool": "delete_record",
         "params": {}, "required_scope": "data:delete"},
        # scraper exfiltrates PII to an external destination -> DENY (RGPD)
        {"agent_id": "agent-scraper-07", "tool": "export_data",
         "params": {"contains_pii": True, "destination": "external"},
         "required_scope": "web:fetch"},
    ]


def main() -> None:
    ctx = {
        "auto_amount_limit": 1000,
        "irreversible_tools": {"bank_transfer", "delete_record"},
        "consent_required": {"issue_refund", "export_data"},
    }
    res, trail = run_governance("Acme Corp", _example_fleet(),
                                _example_attempts(), ctx)

    # --- Human-facing markdown report (the board reads + signs this) ---
    print(render_markdown(res))

    # --- A peek at the tamper-evident audit trail behind the numbers (J9) ---
    print("\nAudit trail (append-only, hash-chained):")
    for e in trail.entries():
        print(f"  {e['agent_id']:<18} {e['tool']:<14} {e['verdict']:<7} "
              f"rule={e['rule']:<22} executed={e['executed']}")

    # --- Adversarial probe: silently rewrite a past decision -> must be caught ---
    print("\nAdversarial probe: flip a past 'DENY' to 'ALLOW' in the trail...")
    if trail._chain:
        trail._chain[-1]["payload"]["verdict"] = "ALLOW"   # the silent edit
    ok, broken = trail.verify()
    print(f"  verify() -> ok={ok}, broken_index={broken} "
          f"(tampering detected at exact position; report would read BROKEN)")

    # --- Machine-facing JSON (replay / diff month-over-month / archive) ---
    print("\nJSON artifact (truncated head):")
    js = render_json(res)
    print("\n".join(js.splitlines()[:12]) + "\n  ...")


if __name__ == "__main__":
    main()
