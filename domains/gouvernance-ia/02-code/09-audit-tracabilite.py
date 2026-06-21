"""Day 9 — Runtime audit, observability & traceability for AI agents.

What this script demonstrates
-----------------------------
A *tamper-evident, append-only* audit trail for agent actions, built with
the Python standard library only. It mirrors three real governance tools in
miniature and names them in comments:

  * a hash-chained audit log  -> the Merkle/Git-style integrity idea, here
    as `hash = SHA256(prev_hash + canonical(payload))`;
  * structured audit entries  -> the OpenTelemetry GenAI semantic conventions
    (spans, trace_id/span_id, tool calls, token usage). NOTE: those
    conventions are *Development/Experimental* as of H1 2026, NOT stable;
  * an integrity verifier + incident reconstruction -> the "machine proof"
    that lets you answer who / what / when / on-what-authorization after the
    fact (the basis of any credible incident response, cf. NIST AI 100-2).

The demo: log a short flight of agent actions (including a 40k EUR transfer),
verify the chain is intact, reconstruct one incident by trace_id, then show
that *any* silent edit to a past entry is detected at its exact position.

# requires: stdlib only
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

GENESIS = "GENESIS"  # sentinel prev_hash for the first entry in the chain


def _utc_now_iso() -> str:
    # WHY: governance traces must be timezone-explicit and comparable across
    # systems; we pin UTC ISO-8601 so timestamps are unambiguous and sortable.
    return datetime.now(timezone.utc).isoformat()


def _canonical(payload: dict) -> str:
    # WHY: the hash must be reproducible byte-for-byte by anyone re-checking
    # the chain. Dict key order is not guaranteed across processes, so we
    # serialize with sort_keys=True -> a single canonical form per payload.
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _hash_entry(prev_hash: str, payload: dict) -> str:
    # WHY: chaining each entry to the previous hash makes a single edit ripple
    # through every following hash -> tamper-EVIDENT (we detect it), even if
    # not tamper-PROOF (we cannot prevent a write). Same idea as a Git commit.
    material = (prev_hash + _canonical(payload)).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


@dataclass
class AuditEntry:
    """One audit record: the who / what / when / authorization / outcome quintuple.

    Field names intentionally echo OTel GenAI semconv (trace_id, span_id, the
    tool/action attributes) — a vendor-neutral, *experimental* H1-2026 standard.
    """

    # --- correlation (OTel-style distributed tracing) ---
    trace_id: str   # one id for the whole end-to-end request
    span_id: str    # one id per step
    parent_span: str | None  # links a step to its caller -> causal tree

    # --- who ---
    agent_id: str
    owner: str      # the accountable human (pillar 2 from day 2)

    # --- what ---
    action: str
    params: dict

    # --- authorization (the field that makes this *governance*, not just a log) ---
    scope: str          # the OAuth-like permission used (cf. day 8)
    policy: str         # which policy was evaluated (cf. day 14)
    decision: str       # "ALLOW" / "DENY"

    # --- outcome + when ---
    status: str
    timestamp: str = field(default_factory=_utc_now_iso)

    def payload(self) -> dict:
        # WHY: the hash covers the full semantic content of the entry EXCEPT the
        # chain metadata itself (prev_hash/entry_hash live on the chain record),
        # so re-hashing is deterministic and order-independent.
        return asdict(self)


@dataclass
class ChainRecord:
    """An AuditEntry plus its position in the hash chain."""

    index: int
    entry: AuditEntry
    prev_hash: str
    entry_hash: str


class AuditLog:
    """Append-only, hash-chained, tamper-evident audit trail (in-memory).

    Real-world equivalents: an immutable/WORM store anchored by periodic
    checkpoints. Here we keep it in a list and expose the same guarantees:
    you can append and verify, but a silent edit is always detected.
    """

    def __init__(self) -> None:
        self._chain: list[ChainRecord] = []

    @property
    def head_hash(self) -> str:
        # The last hash = a fingerprint of the ENTIRE history. This is what you
        # would periodically "anchor" (checkpoint) to an external append-only
        # system so even a full-file recompute attack is caught.
        return self._chain[-1].entry_hash if self._chain else GENESIS

    def append(self, entry: AuditEntry) -> ChainRecord:
        prev = self.head_hash
        entry_hash = _hash_entry(prev, entry.payload())
        record = ChainRecord(
            index=len(self._chain),
            entry=entry,
            prev_hash=prev,
            entry_hash=entry_hash,
        )
        self._chain.append(record)
        return record

    def verify(self) -> tuple[bool, int | None]:
        # WHY: re-walk the chain and recompute each hash from scratch. The first
        # position where the recomputed hash != stored hash (or prev_hash breaks)
        # is exactly where tampering occurred. Returns (ok, broken_index).
        prev = GENESIS
        for record in self._chain:
            if record.prev_hash != prev:
                return False, record.index  # broken link to previous entry
            recomputed = _hash_entry(prev, record.entry.payload())
            if recomputed != record.entry_hash:
                return False, record.index  # entry content was altered
            prev = record.entry_hash
        return True, None

    def checkpoint(self) -> dict:
        # WHY: an "anchor" you would ship to a third party. Freezes "the log was
        # in THIS state at THIS time" beyond an attacker who controls the file.
        return {
            "checkpoint_at": _utc_now_iso(),
            "entries": len(self._chain),
            "head_hash": self.head_hash,
        }

    def reconstruct_incident(self, trace_id: str) -> list[ChainRecord]:
        # WHY: incident response asks "what did the agent actually do?". We
        # filter the whole trail by trace_id and return the ordered steps so a
        # defensible narrative can be written (who / what / when / authorization).
        return [r for r in self._chain if r.entry.trace_id == trace_id]


def _demo() -> None:
    log = AuditLog()

    # --- A flight of agent actions sharing one trace_id (one user request) ---
    trace = "T-7f3a"
    log.append(AuditEntry(
        trace_id=trace, span_id="A", parent_span=None,
        agent_id="orchestrator", owner="a.dupont",
        action="plan_request", params={"goal": "pay supplier"},
        scope="orchestration:run", policy="default", decision="ALLOW",
        status="success",
    ))
    log.append(AuditEntry(
        trace_id=trace, span_id="B", parent_span="A",
        agent_id="finance-ops", owner="a.dupont",
        action="bank_transfer", params={"amount": 40000, "currency": "EUR"},
        scope="payments:execute", policy="auto<50k", decision="ALLOW",
        status="success",
    ))
    log.append(AuditEntry(
        trace_id=trace, span_id="C", parent_span="B",
        agent_id="banking-api-tool", owner="a.dupont",
        action="tool_call", params={"endpoint": "POST /transfers", "http": 200},
        scope="payments:execute", policy="auto<50k", decision="ALLOW",
        status="success",
    ))
    # An unrelated action by another agent (different trace_id) — proves the
    # incident reconstruction filters correctly.
    log.append(AuditEntry(
        trace_id="T-9c11", span_id="A", parent_span=None,
        agent_id="support-bot", owner="m.martin",
        action="send_email", params={"to": "client@example.com"},
        scope="email:send", policy="default", decision="ALLOW",
        status="success",
    ))

    print("=== 1. Audit trail written (append-only) ===")
    for r in log._chain:
        e = r.entry
        print(f"  #{r.index} {e.timestamp}  trace={e.trace_id} span={e.span_id}"
              f"  {e.agent_id}({e.owner}) {e.action} -> {e.status}"
              f"  [hash {r.entry_hash[:12]}...]")

    print("\n=== 2. Integrity check on the intact chain ===")
    ok, broken = log.verify()
    print(f"  verify() -> ok={ok}, broken_index={broken}")

    print("\n=== 3. External checkpoint (anchor the head hash) ===")
    print(f"  {json.dumps(log.checkpoint(), indent=None)}")

    print("\n=== 4. Incident reconstruction by trace_id (the 40k EUR transfer) ===")
    steps = log.reconstruct_incident(trace)
    for r in steps:
        e = r.entry
        print(f"  {e.span_id}<-{e.parent_span}  {e.agent_id} {e.action}"
              f"  scope={e.scope} policy={e.policy} decision={e.decision}"
              f"  params={e.params}")
    transfer = next(r.entry for r in steps if r.entry.action == "bank_transfer")
    print("  Narrative: at {ts}, agent '{a}' owned by '{o}' executed "
          "{act}({amt} {cur}); scope '{s}', policy '{p}' returned {d}; "
          "status={st}. Chain verified.".format(
              ts=transfer.timestamp, a=transfer.agent_id, o=transfer.owner,
              act=transfer.action, amt=transfer.params["amount"],
              cur=transfer.params["currency"], s=transfer.scope,
              p=transfer.policy, d=transfer.decision, st=transfer.status))

    print("\n=== 5. Adversarial: silently rewrite the 40k transfer to 400 ===")
    # WHY: simulate an insider editing a past entry to hide the real amount.
    # The stored entry_hash is NOT recomputed -> verify() must catch the lie.
    tampered = log._chain[1].entry
    print(f"  before: amount={tampered.params['amount']}")
    tampered.params["amount"] = 400  # the silent edit
    print(f"  after:  amount={tampered.params['amount']}")
    ok, broken = log.verify()
    print(f"  verify() -> ok={ok}, broken_index={broken}")
    print("  -> tampering detected at the exact position of the edit "
          "(this is what a mutable app.log could never prove).")


if __name__ == "__main__":
    _demo()
