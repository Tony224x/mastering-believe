"""Day 9 — Solutions: runtime audit, observability & traceability.

One file covering the three exercise levels:

  # === EASY ===   one governable audit entry (who/what/when/authorization/outcome)
  # === MEDIUM === append-only hash-chained log + tamper-evident verifier
  # === HARD ===   incident reconstruction by trace_id + checkpoint anchoring

# requires: stdlib only
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone


def _utc_now_iso() -> str:
    # Governance traces must be timezone-explicit and sortable -> pin UTC ISO-8601.
    return datetime.now(timezone.utc).isoformat()


# === EASY ===
# Build ONE complete audit entry. The `authorization` block is what turns a
# plain application log into a governance audit trail.

def make_entry(agent_id, owner, action, params, scope, policy, decision, status):
    """Return a dict audit entry with the who/what/when/authorization/outcome quintuple."""
    return {
        "who": {"agent_id": agent_id, "owner": owner},
        "what": {"action": action, "params": params},
        "when": _utc_now_iso(),  # auto-generated UTC ISO-8601 timestamp
        "authorization": {"scope": scope, "policy": policy, "decision": decision},
        "outcome": {"status": status},
    }


def is_governable(entry):
    """True only if the 5 blocks exist AND scope/policy/decision are non-empty."""
    required_blocks = ("who", "what", "when", "authorization", "outcome")
    if not all(block in entry for block in required_blocks):
        return False
    auth = entry.get("authorization", {})
    # WHY: an entry that does not record on-what-authorization the action passed
    # cannot support incident reconstruction -> it is not governable.
    return all(auth.get(k) for k in ("scope", "policy", "decision"))


def _demo_easy():
    print("=== EASY: one governable audit entry ===")
    good = make_entry(
        agent_id="finance-ops", owner="a.dupont",
        action="bank_transfer", params={"amount": 40000, "currency": "EUR"},
        scope="payments:execute", policy="auto<50k", decision="ALLOW",
        status="success",
    )
    print(f"  entry: {json.dumps(good, sort_keys=True)}")
    print(f"  is_governable -> {is_governable(good)}")

    bad = make_entry(
        agent_id="finance-ops", owner="a.dupont",
        action="bank_transfer", params={"amount": 40000},
        scope="", policy="auto<50k", decision="ALLOW",  # empty scope!
        status="success",
    )
    print(f"  entry with empty scope -> is_governable = {is_governable(bad)}")
    assert is_governable(good) is True
    assert is_governable(bad) is False


# === MEDIUM ===
# Append-only, hash-chained, tamper-evident log + integrity verifier.

GENESIS = "GENESIS"


def _canonical(entry: dict) -> str:
    # WHY: deterministic serialization so anyone can recompute the same hash.
    return json.dumps(entry, sort_keys=True, separators=(",", ":"))


def _hash(prev_hash: str, entry: dict) -> str:
    return hashlib.sha256((prev_hash + _canonical(entry)).encode("utf-8")).hexdigest()


class HashChainLog:
    """Append-only log where each entry is chained to the previous hash."""

    def __init__(self):
        self.chain = []  # list of {index, entry, prev_hash, entry_hash}

    @property
    def head_hash(self) -> str:
        return self.chain[-1]["entry_hash"] if self.chain else GENESIS

    def append(self, entry: dict):
        prev = self.head_hash
        record = {
            "index": len(self.chain),
            "entry": entry,
            "prev_hash": prev,
            "entry_hash": _hash(prev, entry),
        }
        self.chain.append(record)
        return record

    def verify(self):
        # Re-walk and recompute every hash. Return (False, index) at the FIRST
        # position whose stored hash/prev-link no longer matches -> exact locus
        # of tampering. Otherwise (True, None).
        prev = GENESIS
        for record in self.chain:
            if record["prev_hash"] != prev:
                return False, record["index"]
            if _hash(prev, record["entry"]) != record["entry_hash"]:
                return False, record["index"]
            prev = record["entry_hash"]
        return True, None


def _demo_medium():
    print("\n=== MEDIUM: tamper-evident hash chain ===")
    log = HashChainLog()
    log.append(make_entry("orchestrator", "a.dupont", "plan", {"goal": "pay"},
                          "orch:run", "default", "ALLOW", "success"))
    log.append(make_entry("finance-ops", "a.dupont", "bank_transfer",
                          {"amount": 40000}, "payments:execute", "auto<50k",
                          "ALLOW", "success"))
    log.append(make_entry("banking-tool", "a.dupont", "tool_call",
                          {"http": 200}, "payments:execute", "auto<50k",
                          "ALLOW", "success"))
    print(f"  verify (intact) -> {log.verify()}")
    assert log.verify() == (True, None)

    # Silent edit of a stored entry: change the amount without re-hashing.
    log.chain[1]["entry"]["what"]["params"]["amount"] = 400
    ok, broken = log.verify()
    print(f"  after silent edit of index 1 -> verify = ({ok}, {broken})")
    assert ok is False and broken == 1


# === HARD ===
# Correlate across agents (trace_id), reconstruct an incident, and anchor the
# chain with checkpoints to resist a full-chain-recompute attacker.

class AuditTrail(HashChainLog):
    """Hash chain + trace correlation + checkpoint anchoring."""

    def log_action(self, *, trace_id, span_id, parent_span, agent_id, owner,
                   action, params, scope, policy, decision, status):
        entry = make_entry(agent_id, owner, action, params, scope, policy,
                           decision, status)
        # Attach correlation metadata (OTel-style trace_id/span_id/parent).
        entry["correlation"] = {"trace_id": trace_id, "span_id": span_id,
                                "parent_span": parent_span}
        return self.append(entry)

    def reconstruct_incident(self, trace_id):
        # Filter the whole trail by trace_id, preserving insertion order.
        return [r for r in self.chain
                if r["entry"]["correlation"]["trace_id"] == trace_id]

    def narrate(self, trace_id, action="bank_transfer"):
        ok, broken = self.verify()
        prefix = "" if ok else f"[WARNING: integrity broken at index {broken}] "
        steps = self.reconstruct_incident(trace_id)
        target = next((r for r in steps if r["entry"]["what"]["action"] == action),
                      None)
        if target is None:
            return prefix + f"no '{action}' found for trace {trace_id}"
        e = target["entry"]
        return (prefix + "At {when}, agent {agent}({owner}) executed "
                "{act}({params}); scope {scope}, policy {policy} -> {decision}; "
                "status {status}.".format(
                    when=e["when"], agent=e["who"]["agent_id"],
                    owner=e["who"]["owner"], act=e["what"]["action"],
                    params=e["what"]["params"], scope=e["authorization"]["scope"],
                    policy=e["authorization"]["policy"],
                    decision=e["authorization"]["decision"],
                    status=e["outcome"]["status"]))

    def checkpoint(self):
        # Anchor: freeze "the log was in THIS state at THIS time". Shipping this
        # head_hash to an external append-only system catches a recompute attack.
        return {"checkpoint_at": _utc_now_iso(),
                "entries": len(self.chain),
                "head_hash": self.head_hash}


def _demo_hard():
    print("\n=== HARD: incident reconstruction + checkpoint anchoring ===")
    trail = AuditTrail()
    trace = "T-7f3a"
    trail.log_action(trace_id=trace, span_id="A", parent_span=None,
                     agent_id="orchestrator", owner="a.dupont",
                     action="plan", params={"goal": "pay supplier"},
                     scope="orch:run", policy="default", decision="ALLOW",
                     status="success")
    trail.log_action(trace_id=trace, span_id="B", parent_span="A",
                     agent_id="finance-ops", owner="a.dupont",
                     action="bank_transfer", params={"amount": 40000, "currency": "EUR"},
                     scope="payments:execute", policy="auto<50k",
                     decision="ALLOW", status="success")
    trail.log_action(trace_id=trace, span_id="C", parent_span="B",
                     agent_id="banking-tool", owner="a.dupont",
                     action="tool_call", params={"http": 200},
                     scope="payments:execute", policy="auto<50k",
                     decision="ALLOW", status="success")
    # Unrelated action under a different trace_id.
    trail.log_action(trace_id="T-9c11", span_id="A", parent_span=None,
                     agent_id="support-bot", owner="m.martin",
                     action="send_email", params={"to": "client@example.com"},
                     scope="email:send", policy="default", decision="ALLOW",
                     status="success")

    steps = trail.reconstruct_incident(trace)
    print(f"  reconstruct_incident({trace}) -> {len(steps)} spans "
          f"({[s['entry']['correlation']['span_id'] for s in steps]})")
    assert len(steps) == 3
    assert all(s["entry"]["correlation"]["trace_id"] == trace for s in steps)

    print(f"  narrate -> {trail.narrate(trace)}")

    # Anchor BEFORE the attack.
    cp = trail.checkpoint()
    print(f"  checkpoint (trusted): entries={cp['entries']} head={cp['head_hash'][:12]}...")

    # Attacker edits a past entry AND recomputes the whole chain so verify() passes.
    trail.chain[1]["entry"]["what"]["params"]["amount"] = 400
    prev = GENESIS
    for record in trail.chain:  # full recompute -> internal consistency restored
        record["prev_hash"] = prev
        record["entry_hash"] = _hash(prev, record["entry"])
        prev = record["entry_hash"]

    ok, broken = trail.verify()
    print(f"  after full-chain recompute attack -> verify = ({ok}, {broken})")
    # WHY verify() is fooled: it only checks INTERNAL consistency; an attacker
    # who controls the whole file can re-chain everything. The external anchor
    # is the safety net.
    tampered_head = trail.head_hash
    detected = tampered_head != cp["head_hash"]
    print(f"  head_hash now != trusted checkpoint head_hash -> tampering detected = {detected}")
    assert ok is True            # verify() alone is fooled
    assert detected is True      # checkpoint comparison catches it


if __name__ == "__main__":
    _demo_easy()
    _demo_medium()
    _demo_hard()
    print("\nAll smoke tests passed.")
