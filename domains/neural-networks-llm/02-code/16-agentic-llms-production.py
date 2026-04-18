"""
Jour 16 — Agentic LLMs en production
=====================================
Pure Python, pas de dependance externe. Le "LLM" est simule pour pouvoir
executer en local sans cle API.

Objectif pedagogique : construire une boucle agentique realiste avec les
garde-fous qu'on trouve dans Claude Code, OpenAI Agents SDK, LangGraph :
  1. Une boucle run-to-completion avec tool calls
  2. Tool registry + JSON schema + validation
  3. Budget tokens / max_turns / timeout par tool
  4. Observabilite structuree de chaque tool call
  5. Detection de context rot et strategie de compaction
  6. Gestion d'une action destructive avec human-in-the-loop

Run : python 02-code/16-agentic-llms-production.py
"""

from __future__ import annotations
import sys
import io
import time
import json
import random
import uuid
import re
from dataclasses import dataclass, field
from typing import Callable, Any

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

random.seed(11)

# ============================================================================
# 1) Tool registry : chaque outil a un schema + une implementation + des limits
# ============================================================================


@dataclass
class Tool:
    name: str
    description: str
    schema: dict          # JSON schema (parameters)
    impl: Callable[[dict], str]
    timeout_s: float = 5.0
    dangerous: bool = False   # si True → human-in-the-loop


REGISTRY: dict[str, Tool] = {}


def register(tool: Tool):
    REGISTRY[tool.name] = tool


# -- Outils exemples -------------------------------------------------------

def _tool_list_files(args):
    path = args.get("path", ".")
    # Simulation deterministe
    fake_fs = {
        ".": ["README.md", "src/", "tests/", "data.csv"],
        "src/": ["main.py", "utils.py"],
        "tests/": ["test_main.py"],
    }
    return "\n".join(fake_fs.get(path, []))


def _tool_read_file(args):
    path = args["path"]
    fake_files = {
        "README.md": "# My project\nHello world.",
        "src/main.py": "def main():\n    print('hi')\n",
        "data.csv": "id,value\n1,10\n2,20\n3,30\n",
    }
    if path not in fake_files:
        raise FileNotFoundError(path)
    return fake_files[path]


def _tool_query_db(args):
    sql = args["sql"]
    # On simule un latence variable, parfois au-dela du timeout pour demontrer
    # la gestion d'erreur (J16 regle #3 : timeout partout).
    time.sleep(random.uniform(0.01, 0.5))
    if "drop" in sql.lower() or "delete" in sql.lower():
        raise PermissionError("Write SQL requires the dangerous=True path")
    return json.dumps([{"id": 1, "name": "ACME"}, {"id": 2, "name": "MASA"}])


def _tool_send_email(args):
    # Outil dangereux : effet de bord externe.
    return f"Email envoye a {args['to']} (objet: {args['subject']})"


register(Tool(
    "list_files",
    "Liste les fichiers dans un chemin. Utile pour explorer le repo avant de lire.",
    {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    _tool_list_files,
))
register(Tool(
    "read_file",
    "Lit le contenu texte d'un fichier. Renvoie FileNotFoundError si absent.",
    {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    _tool_read_file,
))
register(Tool(
    "query_db",
    "Execute une requete SELECT read-only sur la DB. Interdit DROP/DELETE/UPDATE.",
    {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]},
    _tool_query_db, timeout_s=0.3,
))
register(Tool(
    "send_email",
    "Envoie un email. Action destructive, validation humaine requise.",
    {"type": "object", "properties": {
        "to": {"type": "string"}, "subject": {"type": "string"},
        "body": {"type": "string"}},
     "required": ["to", "subject", "body"]},
    _tool_send_email, dangerous=True,
))

# ============================================================================
# 2) Simulation d'un LLM qui emet des tool_calls structurees
# ============================================================================


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)


class MockAgentLLM:
    """
    Simule un agent qui doit resoudre la tache : "compte le nombre de lignes
    non-header dans data.csv et envoie un rapport a admin@example.com". On
    hardcode le plan pour pouvoir executer en local, mais la forme est exactement
    celle d'un vrai LLM (tool_calls).
    """

    def __init__(self):
        self.step = 0

    def respond(self, history: list[dict]) -> LLMResponse:
        self.step += 1
        if self.step == 1:
            return LLMResponse(tool_calls=[ToolCall(uuid.uuid4().hex, "list_files",
                                                   {"path": "."})])
        if self.step == 2:
            return LLMResponse(tool_calls=[ToolCall(uuid.uuid4().hex, "read_file",
                                                   {"path": "data.csv"})])
        if self.step == 3:
            # Reflechit : ici un vrai LLM compterait dans son contexte.
            return LLMResponse(tool_calls=[ToolCall(uuid.uuid4().hex, "send_email",
                                                   {"to": "admin@example.com",
                                                    "subject": "Rapport CSV",
                                                    "body": "3 lignes de donnees."})])
        if self.step == 4:
            return LLMResponse(text="Done. 3 lignes de donnees, email envoye.")
        return LLMResponse(text="Agent idle.")


# ============================================================================
# 3) Validation du JSON schema (minimale, mais fidele a l'esprit)
# ============================================================================


def validate_args(schema: dict, args: dict) -> None:
    required = schema.get("required", [])
    for r in required:
        if r not in args:
            raise ValueError(f"Missing required field: {r}")
    props = schema.get("properties", {})
    for k, v in args.items():
        if k in props:
            expected_type = props[k].get("type")
            type_map = {"string": str, "number": (int, float),
                        "integer": int, "boolean": bool, "object": dict,
                        "array": list}
            if expected_type in type_map and not isinstance(v, type_map[expected_type]):
                raise ValueError(f"Field {k} expected {expected_type}, got {type(v).__name__}")


# ============================================================================
# 4) Boucle agentique avec tous les garde-fous
# ============================================================================


@dataclass
class AgentConfig:
    max_turns: int = 20
    max_tokens: int = 50_000           # budget "fake" (on compte les chars)
    per_call_timeout_s: float = 2.0
    confirm_dangerous: Callable[[ToolCall], bool] = lambda c: True  # override en prod
    compaction_threshold: int = 30_000  # chars dans l'historique avant compaction


@dataclass
class AgentTrace:
    events: list[dict] = field(default_factory=list)

    def log(self, kind: str, **kwargs):
        self.events.append({"kind": kind, "t": time.time(), **kwargs})


def run_agent(task: str, llm: MockAgentLLM, cfg: AgentConfig) -> tuple[str, AgentTrace]:
    history: list[dict] = [{"role": "user", "content": task}]
    trace = AgentTrace()
    total_chars = len(task)

    for turn in range(cfg.max_turns):
        # --- Context rot mitigation : compaction si l'historique devient lourd
        if total_chars > cfg.compaction_threshold:
            trace.log("compaction", before_chars=total_chars)
            history = _compact(history)
            total_chars = sum(len(str(m.get("content", ""))) for m in history)
            trace.log("compaction_done", after_chars=total_chars)

        if total_chars > cfg.max_tokens:
            trace.log("kill", reason="token_budget_exceeded", chars=total_chars)
            return "[ABORTED: token budget]", trace

        resp = llm.respond(history)
        if resp.text and not resp.tool_calls:
            trace.log("final", text=resp.text)
            return resp.text, trace

        for call in resp.tool_calls:
            tool = REGISTRY.get(call.name)
            if tool is None:
                err = f"Unknown tool: {call.name}"
                trace.log("tool_error", call_id=call.id, error=err)
                history.append({"role": "tool", "tool_call_id": call.id,
                                "content": f"Error: {err}"})
                continue

            # Validation args
            try:
                validate_args(tool.schema, call.arguments)
            except Exception as e:
                trace.log("tool_error", call_id=call.id, tool=call.name, error=str(e))
                history.append({"role": "tool", "tool_call_id": call.id,
                                "content": f"Error: {e}"})
                continue

            # Human-in-the-loop pour outils destructifs
            if tool.dangerous:
                approved = cfg.confirm_dangerous(call)
                trace.log("dangerous_prompt", tool=call.name,
                          args=call.arguments, approved=approved)
                if not approved:
                    history.append({"role": "tool", "tool_call_id": call.id,
                                    "content": "Error: user denied"})
                    continue

            # Execution avec timeout
            t0 = time.time()
            try:
                result = _execute_with_timeout(tool, call.arguments,
                                               max(cfg.per_call_timeout_s,
                                                   tool.timeout_s))
                elapsed = time.time() - t0
                trace.log("tool_ok", tool=call.name, call_id=call.id,
                          elapsed_ms=int(elapsed * 1000), chars=len(result))
                history.append({"role": "tool", "tool_call_id": call.id,
                                "content": result})
                total_chars += len(result)
            except Exception as e:
                trace.log("tool_error", tool=call.name, call_id=call.id,
                          error=str(e))
                history.append({"role": "tool", "tool_call_id": call.id,
                                "content": f"Error: {e}"})

    trace.log("kill", reason="max_turns")
    return "[ABORTED: max turns]", trace


def _execute_with_timeout(tool: Tool, args: dict, timeout_s: float) -> str:
    # En Python synchro pur, on fait une execution simple et on mesure le
    # temps ecoule. En prod on utiliserait signals ou un ProcessPoolExecutor.
    t0 = time.time()
    out = tool.impl(args)
    elapsed = time.time() - t0
    if elapsed > timeout_s:
        raise TimeoutError(f"{tool.name} took {elapsed:.2f}s > {timeout_s}s")
    return str(out)


def _compact(history: list[dict]) -> list[dict]:
    """Strategie naive de compaction : garder le user initial + un resume
    texte des tool calls et le dernier message. Un vrai systeme appelerait
    un LLM pour faire un resume semantique."""
    if len(history) < 3:
        return history
    user = history[0]
    summary = {"role": "system",
               "content": f"[compacted {len(history) - 2} messages — "
                          "summary: tool calls inspected local filesystem]"}
    last = history[-1]
    return [user, summary, last]


# ============================================================================
# 5) Run : deux runs avec et sans approbation humaine de l'outil dangereux
# ============================================================================


def demo(confirm_flag: bool, title: str):
    print("=" * 70)
    print(title)
    print("=" * 70)
    cfg = AgentConfig(confirm_dangerous=lambda c: confirm_flag)
    llm = MockAgentLLM()
    out, trace = run_agent(
        task="Compte les lignes non-header de data.csv et envoie un rapport.",
        llm=llm, cfg=cfg,
    )
    for e in trace.events:
        kind = e["kind"]
        rest = {k: v for k, v in e.items() if k not in ("kind", "t")}
        print(f"  [{kind}] {rest}")
    print(f"  => RESULT: {out}\n")


demo(confirm_flag=True, title="Run 1 : humain approuve l'envoi d'email")
demo(confirm_flag=False, title="Run 2 : humain refuse — l'agent s'adapte")


# ============================================================================
# 6) Bonus : MCP-like tool discovery (format standardise)
# ============================================================================

print("=" * 70)
print("Bonus : manifest MCP-like des outils exposes")
print("=" * 70)
manifest = {
    "tools": [
        {"name": t.name, "description": t.description,
         "inputSchema": t.schema, "dangerous": t.dangerous}
        for t in REGISTRY.values()
    ]
}
print(json.dumps(manifest, indent=2)[:600] + "\n...")
print("  Lecon : ce manifest est ce que ton MCP server expose via tools/list.")
print("  N'importe quelle app compatible MCP (Claude Code, Cursor, ...) le decouvre.")
