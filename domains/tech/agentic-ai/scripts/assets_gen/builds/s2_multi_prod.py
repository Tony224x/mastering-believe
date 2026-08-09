"""Builders SVG — modules pédagogiques."""
from __future__ import annotations

from ..kit import (
    FONT,
    P,
    arrow_h,
    card,
    card_header,
    esc,
    node_box,
    title_block,
    wrap_svg,
)

def build_08() -> str:
    body = title_block(
        "RAG vanilla vs RAG agentique",
        "Une seule récupération vs boucle de raisonnement",
        "Module 08 · agentic-ai",
    )
    body += f"""  {card(80, 140, 500, 420)}
  {card_header(80, 140, 500, "RAG vanilla", P["muted"])}
  <text x="330" y="240" text-anchor="middle" font-family="{FONT}" font-size="15" fill="{P["ink"]}">Query → embed → top-k → LLM</text>
  <text x="330" y="290" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["muted"]}">1 retrieval, 1 génération</text>
  <text x="330" y="340" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["muted"]}">Fragile si la query est ambigüe</text>
  <text x="330" y="390" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["muted"]}">Pas de multi-hop ni re-query</text>
  <text x="330" y="460" text-anchor="middle" font-family="{FONT}" font-size="14" font-weight="600" fill="{P["muted"]}">OK pour FAQ simple</text>

  {card(620, 140, 500, 420)}
  {card_header(620, 140, 500, "RAG agentique", P["primary"])}
  <text x="870" y="240" text-anchor="middle" font-family="{FONT}" font-size="15" fill="{P["ink"]}">Décompose · route · retrieve</text>
  <text x="870" y="290" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["muted"]}">Grade les docs · multi-hop</text>
  <text x="870" y="340" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["muted"]}">Re-query si insuffisant</text>
  <text x="870" y="390" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["muted"]}">Le LLM pilote la recherche</text>
  <text x="870" y="460" text-anchor="middle" font-family="{FONT}" font-size="14" font-weight="600" fill="{P["primary"]}">Pour questions complexes</text>"""
    return wrap_svg(
        "RAG vanilla vs agentique",
        "À gauche RAG vanilla (query embed top-k LLM). À droite RAG agentique : décomposition, routing, grading, multi-hop, re-query.",
        body,
    )

def build_09() -> str:
    body = title_block(
        "4 patterns multi-agent",
        "Supervisor · Swarm · Hiérarchie · Débat",
        "Module 09 · agentic-ai",
    )
    patterns = [
        (80, "Supervisor", "Chef délègue\naux workers"),
        (340, "Swarm", "Handoff peer\nà peer"),
        (600, "Hiérarchie", "Niveaux de\nsupervision"),
        (860, "Débat", "Critique\ncroisée"),
    ]
    for x, t, s in patterns:
        body += f"""  {card(x, 180, 240, 300)}
  <rect x="{x + 70}" y="220" width="100" height="100" rx="50" fill="{P["teal_soft"]}" stroke="{P["primary"]}" stroke-width="2"/>
  <text x="{x + 120}" y="278" text-anchor="middle" font-family="{FONT}" font-size="15" font-weight="700" fill="{P["primary"]}">{esc(t[:3].upper())}</text>
  <text x="{x + 120}" y="360" text-anchor="middle" font-family="{FONT}" font-size="18" font-weight="700" fill="{P["ink"]}">{esc(t)}</text>"""
        for i, line in enumerate(s.split("\n")):
            body += f'  <text x="{x + 120}" y="{400 + i * 26}" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["muted"]}">{esc(line)}</text>'
    body += f'  <text x="600" y="560" text-anchor="middle" font-family="{FONT}" font-size="15" fill="{P["ink"]}">Commence single-agent. Multi seulement si rôles distincts.</text>'
    return wrap_svg(
        "Patterns multi-agent",
        "Quatre archétypes : Supervisor (chef + workers), Swarm (handoffs), Hiérarchie, Débat (critique croisée). Commencer single-agent.",
        body,
    )

def build_10() -> str:
    body = title_block(
        "MCP : le USB des LLMs",
        "Un protocole pour brancher tools, resources, prompts",
        "Module 10 · agentic-ai",
    )
    body += node_box(
        120,
        250,
        220,
        120,
        "Host / Agent",
        "Claude, app…",
        P["teal_soft"],
        P["primary_deep"],
    )
    body += f"  {arrow_h(340, 310, 420)}"
    body += node_box(
        420, 250, 220, 120, "MCP Client", "session protocole", P["card"], P["ink"]
    )
    body += f"  {arrow_h(640, 310, 720)}"
    body += node_box(
        720,
        200,
        360,
        220,
        "MCP Server",
        "tools · resources · prompts",
        P["amber_soft"],
        "#92400E",
    )
    body += f"""  <text x="900" y="380" text-anchor="middle" font-family="{FONT}" font-size="13" fill="{P["muted"]}">FS, DB, APIs métier…</text>
  {card(200, 480, 800, 100)}
  <text x="600" y="525" text-anchor="middle" font-family="{FONT}" font-size="16" fill="{P["ink"]}">Sans MCP : N×M intégrations custom. Avec : un serveur standard, plusieurs hosts.</text>
  <text x="600" y="555" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["muted"]}">Complémentaire des protocoles agent-to-agent (J19)</text>"""
    return wrap_svg(
        "Architecture MCP",
        "Host/Agent connecté via MCP Client à un MCP Server qui expose tools, resources et prompts (fichiers, DB, APIs).",
        body,
    )

def build_11() -> str:
    body = title_block(
        "Pyramide d'évaluation d'un agent",
        "Unit → trajectory → end-to-end → LLM-as-judge → prod",
        "Module 11 · agentic-ai",
    )
    levels = [
        (480, 130, 240, "Prod / online", "drift, feedback user", P["danger"]),
        (420, 230, 360, "E2E + judge", "pass^k, scorers", P["accent"]),
        (340, 330, 520, "Trajectory tests", "outils appelés, ordre", P["primary"]),
        (
            260,
            430,
            680,
            "Unit tools / prompts",
            "fast, déterministe",
            P["primary_deep"],
        ),
    ]
    for x, y, w, t, s, c in levels:
        body += f"""  <rect x="{x}" y="{y}" width="{w}" height="80" rx="12" fill="{c}" filter="url(#s)"/>
  <text x="{x + w // 2}" y="{y + 35}" text-anchor="middle" font-family="{FONT}" font-size="16" font-weight="700" fill="#fff">{esc(t)}</text>
  <text x="{x + w // 2}" y="{y + 58}" text-anchor="middle" font-family="{FONT}" font-size="13" fill="#E2E8F0">{esc(s)}</text>"""
    body += f'  <text x="600" y="560" text-anchor="middle" font-family="{FONT}" font-size="15" fill="{P["ink"]}">Sans tests de trajectoire, tu ne sais pas *comment* l\'agent a réussi (ou triché).</text>'
    return wrap_svg(
        "Pyramide d'évaluation",
        "Quatre niveaux : unit tools/prompts, tests de trajectoire, E2E + LLM-as-judge, puis monitoring prod/online.",
        body,
    )

def build_12() -> str:
    body = title_block(
        "Observabilité en production",
        "Traces · coûts · latence · erreurs · guardrails",
        "Module 12 · agentic-ai",
    )
    items = [
        (100, "Traces", "chaque nœud\n& tool call"),
        (340, "Coûts", "tokens ×\nprix modèle"),
        (580, "Latence", "p50 / p95\npar étape"),
        (820, "Recovery", "retry, fallback,\nHITL"),
    ]
    for x, t, s in items:
        body += f"""  {card(x, 200, 220, 260)}
  <text x="{x + 110}" y="280" text-anchor="middle" font-family="{FONT}" font-size="20" font-weight="700" fill="{P["primary"]}">{esc(t)}</text>"""
        for i, line in enumerate(s.split("\n")):
            body += f'  <text x="{x + 110}" y="{340 + i * 28}" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["muted"]}">{esc(line)}</text>'
    body += f'  <text x="600" y="540" text-anchor="middle" font-family="{FONT}" font-size="15" fill="{P["ink"]}">Langfuse / LangSmith : voir la trajectoire, pas seulement le log final.</text>'
    return wrap_svg(
        "Observabilité agent",
        "Quatre piliers prod : traces (nœuds et tools), coûts tokens, latence p50/p95, recovery (retry, fallback, HITL).",
        body,
    )

def build_13() -> str:
    body = title_block(
        "Surfaces d'attaque d'un agent",
        "Injection · abus d'outils · fuite · boucles — défense en couches",
        "Module 13 · agentic-ai",
    )
    threats = [
        (100, "Prompt\ninjection", P["danger"]),
        (340, "Tool\nabuse", P["accent"]),
        (580, "Data\nexfil", P["danger"]),
        (820, "Infinite\nloops", P["accent"]),
    ]
    for x, t, c in threats:
        body += f"""  {card(x, 160, 220, 160)}
  <text x="{x + 110}" y="230" text-anchor="middle" font-family="{FONT}" font-size="16" font-weight="700" fill="{c}">{esc(t.split(chr(10))[0])}</text>
  <text x="{x + 110}" y="260" text-anchor="middle" font-family="{FONT}" font-size="16" font-weight="700" fill="{c}">{esc(t.split(chr(10))[1])}</text>"""
    body += f"""  {card(150, 380, 900, 180)}
  <text x="600" y="430" text-anchor="middle" font-family="{FONT}" font-size="17" font-weight="700" fill="{P["primary"]}">Défense en profondeur</text>
  <text x="600" y="475" text-anchor="middle" font-family="{FONT}" font-size="15" fill="{P["ink"]}">Whitelist tools · sandbox · validation args · rate limits · human-in-the-loop</text>
  <text x="600" y="515" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["muted"]}">Ne traite jamais le contenu récupéré comme une instruction</text>"""
    return wrap_svg(
        "Sécurité agents",
        "Quatre menaces : prompt injection, abus d'outils, exfiltration de données, boucles infinies. Défense : whitelist, sandbox, validation, HITL.",
        body,
    )

def build_14() -> str:
    body = title_block(
        "Capstone S1-S2 : assistant recherche multi-agent",
        "Supervisor + workers + eval + observabilité",
        "Module 14 · agentic-ai",
    )
    body += node_box(480, 140, 240, 80, "Supervisor", "orchestre", P["primary"], "#fff")
    body += node_box(
        100, 300, 200, 90, "Researcher", "web / docs", P["teal_soft"], P["primary_deep"]
    )
    body += node_box(
        360, 300, 200, 90, "Analyst", "synthèse", P["amber_soft"], "#92400E"
    )
    body += node_box(620, 300, 200, 90, "Critic", "qualité", "#DBEAFE", "#1E3A8A")
    body += node_box(880, 300, 200, 90, "Writer", "livrable", P["card"], P["ink"])
    for x in (200, 460, 720, 980):
        body += f'  <path d="M600 220 L{x} 300" fill="none" stroke="{P["primary"]}" stroke-width="2" marker-end="url(#arrT)"/>'
    body += f"""  {card(200, 450, 800, 140)}
  <text x="600" y="505" text-anchor="middle" font-family="{FONT}" font-size="16" font-weight="700" fill="{P["ink"]}">Livrable : runnable + traces + tests de trajectoire + garde-fous</text>
  <text x="600" y="545" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["muted"]}">Pas un démo happy-path : erreurs, timeouts, et cas ambigus doivent être gérés</text>"""
    return wrap_svg(
        "Capstone multi-agent",
        "Supervisor au centre qui délègue à Researcher, Analyst, Critic et Writer ; livrable avec traces, tests et garde-fous.",
        body,
    )

