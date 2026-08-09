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

def build_15() -> str:
    body = title_block(
        "Context engineering : combattre le context rot",
        "Compaction · offloading · budgets tokens",
        "Module 15 · agentic-ai",
    )
    body += f"""  {card(80, 160, 340, 360)}
  <text x="250" y="220" text-anchor="middle" font-family="{FONT}" font-size="18" font-weight="700" fill="{P["danger"]}">Sans curation</text>
  <text x="250" y="280" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["muted"]}">Historique monstrueux</text>
  <text x="250" y="320" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["muted"]}">Signal noyé dans le bruit</text>
  <text x="250" y="360" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["muted"]}">Coût ↗ qualité ↘</text>
  <text x="250" y="420" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["muted"]}">Sous-agents qui se polluent</text>

  {card(440, 160, 340, 360)}
  <text x="610" y="220" text-anchor="middle" font-family="{FONT}" font-size="18" font-weight="700" fill="{P["primary"]}">Compaction</text>
  <text x="610" y="280" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["muted"]}">Résumer l'historique</text>
  <text x="610" y="320" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["muted"]}">Garder faits + décisions</text>
  <text x="610" y="360" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["muted"]}">Jeter le bruit</text>
  <text x="610" y="420" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["muted"]}">Trigger par seuil tokens</text>

  {card(800, 160, 320, 360)}
  <text x="960" y="220" text-anchor="middle" font-family="{FONT}" font-size="18" font-weight="700" fill="{P["accent"]}">Offloading</text>
  <text x="960" y="280" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["muted"]}">Scratchpad / FS virtuel</text>
  <text x="960" y="320" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["muted"]}">Notes hors fenêtre</text>
  <text x="960" y="360" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["muted"]}">Budget par sous-agent</text>
  <text x="960" y="420" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["muted"]}">Isolation de contexte</text>"""
    return wrap_svg(
        "Context engineering",
        "Trois colonnes : sans curation (context rot), compaction (résumer l'historique), offloading (scratchpad hors fenêtre + budgets).",
        body,
    )

def build_16() -> str:
    body = title_block(
        "Mémoire long-horizon",
        "Épisodique · sémantique · procédurale + consolidation",
        "Module 16 · agentic-ai",
    )
    cards = [
        (100, "Épisodique", "Ce qui s'est\npassé (événements)"),
        (430, "Sémantique", "Faits stables\n& connaissances"),
        (760, "Procédurale", "Comment faire\n(skills, recettes)"),
    ]
    for x, t, s in cards:
        body += f"""  {card(x, 180, 300, 260)}
  <text x="{x + 150}" y="260" text-anchor="middle" font-family="{FONT}" font-size="20" font-weight="700" fill="{P["primary"]}">{esc(t)}</text>"""
        for i, line in enumerate(s.split("\n")):
            body += f'  <text x="{x + 150}" y="{320 + i * 28}" text-anchor="middle" font-family="{FONT}" font-size="15" fill="{P["muted"]}">{esc(line)}</text>'
    body += f"""  <text x="600" y="520" text-anchor="middle" font-family="{FONT}" font-size="15" fill="{P["ink"]}">Scoring : récence + importance + similarité → ce qu'on charge dans le contexte</text>
  <text x="600" y="560" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["muted"]}">Consolidation (style Generative Agents) : réfléchir pour abstraire des leçons</text>"""
    return wrap_svg(
        "Mémoire long-horizon",
        "Trois types : épisodique (événements), sémantique (faits), procédurale (skills). Scoring récence/importance/similarité et consolidation.",
        body,
    )

def build_17() -> str:
    body = title_block(
        "Verifiers & self-improvement",
        "Générer N · scorer · garder le meilleur · persister les leçons",
        "Module 17 · agentic-ai",
    )
    body += node_box(
        100, 220, 200, 100, "Generate N", "candidats", P["teal_soft"], P["primary_deep"]
    )
    body += f"  {arrow_h(300, 270, 360)}"
    body += node_box(
        360, 220, 200, 100, "Verifier", "PRM / règles", P["amber_soft"], "#92400E"
    )
    body += f"  {arrow_h(560, 270, 620)}"
    body += node_box(620, 220, 200, 100, "Select", "best-of-N", P["card"], P["ink"])
    body += f"  {arrow_h(820, 270, 880)}"
    body += node_box(880, 220, 200, 100, "Persist", "leçons", P["primary"], "#fff")
    body += f"""  {card(200, 400, 800, 160)}
  <text x="600" y="460" text-anchor="middle" font-family="{FONT}" font-size="16" font-weight="700" fill="{P["ink"]}">Outcome reward = résultat final · Process reward = qualité des étapes</text>
  <text x="600" y="510" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["muted"]}">Sans persistance, l'agent réapprend les mêmes erreurs</text>"""
    return wrap_svg(
        "Boucle verifier",
        "Generate N candidats, verifier (PRM/règles), select best-of-N, persist les leçons pour s'améliorer entre les runs.",
        body,
    )

def build_18() -> str:
    body = title_block(
        "Orchestration : choisir son framework",
        "Trade-offs — pas un classement absolu",
        "Module 18 · agentic-ai",
    )
    frameworks = [
        (80, "LangGraph", "Graphes &\ncontrôle fin"),
        (300, "CrewAI", "Rôles &\néquipes rapides"),
        (520, "AutoGen", "Conversations\nmulti-agent"),
        (740, "OpenAI SDK", "Agents +\ntools natifs"),
        (960, "Swarm/ADK", "Handoffs &\nlight"),
    ]
    for x, t, s in frameworks:
        body += f"""  {card(x, 180, 200, 260)}
  <text x="{x + 100}" y="250" text-anchor="middle" font-family="{FONT}" font-size="15" font-weight="700" fill="{P["primary"]}">{esc(t)}</text>"""
        for i, line in enumerate(s.split("\n")):
            body += f'  <text x="{x + 100}" y="{310 + i * 28}" text-anchor="middle" font-family="{FONT}" font-size="13" fill="{P["muted"]}">{esc(line)}</text>'
    body += f'  <text x="600" y="520" text-anchor="middle" font-family="{FONT}" font-size="15" fill="{P["ink"]}">Failure modes : loops, ping-pong, contexte pollué, coût explosif</text>'
    body += f'  <text x="600" y="560" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["muted"]}">Choisir selon contrôle vs vitesse de delivery, pas la hype</text>'
    return wrap_svg(
        "Frameworks d'orchestration",
        "Cinq familles : LangGraph, CrewAI, AutoGen, OpenAI Agents SDK, Swarm/ADK — trade-offs contrôle vs vitesse ; failure modes multi-agent.",
        body,
    )

def build_19() -> str:
    body = title_block(
        "Protocoles inter-agents",
        "MCP (outils) + A2A/ACP (agents entre eux)",
        "Module 19 · agentic-ai",
    )
    body += node_box(
        150, 240, 240, 120, "Agent A", "vendor X", P["teal_soft"], P["primary_deep"]
    )
    body += node_box(
        810, 240, 240, 120, "Agent B", "vendor Y", P["amber_soft"], "#92400E"
    )
    body += f"""  {card(450, 220, 300, 160)}
  <text x="600" y="280" text-anchor="middle" font-family="{FONT}" font-size="18" font-weight="700" fill="{P["primary"]}">A2A / ACP</text>
  <text x="600" y="320" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["muted"]}">découverte · cards</text>
  <text x="600" y="350" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["muted"]}">confiance · messages</text>"""
    body += f"  {arrow_h(390, 300, 450)}"
    body += f"  {arrow_h(750, 300, 810)}"
    body += f"""  {card(200, 450, 800, 120)}
  <text x="600" y="500" text-anchor="middle" font-family="{FONT}" font-size="16" fill="{P["ink"]}">MCP = brancher des capacités · A2A = faire collaborer des agents hétérogènes</text>
  <text x="600" y="540" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["muted"]}">Complémentaires, pas concurrents</text>"""
    return wrap_svg(
        "Protocoles inter-agents",
        "Agent A et Agent B (vendeurs différents) communiquent via A2A/ACP (découverte, cards, confiance). MCP reste pour les tools.",
        body,
    )

def build_20() -> str:
    body = title_block(
        "Durable execution vs checkpoint",
        "Survivre au crash · event-driven · HITL avancé",
        "Module 20 · agentic-ai",
    )
    body += f"""  {card(80, 150, 520, 380)}
  {card_header(80, 150, 520, "Checkpoint (J6)", P["muted"])}
  <text x="340" y="260" text-anchor="middle" font-family="{FONT}" font-size="15" fill="{P["ink"]}">Snapshot de l'état du graph</text>
  <text x="340" y="310" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["muted"]}">Reprendre le même process</text>
  <text x="340" y="360" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["muted"]}">Super pour debug / HITL local</text>
  <text x="340" y="420" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["muted"]}">Limite : orchestration longue</text>

  {card(640, 150, 480, 380)}
  {card_header(640, 150, 480, "Durable (Temporal…)", P["primary"])}
  <text x="880" y="260" text-anchor="middle" font-family="{FONT}" font-size="15" fill="{P["ink"]}">Workflow rejouable</text>
  <text x="880" y="310" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["muted"]}">Crash machine ≠ perte</text>
  <text x="880" y="360" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["muted"]}">Timers, events, retries</text>
  <text x="880" y="420" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["muted"]}">Idéal runs multi-heures</text>"""
    return wrap_svg(
        "Durable vs checkpoint",
        "Checkpoint = snapshot d'état pour reprendre/debug. Durable execution = workflow rejouable survivant aux crashs, timers et events.",
        body,
    )

def build_21() -> str:
    body = title_block(
        "Coding agent : boucle edit / search / run",
        "ACI — l'agent est un utilisateur du dépôt",
        "Module 21 · agentic-ai",
    )
    steps = [
        (120, "SEARCH", "trouver le code"),
        (400, "EDIT", "patch minimal"),
        (680, "RUN", "tests / CLI"),
        (960, "OBSERVE", "logs, diffs"),
    ]
    for x, t, s in steps:
        body += node_box(
            x - 60,
            240,
            200,
            120,
            t,
            s,
            P["teal_soft"] if t != "EDIT" else P["amber_soft"],
            P["primary_deep"],
        )
    for x in (260, 540, 820):
        body += f"  {arrow_h(x, 300, x + 80)}"
    body += f'  <path d="M1040 360 Q1040 480 160 480 Q100 480 100 360" fill="none" stroke="{P["primary"]}" stroke-width="2.5" marker-end="url(#arrT)"/>'
    body += f'  <text x="600" y="520" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["primary"]}">boucle jusqu\'aux tests verts (ou abandon)</text>'
    body += f'  <text x="600" y="580" text-anchor="middle" font-family="{FONT}" font-size="15" fill="{P["ink"]}">SWE-bench mesure cette boucle sur de vrais issues GitHub</text>'
    return wrap_svg(
        "Boucle coding agent",
        "SEARCH → EDIT → RUN → OBSERVE en boucle jusqu'aux tests verts. L'ACI définit comment l'agent perçoit et modifie le dépôt.",
        body,
    )

