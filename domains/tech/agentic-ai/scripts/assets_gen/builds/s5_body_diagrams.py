"""Builders SVG — schémas corps de module (remplacent ASCII / mermaid denses)."""

from __future__ import annotations

from ..kit import (
    FONT,
    P,
    arrow_h,
    arrow_v,
    card,
    esc,
    node_box,
    title_block,
    wrap_svg,
)


def build_08_pipeline() -> str:
    """RAG agentique — pipeline complet (corps J8), layout compact centré."""
    body = title_block(
        "Pipeline RAG agentique complet",
        "Decomposer → router → sources → grader → multi-hop → synthèse",
        "Module 08 · schéma détaillé",
    )
    # R1 — USER QUERY (centré)
    body += node_box(420, 100, 360, 48, "USER QUERY", "", P["amber_soft"], "#92400E")
    # R2 — DECOMPOSER → ROUTER (paire centrée : 260+70+260=590 → start 305)
    body += f"""  <path d="M600 148 L600 165 L435 165 L435 180" fill="none" stroke="{P["primary"]}" stroke-width="2.5" marker-end="url(#arrT)"/>"""
    body += node_box(
        305,
        180,
        260,
        56,
        "DECOMPOSER",
        "→ sub-queries",
        P["teal_soft"],
        P["primary_deep"],
    )
    body += node_box(
        635,
        180,
        260,
        56,
        "ROUTER",
        "source par sub-q",
        P["teal_soft"],
        P["primary_deep"],
    )
    body += f"  {arrow_h(565, 208, 625)}"
    # R3 — sources parallèles (3 × 180 + 2 × 50 = 640 → start 280)
    body += f"""  <path d="M765 236 L765 255 L370 255 L370 270" fill="none" stroke="{P["primary"]}" stroke-width="2.5" marker-end="url(#arrT)"/>
  <path d="M765 236 L765 255 L600 255 L600 270" fill="none" stroke="{P["primary"]}" stroke-width="2.5" marker-end="url(#arrT)"/>
  <path d="M765 236 L765 255 L830 255 L830 270" fill="none" stroke="{P["primary"]}" stroke-width="2.5" marker-end="url(#arrT)"/>"""
    body += f'  <text x="600" y="252" text-anchor="middle" font-family="{FONT}" font-size="11" fill="{P["muted"]}">retrieve parallèle</text>'
    for x, lab in [(280, "source 1"), (510, "source 2"), (740, "source 3")]:
        body += node_box(x, 275, 180, 48, lab, "", P["card"], P["ink"])
    # Fan-in → GRADER
    body += f"""  <path d="M370 323 L370 340 L600 340 L600 355" fill="none" stroke="{P["primary"]}" stroke-width="2.5" marker-end="url(#arrT)"/>
  <path d="M600 323 L600 355" fill="none" stroke="{P["primary"]}" stroke-width="2.5" marker-end="url(#arrT)"/>
  <path d="M830 323 L830 340 L600 340 L600 355" fill="none" stroke="{P["primary"]}" stroke-width="2.5" marker-end="url(#arrT)"/>"""
    # R4 — GRADER
    body += node_box(
        420, 360, 360, 50, "GRADER", "garde le pertinent", P["card"], P["ink"]
    )
    # R5 — REFORMULATE | NEXT HOP | SYNTHESIZER (3 × 240 + 2 × 40 = 800 → start 200)
    body += f"""  <path d="M600 410 L600 425 L320 425 L320 440" fill="none" stroke="{P["primary"]}" stroke-width="2.5" marker-end="url(#arrT)"/>
  <path d="M600 410 L600 425 L600 440" fill="none" stroke="{P["primary"]}" stroke-width="2.5" marker-end="url(#arrT)"/>
  <path d="M600 410 L600 425 L880 425 L880 440" fill="none" stroke="{P["primary"]}" stroke-width="2.5" marker-end="url(#arrT)"/>"""
    body += node_box(
        200, 445, 240, 60, "REFORMULATE", "retry ≤ 3", P["red_soft"], "#991B1B"
    )
    body += node_box(
        480, 445, 240, 60, "NEXT HOP", "multi-hop loop", P["amber_soft"], "#92400E"
    )
    body += node_box(
        760,
        445,
        240,
        60,
        "SYNTHESIZER",
        "→ answer",
        P["teal_soft"],
        P["primary_deep"],
    )
    # Boucle dashed REFORMULATE → ROUTER (gauche du canvas)
    body += f"""  <path d="M200 475 L90 475 L90 208 L635 208" fill="none" stroke="{P["danger"]}" stroke-width="2.5" stroke-dasharray="7 5" marker-end="url(#arrA)"/>
  <text x="105" y="350" font-family="{FONT}" font-size="11" font-weight="600" fill="{P["danger"]}" transform="rotate(-90 105 350)">docs faibles</text>"""
    # Labels courts sous R5
    body += f'  <text x="320" y="530" text-anchor="middle" font-family="{FONT}" font-size="12" fill="{P["muted"]}">loop → router</text>'
    body += f'  <text x="600" y="530" text-anchor="middle" font-family="{FONT}" font-size="12" fill="{P["muted"]}">oui → nouvelle sub-q</text>'
    body += f'  <text x="880" y="530" text-anchor="middle" font-family="{FONT}" font-size="12" fill="{P["muted"]}">sinon → answer</text>'
    # Takeaway
    body += f'  <text x="600" y="580" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["ink"]}">2–4 hops max · chaque hop = latence + coût</text>'
    return wrap_svg(
        "Pipeline RAG agentique",
        "Flux : query, décomposition, routing multi-sources, grading, reformulation, multi-hop, synthèse de la réponse finale.",
        body,
    )


def build_14_pipeline() -> str:
    """Capstone — stack de production (corps J14)."""
    body = title_block(
        "Capstone : pipeline de production",
        "Guardrails → budget → supervisor + workers → HITL → rapport",
        "Module 14 · schéma détaillé",
    )
    # Note tracing discrète en haut
    body += f'  <text x="600" y="108" text-anchor="middle" font-family="{FONT}" font-size="12" fill="{P["muted"]}">tracing · cost budget · eval hooks</text>'
    # Colonne gauche : Input → Rate limit → Budget
    left = [
        (120, "INPUT GUARDRAIL", "injection · PII"),
        (210, "RATE LIMIT", "quotas / user"),
        (300, "BUDGET", "tokens · coût max"),
    ]
    for y, lab, sub in left:
        body += node_box(80, y, 260, 70, lab, sub, P["card"], P["ink"])
    body += f"  {arrow_v(210, 190, 208)}"
    body += f"  {arrow_v(210, 280, 298)}"
    # Flèche budget → SUPERVISOR
    body += f"  {arrow_h(340, 335, 430)}"
    # SUPERVISOR au centre
    body += node_box(
        440,
        280,
        320,
        90,
        "SUPERVISOR",
        "délégué + synthèse",
        P["teal_soft"],
        P["primary_deep"],
    )
    # Fan-out vers 3 workers alignés
    body += f'  <text x="780" y="270" text-anchor="middle" font-family="{FONT}" font-size="12" fill="{P["muted"]}">fan-out</text>'
    body += f"""  <path d="M600 370 L600 385 L340 385 L340 400" fill="none" stroke="{P["primary"]}" stroke-width="2.5" marker-end="url(#arrT)"/>
  <path d="M600 370 L600 400" fill="none" stroke="{P["primary"]}" stroke-width="2.5" marker-end="url(#arrT)"/>
  <path d="M600 370 L600 385 L860 385 L860 400" fill="none" stroke="{P["primary"]}" stroke-width="2.5" marker-end="url(#arrT)"/>"""
    workers = [
        (250, "RESEARCHER", "RAG"),
        (510, "ANALYZER", "raisonne"),
        (770, "WRITER", "rédige"),
    ]
    for x, lab, sub in workers:
        body += node_box(x, 405, 180, 64, lab, sub, P["amber_soft"], "#92400E")
    # Flèche retour « synthèse » vers supervisor (côté droit)
    body += f"""  <path d="M950 437 Q1040 437 1040 325 Q1040 280 760 280" fill="none" stroke="{P["accent"]}" stroke-width="2.5" marker-end="url(#arrA)"/>
  <text x="1060" y="365" font-family="{FONT}" font-size="12" font-weight="600" fill="{P["accent"]}">synthèse</text>"""
    # OUTPUT GUARDRAIL puis HITL | FINAL REPORT
    body += f"  {arrow_v(600, 469, 500)}"
    body += node_box(
        440, 505, 320, 52, "OUTPUT GUARDRAIL", "schema · filtre", P["card"], P["ink"]
    )
    body += f"""  <path d="M600 557 L600 570 L450 570 L450 585" fill="none" stroke="{P["primary"]}" stroke-width="2.5" marker-end="url(#arrT)"/>
  <path d="M600 557 L600 570 L750 570 L750 585" fill="none" stroke="{P["primary"]}" stroke-width="2.5" marker-end="url(#arrT)"/>"""
    body += node_box(
        350, 590, 200, 48, "HITL GATE", "approval", P["red_soft"], "#991B1B"
    )
    body += node_box(
        650, 590, 200, 48, "FINAL REPORT", "", P["teal_soft"], P["primary_deep"]
    )
    return wrap_svg(
        "Pipeline capstone production",
        "Entrée gardée, rate limit, budget, supervisor qui fan-out vers researcher analyzer writer, output guardrail, HITL optionnel, rapport final.",
        body,
    )


def build_03_production() -> str:
    """Mémoire production — assemblage (corps J3)."""
    body = title_block(
        "Mémoire d'un agent en production",
        "Boucle agent + 3 mémoires + stores externes",
        "Module 03 · schéma détaillé",
    )
    # outer card for agent loop
    body += f"  {card(80, 120, 1040, 380)}"
    body += f'  <text x="100" y="155" font-family="{FONT}" font-size="14" font-weight="700" fill="{P["primary"]}">AGENT LOOP</text>'
    body += node_box(
        120,
        180,
        280,
        130,
        "Context window",
        "system · summary · msgs",
        P["teal_soft"],
        P["primary_deep"],
    )
    body += node_box(
        440,
        180,
        280,
        130,
        "Working memory",
        "task · step · findings",
        P["amber_soft"],
        "#92400E",
    )
    body += node_box(
        760, 200, 280, 90, "LLM", "décide next action", P["card"], P["ink"]
    )
    body += f"  {arrow_h(400, 245, 430)}"
    body += f"  {arrow_h(720, 245, 750)}"
    body += node_box(
        200, 360, 300, 80, "Tool execution", "APIs · fichiers", P["card"], P["ink"]
    )
    body += node_box(
        600,
        360,
        300,
        80,
        "Checkpoint",
        "save state / resume",
        P["teal_soft"],
        P["primary_deep"],
    )
    body += f"  {arrow_v(900, 290, 360)}"
    # external stores
    body += f"  {arrow_v(600, 500, 530)}"
    body += node_box(
        200,
        540,
        320,
        80,
        "Vector store",
        "long-term sémantique",
        P["amber_soft"],
        "#92400E",
    )
    body += node_box(
        680,
        540,
        320,
        80,
        "Key-value store",
        "prefs · faits stables",
        P["card"],
        P["ink"],
    )
    body += f'  <text x="600" y="655" text-anchor="middle" font-family="{FONT}" font-size="13" fill="{P["muted"]}">Short-term dans le prompt · Working hors prompt · Long-term en store</text>'
    return wrap_svg(
        "Architecture mémoire production",
        "Dans la boucle : context window, working memory, LLM, tools et checkpoint. En dehors : vector store et key-value pour le long terme.",
        body,
    )


def build_13_defense() -> str:
    """Défense en profondeur — 5 couches (corps J13)."""
    body = title_block(
        "Défense en profondeur",
        "Cinq couches — l'attaquant doit toutes les contourner",
        "Module 13 · schéma détaillé",
    )
    layers = [
        (
            "L1 · Input guardrails",
            "length · rate · PII · injection patterns",
            P["teal_soft"],
            P["primary_deep"],
        ),
        (
            "L2 · Trust boundaries",
            "marquer untrusted · séparer system/user/web",
            P["card"],
            P["ink"],
        ),
        (
            "L3 · Tool guardrails",
            "whitelist · sandbox · least privilege · HITL",
            P["amber_soft"],
            "#92400E",
        ),
        (
            "L4 · Output guardrails",
            "schema · filtre contenu · judge risque",
            P["card"],
            P["ink"],
        ),
        (
            "L5 · Monitoring + audit",
            "logs · anomalies · kill switch",
            P["red_soft"],
            "#991B1B",
        ),
    ]
    y0 = 120
    for i, (lab, sub, bg, ink) in enumerate(layers):
        y = y0 + i * 95
        body += node_box(180, y, 840, 78, lab, sub, bg, ink)
        if i < len(layers) - 1:
            body += f"  {arrow_v(600, y + 78, y + 95)}"
    return wrap_svg(
        "Défense en profondeur 5 couches",
        "Couches empilées : input guardrails, trust boundaries, tool guardrails, output guardrails, monitoring et audit.",
        body,
    )


def build_07_research_archi() -> str:
    """Research agent — architecture (corps J7)."""
    body = title_block(
        "Research agent : architecture",
        "Planner → executor (boucle) → analyzer → synthesizer",
        "Module 07 · schéma détaillé",
    )
    body += node_box(80, 200, 140, 80, "START", "", P["line"], P["ink"])
    body += f"  {arrow_h(220, 240, 260)}"
    body += node_box(
        260,
        180,
        180,
        120,
        "PLANNER",
        "list[str] steps",
        P["teal_soft"],
        P["primary_deep"],
    )
    body += f"  {arrow_h(440, 240, 490)}"
    body += node_box(
        490, 180, 200, 120, "EXECUTOR", "tool + scratchpad", P["amber_soft"], "#92400E"
    )
    body += f"  {arrow_h(690, 240, 740)}"
    body += node_box(
        740, 180, 180, 120, "ANALYZER", "faits · check", P["card"], P["ink"]
    )
    body += f"  {arrow_h(920, 240, 960)}"
    body += node_box(
        960, 200, 140, 80, "END", "answer", P["teal_soft"], P["primary_deep"]
    )
    # replan loop
    body += f"""  <path d="M590 300 Q590 380 350 380 Q280 380 280 300" fill="none" stroke="{P["accent"]}" stroke-width="2.5" marker-end="url(#arrA)"/>
  <text x="430" y="405" text-anchor="middle" font-family="{FONT}" font-size="13" font-weight="600" fill="{P["accent"]}">replan si bloqué</text>"""
    # state card
    body += f"""  {card(120, 440, 960, 160)}
  <text x="600" y="480" text-anchor="middle" font-family="{FONT}" font-size="16" font-weight="700" fill="{P["ink"]}">State partagé</text>
  <text x="600" y="515" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["muted"]}">question · plan · short_term · long_term · findings · final_answer</text>
  <text x="600" y="555" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["muted"]}">Tools : mock_web_search · read_doc · summarize</text>
  <text x="600" y="580" text-anchor="middle" font-family="{FONT}" font-size="13" fill="{P["primary"]}">Patterns J4 (plan-and-execute) + J5 (LangGraph nodes)</text>"""
    return wrap_svg(
        "Architecture research agent",
        "Pipeline START, planner, executor avec replan, analyzer, synthesizer, END. State partagé et outils de recherche.",
        body,
    )


def build_11_eval_pipeline() -> str:
    """Pipeline d'eval Dev → Prod (corps J11) — optionnel P0 cleanup as SVG."""
    body = title_block(
        "Pipeline d'évaluation bout en bout",
        "Dev → CI → Nightly → Prod online",
        "Module 11 · schéma détaillé",
    )
    stages = [
        (100, "1 · DEV", "unit tests\nrapides, déterministes", P["teal_soft"]),
        (380, "2 · CI", "20–50 cas / PR\nbloque si fail", P["card"]),
        (660, "3 · NIGHTLY", "200–500 cas\nrapport qualité", P["amber_soft"]),
        (940, "4 · PROD", "tracing + scoring\ntraffic réel", P["red_soft"]),
    ]
    for i, (x, lab, sub, bg) in enumerate(stages):
        body += f"  {card(x, 200, 240, 280, bg)}"
        body += f'  <text x="{x + 120}" y="280" text-anchor="middle" font-family="{FONT}" font-size="18" font-weight="700" fill="{P["ink"]}">{esc(lab)}</text>'
        for j, line in enumerate(sub.split("\n")):
            body += f'  <text x="{x + 120}" y="{340 + j * 28}" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["muted"]}">{esc(line)}</text>'
        if i < 3:
            body += f"  {arrow_h(x + 240, 340, x + 280)}"
    body += f'  <text x="600" y="560" text-anchor="middle" font-family="{FONT}" font-size="15" fill="{P["ink"]}">Niveaux 1–2 = déterministes · Niveau 4 = live (LLM-as-judge possible)</text>'
    return wrap_svg(
        "Pipeline d'évaluation",
        "Quatre étages : tests dev, regression CI, eval nightly large, monitoring production sur le traffic réel.",
        body,
    )
