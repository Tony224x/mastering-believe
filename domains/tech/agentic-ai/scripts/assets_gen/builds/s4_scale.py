"""Builders SVG — modules pédagogiques."""
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

def build_22() -> str:
    body = title_block(
        "GUI / computer-use agents",
        "Perception visuelle → grounding → action souris/clavier",
        "Module 22 · agentic-ai",
    )
    body += node_box(
        100, 220, 240, 140, "Screenshot", "pixels / DOM", P["card"], P["ink"]
    )
    body += f"  {arrow_h(340, 290, 400)}"
    body += node_box(
        400,
        220,
        280,
        140,
        "Grounding",
        "set-of-marks, coords",
        P["amber_soft"],
        "#92400E",
    )
    body += f"  {arrow_h(680, 290, 740)}"
    body += node_box(
        740,
        220,
        340,
        140,
        "Action",
        "click, type, scroll",
        P["teal_soft"],
        P["primary_deep"],
    )
    body += f"""  {card(200, 430, 800, 140)}
  <text x="600" y="490" text-anchor="middle" font-family="{FONT}" font-size="16" font-weight="700" fill="{P["ink"]}">Fragilité : un pixel de décalage = clic raté</text>
  <text x="600" y="530" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["muted"]}">Claude computer use · OpenAI CUA · browser-use — toujours sandboxer</text>"""
    return wrap_svg(
        "GUI agent loop",
        "Screenshot (perception) → grounding (set-of-marks, coordonnées) → action (click/type/scroll). Grounding fragile, sandbox obligatoire.",
        body,
    )

def build_23() -> str:
    body = title_block(
        "Sandboxing : défense en couches",
        "Process → container → gVisor/microVM → réseau filtré",
        "Module 23 · agentic-ai",
    )
    layers = [
        (160, "Réseau : egress filter / allowlist", P["danger"]),
        (260, "microVM / gVisor (noyau isolé)", P["accent"]),
        (360, "Container / cgroups / seccomp", P["primary"]),
        (460, "Process limité (user, caps, timeout)", P["primary_deep"]),
    ]
    for y, t, c in layers:
        body += f"""  <rect x="200" y="{y}" width="800" height="80" rx="14" fill="{c}" filter="url(#s)"/>
  <text x="600" y="{y + 48}" text-anchor="middle" font-family="{FONT}" font-size="16" font-weight="600" fill="#fff">{esc(t)}</text>"""
    body += f'  <text x="600" y="600" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["muted"]}">Plus on descend dans la pile, plus c\'est sûr — et plus c\'est coûteux à opérer</text>'
    return wrap_svg(
        "Couches de sandbox",
        "Quatre couches du bas vers le haut : process limité, container, gVisor/microVM, filtrage egress réseau.",
        body,
    )

def build_24() -> str:
    body = title_block(
        "Inference engineering pour agents",
        "Structured outputs · routing · prompt caching",
        "Module 24 · agentic-ai",
    )
    cards = [
        (100, "Structured", "JSON schema /\nconstrained decode", "tool calls fiables"),
        (430, "Routing", "petit modèle si\ntâche simple", "−50–70 % coût"),
        (760, "Caching", "préfixe système\nréutilisé", "latence ↘"),
    ]
    for x, t, s, r in cards:
        body += f"""  {card(x, 180, 300, 320)}
  <text x="{x + 150}" y="250" text-anchor="middle" font-family="{FONT}" font-size="20" font-weight="700" fill="{P["primary"]}">{esc(t)}</text>"""
        for i, line in enumerate(s.split("\n")):
            body += f'  <text x="{x + 150}" y="{310 + i * 28}" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["muted"]}">{esc(line)}</text>'
        body += f'  <text x="{x + 150}" y="420" text-anchor="middle" font-family="{FONT}" font-size="14" font-weight="600" fill="{P["ink"]}">{esc(r)}</text>'
    return wrap_svg(
        "Inference engineering",
        "Trois leviers : structured outputs (tool calls fiables), model routing (coût), prompt caching (latence).",
        body,
    )

def build_25() -> str:
    body = title_block(
        "Serving stateful à l'échelle",
        "Checkpointer partagé · workers stateless · sessions",
        "Module 25 · agentic-ai",
    )
    body += node_box(100, 220, 280, 120, "Clients", "N sessions", P["card"], P["ink"])
    body += f"  {arrow_h(380, 280, 440)}"
    body += node_box(
        440,
        200,
        280,
        160,
        "Workers",
        "stateless × K",
        P["teal_soft"],
        P["primary_deep"],
    )
    body += f"  {arrow_v(580, 360, 420)}"
    body += node_box(
        440,
        430,
        280,
        120,
        "Checkpointer",
        "Postgres / Redis",
        P["amber_soft"],
        "#92400E",
    )
    body += f"""  {card(800, 220, 300, 280)}
  <text x="950" y="300" text-anchor="middle" font-family="{FONT}" font-size="16" font-weight="700" fill="{P["ink"]}">Aussi</text>
  <text x="950" y="350" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["muted"]}">online eval</text>
  <text x="950" y="390" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["muted"]}">drift detection</text>
  <text x="950" y="430" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["muted"]}">limites session</text>"""
    return wrap_svg(
        "Serving stateful",
        "Clients vers workers stateless qui lisent/écrivent un checkpointer partagé (Postgres/Redis) ; online eval et drift.",
        body,
    )

def build_26() -> str:
    body = title_block(
        "Harness d'évaluation sur TON agent",
        "Dataset · scorers · pass^k · rapport de régression",
        "Module 26 · agentic-ai",
    )
    pipeline = ["Cas de test", "Run × k", "Score", "pass^k", "Rapport"]
    for i, label in enumerate(pipeline):
        x = 80 + i * 220
        body += node_box(
            x,
            250,
            180,
            100,
            label,
            "",
            P["primary"] if i == 3 else P["card"],
            "#fff" if i == 3 else P["ink"],
        )
        if i < len(pipeline) - 1:
            body += f"  {arrow_h(x + 180, 300, x + 220)}"
    body += f"""  {card(200, 420, 800, 140)}
  <text x="600" y="480" text-anchor="middle" font-family="{FONT}" font-size="16" fill="{P["ink"]}">pass^k = proba de ≥1 succès sur k essais indépendants</text>
  <text x="600" y="520" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["muted"]}">Compare toujours à une baseline figée — sinon tu ne vois pas les régressions</text>"""
    return wrap_svg(
        "Harness pass^k",
        "Pipeline : cas de test → run × k → score → pass^k → rapport de régression vs baseline.",
        body,
    )

def build_27() -> str:
    body = title_block(
        "Capstone avancé : architecture deep ops",
        "Durable · isolé · routé · évaluable",
        "Module 27 · agentic-ai",
    )
    boxes = [
        (100, 180, "Ingest", "ticket / alert"),
        (360, 180, "Planner", "plan + budget"),
        (620, 180, "Workers", "isolés + sandbox"),
        (880, 180, "Verifier", "accept / retry"),
        (100, 380, "Memory", "long-horizon"),
        (360, 380, "Durable", "reprise crash"),
        (620, 380, "Observability", "traces / coût"),
        (880, 380, "Eval harness", "pass^k"),
    ]
    for x, y, t, s in boxes:
        body += node_box(
            x,
            y,
            220,
            100,
            t,
            s,
            P["teal_soft"] if y == 180 else P["card"],
            P["primary_deep"],
        )
    return wrap_svg(
        "Architecture deep ops",
        "Huit briques : Ingest, Planner, Workers sandboxed, Verifier, Memory, Durable, Observability, Eval harness.",
        body,
    )

def build_28() -> str:
    body = title_block(
        "Capstone : build & eval bout en bout",
        "Réparer un bug · reprendre après crash · prouver pass^k",
        "Module 28 · agentic-ai",
    )
    steps = [
        (100, "1. Build", "agent runnable"),
        (360, "2. Scenario", "bug → fix"),
        (620, "3. Crash test", "reprise durable"),
        (880, "4. Eval", "rapport final"),
    ]
    for x, t, s in steps:
        body += node_box(x, 240, 220, 120, t, s, P["teal_soft"], P["primary_deep"])
    for x in (320, 580, 840):
        body += f"  {arrow_h(x, 300, x + 40)}"
    body += f"""  {card(200, 430, 800, 140)}
  <text x="600" y="490" text-anchor="middle" font-family="{FONT}" font-size="16" font-weight="700" fill="{P["ink"]}">Done = démonstration live + métriques, pas seulement le code</text>
  <text x="600" y="530" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["muted"]}">Si tu ne peux pas rejouer et expliquer un échec : pas prêt</text>"""
    return wrap_svg(
        "Capstone build eval",
        "Quatre étapes : build runnable, scénario bug→fix, crash test avec reprise, eval pass^k et rapport.",
        body,
    )

def build_parcours() -> str:
    body = title_block(
        "Parcours agentic-ai — 28 jours",
        "S1-S2 fondations · S3-S4 frontier",
        "README · agentic-ai",
    )
    blocks = [
        (80, "S1", "J1–J7", "Agent single\n+ LangGraph", P["primary"]),
        (340, "S2", "J8–J14", "Multi-agent\n+ prod", P["primary_deep"]),
        (600, "S3", "J15–J21", "Frontier\npatterns", P["accent"]),
        (860, "S4", "J22–J28", "Scale +\ncapstone", P["danger"]),
    ]
    for x, s, j, d, c in blocks:
        body += f"""  {card(x, 180, 240, 300)}
  <rect x="{x}" y="180" width="240" height="56" rx="18" fill="{c}"/>
  <rect x="{x}" y="210" width="240" height="26" fill="{c}"/>
  <text x="{x + 120}" y="218" text-anchor="middle" font-family="{FONT}" font-size="20" font-weight="700" fill="#fff">{esc(s)}</text>
  <text x="{x + 120}" y="280" text-anchor="middle" font-family="{FONT}" font-size="16" font-weight="600" fill="{P["ink"]}">{esc(j)}</text>"""
        for i, line in enumerate(d.split("\n")):
            body += f'  <text x="{x + 120}" y="{340 + i * 30}" text-anchor="middle" font-family="{FONT}" font-size="15" fill="{P["muted"]}">{esc(line)}</text>'
    body += f'  <text x="600" y="560" text-anchor="middle" font-family="{FONT}" font-size="15" fill="{P["ink"]}">Même méthode : théorie → code → exercices → capstone</text>'
    return wrap_svg(
        "Parcours 28 jours agentic-ai",
        "Quatre blocs : S1 J1-J7 fondations agent, S2 J8-J14 multi-agent et prod, S3 J15-J21 frontier, S4 J22-J28 scale et capstone.",
        body,
    )


# Map: slug → (builder that returns full SVG string OR needs wrap)

