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
    pill,
    title_block,
    wrap_svg,
)

def build_01() -> str:
    body = title_block(
        "Boucle fondamentale d'un agent",
        "Perceive → Think → Act — le LLM pilote le flux",
        "Module 01 · agentic-ai",
    )
    # three big steps in a loop
    body += node_box(
        120,
        160,
        240,
        120,
        "PERCEIVE",
        "observations, outils",
        P["teal_soft"],
        P["primary_deep"],
    )
    body += node_box(
        480, 160, 240, 120, "THINK", "raisonne, planifie", P["amber_soft"], "#92400E"
    )
    body += node_box(
        840, 160, 240, 120, "ACT", "outil ou réponse", "#DBEAFE", "#1E3A8A"
    )
    body += f"  {arrow_h(360, 220, 470)}"
    body += f"  {arrow_h(720, 220, 830)}"
    # return arrow
    body += f"""  <path d="M960 280 Q960 420 600 420 Q240 420 240 280" fill="none" stroke="{P["primary"]}" stroke-width="2.5" marker-end="url(#arrT)"/>
  <text x="600" y="450" text-anchor="middle" font-family="{FONT}" font-size="14" font-weight="600" fill="{P["primary"]}">boucle jusqu'à l'objectif</text>"""
    body += f"""  {card(200, 500, 800, 100)}
  <text x="600" y="545" text-anchor="middle" font-family="{FONT}" font-size="16" font-weight="600" fill="{P["ink"]}">Pipeline = chemin fixe · Agent = chemin dynamique</text>
  <text x="600" y="575" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["muted"]}">Debugger = repérer où la boucle déraille</text>"""
    return wrap_svg(
        "Boucle Perceive Think Act",
        "Trois étapes en boucle : Perceive (observations), Think (raisonnement), Act (outil ou réponse finale), jusqu'à l'objectif.",
        body,
    )

def build_02() -> str:
    body = title_block(
        "Tool use : le superpouvoir de l'agent",
        "Schéma d'outil → appel → résultat → nouvelle décision",
        "Module 02 · agentic-ai",
    )
    steps = [
        (80, "1. Schéma", "JSON du tool", P["primary"]),
        (320, "2. LLM choisit", "nom + args", P["accent"]),
        (560, "3. Exécution", "code / API", P["primary_deep"]),
        (800, "4. Observation", "résultat injecté", P["primary"]),
    ]
    for x, t, s, c in steps:
        body += node_box(x, 220, 200, 140, t, s, P["card"], P["ink"])
        body += f'  <rect x="{x}" y="220" width="200" height="8" rx="4" fill="{c}"/>'
    for x in (280, 520, 760):
        body += f"  {arrow_h(x, 290, x + 40)}"
    body += f"""  {card(200, 420, 800, 160)}
  <text x="600" y="470" text-anchor="middle" font-family="{FONT}" font-size="17" font-weight="700" fill="{P["ink"]}">Règles d'or</text>
  <text x="240" y="510" font-family="{FONT}" font-size="15" fill="{P["ink"]}">• Description claire du tool (quand l'appeler)</text>
  <text x="240" y="540" font-family="{FONT}" font-size="15" fill="{P["ink"]}">• Arguments typés + validation · erreurs renvoyées au LLM (pas crash silencieux)</text>
  <text x="240" y="570" font-family="{FONT}" font-size="15" fill="{P["ink"]}">• Structured output = appels plus fiables qu'un free-text</text>"""
    return wrap_svg(
        "Flux tool use",
        "Quatre étapes : définir le schéma d'outil, le LLM choisit nom et arguments, exécution, observation réinjectée pour la suite.",
        body,
    )


def build_03_fixed() -> str:
    body = title_block(
        "Trois mémoires d'un agent",
        "Court terme · Working · Long terme — + checkpointing",
        "Module 03 · agentic-ai",
    )
    layers = [
        (
            130,
            "Short-term",
            "Fenêtre de contexte — messages de la session",
            P["teal_soft"],
            P["primary_deep"],
        ),
        (
            280,
            "Working memory",
            "Scratchpad / state — plan et faits en cours",
            P["amber_soft"],
            "#92400E",
        ),
        (
            430,
            "Long-term",
            "Vector store / DB — faits stables, préférences",
            "#DBEAFE",
            "#1E3A8A",
        ),
    ]
    for y, t, s, bg, ink in layers:
        body += f"""  <rect x="100" y="{y}" width="720" height="120" rx="18" fill="{bg}" filter="url(#s)"/>
  <text x="140" y="{y + 50}" font-family="{FONT}" font-size="22" font-weight="700" fill="{ink}">{esc(t)}</text>
  <text x="140" y="{y + 85}" font-family="{FONT}" font-size="15" fill="{P["ink"]}">{esc(s)}</text>"""
    body += f"""  {card(860, 200, 280, 300)}
  <text x="1000" y="250" text-anchor="middle" font-family="{FONT}" font-size="16" font-weight="700" fill="{P["primary"]}">Checkpoint</text>
  <text x="1000" y="300" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["ink"]}">Sauvegarde l'état</text>
  <text x="1000" y="330" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["ink"]}">à chaque nœud</text>
  <text x="1000" y="380" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["muted"]}">Reprise, debug,</text>
  <text x="1000" y="405" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["muted"]}">time-travel</text>"""
    return body

def build_04() -> str:
    body = title_block(
        "Planning & reasoning : choisir le bon pattern",
        "Du simple (CoT) au coûteux (ToT / multi-critique)",
        "Module 04 · agentic-ai",
    )
    items = [
        (100, "CoT", "Chaîne de\nraisonnement", "1 passe", P["primary"]),
        (320, "ReAct", "Raisonner +\noutils en boucle", "itératif", P["primary"]),
        (540, "Plan→Exec", "Plan d'abord,\npuis exécuter", "2 phases", P["accent"]),
        (760, "ToT / Reflexion", "Arbre / critique\nde soi", "cher", P["danger"]),
    ]
    for x, t, s, cost, c in items:
        body += f"""  {card(x, 180, 200, 280)}
  <rect x="{x}" y="180" width="200" height="10" fill="{c}"/>
  <text x="{x + 100}" y="240" text-anchor="middle" font-family="{FONT}" font-size="18" font-weight="700" fill="{P["ink"]}">{esc(t)}</text>"""
        for i, line in enumerate(s.split("\n")):
            body += f'  <text x="{x + 100}" y="{290 + i * 28}" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["muted"]}">{esc(line)}</text>'
        body += pill(x + 40, 380, 120, 36, cost, c if c != P["danger"] else P["danger"])
    body += f"""  <text x="600" y="540" text-anchor="middle" font-family="{FONT}" font-size="16" font-weight="600" fill="{P["ink"]}">Règle : commence simple. Monte en complexité seulement si la tâche le exige.</text>
  <text x="600" y="580" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["muted"]}">Plus de raisonnement ≠ toujours mieux — tokens, latence, et overthinking</text>"""
    return wrap_svg(
        "Patterns de planning",
        "Quatre patterns : CoT (une passe), ReAct (boucle outils), Plan-and-Execute (deux phases), ToT/Reflexion (plus cher). Commencer simple.",
        body,
    )

def build_05() -> str:
    body = title_block(
        "LangGraph : StateGraph mental model",
        "Nodes purs + edges (conditionnels) + state qui circule",
        "Module 05 · agentic-ai",
    )
    # Vertical main flow, clear branches
    body += node_box(480, 120, 240, 70, "START", "", P["primary"], "#fff")
    body += f"  {arrow_v(600, 190, 220)}"
    body += node_box(460, 220, 280, 90, "agent", "lit state → update", P["teal_soft"], P["primary_deep"])
    # diamond-like decision label
    body += f"""  <text x="600" y="350" text-anchor="middle" font-family="{FONT}" font-size="14" font-weight="600" fill="{P["muted"]}">needs_tool ?</text>"""
    body += f"  {arrow_v(600, 310, 340)}"
    # left: tools, right: END
    body += node_box(160, 400, 220, 90, "tools", "exécute", P["amber_soft"], "#92400E")
    body += node_box(820, 400, 220, 90, "END", "réponse finale", P["primary"], "#fff")
    # branch lines from under agent
    body += f"""  <path d="M520 310 L270 310 L270 400" fill="none" stroke="{P["primary"]}" stroke-width="2.5" marker-end="url(#arrT)"/>
  <path d="M680 310 L930 310 L930 400" fill="none" stroke="{P["slate"]}" stroke-width="2.5" marker-end="url(#arr)"/>
  <text x="360" y="300" font-family="{FONT}" font-size="13" font-weight="600" fill="{P["primary"]}">oui</text>
  <text x="780" y="300" font-family="{FONT}" font-size="13" font-weight="600" fill="{P["muted"]}">non</text>"""
    # return tools → agent
    body += f"""  <path d="M270 490 L270 520 L600 520 L600 310" fill="none" stroke="{P["primary"]}" stroke-width="2.5" marker-end="url(#arrT)"/>
  <text x="420" y="545" font-family="{FONT}" font-size="13" fill="{P["primary"]}">retour → agent</text>"""
    body += f"""  {card(160, 570, 880, 60)}
  <text x="600" y="608" text-anchor="middle" font-family="{FONT}" font-size="15" fill="{P["ink"]}">State partagé · HITL = interrupt · stream des events</text>"""
    return wrap_svg(
        "StateGraph LangGraph",
        "START vers agent ; edge conditionnel needs_tool : oui vers tools puis retour agent, non vers END. Le state circule entre les nœuds.",
        body,
    )

def build_06() -> str:
    body = title_block(
        "LangGraph avancé : 4 leviers production",
        "Subgraphs · parallèle · persistence · time-travel",
        "Module 06 · agentic-ai",
    )
    cards = [
        (80, "Subgraphs", "Encapsuler un\nsous-agent propre"),
        (340, "Parallèle", "Plusieurs nœuds\nen même temps"),
        (600, "Persistence", "Checkpointer\nSQLite / Postgres"),
        (860, "Time-travel", "Rejouer un\ncheckpoint précis"),
    ]
    for x, t, s in cards:
        body += f"""  {card(x, 200, 240, 280)}
  <circle cx="{x + 120}" cy="280" r="36" fill="{P["teal_soft"]}" stroke="{P["primary"]}" stroke-width="2"/>
  <text x="{x + 120}" y="288" text-anchor="middle" font-family="{FONT}" font-size="18" font-weight="700" fill="{P["primary"]}">{esc(t[0])}</text>
  <text x="{x + 120}" y="360" text-anchor="middle" font-family="{FONT}" font-size="18" font-weight="700" fill="{P["ink"]}">{esc(t)}</text>"""
        for i, line in enumerate(s.split("\n")):
            body += f'  <text x="{x + 120}" y="{400 + i * 26}" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["muted"]}">{esc(line)}</text>'
    body += f'  <text x="600" y="560" text-anchor="middle" font-family="{FONT}" font-size="15" fill="{P["ink"]}">Sans persistence : crash = tout perdre. Avec : reprendre, debugger, HITL durable.</text>'
    return wrap_svg(
        "LangGraph avancé",
        "Quatre leviers : subgraphs (sous-agents), exécution parallèle, persistence (checkpointer), time-travel pour rejouer un état.",
        body,
    )

def build_07() -> str:
    body = title_block(
        "Agent complet : assembler les briques",
        "Question → plan → tools → mémoire → synthèse",
        "Module 07 · agentic-ai",
    )
    pipeline = ["Question", "Plan", "Tools", "Mémoire", "Synthèse"]
    for i, label in enumerate(pipeline):
        x = 80 + i * 220
        body += node_box(
            x,
            260,
            180,
            100,
            label,
            "",
            P["teal_soft"] if i % 2 == 0 else P["card"],
            P["primary_deep"],
        )
        if i < len(pipeline) - 1:
            body += f"  {arrow_h(x + 180, 310, x + 220)}"
    body += f"""  {card(150, 430, 900, 140)}
  <text x="600" y="485" text-anchor="middle" font-family="{FONT}" font-size="16" font-weight="700" fill="{P["ink"]}">Capstone semaine 1</text>
  <text x="600" y="525" text-anchor="middle" font-family="{FONT}" font-size="15" fill="{P["muted"]}">Recherche + analyse : erreurs gérées, state checkpointé, réponse sourcée</text>
  <text x="600" y="555" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["primary"]}">Si une brique manque, l'agent « a l'air » de marcher… jusqu'au premier cas réel</text>"""
    return wrap_svg(
        "Pipeline agent complet",
        "Cinq étapes assemblées : Question, Plan, Tools, Mémoire, Synthèse — capstone semaine 1 recherche et analyse.",
        body,
    )

