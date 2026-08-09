"""Builders SVG — lot B/C : enrichissement visuel (ASCII denses → SVG)."""

from __future__ import annotations

from ..kit import (
    FONT,
    P,
    arrow_h,
    arrow_v,
    card,
    node_box,
    title_block,
    wrap_svg,
)


# --- J9 multi-agent ---


def build_09_hierarchical() -> str:
    """Hiérarchie CEO → 2 sub-supervisors → workers nommés (2×2)."""
    body = title_block(
        "Pattern hiérarchique",
        "Supervisors de supervisors — plusieurs niveaux de délégation",
        "Module 09 · schéma détaillé",
    )
    # CEO centré en haut
    body += node_box(
        450,
        110,
        300,
        64,
        "CEO agent",
        "objectif global",
        P["teal_soft"],
        P["primary_deep"],
    )
    # Split CEO → A / B (centrés côte à côte)
    body += f"""  <path d="M600 174 L600 200 L340 200 L340 220" fill="none" stroke="{P["primary"]}" stroke-width="2.5" marker-end="url(#arrT)"/>
  <path d="M600 174 L600 200 L860 200 L860 220" fill="none" stroke="{P["primary"]}" stroke-width="2.5" marker-end="url(#arrT)"/>"""
    # Sub-supervisors (A/B centrés sous le CEO)
    body += node_box(
        200,
        225,
        280,
        64,
        "Sub-supervisor A",
        "domaine A",
        P["amber_soft"],
        "#92400E",
    )
    body += node_box(
        720,
        225,
        280,
        64,
        "Sub-supervisor B",
        "domaine B",
        P["amber_soft"],
        "#92400E",
    )
    # Workers nommés sous chaque supervisor (2 + 2), gap 20
    body += f"  {arrow_v(255, 289, 330)}"
    body += f"  {arrow_v(425, 289, 330)}"
    body += f"  {arrow_v(775, 289, 330)}"
    body += f"  {arrow_v(945, 289, 330)}"
    workers_a = [(180, "Research", "web · papers"), (350, "Docs", "fichiers · RAG")]
    workers_b = [(700, "Code", "impl · tests"), (870, "Review", "critique · QA")]
    for x, lab, sub in workers_a + workers_b:
        body += node_box(x, 335, 150, 64, lab, sub, P["card"], P["ink"])
    # Takeaway court
    body += f'  <text x="600" y="460" text-anchor="middle" font-family="{FONT}" font-size="15" fill="{P["ink"]}">Chaque niveau synthétise avant de remonter</text>'
    body += f"""  {card(200, 490, 800, 90)}
  <text x="600" y="530" text-anchor="middle" font-family="{FONT}" font-size="15" fill="{P["ink"]}">Org en couches (équipe → pôle → direction)</text>
  <text x="600" y="558" text-anchor="middle" font-family="{FONT}" font-size="13" fill="{P["muted"]}">Coût ↗ avec la profondeur — borner les niveaux</text>"""
    return wrap_svg(
        "Pattern multi-agent hiérarchique",
        "CEO agent délègue à des sub-supervisors qui pilotent des workers spécialisés.",
        body,
    )


def build_09_debate() -> str:
    body = title_block(
        "Pattern débat collaboratif",
        "Plusieurs agents critiquent, un merger consolide",
        "Module 09 · schéma détaillé",
    )
    # 3 agents centrés : w=220, gap flèche=50 → total 800, start=200
    body += node_box(
        200, 160, 220, 90, "Agent A", "proposition", P["teal_soft"], P["primary_deep"]
    )
    body += f"  {arrow_h(420, 205, 470)}"
    body += node_box(
        470, 160, 220, 90, "Agent B", "critique A", P["amber_soft"], "#92400E"
    )
    body += f"  {arrow_h(690, 205, 740)}"
    body += node_box(740, 160, 220, 90, "Agent C", "critique B", P["card"], P["ink"])
    body += f'  <text x="310" y="275" text-anchor="middle" font-family="{FONT}" font-size="12" fill="{P["muted"]}">round 1</text>'
    body += f'  <text x="580" y="275" text-anchor="middle" font-family="{FONT}" font-size="12" fill="{P["muted"]}">round 2</text>'
    body += f'  <text x="850" y="275" text-anchor="middle" font-family="{FONT}" font-size="12" fill="{P["muted"]}">round 3</text>'
    body += f"  {arrow_v(600, 290, 330)}"
    body += node_box(
        420,
        340,
        360,
        80,
        "MERGE / JUDGE",
        "liste consolidée",
        P["teal_soft"],
        P["primary_deep"],
    )
    body += f"  {arrow_v(600, 420, 460)}"
    body += node_box(
        460, 470, 280, 60, "ANSWER", "synthèse finale", P["amber_soft"], "#92400E"
    )
    body += f'  <text x="600" y="580" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["ink"]}">Borner les rounds — éviter coût et boucles</text>'
    return wrap_svg(
        "Pattern débat multi-agent",
        "Agents A B C s'échangent critiques sur plusieurs rounds, puis un merge/judge produit la réponse.",
        body,
    )


def build_09_swarm() -> str:
    body = title_block(
        "Pattern Swarm / Handoff",
        "Transfert peer-to-peer du contrôle (pas de chef central)",
        "Module 09 · schéma détaillé",
    )
    # User → A → B → C (chemin handoff linéaire, centré, marges ≥80)
    body += node_box(100, 240, 160, 90, "User", "requête", P["amber_soft"], "#92400E")
    body += f"  {arrow_h(260, 285, 310)}"
    body += f'  <text x="285" y="270" text-anchor="middle" font-family="{FONT}" font-size="11" fill="{P["muted"]}">handoff</text>'
    body += node_box(
        310, 225, 200, 120, "Agent A", "triage", P["teal_soft"], P["primary_deep"]
    )
    body += f"  {arrow_h(510, 285, 560)}"
    body += f'  <text x="535" y="270" text-anchor="middle" font-family="{FONT}" font-size="11" fill="{P["muted"]}">handoff</text>'
    body += node_box(560, 225, 200, 120, "Agent B", "coding", P["card"], P["ink"])
    body += f"  {arrow_h(760, 285, 810)}"
    body += f'  <text x="785" y="270" text-anchor="middle" font-family="{FONT}" font-size="11" fill="{P["muted"]}">handoff</text>'
    body += node_box(
        810, 240, 200, 90, "Agent C", "review / done", P["teal_soft"], P["primary_deep"]
    )
    body += f'  <text x="600" y="400" text-anchor="middle" font-family="{FONT}" font-size="14" font-weight="600" fill="{P["accent"]}">chemin : User → A → B → C (peer-to-peer)</text>'
    body += f'  <text x="600" y="440" text-anchor="middle" font-family="{FONT}" font-size="13" fill="{P["muted"]}">handoff = message + contexte minimal</text>'
    body += f'  <text x="600" y="520" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["ink"]}">Risque boucles A↔B : max handoffs + stop</text>'
    return wrap_svg(
        "Pattern Swarm handoff",
        "User vers agent triage, handoff vers agent spécialisé, puis éventuellement review — contrôle peer-to-peer.",
        body,
    )


def build_09_decision_tree() -> str:
    body = title_block(
        "Single-agent ou multi-agent ?",
        "Arbre de décision pragmatique",
        "Module 09 · schéma détaillé",
    )
    body += node_box(
        400,
        115,
        400,
        64,
        "Tâche simple ?",
        "< 10 étapes · 1 compétence",
        P["amber_soft"],
        "#92400E",
    )
    # Bifurcation centrée sous la racine
    body += f'  <line x1="600" y1="179" x2="600" y2="205" stroke="{P["primary"]}" stroke-width="2.5"/>'
    body += f'  <line x1="280" y1="205" x2="920" y2="205" stroke="{P["primary"]}" stroke-width="2.5"/>'
    body += f'  <line x1="280" y1="205" x2="280" y2="230" stroke="{P["primary"]}" stroke-width="2.5" marker-end="url(#arrT)"/>'
    body += f'  <line x1="920" y1="205" x2="920" y2="230" stroke="{P["primary"]}" stroke-width="2.5" marker-end="url(#arrT)"/>'
    body += f'  <text x="400" y="200" font-family="{FONT}" font-size="13" fill="{P["primary"]}">Oui</text>'
    body += f'  <text x="750" y="200" font-family="{FONT}" font-size="13" fill="{P["danger"]}">Non</text>'
    body += node_box(
        140,
        240,
        280,
        80,
        "Single agent",
        "tools + bon prompt",
        P["teal_soft"],
        P["primary_deep"],
    )
    body += node_box(
        760,
        240,
        320,
        80,
        "Plusieurs rôles ?",
        "parallèle / critique",
        P["card"],
        P["ink"],
    )
    # 2e niveau sous « Plusieurs rôles ? »
    body += f'  <line x1="920" y1="320" x2="920" y2="348" stroke="{P["primary"]}" stroke-width="2.5"/>'
    body += f'  <line x1="560" y1="348" x2="1000" y2="348" stroke="{P["primary"]}" stroke-width="2.5"/>'
    body += f'  <line x1="560" y1="348" x2="560" y2="375" stroke="{P["primary"]}" stroke-width="2.5" marker-end="url(#arrT)"/>'
    body += f'  <line x1="1000" y1="348" x2="1000" y2="375" stroke="{P["primary"]}" stroke-width="2.5" marker-end="url(#arrT)"/>'
    body += f'  <text x="640" y="343" font-family="{FONT}" font-size="13" fill="{P["primary"]}">Oui</text>'
    body += f'  <text x="1020" y="343" font-family="{FONT}" font-size="13" fill="{P["danger"]}">Non</text>'
    body += node_box(
        420,
        385,
        280,
        70,
        "Multi-agent",
        "supervisor d'abord",
        P["amber_soft"],
        "#92400E",
    )
    body += node_box(
        860,
        385,
        280,
        70,
        "Single + tools",
        "plus simple, cheap",
        P["teal_soft"],
        P["primary_deep"],
    )
    body += f'  <text x="600" y="520" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["ink"]}">Commence single ; multi si rôles distincts</text>'
    return wrap_svg(
        "Arbre décision single vs multi-agent",
        "Si tâche simple : single agent. Sinon, multi seulement si plusieurs rôles ou besoin de critique parallèle.",
        body,
    )


# --- J6 LangGraph avancé ---


def build_06_agent_prod() -> str:
    """Flux horizontal + bandeau checkpointer sous le flux."""
    body = title_block(
        "Agent production LangGraph",
        "Subgraphs + parallèle + checkpointer + HITL",
        "Module 06 · schéma détaillé",
    )
    body += f'  <text x="600" y="115" text-anchor="middle" font-family="{FONT}" font-size="13" font-weight="700" fill="{P["primary"]}">MAIN GRAPH</text>'
    # Flux horizontal centré : START → Router → Subgraph → Merge → HITL → END
    y, h = 200, 88
    nodes = [
        (80, 110, "START", "", P["line"], P["ink"]),
        (230, 160, "Router", "route intent", P["teal_soft"], P["primary_deep"]),
        (430, 200, "Subgraph workers", "parallel Send", P["amber_soft"], "#92400E"),
        (670, 140, "Merge", "fan-in", P["card"], P["ink"]),
        (850, 160, "HITL", "approval", P["red_soft"], "#991B1B"),
        (1050, 110, "END", "réponse", P["teal_soft"], P["primary_deep"]),
    ]
    for x, w, lab, sub, bg, ink in nodes:
        body += node_box(x, y, w, h, lab, sub, bg, ink)
    # Flèches entre nœuds (milieu vertical y+h/2 = 244)
    mid = y + h // 2
    gaps = [(190, 220), (390, 420), (630, 660), (810, 840), (1010, 1040)]
    for x1, x2 in gaps:
        body += f"  {arrow_h(x1, mid, x2)}"
    # Bandeau checkpointer traversant sous le flux
    body += f"  {card(80, 360, 1040, 120, P['teal_soft'])}"
    body += f'  <text x="600" y="405" text-anchor="middle" font-family="{FONT}" font-size="18" font-weight="700" fill="{P["primary_deep"]}">Checkpointer</text>'
    body += f'  <text x="600" y="438" text-anchor="middle" font-family="{FONT}" font-size="15" fill="{P["ink"]}">thread_id · resume · time-travel</text>'
    body += f'  <text x="600" y="520" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["danger"]}">crash sans persistence = tout perdre</text>'
    body += f'  <text x="600" y="560" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["muted"]}">Persistence sous tout le graphe — reprise après interrupt HITL</text>'
    return wrap_svg(
        "Patterns LangGraph combinés production",
        "Main graph avec router, subgraph workers en parallèle, merge, HITL, le tout sur checkpointer.",
        body,
    )


def build_06_parallel() -> str:
    body = title_block(
        "Séquentiel vs parallèle (Send API)",
        "Fan-out pour réduire la latence totale",
        "Module 06 · schéma détaillé",
    )
    body += f'  <text x="300" y="140" text-anchor="middle" font-family="{FONT}" font-size="16" font-weight="700" fill="{P["ink"]}">Séquentiel (naïf)</text>'
    body += node_box(
        80,
        170,
        440,
        200,
        "agent → s1 → s2 → s3 → s4",
        "Total ≈ 4 × latence_source",
        P["red_soft"],
        "#991B1B",
    )
    body += f'  <text x="900" y="140" text-anchor="middle" font-family="{FONT}" font-size="16" font-weight="700" fill="{P["ink"]}">Parallèle (Send)</text>'
    body += node_box(
        680,
        170,
        440,
        200,
        "agent → [s1|s2|s3|s4] → merge",
        "Total ≈ max(latences)",
        P["teal_soft"],
        P["primary_deep"],
    )
    body += f'  <text x="600" y="450" text-anchor="middle" font-family="{FONT}" font-size="15" fill="{P["ink"]}">Paralléliser seulement si les sous-tâches sont indépendantes</text>'
    body += f'  <text x="600" y="500" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["muted"]}">Fan-in : reducer sur le state pour collecter les résultats</text>'
    return wrap_svg(
        "Exécution séquentielle vs parallèle",
        "À gauche chaîne séquentielle coûteuse ; à droite fan-out Send API puis merge.",
        body,
    )


# --- J16 mémoire ---


def build_16_main_external() -> str:
    body = title_block(
        "Main context vs External context",
        "Modèle MemGPT / Letta — mémoire comme OS",
        "Module 16 · schéma détaillé",
    )
    # Deux colonnes, marges ≥ 80
    body += f"  {card(80, 120, 500, 400)}"
    body += f'  <text x="330" y="155" text-anchor="middle" font-family="{FONT}" font-size="15" font-weight="700" fill="{P["primary"]}">MAIN CONTEXT (fenêtre LLM)</text>'
    body += node_box(
        110,
        180,
        440,
        70,
        "System + core memory",
        "toujours chargé",
        P["teal_soft"],
        P["primary_deep"],
    )
    body += node_box(
        110,
        270,
        440,
        70,
        "Working context",
        "messages récents",
        P["amber_soft"],
        "#92400E",
    )
    body += node_box(
        110, 360, 440, 70, "Functions / tools", "paging in-out", P["card"], P["ink"]
    )
    body += f"  {card(620, 120, 500, 400)}"
    body += f'  <text x="870" y="155" text-anchor="middle" font-family="{FONT}" font-size="15" font-weight="700" fill="{P["ink"]}">EXTERNAL CONTEXT</text>'
    body += node_box(
        650,
        190,
        440,
        70,
        "Archival store",
        "vector / long-term",
        P["teal_soft"],
        P["primary_deep"],
    )
    body += node_box(
        650,
        280,
        440,
        70,
        "Recall storage",
        "historique complet",
        P["amber_soft"],
        "#92400E",
    )
    body += node_box(
        650, 370, 440, 70, "Fichiers / VFS", "offload", P["card"], P["ink"]
    )
    body += f'  <text x="600" y="560" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["ink"]}">Paging : charger à la demande le pertinent</text>'
    return wrap_svg(
        "Main context versus external context",
        "À gauche la fenêtre LLM (system, working, tools). À droite archival, recall et fichiers externes.",
        body,
    )


def build_16_memory_flow() -> str:
    body = title_block(
        "Flux mémoire d'un agent long-horizon",
        "Événement → score → store → injection",
        "Module 16 · schéma détaillé",
    )
    # 5 étapes w=160, gap=48 → total 992, start=104 (marges ≥80, pas d'overflow)
    steps = [
        (104, "Événement", "tool / user msg"),
        (312, "Score", "récence · import."),
        (520, "Store", "episodic / sém."),
        (728, "Retrieve", "top-k pertinent"),
        (936, "Inject", "dans le prompt"),
    ]
    for i, (x, lab, sub) in enumerate(steps):
        body += node_box(
            x,
            230,
            160,
            100,
            lab,
            sub,
            P["teal_soft"] if i % 2 == 0 else P["amber_soft"],
            P["primary_deep"] if i % 2 == 0 else "#92400E",
        )
        if i < len(steps) - 1:
            body += f"  {arrow_h(x + 160, 280, x + 208)}"
    body += f"""  {card(150, 380, 900, 120)}
  <text x="600" y="430" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["ink"]}">Consolidation : observations → faits stables</text>
  <text x="600" y="465" text-anchor="middle" font-family="{FONT}" font-size="13" fill="{P["muted"]}">Decay : oublier le bruit pour rester pertinent</text>"""
    return wrap_svg(
        "Flux mémoire agent long-horizon",
        "Pipeline événement, scoring, stockage, retrieval et injection dans le contexte.",
        body,
    )


# --- J10 MCP ---


def build_10_lifecycle() -> str:
    body = title_block(
        "Cycle de vie d'une connexion MCP",
        "Du lancement subprocess à l'appel d'outil",
        "Module 10 · schéma détaillé",
    )
    # 4 étapes centrées : w=200, gap=50 → total 950, start=125
    steps = [
        (125, "1 · Launch", "host démarre server"),
        (375, "2 · Initialize", "handshake JSON-RPC"),
        (625, "3 · List", "tools / resources"),
        (875, "4 · Call", "tools/call + result"),
    ]
    for i, (x, lab, sub) in enumerate(steps):
        body += node_box(
            x,
            200,
            200,
            110,
            lab,
            sub,
            P["teal_soft"] if i < 3 else P["amber_soft"],
            P["primary_deep"] if i < 3 else "#92400E",
        )
        if i < 3:
            body += f"  {arrow_h(x + 200, 255, x + 250)}"
    body += f"""  {card(150, 370, 900, 140)}
  <text x="600" y="420" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["ink"]}">Transport : stdio local ou HTTP distant</text>
  <text x="600" y="455" text-anchor="middle" font-family="{FONT}" font-size="13" fill="{P["muted"]}">Découvrir les capacités avant d'invoquer</text>
  <text x="600" y="485" text-anchor="middle" font-family="{FONT}" font-size="13" fill="{P["primary"]}">HITL sur tools sensibles</text>"""
    return wrap_svg(
        "Lifecycle connexion MCP",
        "Quatre étapes : launch, initialize, list des capacités, call d'outil avec résultat.",
        body,
    )


# --- J4 planning ---


def build_04_plan_execute() -> str:
    body = title_block(
        "Plan-and-Execute",
        "Planifier d'abord, exécuter ensuite",
        "Module 04 · schéma détaillé",
    )
    # Chaîne centrée, marges ≥80
    body += node_box(
        100, 200, 180, 100, "Question", "user goal", P["amber_soft"], "#92400E"
    )
    body += f"  {arrow_h(280, 250, 325)}"
    body += node_box(
        325,
        180,
        220,
        140,
        "PLANNER",
        "liste d'étapes",
        P["teal_soft"],
        P["primary_deep"],
    )
    body += f"  {arrow_h(545, 250, 590)}"
    body += node_box(
        590, 180, 220, 140, "EXECUTOR", "tools step by step", P["card"], P["ink"]
    )
    body += f"  {arrow_h(810, 250, 855)}"
    body += node_box(
        855, 200, 180, 100, "Answer", "synthèse", P["teal_soft"], P["primary_deep"]
    )
    body += f"""  <path d="M700 320 Q700 400 435 400 Q380 400 380 320" fill="none" stroke="{P["accent"]}" stroke-width="2.5" marker-end="url(#arrA)"/>
  <text x="560" y="430" text-anchor="middle" font-family="{FONT}" font-size="13" font-weight="600" fill="{P["accent"]}">replan si étape échoue</text>"""
    body += f'  <text x="600" y="500" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["ink"]}">Moins de tokens que ReAct sur tâches longues</text>'
    return wrap_svg(
        "Pattern Plan-and-Execute",
        "Question, planner qui produit des étapes, executor outillé, réponse ; boucle de replan si besoin.",
        body,
    )


def build_04_tot() -> str:
    body = title_block(
        "Tree-of-Thought (ToT)",
        "Explorer plusieurs branches, scorer, élaguer",
        "Module 04 · schéma détaillé",
    )
    # Racine centrée
    body += node_box(
        480, 115, 240, 56, "Racine", "problème", P["teal_soft"], P["primary_deep"]
    )
    # Branches vers A1 / A2 / A3 (centres 220, 600, 980)
    body += f'  <line x1="600" y1="171" x2="600" y2="195" stroke="{P["primary"]}" stroke-width="2.5"/>'
    body += f'  <line x1="220" y1="195" x2="980" y2="195" stroke="{P["primary"]}" stroke-width="2.5"/>'
    body += f'  <line x1="220" y1="195" x2="220" y2="220" stroke="{P["primary"]}" stroke-width="2.5" marker-end="url(#arrT)"/>'
    body += f'  <line x1="600" y1="195" x2="600" y2="220" stroke="{P["primary"]}" stroke-width="2.5" marker-end="url(#arrT)"/>'
    body += f'  <line x1="980" y1="195" x2="980" y2="220" stroke="{P["primary"]}" stroke-width="2.5" marker-end="url(#arrT)"/>'
    body += node_box(
        120, 230, 200, 70, "A1 · score 0.8", "gardée", P["amber_soft"], "#92400E"
    )
    body += node_box(
        500, 230, 200, 70, "A2 · score 0.4", "élaguée", P["red_soft"], "#991B1B"
    )
    body += node_box(
        880, 230, 200, 70, "A3 · score 0.7", "gardée", P["amber_soft"], "#92400E"
    )
    body += f"  {arrow_v(220, 300, 335)}"
    body += f"  {arrow_v(980, 300, 335)}"
    body += node_box(
        80, 345, 160, 58, "A1.1 · 0.9", "meilleure", P["teal_soft"], P["primary_deep"]
    )
    body += node_box(
        260, 345, 160, 58, "A1.2 · 0.3", "élaguée", P["red_soft"], "#991B1B"
    )
    body += node_box(900, 345, 160, 58, "A3.1 · 0.6", "gardée", P["card"], P["ink"])
    body += f'  <text x="600" y="450" text-anchor="middle" font-family="{FONT}" font-size="13" fill="{P["danger"]}">rouge = branche élaguée</text>'
    body += f'  <text x="600" y="500" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["ink"]}">Cher (N appels LLM) — puzzles / planning dur</text>'
    body += f'  <text x="600" y="535" text-anchor="middle" font-family="{FONT}" font-size="13" fill="{P["muted"]}">generate · evaluate · select · expand</text>'
    return wrap_svg(
        "Tree-of-Thought",
        "Arbre de pensées avec scores : branches gardées ou élaguées jusqu'à la meilleure feuille.",
        body,
    )


# --- J22 GUI ---


def build_22_pma() -> str:
    body = title_block(
        "Boucle perceive → mark → act",
        "Agent GUI / computer-use",
        "Module 22 · schéma détaillé",
    )
    # 3 boîtes centrées : w=260, gap=50 → total 880, start=160
    body += node_box(
        160,
        180,
        260,
        110,
        "PERCEIVE",
        "screenshot / DOM",
        P["teal_soft"],
        P["primary_deep"],
    )
    body += f"  {arrow_h(420, 235, 470)}"
    body += node_box(
        470, 180, 260, 110, "MARK / ground", "SoM · coords", P["amber_soft"], "#92400E"
    )
    body += f"  {arrow_h(730, 235, 780)}"
    body += node_box(
        780, 180, 260, 110, "ACT", "click · type · scroll", P["card"], P["ink"]
    )
    body += f"""  <path d="M910 290 Q910 380 290 380 Q220 380 220 290" fill="none" stroke="{P["primary"]}" stroke-width="2.5" marker-end="url(#arrT)"/>
  <text x="600" y="410" text-anchor="middle" font-family="{FONT}" font-size="14" font-weight="600" fill="{P["primary"]}">boucle jusqu'à objectif UI</text>"""
    body += f"""  {card(150, 460, 900, 100)}
  <text x="600" y="505" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["ink"]}">Grounding fragile → sandbox + allowlist</text>
  <text x="600" y="535" text-anchor="middle" font-family="{FONT}" font-size="13" fill="{P["muted"]}">Environnement = écran / navigateur isolé</text>"""
    return wrap_svg(
        "Boucle perceive mark act GUI",
        "Screenshot ou DOM, grounding Set-of-Marks, action souris clavier, en boucle.",
        body,
    )


def build_22_som() -> str:
    body = title_block(
        "Set-of-Marks (SoM)",
        "Numéroter les éléments cliquables pour le LLM",
        "Module 22 · schéma détaillé",
    )
    # Deux colonnes, marges ≥80
    body += f"  {card(80, 140, 480, 340)}"
    body += f'  <text x="320" y="180" text-anchor="middle" font-family="{FONT}" font-size="16" font-weight="700" fill="{P["muted"]}">Screenshot brut</text>'
    body += node_box(140, 220, 360, 50, "Username", "", P["card"], P["ink"])
    body += node_box(140, 290, 360, 50, "Password", "", P["card"], P["ink"])
    body += node_box(140, 360, 160, 50, "Submit", "", P["teal_soft"], P["primary_deep"])
    body += node_box(340, 360, 160, 50, "Reset", "", P["line"], P["ink"])
    body += f"  {card(640, 140, 480, 340)}"
    body += f'  <text x="880" y="180" text-anchor="middle" font-family="{FONT}" font-size="16" font-weight="700" fill="{P["primary"]}">Screenshot SoM</text>'
    body += node_box(
        700, 220, 360, 50, "Username  [1]", "", P["teal_soft"], P["primary_deep"]
    )
    body += node_box(
        700, 290, 360, 50, "Password  [2]", "", P["teal_soft"], P["primary_deep"]
    )
    body += node_box(700, 360, 160, 50, "Submit [3]", "", P["amber_soft"], "#92400E")
    body += node_box(900, 360, 160, 50, "Reset [4]", "", P["amber_soft"], "#92400E")
    body += f'  <text x="600" y="540" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["ink"]}">Prompt : « clique [1], tape alice, clique [3] »</text>'
    return wrap_svg(
        "Set-of-Marks prompting",
        "Comparaison screenshot brut versus éléments numérotés pour guider les actions du LLM.",
        body,
    )


# --- J25 serving ---


def build_25_stateful() -> str:
    body = title_block(
        "Le piège du stateful en prod",
        "État dans le worker = scaling cassé",
        "Module 25 · schéma détaillé",
    )
    # Anti-pattern en haut (3 workers rouge, centrés)
    body += f'  <text x="600" y="125" text-anchor="middle" font-family="{FONT}" font-size="15" font-weight="700" fill="{P["danger"]}">Anti-pattern : state dans le process</text>'
    for i, x in enumerate([290, 510, 730]):
        body += node_box(
            x,
            145,
            180,
            70,
            f"Worker {i + 1}",
            "mémoire locale",
            P["red_soft"],
            "#991B1B",
        )
    body += f'  <text x="600" y="250" text-anchor="middle" font-family="{FONT}" font-size="13" fill="{P["muted"]}">LB envoie la suite ailleurs → état perdu</text>'
    # Cible en bas (3 workers teal + checkpointer)
    body += f'  <text x="600" y="295" text-anchor="middle" font-family="{FONT}" font-size="15" font-weight="700" fill="{P["primary"]}">Cible : workers stateless + checkpointer</text>'
    body += node_box(
        290, 320, 180, 70, "Worker A", "stateless", P["teal_soft"], P["primary_deep"]
    )
    body += node_box(
        510, 320, 180, 70, "Worker B", "stateless", P["teal_soft"], P["primary_deep"]
    )
    body += node_box(
        730, 320, 180, 70, "Worker C", "stateless", P["teal_soft"], P["primary_deep"]
    )
    # 3 flèches workers → checkpointer
    body += f"  {arrow_v(380, 390, 440)}"
    body += f"  {arrow_v(600, 390, 440)}"
    body += f"  {arrow_v(820, 390, 440)}"
    body += node_box(
        400,
        450,
        400,
        70,
        "Checkpointer",
        "Postgres / Redis",
        P["amber_soft"],
        "#92400E",
    )
    body += f'  <text x="600" y="570" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["ink"]}">Workers interchangeables · état partagé</text>'
    return wrap_svg(
        "Stateful vs workers stateless",
        "Anti-pattern workers avec état local ; cible workers interchangeables et checkpointer partagé.",
        body,
    )


# --- J3 hybrid memory ---


def build_03_hybrid() -> str:
    body = title_block(
        "Hybrid memory",
        "Summary des vieux tours + fenêtre récente",
        "Module 03 · schéma détaillé",
    )
    body += f"  {card(80, 140, 1040, 360)}"
    body += node_box(
        120,
        180,
        420,
        120,
        "Summary messages 1–N",
        "compressé · bas coût",
        P["amber_soft"],
        "#92400E",
    )
    body += node_box(
        660,
        180,
        420,
        120,
        "Recent window",
        "verbatim · haute fidélité",
        P["teal_soft"],
        P["primary_deep"],
    )
    body += f"  {arrow_v(600, 320, 360)}"
    body += node_box(
        350, 370, 500, 70, "Prompt assemblé → LLM", "", P["card"], P["ink"]
    )
    body += f'  <text x="600" y="500" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["ink"]}">Summary avant le plafond de tokens</text>'
    return wrap_svg(
        "Hybrid memory summary plus fenêtre",
        "Résumé des anciens messages à gauche, fenêtre récente à droite, assemblés pour le LLM.",
        body,
    )


# --- J8 hybrid RRF ---


def build_08_hybrid_rrf() -> str:
    body = title_block(
        "Hybrid search + RRF + rerank",
        "Pattern retrieval production",
        "Module 08 · schéma détaillé",
    )
    body += node_box(450, 115, 300, 56, "Query", "", P["amber_soft"], "#92400E")
    # Bifurcation dense / sparse depuis le centre de Query
    body += f'  <line x1="600" y1="171" x2="600" y2="195" stroke="{P["primary"]}" stroke-width="2.5"/>'
    body += f'  <line x1="280" y1="195" x2="920" y2="195" stroke="{P["primary"]}" stroke-width="2.5"/>'
    body += f'  <line x1="280" y1="195" x2="280" y2="220" stroke="{P["primary"]}" stroke-width="2.5" marker-end="url(#arrT)"/>'
    body += f'  <line x1="920" y1="195" x2="920" y2="220" stroke="{P["primary"]}" stroke-width="2.5" marker-end="url(#arrT)"/>'
    body += node_box(
        120,
        230,
        320,
        80,
        "Dense retriever",
        "embeddings · top 50",
        P["teal_soft"],
        P["primary_deep"],
    )
    body += node_box(
        760,
        230,
        320,
        80,
        "Sparse retriever",
        "BM25 · top 50",
        P["teal_soft"],
        P["primary_deep"],
    )
    body += f"  {arrow_v(280, 310, 355)}"
    body += f"  {arrow_v(920, 310, 355)}"
    body += f'  <line x1="280" y1="355" x2="920" y2="355" stroke="{P["primary"]}" stroke-width="2.5"/>'
    body += f'  <line x1="600" y1="355" x2="600" y2="375" stroke="{P["primary"]}" stroke-width="2.5" marker-end="url(#arrT)"/>'
    body += node_box(
        350, 385, 500, 70, "RRF fusion", "fusion des ranks", P["amber_soft"], "#92400E"
    )
    body += f"  {arrow_v(600, 455, 490)}"
    body += node_box(
        350,
        500,
        500,
        64,
        "Cross-encoder rerank",
        "top 5–10 pour le LLM",
        P["card"],
        P["ink"],
    )
    body += f'  <text x="600" y="610" text-anchor="middle" font-family="{FONT}" font-size="13" fill="{P["muted"]}">Dense · sparse · RRF · rerank → top-k LLM</text>'
    return wrap_svg(
        "Hybrid retrieval RRF rerank",
        "Query vers dense et sparse, fusion RRF, puis cross-encoder rerank avant le LLM.",
        body,
    )


# --- J14 deployment ---


def build_14_deployment() -> str:
    body = title_block(
        "Déploiement capstone (théorique)",
        "API HTTP + streaming + stack agent",
        "Module 14 · schéma détaillé",
    )
    layers = [
        (150, "Clients", "web · CLI · jobs", P["card"]),
        (250, "FastAPI", "HTTP + SSE streaming", P["teal_soft"]),
        (350, "Agent runtime", "supervisor + workers", P["amber_soft"]),
        (450, "Stores", "vector · checkpoint · traces", P["card"]),
        (550, "Observability", "OTel · cost · eval hooks", P["red_soft"]),
    ]
    for y, lab, sub, bg in layers:
        ink = (
            P["primary_deep"]
            if bg == P["teal_soft"]
            else (P["ink"] if bg != P["red_soft"] else "#991B1B")
        )
        if bg == P["amber_soft"]:
            ink = "#92400E"
        body += node_box(250, y, 700, 80, lab, sub, bg, ink)
    return wrap_svg(
        "Stack déploiement capstone",
        "Couches clients, FastAPI SSE, runtime agent, stores et observabilité.",
        body,
    )


# --- J19 A2A ---


def build_19_a2a_lifecycle() -> str:
    body = title_block(
        "Cycle de vie d'une tâche A2A",
        "États d'une tâche inter-agents",
        "Module 19 · schéma détaillé",
    )
    # Chemin principal centré : submitted → working → completed
    body += node_box(100, 260, 170, 80, "submitted", "", P["line"], P["ink"])
    body += f"  {arrow_h(270, 300, 320)}"
    body += node_box(
        320, 240, 200, 120, "working", "exécution", P["teal_soft"], P["primary_deep"]
    )
    body += f"  {arrow_h(520, 300, 580)}"
    body += node_box(580, 260, 180, 80, "completed", "", P["amber_soft"], "#92400E")
    # Flèches failed / canceled depuis working
    body += f'  <path d="M420 240 Q420 175 900 175" fill="none" stroke="{P["danger"]}" stroke-width="2" marker-end="url(#arrA)"/>'
    body += node_box(900, 145, 180, 60, "failed", "", P["red_soft"], "#991B1B")
    body += f'  <path d="M520 300 Q720 300 900 360" fill="none" stroke="{P["slate"]}" stroke-width="2" marker-end="url(#arr)"/>'
    body += node_box(900, 340, 180, 60, "canceled", "", P["line"], P["ink"])
    # Boucle input-required
    body += f'  <path d="M420 360 Q420 470 200 470" fill="none" stroke="{P["accent"]}" stroke-width="2" marker-end="url(#arrA)"/>'
    body += node_box(
        100, 440, 200, 60, "input-required", "HITL info", P["amber_soft"], "#92400E"
    )
    body += f'  <path d="M200 440 Q200 400 320 360" fill="none" stroke="{P["accent"]}" stroke-width="2" marker-end="url(#arrA)"/>'
    body += f'  <text x="280" y="520" text-anchor="middle" font-family="{FONT}" font-size="12" fill="{P["accent"]}">HITL → working</text>'
    body += f'  <text x="600" y="580" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["ink"]}">A2A orchestre agents · MCP = outils</text>'
    return wrap_svg(
        "Lifecycle tâche A2A",
        "États submitted, working, completed, avec branches input-required, failed et canceled.",
        body,
    )


# --- J20 durable ---


def build_20_combined() -> str:
    body = title_block(
        "Durable + event-driven + HITL",
        "Architecture combinée pour longs runs",
        "Module 20 · schéma détaillé",
    )
    body += f"  {card(80, 130, 680, 400)}"
    body += f'  <text x="100" y="170" font-family="{FONT}" font-size="14" font-weight="700" fill="{P["primary"]}">Workflow durable (Temporal…)</text>'
    body += node_box(
        120, 200, 280, 56, "Activity: LLM plan", "", P["teal_soft"], P["primary_deep"]
    )
    body += node_box(
        120, 275, 280, 56, "Activity: tools", "", P["amber_soft"], "#92400E"
    )
    body += node_box(
        120, 350, 280, 56, "Wait signal HITL", "", P["red_soft"], "#991B1B"
    )
    body += node_box(120, 425, 280, 56, "Activity: finalize", "", P["card"], P["ink"])
    body += f"  {card(820, 160, 280, 340)}"
    body += f'  <text x="960" y="200" text-anchor="middle" font-family="{FONT}" font-size="14" font-weight="700" fill="{P["ink"]}">Event bus</text>'
    body += node_box(
        850, 230, 220, 50, "user.reply", "", P["teal_soft"], P["primary_deep"]
    )
    body += node_box(850, 300, 220, 50, "timer.fire", "", P["amber_soft"], "#92400E")
    body += node_box(850, 370, 220, 50, "tool.done", "", P["card"], P["ink"])
    body += f'  <text x="600" y="580" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["muted"]}">Crash ≠ perte — reprise au signal</text>'
    return wrap_svg(
        "Architecture durable event-driven HITL",
        "Workflow durable avec activities LLM et tools, attente signal HITL, alimenté par un bus d'événements.",
        body,
    )


# --- J24 inference ---


def build_24_levers() -> str:
    body = title_block(
        "Trois leviers d'inference engineering",
        "Routing · structured outputs · caching",
        "Module 24 · schéma détaillé",
    )
    body += node_box(450, 120, 300, 56, "Requête user", "", P["amber_soft"], "#92400E")
    body += f"  {arrow_v(600, 176, 215)}"
    body += node_box(
        400,
        225,
        400,
        70,
        "ModelRouter",
        "weak si simple · strong si dur",
        P["teal_soft"],
        P["primary_deep"],
    )
    body += f"  {arrow_v(600, 295, 340)}"
    body += node_box(
        140,
        350,
        280,
        90,
        "Structured out",
        "schema / tools fiables",
        P["card"],
        P["ink"],
    )
    body += node_box(
        460,
        350,
        280,
        90,
        "Prompt cache",
        "préfixe réutilisé",
        P["amber_soft"],
        "#92400E",
    )
    body += node_box(
        780,
        350,
        280,
        90,
        "Réponse",
        "latence ↓ coût ↓",
        P["teal_soft"],
        P["primary_deep"],
    )
    body += f'  <text x="600" y="500" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["ink"]}">Router → format contraint → cache du stable</text>'
    return wrap_svg(
        "Trois leviers inference engineering",
        "Requête, model router, puis structured outputs et prompt caching vers la réponse.",
        body,
    )


# --- J23 defense layers list as stack ---


def build_23_layers() -> str:
    body = title_block(
        "Isolation : défense en profondeur infra",
        "process → microVM → egress → audit",
        "Module 23 · schéma détaillé",
    )
    # L7 (audit) en haut → L1 (process) en bas ; sous-titre aligné sur le contenu
    layers = [
        ("L7  Audit log immuable", P["card"]),
        ("L6  Egress filtering", P["teal_soft"]),
        ("L5  Capability tools", P["amber_soft"]),
        ("L4  MicroVM / Firecracker", P["card"]),
        ("L3  gVisor / sandbox runtime", P["teal_soft"]),
        ("L2  Container / cgroups", P["amber_soft"]),
        ("L1  Process limits", P["red_soft"]),
    ]
    for i, (lab, bg) in enumerate(layers):
        y = 110 + i * 62
        ink = (
            P["ink"]
            if bg in (P["card"], P["line"])
            else (
                P["primary_deep"]
                if bg == P["teal_soft"]
                else ("#92400E" if bg == P["amber_soft"] else "#991B1B")
            )
        )
        body += node_box(180, y, 840, 54, lab, "", bg, ink)
    return wrap_svg(
        "Couches d'isolation sandbox",
        "Sept couches empilées de l'audit log jusqu'aux limites process.",
        body,
    )


# --- J15 context isolation ---


def build_15_isolation() -> str:
    body = title_block(
        "Isolation de contexte par sous-agent",
        "Le worker ne voit pas les 80k du superviseur",
        "Module 15 · schéma détaillé",
    )
    body += node_box(
        100,
        180,
        400,
        260,
        "Superviseur",
        "80k historique · budget global",
        P["amber_soft"],
        "#92400E",
    )
    body += f"  {arrow_h(500, 310, 560)}"
    body += node_box(
        560,
        200,
        480,
        220,
        "Sous-agent",
        "prompt 500 tokens · tâche X",
        P["teal_soft"],
        P["primary_deep"],
    )
    body += f'  <text x="600" y="500" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["ink"]}">Déléguer = minimum utile, pas tout le contexte</text>'
    body += f'  <text x="600" y="540" text-anchor="middle" font-family="{FONT}" font-size="13" fill="{P["muted"]}">Réduit coût, noise et fuites entre tâches</text>'
    return wrap_svg(
        "Isolation contexte sous-agent",
        "Superviseur à gros historique qui délègue un prompt minimal à un sous-agent.",
        body,
    )


# --- J17 self-refine ---


def build_17_self_refine() -> str:
    body = title_block(
        "Self-Refine / verifier loop",
        "Generate → critique → refine → select",
        "Module 17 · schéma détaillé",
    )
    # 4 boîtes centrées, marges ≥80
    body += node_box(
        100, 220, 200, 100, "Generator", "draft", P["teal_soft"], P["primary_deep"]
    )
    body += f"  {arrow_h(300, 270, 345)}"
    body += node_box(
        345, 220, 200, 100, "Verifier", "ORM / PRM", P["amber_soft"], "#92400E"
    )
    body += f"  {arrow_h(545, 270, 590)}"
    body += node_box(590, 220, 200, 100, "Refiner", "corrige", P["card"], P["ink"])
    body += f"  {arrow_h(790, 270, 835)}"
    body += node_box(
        835, 220, 200, 100, "Select", "best / accept", P["teal_soft"], P["primary_deep"]
    )
    body += f"""  <path d="M690 320 Q690 420 200 420 Q150 420 150 320" fill="none" stroke="{P["accent"]}" stroke-width="2.5" marker-end="url(#arrA)"/>
  <text x="420" y="450" text-anchor="middle" font-family="{FONT}" font-size="13" fill="{P["accent"]}">retry borné</text>"""
    body += f'  <text x="600" y="520" text-anchor="middle" font-family="{FONT}" font-size="14" fill="{P["ink"]}">Option : persister les leçons entre runs</text>'
    return wrap_svg(
        "Boucle Self-Refine verifier",
        "Generator, verifier, refiner, select, avec boucle de retry bornée et leçons optionnelles.",
        body,
    )
