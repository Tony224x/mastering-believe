# Jour 16 — Agentic LLMs en production : tool use, MCP, computer use

> **Temps estime** : 5h | **Prerequis** : J1-J15 (LLMs, reasoning models)

---

## 1. Ce qui a change entre 2023 et 2026

En 2023, "agent" signifiait ReAct + LangChain + beaucoup d'espoir. Taux de succes < 40% sur des taches simples. En 2026, les agents sont **une ligne produit** dans tous les labs :

- **Claude Code** (Anthropic, 2025) → agent de dev dans le terminal, utilise en prod par des equipes entieres
- **Computer Use** (Claude 3.5 Sonnet, oct 2024 → Claude 4.5 Opus, 2025) → controle pixel-perfect d'un ecran
- **ChatGPT Agent** (OpenAI, 2025) → browsing + code + filesystem unifies
- **Gemini CLI / AI Studio** → tool use natif + execution sandboxee
- **OSWorld / SWE-Bench Verified / TAC-Bench** sont devenus les benchmarks qui comptent

Trois raisons de ce saut :
1. **Reasoning models** (J15) → meilleure planification, meilleur recovery d'erreur
2. **Tool use natif au training** → les modeles sont pre-entraines sur des traces d'outils
3. **MCP** (Model Context Protocol, Anthropic, nov 2024) → standard ouvert pour connecter outils et LLMs

### Definition operationnelle d'un agent en 2026

> Un **agent** est un LLM place dans une boucle ou il choisit des actions (outils) jusqu'a avoir atteint un objectif, sans intervention humaine a chaque tour.

Ce qui distingue d'un "chatbot qui appelle une API" :
- Multi-tour (5-200+ tool calls)
- Decision dynamique (le plan change en fonction des observations)
- Recovery d'erreur autonome (re-essayer avec une approche differente)
- Etat persistant entre les tours (memoire de travail)

---

## 2. L'anatomie d'une boucle agentique moderne

```python
# Pseudocode simplifie de la boucle Claude Code / OpenAI agents SDK
messages = [system_prompt, user_task]
while not done and turn < max_turns:
    response = llm.generate(messages, tools=TOOLS)
    messages.append(response)

    if response.tool_calls:
        for call in response.tool_calls:
            result = execute_tool(call.name, call.arguments)
            messages.append(ToolResult(call.id, result))
    else:
        done = True  # le modele a rendu une reponse finale
    turn += 1
```

**Les 4 composants critiques** :
1. **Le system prompt** : definit le role, les contraintes, les outils disponibles, comment reporter le resultat
2. **La definition des outils** : JSON schema + description textuelle claire (ce qui compte le plus pour un bon tool use)
3. **La boucle** : timeout, max_turns, token budget, circuit breakers
4. **L'observabilite** : logs structures de chaque tool call, latence, succes/echec

### Formats de tool calling en 2026

| Format | Qui | Robustesse | Note |
|---|---|---|---|
| **OpenAI tools API** | OpenAI, compatibles | Excellent | De facto standard |
| **Claude tool_use** | Anthropic | Excellent | Syntaxe plus verbose mais mieux type |
| **XML tags** | Anthropic historique, jailbreak-resistant | Bon | Utile en texte libre |
| **JSON mode libre** | Legacy | Mediocre | A eviter : parsing fragile |
| **MCP** | Standard ouvert 2024+ | Excellent | Abstrait le transport |

Regle : tu ne definis JAMAIS tes outils en "le modele doit repondre en JSON avec X, Y, Z". Tu utilises l'API tool calling native. C'est 10x plus fiable.

---

## 3. MCP — Model Context Protocol

### Le probleme que MCP resout

Avant MCP, chaque combo (LLM app × outil) etait un adapter custom :
```
Claude Code ↔ GitHub    : adapter custom 1
Cursor      ↔ GitHub    : adapter custom 2
ChatGPT     ↔ GitHub    : adapter custom 3
Claude Code ↔ Postgres  : adapter custom 4
...
```
N apps × M outils = N*M adapters. C'etait le cauchemar.

### L'architecture MCP

```
  [LLM host app]         [MCP client]          [MCP server]
  Claude Code,     ←→    protocole     ←→     github, slack,
  Cursor, etc.           JSON-RPC              filesystem, db
```

- **Server** expose des resources (read-only), tools (actions), prompts (templates)
- **Client** (inclus dans l'app LLM) parle un JSON-RPC standard
- **Host** (l'app utilisateur) gere la session et les permissions

Ainsi, ecrire un MCP server pour ton API = tu es branchable a Claude Code, Cursor, Cline, Windsurf, ChatGPT, etc., sans code specifique.

### Les concepts MCP que tu dois connaitre comme AI engineer

- **`initialize`** : handshake capabilities entre client et server
- **`tools/list`**, **`tools/call`** : decouverte et execution d'outils
- **`resources/read`** : lecture de donnees (fichiers, rows DB)
- **`prompts/get`** : templates prepackagees
- **Roots** : sandbox filesystem partage
- **Sampling** : le server peut demander une generation LLM au host (chaine renversee — utile pour agents dans l'agent)

### Quand utiliser MCP vs tool calling natif

| Situation | Choix |
|---|---|
| Outil custom interne a un seul produit | tool calling natif (plus simple) |
| Outil reutilisable cross-apps (ex: ton SaaS) | MCP server (reach maximal) |
| Outil tres latence-critique | tool calling natif (moins de hops) |
| Outil sensible (secrets, ecriture DB) | MCP avec permission prompts obligatoires |

---

## 4. Computer Use — l'agent qui voit l'ecran

Sorti en oct 2024 (Claude 3.5 Sonnet), generalise en 2025-2026 (Claude 4.5, Gemini 2.5 Computer Use, OpenAI Operator). L'idee : donner au modele une capture d'ecran + un clavier/souris virtuel.

```
Loop:
  screenshot = take_screenshot()
  action = model(task, screenshot, history)
  # action ∈ {click(x,y), type(text), scroll(dy), key(combo), wait, done}
  execute(action)
```

### Pourquoi c'est un gros deal

- **Pas besoin d'API** : tout site web / app desktop est "programmable"
- **Tout software legacy** devient automatisable sans integration
- **Les agents generalistes** peuvent enfin agir dans le monde numerique

### Les vraies limites 2026 (honnetete d'AI engineer)

- Taux de succes OSWorld 2026 : ~60% (humain ~90%). Tres loin du "remplace un stagiaire".
- **Latence** : 3-15s par action. Une tache "reserver un vol" prend 3-10 minutes.
- **Cout** : chaque screenshot = 1-2k tokens d'image. Sessions longues = $$.
- **Securite** : prompt injection via contenu affiche (spam email, site malveillant) → le modele execute des actions destructrices. Sandboxing obligatoire.
- **Captchas et anti-bot** : les sites font la chasse aux agents.

Pour la prod aujourd'hui : **computer use est un fallback**, pas le default. Tool calling > MCP > browser automation (Playwright + LLM) > computer use, dans cet ordre de preference.

---

## 5. Patterns d'architecture d'agents en 2026

### Pattern 1 : Single agent avec tool use (le defaut)

```
User → Agent(reasoning model) ↔ Tools → Response
```

80% des cas en 2026. Claude 4.5 Sonnet/Opus + 10-30 outils + bon system prompt + boucle robuste. C'est ce que fait Claude Code.

### Pattern 2 : Sub-agents / delegation

```
Orchestrator → spawn(SubAgent A) → result
             → spawn(SubAgent B) → result
             → synthesize → Response
```

Utile quand : taches independantes parallelisables, besoin de contextes separes pour eviter la pollution, specialisation par role. Pattern "Anthropic research system" (2025) : un lead agent spawn N search agents, chacun avec sa fenetre de contexte propre.

### Pattern 3 : Supervisor + swarm (LangGraph)

Utilise dans des systemes multi-agent complexes. Le supervisor route vers un specialiste, qui peut re-router via le swarm. Cf `domains/agentic-ai/02-code/` et le projet guide supervisor-swarm-interarmes dans ce repo.

### Pattern 4 : Deep research / long-horizon

```
Phase 1 : Planner (reasoning) → plan detaille (1 appel)
Phase 2 : Executors paralleles (N appels rapides)
Phase 3 : Synthesizer (reasoning) → rapport final (1 appel)
```

Utilise par OpenAI "Deep Research", Gemini "Deep Think". Budget typique : 100-1000 tool calls, 20-60 min, rapport de 10-50 pages. Cout : $5-50 par query.

### Pattern 5 : Human-in-the-loop pour actions risquees

```
Agent propose une action destructive (rm -rf, DROP TABLE, send email)
    ↓
Pause + demande explicite humaine
    ↓
Humain approuve / modifie / annule
    ↓
Agent continue
```

Non-optionnel pour la prod. Claude Code, Cursor, etc. implementent tous cette pause. Les outils d'ecriture DB, envoi de messages, paiement DOIVENT passer par une confirmation.

---

## 6. Les 10 regles d'or d'un agent en prod

1. **Moins d'outils, mieux decrits** : 10 outils avec descriptions excellentes battent 50 outils moyens.
2. **Idempotence** : chaque tool call doit pouvoir etre rejoue sans effet de bord negatif.
3. **Timeout partout** : par tool call, par boucle, par session.
4. **Budget tokens dur** : kill l'agent a N tokens. Les boucles infinies sont la premiere source de pertes.
5. **Observability first** : trace chaque tool call avec ID, latence, input/output, erreur. OpenTelemetry ou equivalent.
6. **Schema d'outil versionne** : changer un tool schema peut casser silencieusement les prompts.
7. **Separate tool implementation from tool definition** : l'implementation peut changer, le schema expose au modele reste stable.
8. **Tool results = texte UTF-8 borne** : max 50k tokens, tronquer avec un marqueur explicite.
9. **Fail loud, not silent** : retourner au modele "Error: X" texte explicite permet au modele de recover. Swallowing les erreurs = boucle morte.
10. **Permissioning en amont** : least privilege par outil, pas au niveau "agent".

---

## 7. Piege : le context rot

Au-dela de ~100 tool calls, meme avec 1M de context window, les agents **oublient leur tache initiale** ou **suivent un faux trail**. C'est le "context rot" : plus le contexte est long, plus le modele est distrait par des details intermediaires.

Solutions 2026 :
- **Summarization periodique** : tous les 30 tool calls, condenser l'historique
- **Scratchpad externe** : un outil `note_to_self` ou le modele ecrit son plan a jour
- **Redemarrage avec handoff** : un sub-agent fresh prend le relais avec un contexte condense
- **Context compaction** : au niveau API (Claude supporte `compaction` depuis 2025)

Cf J18 pour le context engineering en profondeur.

---

## Key takeaways (flashcards)

**Q1** — Quels sont les 4 composants critiques d'une boucle agentique en prod ?
> System prompt + definition des outils + boucle (timeout/max_turns/budget) + observabilite.

**Q2** — Pourquoi MCP a gagne en 2025 ?
> Standard ouvert qui evite le N×M d'adapters. Un MCP server pour ton API = branchable a Claude Code, Cursor, ChatGPT, etc. sans code specifique.

**Q3** — Quand utiliser computer use plutot que tool calling ?
> En dernier recours : quand l'outil n'a pas d'API et que MCP/browser automation ne suffisent pas. Latence 10x, cout 5x, fiabilite moindre. A sandboxer strictement.

**Q4** — Qu'est-ce que le context rot et comment le mitiger ?
> Au-dela de ~100 tool calls, l'agent oublie ou se distrait. Mitigations : summarization periodique, scratchpad externe, handoff a un sub-agent fresh, context compaction native API.

**Q5** — Pourquoi separer tool schema et tool implementation ?
> Le schema expose au modele doit rester stable (changer casse les prompts). L'implementation peut evoluer sous-jacente. Cette indirection permet aussi de mocker pour les tests.

**Q6** — Quel est le pattern "Deep Research" ?
> Planner (reasoning, 1 appel) → Executors paralleles (N appels rapides) → Synthesizer (reasoning, 1 appel). Utilise par OpenAI Deep Research, Gemini Deep Think. Cost/qualite optimal pour long-horizon.
