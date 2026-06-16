# Exercices Medium — Multi-agent patterns (J9)

---

## Exercice 1 : Supervisor qui route vers le bon specialiste (avec fallback)

### Objectif
Aller au-dela du supervisor "plan fige" du module : construire un supervisor qui **classe** chaque sous-tache (routing pattern), choisit le specialiste adapte parmi plusieurs, et retombe sur un agent generaliste de **fallback** quand aucun specialiste ne matche. C'est le piege classique du supervisor : il doit etre bon a router, pas seulement a synthetiser.

### Consigne
En t'inspirant du `SupervisorPattern` du module 09 (tu peux l'importer via `sys.path` ou tout reembarquer dans ta solution) :

1. Ecris un `RoutingMockLLM` deterministe qui expose au moins 3 specialistes (`math_agent`, `text_agent`, `code_agent`) plus un `generalist` de fallback. Chacun produit une sortie reconnaissable (ex : `math_agent` prefixe `[math]`, etc.).
2. Implemente une classe `RoutingSupervisor` avec une methode `route(subtask: str) -> str` qui retourne le **nom** du specialiste choisi :
   - une sous-tache contenant des chiffres / operateurs (`+ - * / =` ou mots `calcul`, `somme`, `moyenne`) → `math_agent`
   - une sous-tache contenant `code`, `fonction`, `python`, `bug` → `code_agent`
   - une sous-tache contenant `resume`, `redige`, `traduis`, `corrige` → `text_agent`
   - sinon → `generalist` (**fallback**, jamais une exception)
3. La methode `run(subtasks: list[str]) -> dict` doit, pour chaque sous-tache : router, invoquer le specialiste choisi, et stocker `(subtask, agent, output)` dans une trace.
4. Ajoute une etape finale de **synthese** par le supervisor qui reecrit toutes les sorties dans un style unifie (pas une simple concatenation).
5. Teste avec un lot de 4-5 sous-taches melangeant les 4 categories (dont **au moins une** qui doit partir au fallback) et **prouve par assertions** que chaque sous-tache a ete routee vers le bon agent.

### Criteres de reussite
- [ ] `route()` est total : il retourne toujours un nom d'agent valide, jamais d'exception sur une entree inattendue
- [ ] Au moins une sous-tache tombe sur le `generalist` (fallback) et c'est verifie par assertion
- [ ] Chaque categorie (math / code / text) est routee vers le bon specialiste, prouve par assertion
- [ ] La trace contient `(subtask, agent, output)` pour chaque sous-tache
- [ ] La synthese finale mentionne les contributions de plusieurs agents, pas un simple `"\n".join`
- [ ] Tout tourne offline, sans cle API ni dependance

---

## Exercice 2 : Swarm avec handoff explicite et report de contexte

### Objectif
Implementer le **handoff tool-like** du module (section 4.3) : un agent termine sa part, construit un **payload de handoff** structure, et passe le controle a un autre agent. Tu dois prouver deux choses que beaucoup d'implementations ratent : (a) le controle est **reellement transfere** (l'agent suivant prend la main), et (b) le **contexte est preserve** a travers le handoff (pas de perte de contexte, section 6.3).

### Consigne
1. Definis un `@dataclass Handoff` avec au minimum : `to_agent: str`, `reason: str`, `payload: dict` (le contexte transmis : ce qui a deja ete fait, les artefacts produits).
2. Modelise 3 agents specialises sous forme de callables purs `agent(task, inbox) -> (output, handoff | None)` :
   - `triage` : ne traite rien, decide qui doit commencer et fait un handoff vers `coder` avec un payload `{"spec": ...}`.
   - `coder` : lit `payload["spec"]`, produit du code, et fait un handoff vers `qa` avec un payload qui **inclut le code produit** + ce qu'il a recu (`spec`).
   - `qa` : lit `payload["code"]`, valide, et **termine** (retourne `handoff = None`).
3. Ecris un orchestrateur `run_swarm(start_agent, task) -> dict` qui boucle sur les handoffs :
   - maintient un `control: str` (l'agent qui a la main) et le met a jour a chaque handoff,
   - **accumule** le contexte : chaque agent recoit dans son `inbox` le payload du handoff precedent,
   - tient une `loop guard` (max hops) qui leve `RuntimeError` si depassee.
4. **Prouve par assertions** :
   - la sequence de controle est exactement `triage → coder → qa` (le controle transfere bien),
   - le `qa` a recu dans son inbox la `spec` d'origine **ET** le `code` produit par le coder (le contexte a traverse 2 handoffs),
   - le run se termine proprement quand `qa` ne fait plus de handoff (pas de boucle).

### Criteres de reussite
- [ ] `Handoff` est un dataclass avec `to_agent`, `reason`, `payload`
- [ ] La sequence de controle observee est `["triage", "coder", "qa"]`, verifiee par assertion
- [ ] Le payload recu par `qa` contient a la fois la `spec` initiale et le `code` produit (contexte preserve sur 2 sauts)
- [ ] L'orchestrateur s'arrete proprement quand le dernier agent retourne `handoff=None`
- [ ] La loop guard leve `RuntimeError` sur un swarm qui bouclerait (test dedie avec 2 agents qui se renvoient la main)
- [ ] Aucune dependance externe, execution offline

---

## Exercice 3 : Blackboard partage avec detection d'achevement

### Objectif
Implementer le pattern **shared state / blackboard** (section 4.2) : plusieurs agents lisent et ecrivent dans un etat global commun, et un orchestrateur detecte **quand la tache est terminee** en inspectant le blackboard — sans plan central rigide. C'est l'alternative "data-driven" au supervisor : les agents se declenchent selon ce qui manque dans l'etat.

### Consigne
1. Definis un blackboard sous forme de dict avec des cles booleennes/donnees, ex :
   ```python
   board = {"task": "...", "research": None, "code": None, "review": None}
   ```
2. Modelise 3 agents comme des callables `agent(board) -> dict` qui retournent **uniquement les cles qu'ils ecrivent** (un patch a merger), et qui declarent une **precondition** (ce qu'ils ont besoin de lire) :
   - `researcher` : precondition `research is None` → ecrit `research`.
   - `coder` : precondition `research is not None and code is None` → lit `research`, ecrit `code`.
   - `reviewer` : precondition `code is not None and review is None` → lit `code`, ecrit `review`.
3. Ecris un orchestrateur `run_blackboard(board, agents, max_rounds=10)` qui :
   - a chaque round, parcourt les agents et **n'execute que ceux dont la precondition est vraie** (eligibles),
   - merge leur patch dans le board,
   - **detecte l'achevement** via une fonction `is_complete(board)` (toutes les cles attendues sont remplies) et s'arrete des qu'elle est vraie,
   - leve `RuntimeError` si `max_rounds` est atteint sans achevement (deadlock / agent manquant).
4. **Prouve par assertions** :
   - l'ordre d'ecriture effectif respecte les dependances (`research` avant `code` avant `review`), meme si les agents sont passes a l'orchestrateur dans un ordre **melange**,
   - chaque agent ne s'execute **qu'une seule fois** (idempotence via les preconditions),
   - `is_complete` devient vraie et le run s'arrete avant `max_rounds`,
   - un board auquel il manque un agent (ex : pas de `coder`) provoque bien le `RuntimeError` de deadlock.

### Criteres de reussite
- [ ] Le blackboard est un dict partage lu et ecrit par plusieurs agents
- [ ] Les agents ne s'executent que si leur precondition est satisfaite (data-driven, pas de plan fige)
- [ ] L'ordre d'ecriture respecte les dependances meme avec un ordre d'agents melange en entree
- [ ] Chaque agent s'execute exactement une fois (verifie par un compteur)
- [ ] `is_complete(board)` declenche l'arret avant `max_rounds`
- [ ] Un blackboard incompletable (agent manquant) leve `RuntimeError` au lieu de tourner a l'infini
- [ ] Execution offline, deterministe, sans dependance
