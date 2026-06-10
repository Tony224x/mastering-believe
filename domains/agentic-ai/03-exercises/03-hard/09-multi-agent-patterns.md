# Exercices Hard — Multi-Agent Patterns (J9)

---

## Exercice 1 : Orchestrateur generique a patterns interchangeables

### Objectif
Extraire l'abstraction commune aux patterns multi-agents : construire un orchestrateur ou supervisor, swarm et debate sont des strategies interchangeables partageant les memes agents, le meme budget et le meme tracing — puis comparer objectivement les 3 patterns sur la meme tache.

### Consigne
1. Definis les abstractions :
   ```python
   class AgentRegistry:
       """Agents partages: name -> (system_prompt, capability_tags)."""

   @dataclass
   class OrchestrationResult:
       pattern: str
       output: str
       llm_calls: int
       handoffs_or_steps: int
       trace: list[dict]      # [{agent, action, output_excerpt, ts}]

   class Pattern(Protocol):
       def run(self, task: str, registry: AgentRegistry,
               budget: CallBudget) -> OrchestrationResult: ...
   ```
2. Implemente les 3 patterns derriere cette interface :
   - `SupervisorStrategy` : plan central, delegation etape par etape
   - `SwarmStrategy` : handoffs decentralises (chaque agent decide du suivant)
   - `DebateStrategy` : proposition initiale + scores + verdict modere
3. **Budget partage** : `CallBudget(max_calls)` injecte dans chaque run ; chaque appel LLM passe par `budget.spend()` ; depassement -> arret propre du pattern avec `output="[BUDGET EXHAUSTED] " + best_effort`
4. **Tracing unifie** : les 3 patterns remplissent `trace` avec le meme schema — c'est ce qui permet la comparaison
5. Lance la MEME tache ("Implement and validate a function that deduplicates a list of orders") avec les 3 patterns et produis un tableau comparatif :
   ```
   Pattern     | calls | steps/hops | output quality (keywords) | budget ok
   supervisor  |   5   |     3      |  code+review+doc          |   yes
   ...
   ```
6. Ajoute un mini "quality check" local : la sortie doit contenir du code (`def `), une trace de review (`VERDICT` ou `review`), et etre non vide — score 0-3
7. Teste aussi le comportement sous budget tres serre (`max_calls=2`) : les 3 patterns doivent s'arreter proprement, sans exception non geree

### Criteres de reussite
- [ ] Les 3 patterns implementent exactement la meme interface et tournent sur les memes agents
- [ ] Le budget partage est respecte par les 3 (asserts sur les compteurs)
- [ ] Les traces des 3 patterns ont un schema identique et exploitable
- [ ] Le tableau comparatif s'affiche avec les metriques reelles
- [ ] Le quality check score chaque pattern sans en favoriser un par construction
- [ ] Le mode budget serre termine proprement pour les 3 patterns

---

## Exercice 2 : Contract-Net Protocol — encheres de taches entre agents

### Objectif
Implementer un protocole de coordination decentralise classique (Contract-Net, Smith 1980) adapte aux agents LLM : un manager met une tache aux encheres, les agents soumissionnent selon leurs capacites, le gagnant peut sous-traiter, et l'echec declenche une re-enchere.

### Consigne
1. Definis les messages du protocole (dataclasses) : `CallForProposal(task_id, description, deadline_hops)`, `Proposal(agent, confidence, cost_estimate, rationale)`, `Award(agent, task_id)`, `TaskResult(agent, task_id, status, output)`
2. Chaque agent a un profil de capacites : `{"coder": {"skills": ["python", "implement"], "cost": 3}, "researcher": {"skills": ["search", "analyze"], "cost": 2}, ...}` (4 agents minimum)
3. **Soumission** : a la reception d'un CFP, chaque agent calcule (mock deterministe) :
   - `confidence` = overlap entre les mots de la tache et ses skills (0 si aucun match -> il decline)
   - `cost_estimate` = son cout de base + 1 par mot-cle non couvert
4. **Attribution** : le manager choisit le meilleur ratio `confidence / cost`, en departageant par ordre alphabetique (determinisme). Trace tous les bids
5. **Sous-traitance** : si la tache contient 2 domaines de skills disjoints ("research best practices THEN implement"), le gagnant detecte la partie hors de ses skills et emet un CFP secondaire pour la sous-partie (1 niveau de sous-traitance max)
6. **Echec et re-enchere** : simule un agent qui echoue (`status="failed"`) a sa premiere tache attribuee ; le manager le blackliste pour ce task_id et relance l'enchere avec les agents restants
7. Demo en 3 scenarios : tache simple (1 gagnant direct), tache composite (sous-traitance), tache avec echec (re-enchere). Affiche pour chacun le deroulement complet du protocole
8. Garde-fous : un CFP sans aucune proposition -> le manager retourne `"no capable agent"` ; profondeur de sous-traitance limitee a 1 ; pas de re-enchere infinie (1 retry max par tache)

### Criteres de reussite
- [ ] Les 4 types de messages structurent tous les echanges (pas d'appel direct hors protocole)
- [ ] Les bids sont deterministes et coherents avec les profils de capacites
- [ ] L'attribution choisit le bon agent et la trace montre tous les bids
- [ ] La sous-traitance ne depasse jamais 1 niveau et produit un resultat assemble
- [ ] La re-enchere apres echec blackliste l'agent fautif et aboutit
- [ ] Les 3 garde-fous sont testes explicitement
