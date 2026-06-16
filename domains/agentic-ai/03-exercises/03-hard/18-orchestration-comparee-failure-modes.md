# Exercices Hard — Orchestration comparee & failure modes (J18)

---

## Exercice 1 : Classifieur de failure modes multi-agent a partir de traces d'execution

### Objectif
Construire un **classifieur de failure modes** : on lui donne des **traces d'execution** multi-agent (sequence d'evenements : appels, handoffs, ecritures d'etat, outputs), et il doit **detecter et categoriser** au moins **4 failure modes distincts** issus du cours (section 4) et de la litterature MAST. C'est l'outillage d'observabilite (section 4.1, "le debug est difficile") qui transforme un log brut en diagnostic exploitable.

### Consigne
Definis un format de trace minimal : une liste d'`Event` (`@dataclass` avec au moins `step: int`, `agent: str`, `kind: str` parmi `call`/`handoff`/`write`/`output`, et un `data: dict`). Puis construis un `FailureClassifier` avec une methode `classify(trace: list[Event]) -> list[str]` (les labels detectes) qui repere **au moins 4** des failure modes suivants :

1. **`info_loss_handoff`** — un handoff transmet un payload qui **perd** une cle presente avant le handoff (perte d'information inter-agents, section 4.1). Detection : compare les cles du contexte avant/apres un evenement `handoff`.
2. **`role_violation`** — un agent ecrit/agit hors de son perimetre declare (ex : le `reviewer` ecrit la cle `code`, ou un agent appelle un tool non autorise pour son role). Detection : table `role -> cles/tools autorises`.
3. **`infinite_loop` / livelock** — le **meme couple (agent, kind)** ou un cycle de handoffs se repete au-dela d'un seuil sans progres d'etat (boucle de desaccord, section 4.2). Detection : fenetre/compteur de repetitions sans nouvelle cle ecrite.
4. **`conflicting_writes`** — deux agents ecrivent des **valeurs differentes** sur la **meme** cle d'etat (actions contradictoires / race sur l'etat partage). Detection : suivi de la derniere valeur par cle.
5. *(bonus)* **`cascade_unvalidated`** — une sortie de faible confiance est relayee comme `output` final sans evenement de validation entre les deux (hallucination cross-agent amplifiee, section 4.4).

Exigences :
- Le classifieur renvoie l'**ensemble** des labels presents dans une trace (une trace peut en cumuler plusieurs, mais privilegie des traces ciblees pour les assertions).
- Fournis pour chaque failure mode une **trace forgee** qui le declenche, plus **une trace saine** qui n'en declenche aucun.
- **Prouve par assertions** : chaque trace fautive est classee avec le **bon** label (et seulement lui, ou au moins le label attendu present), et la trace saine renvoie une liste vide.

### Criteres de reussite
- [ ] Un format `Event` structure represente la trace d'execution multi-agent
- [ ] `classify()` detecte **>= 4** failure modes distincts (info_loss, role_violation, infinite_loop, conflicting_writes)
- [ ] Chaque failure mode a une trace forgee qui le declenche, verifiee par assertion (bon label detecte)
- [ ] Une trace saine renvoie une liste de labels vide (pas de faux positif), verifiee par assertion
- [ ] La detection de boucle distingue "repetition sans progres d'etat" d'une iteration legitime
- [ ] Execution offline, deterministe, stdlib uniquement

---

## Exercice 2 : Orchestrateur robuste — comparer les topologies ET recuperer sur failure mode

### Objectif
Boucler la boucle du module : un **orchestrateur robuste** qui (a) choisit/compare des topologies, (b) **detecte** un failure mode pendant l'execution (en reutilisant l'esprit de l'Ex 1), et (c) applique une **strategie de recuperation** graduee — **retry** (erreur transitoire), **reroute** (sous-agent alternatif), puis **fallback to single-agent** (circuit-breaker, section 6 + debat Cognition section 5.1). Tu dois prouver **end-to-end** que le run recupere reussit la ou le run naif echoue.

### Consigne
Construis un `RobustOrchestrator` qui execute une tache decomposee en sous-taches via une topologie multi-agent, mais sous surveillance :

1. **Agents faillibles deterministes** : modelise des sous-agents dont l'echec est **scripte** (pas aleatoire) — ex : le `coder_agent` echoue (sortie malformee / exception) aux 2 premiers appels puis reussit (erreur transitoire), un autre agent echoue **toujours** (erreur permanente → exige un reroute), etc. Garde le tout deterministe.
2. **Detection** : apres chaque sous-etape, valide la sortie (schema + un detecteur de failure mode minimal, ex : sortie vide/malformee ou confiance trop basse). Une etape invalide declenche la politique de recuperation.
3. **Politique de recuperation graduee** (dans cet ordre, configurable) :
   - **retry** avec un compteur `max_retries` (gere l'echec transitoire),
   - **reroute** : si l'agent echoue encore, router la sous-tache vers un agent **alternatif** equivalent (specialisation de secours),
   - **fallback single-agent** : si la topologie multi-agent echoue malgre retry+reroute, basculer **toute** la tache sur un agent unique generaliste (circuit-breaker / "un bon agent outille > N agents") qui, lui, reussit.
4. **Telemetrie** : l'orchestrateur renvoie un rapport `{success, strategy_used: list[str], retries, rerouted, fell_back: bool, steps}`.
5. **Prouve end-to-end** :
   - un `naive_run` (sans recuperation) sur le **meme** scenario echoue (`success=False`),
   - le `RobustOrchestrator` sur le **meme** scenario reussit (`success=True`), et `strategy_used` montre l'escalade reelle (`retry` puis, si besoin, `reroute`, puis `fallback`),
   - un scenario "tout casse en multi-agent" prouve que le **fallback single-agent** (`fell_back=True`) est ce qui sauve le run,
   - aucune boucle infinie : `max_retries` et un budget d'etapes global bornent l'execution (sinon `RuntimeError`).

### Criteres de reussite
- [ ] Les sous-agents ont des echecs **deterministes** (transitoire vs permanent), pas aleatoires
- [ ] La detection valide chaque sous-etape (schema + failure mode minimal) et declenche la recuperation
- [ ] La politique applique **retry → reroute → fallback single-agent** dans l'ordre, de facon configurable
- [ ] `naive_run` echoue et `RobustOrchestrator` reussit sur le MEME scenario (prouve par assertions)
- [ ] Un scenario prouve que `fell_back=True` (single-agent) est ce qui rend le run vert
- [ ] `strategy_used` reflete l'escalade reelle ; un budget global borne l'execution (pas de boucle infinie)
- [ ] Execution offline, deterministe, stdlib uniquement
