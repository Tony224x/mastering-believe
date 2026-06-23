# Exercices Hard — Agent Systems Architecture

---

## Exercice 1 : Concevoir un agent autonome de resolution d'incidents (SRE copilot)

### Objectif
Concevoir un systeme d'agent a fort enjeu : un copilote SRE qui aide a diagnostiquer et remediar des incidents prod. Latence, securite des actions, observability, arret. Design d'entretien senior.

### Consigne
Tu concois un agent qui assiste l'astreinte pendant un incident prod.

**Contexte & contraintes :**
- L'agent recoit une alerte (ex: "p99 latency > 2s sur le service checkout")
- Il peut : lire les dashboards (metrics), lire les logs, lire les traces, proposer une hypothese, et executer des actions de remediation (restart pod, rollback deploy, scale up) — **mais les actions destructives requierent une validation humaine**
- Tools : ~25 tools repartis en modules (metrics.*, logs.*, traces.*, deploy.*, k8s.*)
- Enjeu : une mauvaise action peut aggraver l'incident ; une bonne action rapide reduit le MTTR
- L'astreinte doit pouvoir suivre le raisonnement en temps reel et reprendre la main

**Livre :**

1. **Pattern d'orchestration** :
   - Single-agent ou multi-agent ? Justifie pour CE cas (domaine, enjeu, latence).
   - Si multi-agent, quels specialistes ? Si single, comment geres-tu les 25 tools ?

2. **Tool routing** :
   - 25 tools = trop pour un seul step. Quelle strategie (filtering, namespacing, registry, MCP) ?
   - Comment l'agent choisit-il le bon module puis le bon tool ?

3. **Securite des actions (human-in-the-loop)** :
   - Comment classes-tu les actions (read-only / reversible / destructive) ?
   - Concois le mecanisme d'approbation humaine pour les actions destructives.
   - Comment empeches-tu l'agent d'executer une action destructive sans go ?

4. **Boucle, arret et budget** :
   - Quelles conditions d'arret ? (hypothese validee, action appliquee + verifiee, budget, human takeover)
   - Comment evites-tu qu'il boucle (re-diagnostique en rond) pendant un incident critique ?
   - Que se passe-t-il si l'agent est incertain ? (ne pas agir au hasard)

5. **Memoire & contexte** :
   - Quel contexte injecter (incident courant, runbooks passes, post-mortems) ?
   - Comment reutilises-tu les incidents passes (memoire long-terme) ?

6. **Observability & failure modes** :
   - Que logges-tu pour rendre chaque decision auditable a posteriori ?
   - Failure modes : l'agent propose une remediation qui aggrave, l'agent reste bloque, un tool ment (donnees stale). Comment tu geres chacun ?

### Criteres de reussite
- [ ] Le choix single vs multi est justifie : single-agent avec tool routing est defendable (domaine SRE coherent, latence critique) ; un superviseur leger se defend aussi
- [ ] Les 25 tools sont geres par namespacing/modules (metrics.*, logs.*, deploy.*...) + filtering par etape
- [ ] Les actions sont classees (read-only / reversible / destructive) avec approbation humaine OBLIGATOIRE sur destructive
- [ ] Le garde-fou empeche techniquement l'execution destructive sans approbation (pas juste "le prompt le demande")
- [ ] Les conditions d'arret incluent hypothese validee, action verifiee, budget, et human takeover
- [ ] L'incertitude mene a l'escalade humaine (jamais d'action destructive au hasard)
- [ ] L'observability rend chaque step auditable (decision + reasoning + tool + resultat) ; les 3 failure modes sont traites (rollback de remediation, detection de blocage, defiance vis-a-vis de donnees stale)

---

## Exercice 2 : Post-mortem — L'agent multi-agent qui a brule le budget et boucle

### Objectif
Analyser un incident d'un systeme multi-agent (boucle inter-agents + explosion de contexte + action erronee), reconstituer la cascade, et concevoir les garde-fous.

### Consigne
Voici le rapport d'incident (resume) d'un assistant de recherche multi-agent.

**Contexte** : Un systeme "deep research" avec un superviseur et 4 specialistes : `planner`, `searcher`, `reader`, `writer`. Le superviseur dispatche, agrege, et decide quand c'est fini. Pas de budget global cable. Le handoff entre agents transmet juste "continue with the research". La condition d'arret est : "le writer a produit un rapport". Pas de detecteur de no-progress. Le `searcher` peut appeler un outil de recherche web payant ($0.01/recherche).

**Timeline de l'incident :**

| Heure | Evenement |
|---|---|
| 14:00 | Un user lance une recherche sur un sujet de niche tres peu documente. |
| 14:00 | Le `planner` decompose en 12 sous-questions. |
| 14:01 | Le `searcher` cherche, mais trouve peu de sources fiables (sujet de niche). |
| 14:01 | Le `reader` lit les rares sources, juge l'info insuffisante, renvoie au superviseur. |
| 14:02 | Le superviseur relance le `searcher` avec un handoff "continue with the research" (aucun contexte sur ce qui a deja ete cherche). |
| 14:02 | Le `searcher` refait les MEMES recherches (il ne sait pas ce qui a deja ete fait). |
| 14:02 - 16:00 | Boucle : searcher -> reader -> superviseur -> searcher. ~2000 recherches web. Le contexte agrege gonfle (toutes les sources empilees). |
| 16:00 | Le superviseur, voyant un contexte enorme, n'arrive plus a planifier (context overflow), produit des plans incoherents. |
| 16:00 | Le `writer` finit par produire un rapport bourre d'infos contradictoires et de sources inventees. |
| 16:05 | Le rapport est livre au user, qui le signale comme faux. |
| 16:30 | Post-mortem : $20 de recherche web + cout LLM eleve sur UN seul run, rapport non fiable. |

**Questions :**

1. **Root cause analysis** :
   - Reconstitue la cascade complete.
   - Pour chaque maillon, le garde-fou manquant.
   - Classe : handoff/architecture, condition d'arret, memoire/contexte, qualite.

2. **Le handoff defaillant** :
   - Pourquoi "continue with the research" cause-t-il les recherches dupliquees ?
   - Reecris un handoff message correct pour superviseur -> searcher (les 5 champs), incluant ce qui a deja ete cherche.

3. **Boucle & arret** :
   - Pourquoi la condition "le writer a produit un rapport" est-elle insuffisante ?
   - Concois 3 conditions d'arret supplementaires, dont un detecteur de no-progress.
   - Comment bornes-tu le budget (recherches, steps, tokens, temps) et le propages dans les handoffs ?

4. **Context overflow** :
   - Pourquoi le superviseur a-t-il fini par produire des plans incoherents ?
   - Quelle strategie de memoire aurait evite l'explosion du contexte agrege ?

5. **Qualite & resilience** :
   - Pourquoi le `writer` a-t-il invente des sources ? Quel garde-fou (vu au J10) manque ?
   - Concois un "completeness check" : quand l'agent doit-il dire "je n'ai pas assez d'info fiable" plutot que d'inventer ?
   - Propose un runbook de 7 etapes pour un run d'agent qui part en vrille.

### Criteres de reussite
- [ ] La cascade complete est reconstituee : sujet de niche -> handoff sans contexte -> recherches dupliquees -> boucle -> contexte agrege explose -> superviseur incoherent -> writer hallucine -> rapport faux
- [ ] Le handoff "continue" est identifie comme cause des doublons ; un handoff correct (5 champs) listant les recherches deja faites est fourni
- [ ] La condition d'arret "rapport produit" est jugee insuffisante ; 3 conditions ajoutees dont un no-progress detector
- [ ] Le budget (recherches/steps/tokens/temps) est borne ET propage/decremente via les handoffs
- [ ] Le context overflow est explique (toutes les sources empilees) ; la mitigation est la summarization / retrieve-on-demand (pas tout dans le prompt)
- [ ] L'hallucination du writer est reliee a l'absence de groundedness check ; un completeness check declenche un "info insuffisante" plutot qu'une invention
- [ ] Le runbook est actionable (couper le run, inspecter la boucle, ajouter no-progress + budget, etc.)
