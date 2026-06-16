# Exercices Medium — Orchestration comparee & failure modes (J18)

---

## Exercice 1 : Harnais de benchmark — comparer 3 topologies sur le meme lot de taches

### Objectif
Aller au-dela des demos isolees du module : construire un **harnais de benchmark** qui execute la **meme suite de taches** sur 3 topologies d'orchestration (single-agent outille, pipeline sequentiel, fan-out/fan-in parallele) et **mesure** cout (tokens/appels LLM), nombre d'etapes et succes. Tu dois ensuite **prouver le trade-off** annonce par le cours (sections 4.3, 4.5, 5) : le single-agent est moins cher sur une tache lineaire, le fan-out parallele est plus rapide (latence percue) sur des sous-taches independantes.

### Consigne
En reembarquant un `mock_llm` deterministe (compteur d'appels + tokens, pas de cle API) :

1. Implemente un compteur instrumente : chaque appel LLM enregistre `(topology, agent, tokens_in, tokens_out, latency)` ou `latency` est un cout simule fixe par appel (ex : 1.0 unite). Pas de `time.sleep`.
2. Modelise 3 topologies prenant une `task` decomposee en `n` sous-taches independantes :
   - `single_agent(subtasks)` : **un seul** appel LLM traitant toutes les sous-taches d'un coup (contexte unique). Latence = 1 appel.
   - `pipeline(subtasks)` : un agent par sous-tache, **sequentiel**, chaque agent reinjecte le contexte accumule des precedents (cout qui croit). Latence = somme des appels.
   - `parallel_fanout(subtasks)` : un agent par sous-tache **independant** (pas de contexte accumule) + 1 agent fan-in de synthese. Latence = `max(appels paralleles) + 1` (fan-in).
3. Ecris `benchmark(subtasks) -> dict` qui lance les 3 topologies sur le **meme** lot et renvoie, par topologie : `{llm_calls, tokens_total, steps, perceived_latency, success}`.
4. **Prouve par assertions** :
   - sur une tache lineaire/dependante (le pipeline est requis), `single_agent` consomme **moins** de tokens que `pipeline` (pas d'accumulation de contexte),
   - sur des sous-taches **independantes**, `parallel_fanout` a une `perceived_latency` **strictement inferieure** a celle du `pipeline` (parallelisme),
   - les 3 topologies aboutissent a `success=True` (le benchmark mesure des trade-offs, pas des echecs).

### Criteres de reussite
- [ ] Le compteur instrumente enregistre tokens **et** latence simulee par appel (sans `time.sleep`)
- [ ] Les 3 topologies tournent sur la **meme** suite de sous-taches
- [ ] `single_agent` consomme moins de tokens que `pipeline` (prouve par assertion)
- [ ] `parallel_fanout` a une latence percue strictement inferieure au `pipeline` (prouve par assertion)
- [ ] `benchmark()` renvoie un dict structure par topologie (calls, tokens, steps, latency, success)
- [ ] Tout tourne offline, deterministe, sans cle API ni dependance

---

## Exercice 2 : Framework d'injection de fautes — handoff perdu & sortie malformee

### Objectif
Implementer un **framework d'injection de fautes** qui injecte un failure mode precis dans un pipeline A → B → C et **prouve qu'un detecteur le repere**. Tu cibles deux failure modes du cours : la **perte d'information au handoff** (payload tronque/vide, section 4.1) et la **sortie malformee** d'un agent (schema casse). Sans validation inter-agents, l'erreur se propage silencieusement (section 4.1) ; avec, elle est attrapee a la frontiere.

### Consigne
1. Modelise un pipeline de 3 agents `agent(payload: dict) -> dict` ou chaque agent **doit** produire un payload respectant un schema attendu (ex : `researcher` produit `{"sources": [...]}`, `writer` produit `{"draft": str}`, `reviewer` produit `{"verdict": str}`).
2. Ecris un `SchemaValidator` (ou des fonctions de validation par etape) qui, **entre** chaque agent, verifie que la sortie contient les cles requises et les bons types ; en cas d'echec il leve `HandoffValidationError(stage, missing_or_bad)`.
3. Cree un `FaultInjector` parametrable qui peut activer au moins **2 fautes** :
   - `dropped_handoff` : vide ou supprime une cle du payload transmis a l'agent suivant (perte d'info),
   - `malformed_output` : fait retourner par un agent un payload de mauvais type (ex : `draft` = `None` ou une liste au lieu d'un `str`).
4. Ecris `run_pipeline(agents, validator, injector=None) -> dict` qui execute le pipeline en validant a chaque frontiere et renvoie soit `{"ok": True, "result": ...}` soit `{"ok": False, "error": <stage>, "detail": ...}`.
5. **Prouve par assertions** :
   - sans injection, le pipeline finit `ok=True`,
   - avec `dropped_handoff` injecte au handoff `writer→reviewer`, le detecteur **flag** l'erreur a la bonne etape (et pas plus loin),
   - avec `malformed_output` injecte sur `writer`, le detecteur **flag** une erreur de type a l'etape `writer`,
   - **sans** validateur (run "naif"), la faute **n'est pas** detectee a la frontiere et se propage (ex : `reviewer` recoit un draft vide / corrompu) — tu montres le contraste.

### Criteres de reussite
- [ ] Le pipeline a 3 agents avec un schema de sortie attendu par etape
- [ ] `HandoffValidationError` indique l'etape fautive et la cle/type en cause
- [ ] `FaultInjector` peut injecter `dropped_handoff` ET `malformed_output`
- [ ] Le run sans faute finit `ok=True` ; chaque faute injectee est flaggee a la BONNE etape (assertions)
- [ ] Le contraste "run naif sans validateur" montre la propagation silencieuse de la faute
- [ ] Execution offline, deterministe, sans dependance

---

## Exercice 3 : Simulateur de cascade d'erreurs + circuit-breaker qui la contient

### Objectif
Reproduire la **propagation d'erreurs en cascade** (section 4.1 du cours) dans une chaine d'agents A → B → C → D, ou une erreur a la source est **amplifiee** a chaque saut, puis cabler un **circuit-breaker** (section 6, "fallback to single agent") qui **detecte** la degradation et **coupe** la chaine avant qu'elle ne corrompe la sortie finale.

### Consigne
1. Modelise une chaine de 4 agents ou chaque agent calcule un `confidence` (score 0..1) sur la sortie recue. Un agent qui recoit une entree de faible confiance **propage et degrade** : `confidence_sortie = confidence_entree * facteur` (`facteur < 1` quand l'entree est suspecte), simulant l'amplification d'erreur.
2. Injecte une **erreur a la source** (l'agent A produit une sortie de confiance faible, ex : 0.4) et montre, sans garde-fou, que la confiance finale s'effondre (`< seuil`) tout en etant **presentee comme valide** par le dernier agent (la fameuse legitimation cross-agent, section 4.4).
3. Implemente un `CircuitBreaker` :
   - un seuil de confiance `min_confidence`,
   - apres chaque saut, il **inspecte** la confiance ; si elle passe sous le seuil, il **ouvre le circuit** (`open=True`), stoppe la chaine et renvoie un resultat marque `degraded=True` au lieu d'une sortie faussement validee,
   - il expose `state` (`closed`/`open`) et la `stage` ou il a coupe.
4. Compare deux runs sur la **meme** erreur injectee : (a) **naif** (sans breaker) → la chaine va jusqu'au bout et renvoie une sortie corrompue presentee comme valide ; (b) **protege** (avec breaker) → la chaine est coupee tot, `degraded=True`, et la sortie finale n'est PAS faussement validee.
5. **Prouve par assertions** : run naif → confiance finale sous le seuil mais marquee "validee" ; run protege → circuit `open`, coupe a une etape `< len(chaine)`, resultat `degraded=True`.

### Criteres de reussite
- [ ] La chaine de 4 agents propage et degrade la confiance a chaque saut
- [ ] Le run naif aboutit a une sortie de faible confiance presentee comme "validee" (legitimation cross-agent)
- [ ] `CircuitBreaker` ouvre le circuit des que la confiance passe sous le seuil
- [ ] Le run protege coupe la chaine AVANT la fin et renvoie `degraded=True`
- [ ] La comparaison naif vs protege est prouvee par assertions (etape de coupure, etat du circuit)
- [ ] Execution offline, deterministe, sans dependance
