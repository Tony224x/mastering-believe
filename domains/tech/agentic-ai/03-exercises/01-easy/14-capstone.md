# Exercices Faciles — Capstone (J14)

---

## Exercice 1 : Ajouter un 4e specialiste (Critic)

### Objectif
Etendre le systeme multi-agent avec un role de "critique" qui relit le draft et propose des corrections avant validation.

### Consigne
En partant de `02-code/14-capstone.py` :

1. Ajoute une skill `critique(draft: str) -> tuple[dict, dict]` sur `MockLLM` qui :
   - Lit le draft
   - Retourne un dict avec `issues: list[str]` et `revised: str`
   - Les issues sont au moins 2 remarques heuristiques (ex: "missing numbers", "too short")
   - Le `revised` est le draft avec un suffixe "\n\n[Critic revision: N issues addressed]"
2. Cree une classe `CriticAgent` (comme `WriterAgent`) avec une methode `run(state)` decoree par `@traced("critic.run")`
3. Modifie `AcmeResearcher` pour inserer l'etape critic **apres** le writer et **avant** les guardrails output
4. Le state doit desormais avoir un champ `critic_report: dict`
5. Teste avec les 3 queries du demo : tu dois voir 4 spans par run (supervisor + researcher + analyzer + writer + critic) ... en realite 5
6. Verifie que le `final_report` est la version revised et non le draft original

### Criteres de reussite
- [ ] `critique` existe sur MockLLM et retourne issues + revised
- [ ] `CriticAgent.run` est trace et modifie le state
- [ ] `AcmeResearcher` appelle le critic dans le bon ordre
- [ ] Le state a un champ `critic_report`
- [ ] Le final_report contient le suffixe de revision
- [ ] Les 3 queries du demo montrent 5 spans (4 agents + critic)

---

## Exercice 2 : Remplacer le supervisor par un swarm

### Objectif
Comprendre comment changer de pattern multi-agent (supervisor -> swarm) sans casser les autres briques du systeme.

### Consigne
1. Cree une nouvelle classe `SwarmAcmeResearcher` qui remplace `AcmeResearcher`
2. Au lieu d'un supervisor qui plan, les agents sont chainees via des handoffs :
   - Entree : `researcher` (pas de supervisor)
   - `researcher` → `analyzer` → `writer`
   - Chaque agent a une methode `should_handoff(state) -> str | None` qui retourne le prochain agent ou None si c'est le dernier
3. Le loop principal itere : tant qu'on a un prochain agent, on l'execute
4. Guardrails input/output et HITL sont les memes
5. Le budget, le tracing et la gestion des erreurs fonctionnent comme avant
6. Teste sur les 3 queries du demo et verifie que le resultat est equivalent

### Criteres de reussite
- [ ] `SwarmAcmeResearcher` n'utilise pas de `SupervisorAgent`
- [ ] Les handoffs sont definis de maniere explicite sur chaque agent
- [ ] Le flow de bout en bout fonctionne sans supervisor
- [ ] Les traces montrent une difference : absence de span "supervisor.plan"
- [ ] Les 3 queries donnent un verdict et un rapport equivalent (keywords comparables)

---

## Exercice 3 : Ajouter un cas d'eval "hors-domaine"

### Objectif
Comprendre comment designer un cas de test qui verifie la robustesse d'un systeme face a des questions qu'il ne peut pas repondre.

### Consigne
1. Ajoute un 4e `EvalCase` dans `build_eval_cases()` :
   - id : `"out-of-domain"`
   - query : `"What is the capital of Mars?"`
   - expected_keywords : `""` (rien d'attendu)
   - expected_verdict : `"ok"` (le systeme doit gracieusement admettre qu'il ne sait pas)
2. Modifie `MockLLM.write` pour :
   - Si les findings sont vides ou pas pertinents (score < 1.0), retourner un rapport qui commence par "I do not have enough information..."
3. Modifie `run_eval` pour considerer un PASS si :
   - Le verdict est "ok"
   - ET le rapport contient soit "do not have" soit "not found" soit les keywords attendus (si non vides)
4. Verifie que :
   - Les 3 cas existants passent toujours
   - Le nouveau cas passe (l'agent admet son ignorance)
5. Bonus : affiche explicitement `"out_of_domain_handled"` pour ce cas

### Criteres de reussite
- [ ] Le 4e cas est ajoute
- [ ] `MockLLM.write` gere les findings vides ou faibles
- [ ] `run_eval` valide le cas out-of-domain
- [ ] Les 4 cas passent (pass rate = 100%)
- [ ] Le rapport affiche clairement que le systeme n'a pas assez d'infos
- [ ] Aucune hallucination sur le cas out-of-domain
