# Exercices Medium — Planning & Reasoning (J4)

---

## Exercice 1 : Plan-and-execute avec replanning sur info nouvelle

### Objectif
Aller au-dela du plan lineaire : implementer un executor qui detecte qu'une etape a invalide le plan, et qui declenche un **replanning** au lieu de continuer aveuglement.

### Consigne
En partant du pattern `plan_and_execute` du module 04 :

1. Cree un `MockLLM` (ou etends-le) capable de produire **deux plans** pour la meme tache : le plan initial et un plan revise.
2. Implemente une boucle `plan_execute_replan(llm, question, max_replans=2)` :
   - Le planner produit un plan initial (3-4 steps).
   - L'executor execute les steps un par un. Chaque resultat d'outil peut contenir un marqueur `REPLAN_NEEDED` (ex : un budget depasse, une donnee manquante, une contrainte violee).
   - Quand un step retourne `REPLAN_NEEDED`, l'executor s'arrete, passe l'historique au **replanner**, recoit un nouveau plan, et reprend l'execution depuis le debut du nouveau plan.
   - Compteur `replan_count` ; au-dela de `max_replans`, on stoppe avec un warning.
3. Scenario : "Reserve un vol Paris-Tokyo sous 800 EUR." Le step "search_flights" retourne d'abord `REPLAN_NEEDED: cheapest is 950 EUR` (budget depasse) → le replanner produit un plan qui inclut "search_flights with flexible dates".
4. La reponse finale doit citer le vol trouve apres replanning.

### Criteres de reussite
- [ ] Le plan initial et le plan revise sont differents et tous deux parses correctement
- [ ] L'executor detecte le marqueur `REPLAN_NEEDED` et declenche le replanner
- [ ] `replan_count` est correctement incremente (1 replan dans le scenario)
- [ ] La reponse finale provient du second plan (vol trouve avec dates flexibles)
- [ ] Le cas "plan initial reussit du premier coup" ne declenche aucun replan
- [ ] La boucle s'arrete proprement si `max_replans` est atteint (warning, pas de crash)

---

## Exercice 2 : Self-consistency avec extraction robuste et tie-breaking

### Objectif
Rendre le vote majoritaire de self-consistency fiable face a des reponses heterogenes (formats varies, ex-aequo, abstentions).

### Consigne
Le `extract_final_answer` naif du module casse des que les reponses ne suivent pas le format `Reponse : X`. Construis un systeme robuste :

1. `extract_answer(text) -> str | None` qui gere **plusieurs formats** :
   - `Reponse : 13`, `Answer: 13`, `Final: 13`, `= 13`, ou un dernier nombre isole.
   - Retourne `None` si aucune reponse exploitable (abstention).
2. `normalize_answer(ans) -> str` qui canonicalise : `"13."`, `" 13 "`, `"13.0"`, `"thirteen"` (mapping mots→chiffres pour 0-20) doivent toutes donner `"13"`.
3. `vote(responses, tie_break="confidence") -> dict` qui :
   - Ignore les abstentions (None).
   - Compte les reponses normalisees.
   - En cas d'egalite, applique une strategie de tie-break configurable : `"first"` (premiere apparue), `"confidence"` (la reponse dont les raisonnements sont en moyenne les plus longs — proxy d'effort), ou `"abstain"` (retourne None si pas de majorite nette).
   - Retourne `{"winner": ..., "votes": {...}, "abstentions": n, "confidence": ratio}`.
4. Teste sur un set de 7 reponses mockees melangeant formats, mots, et 1-2 abstentions, avec au moins un scenario d'ex-aequo.

### Criteres de reussite
- [ ] `extract_answer` gere au moins 4 formats differents et retourne None sur abstention
- [ ] `normalize_answer` unifie nombres, ponctuation, et mots (0-20)
- [ ] Les abstentions sont exclues du vote mais comptees dans le rapport
- [ ] Le tie-break `"confidence"` choisit la reponse aux raisonnements les plus longs
- [ ] Le tie-break `"abstain"` retourne None quand aucune majorite n'emerge
- [ ] `confidence` = votes_gagnant / votes_valides est correctement calcule

---

## Exercice 3 : Router de strategie de raisonnement (cost-aware)

### Objectif
Implementer le meta-niveau du module : un routeur qui **choisit la strategie de raisonnement** (direct, CoT, self-consistency, plan-and-execute) selon la nature de la question — et qui respecte un budget de tokens.

### Consigne
Cree un `ReasoningRouter` :

1. `classify(question) -> str` qui categorise la question via des heuristiques locales (pas de LLM pour juger) :
   - `"factual"` : courte, question fermee → strategie `direct`
   - `"arithmetic"` : contient des nombres + operation → strategie `cot`
   - `"critical"` : contient un marqueur d'enjeu ("medical", "legal", "budget", "irreversible") → strategie `self_consistency`
   - `"multi_step"` : contient "et", "puis", plusieurs sous-questions, ou > 25 mots → strategie `plan_execute`
2. `estimate_cost(strategy) -> int` : cout en appels LLM par strategie (direct=1, cot=1, self_consistency=5, plan_execute=~5).
3. `route(question, token_budget) -> dict` :
   - Choisit la strategie via `classify`.
   - Si le cout estime depasse le budget, **degrade** vers une strategie moins chere (self_consistency → cot, plan_execute → cot).
   - Retourne `{"category", "chosen_strategy", "fallback_applied", "estimated_calls"}`.
4. `run(question, token_budget)` qui execute reellement la strategie choisie (sur un MockLLM) et retourne la reponse + les metadonnees de routage.
5. Teste avec 4-5 questions couvrant chaque categorie, dont au moins une qui declenche une degradation par budget.

### Criteres de reussite
- [ ] Les 4 categories sont detectees correctement sur les questions de test
- [ ] Chaque categorie mappe vers la bonne strategie par defaut
- [ ] La degradation par budget fonctionne (ex : self_consistency → cot quand budget serre)
- [ ] `fallback_applied` est True uniquement quand une degradation a eu lieu
- [ ] `run` execute reellement la strategie et retourne une reponse coherente
- [ ] Le routeur est deterministe (memes entrees → meme routage)
