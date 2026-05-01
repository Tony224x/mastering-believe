# Exercices Faciles — Production & Observabilite (J12)

---

## Exercice 1 : Trace summary report

### Objectif
Comprendre comment extraire des metriques synthetiques d'un set de traces.

### Consigne
En partant de `02-code/12-production-observabilite.py` :

1. Ecris une fonction `trace_report(tracer: Tracer, trace_id: str) -> dict` qui retourne :
   - `trace_id`
   - `total_spans` : nombre de spans dans le trace
   - `total_duration_ms` : somme des durees de tous les spans
   - `total_cost_usd` : somme des couts
   - `total_tokens_in`, `total_tokens_out`
   - `errors` : liste des span names qui ont eu une erreur
   - `slowest_span` : dict du span le plus lent
   - `most_expensive_span` : dict du span le plus cher
2. Lance 3 traces avec des operations variees
3. Affiche les reports pour les 3 traces et verifie la coherence
4. Bonus : ajoute un test qui leve volontairement une erreur et verifie qu'elle apparait dans `errors`

### Criteres de reussite
- [ ] `trace_report` extrait tous les spans d'un trace_id donne
- [ ] Les totaux (duration, cost, tokens) sont corrects
- [ ] `slowest_span` et `most_expensive_span` sont identifies correctement
- [ ] Les erreurs sont listees
- [ ] Le rapport est printable et lisible

---

## Exercice 2 : Per-user daily budget

### Objectif
Passer d'un budget par requete a un budget cumule par utilisateur par jour.

### Consigne
1. Cree une classe `DailyUserBudget` :
   - `__init__(self, max_cost_per_user_per_day: float)`
   - `charge(self, user_id: str, model: str, tokens_in: int, tokens_out: int) -> float`
   - Maintient un dict `{user_id: {"date": <YYYY-MM-DD>, "cost": float}}`
   - Si on change de jour, le budget du user est reset
   - Si le charge depasse le budget, leve `BudgetExceeded(user_id, current_cost)`
2. Cree une methode `get_remaining(user_id: str) -> float` qui retourne le budget restant
3. Teste avec 3 users :
   - user_A : 5 charges qui restent sous le budget
   - user_B : charges qui depassent le budget au 4e appel
   - user_C : charge, simulate un changement de date (en modifiant manuellement le dict), charge a nouveau, verifier que le budget est reset
4. Affiche les charges et les verdicts

### Criteres de reussite
- [ ] `DailyUserBudget` isole correctement les budgets par user
- [ ] Le reset quotidien fonctionne (simule par changement de date)
- [ ] Le depassement leve `BudgetExceeded` avec le user_id dans le message
- [ ] `get_remaining` est coherent avec les charges effectuees
- [ ] Les 3 scenarios de test tournent sans erreur non geree

---

## Exercice 3 : Circuit breaker

### Objectif
Implementer un circuit breaker simple qui protege une dependance externe en difficulte.

### Consigne
1. Implemente une classe `CircuitBreaker` :
   - `__init__(self, failure_threshold: int = 3, reset_seconds: float = 2.0)`
   - Etats : `CLOSED` (normal), `OPEN` (on rejette tout), `HALF_OPEN` (on retente)
   - Methode `call(self, fn: Callable) -> Any`
   - Si CLOSED : appelle `fn`, si exception increment les failures, si threshold atteint passe a OPEN
   - Si OPEN : leve `CircuitOpen` sans appeler fn, tant que le temps ecoule < reset_seconds
   - Apres reset_seconds, passe a HALF_OPEN : le prochain appel teste la sante
   - Si le test HALF_OPEN reussit, retour a CLOSED (failures reset)
   - Si echec, retour a OPEN
2. Cree une fonction `flaky_service()` qui echoue N fois puis marche
3. Wrappe la dans un circuit breaker avec threshold=3, reset=0.5s
4. Fais 5 appels :
   - 1-2 : echecs normaux
   - 3 : echec, le breaker passe OPEN
   - 4 : bloque immediatement par le breaker
   - Apres 0.5s + 1 appel reussi : retour CLOSED
5. Affiche les etats a chaque etape

### Criteres de reussite
- [ ] Les 3 etats sont bien implementes
- [ ] Le breaker passe bien a OPEN apres N echecs
- [ ] Quand OPEN, les appels sont rejetes sans appeler fn
- [ ] Apres reset_seconds, HALF_OPEN permet un test
- [ ] Un test reussi en HALF_OPEN retourne a CLOSED
- [ ] Un test echoue en HALF_OPEN retourne a OPEN
