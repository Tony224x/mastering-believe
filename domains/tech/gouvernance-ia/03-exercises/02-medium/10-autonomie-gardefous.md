# Exercices Moyens — Autonomie, garde-fous & operations (J10)

---

## Exercice 1 : Budget avec fenetre glissante

### Objectif
Plafonner le **cumul** d'actions qu'un garde-fou par action ne voit pas, sur une fenetre de temps.

### Consigne
1. Ecris une classe `RollingBudget` avec `max_cost: float`, `max_actions: int`, `window_seconds: float`.
2. Methode `would_exceed(cost, now) -> bool` : purge les evenements plus vieux que `now - window_seconds`, puis dit si ajouter `cost` depasserait `max_cost` **ou** si le nombre d'actions depasserait `max_actions`.
3. Methode `record(cost, now)` : enregistre `(now, cost)`.
4. Important : `now` est **injecte** (pas `time.time()`) pour rendre le test deterministe.
5. Scenario de test : `max_cost=100`, `max_actions=3`, `window=60`. Enregistre 3 actions de 30 a `t=0`. Verifie qu'une 4e action est refusee a `t=10` (cap d'actions) mais **acceptee a `t=70`** (la fenetre a roule, les premieres ont expire).

### Criteres de reussite
- [ ] La 4e action a `t=10` declenche `would_exceed == True`.
- [ ] La meme action a `t=70` declenche `would_exceed == False` (fenetre glissante OK).
- [ ] Le cap de **cout** ET le cap de **nombre** sont tous deux testes (un cas qui casse chacun).
- [ ] Aucun appel a `time.time()` dans le test (horloge injectee).

---

## Exercice 2 : Autonomy gate combinant les 3 controles

### Objectif
Chainer kill-switch -> budget -> autonomie dans **un seul** point de decision, dans le bon ordre.

### Consigne
1. Reutilise `RollingBudget` (Ex. 1) et un kill-switch (dict `{agent_id: state}`).
2. Ecris `gate(agent_id, cost, impact, irreversible, switch, budget, now) -> (decision, reason)`.
3. Ordre d'evaluation **non negociable** :
   - si l'agent n'est pas `"active"` -> `("DENY", "kill-switch")` ;
   - sinon si `budget.would_exceed(cost, now)` -> `("ESCALATE", "budget")` ;
   - sinon calcule `score = impact (+2 si irreversible)` ; si `score >= 5` -> `("ESCALATE", "risk")` ;
   - sinon `budget.record(cost, now)` et `("ALLOW", "ok")`.
4. Teste les 4 chemins (un cas par branche) en gardant `now` fixe.

### Criteres de reussite
- [ ] Un agent `"killed"` renvoie `DENY` **meme si** le budget et le risque seraient OK (le kill-switch passe en premier).
- [ ] Un depassement de budget renvoie `ESCALATE`.
- [ ] Une action a haut risque (score >= 5) renvoie `ESCALATE`.
- [ ] Une action ALLOW est bien **comptabilisee** dans le budget (verifie via un `record` effectif).

---

## Exercice 3 : Machine a etats d'incident

### Objectif
Modeliser le cycle detect -> contain -> eradicate -> recover -> close avec transitions interdites.

### Consigne
1. Definis l'ordre des phases : `OPEN, CONTAINED, ERADICATED, RECOVERED, CLOSED`.
2. Ecris une classe `Incident` avec `.phase` (depart `OPEN`) et `.advance(note)` qui passe **seulement** a la phase suivante autorisee.
3. Si aucune transition n'est possible (deja `CLOSED`), `advance` doit lever `ValueError`.
4. Tente une transition interdite : forcer `phase = CLOSED` puis appeler `advance` -> doit lever `ValueError`.
5. Conserve un historique `(phase, note)` a chaque etape.

### Criteres de reussite
- [ ] Un incident parcourt `OPEN -> CONTAINED -> ERADICATED -> RECOVERED -> CLOSED` sans erreur.
- [ ] Tenter `advance` depuis `CLOSED` leve `ValueError`.
- [ ] On ne peut **pas** sauter une phase (ex. `OPEN` n'autorise que `CONTAINED`).
- [ ] L'historique contient une entree par transition, avec sa note.
