# Exercices Difficiles — Autonomie, garde-fous & operations (J10)

---

## Exercice 1 : Soft cap avec escalade ET decremente apres approbation

### Objectif
Aller au-dela du simple ALLOW/DENY : modeliser une file d'approbation humaine (HITL) qui, une fois l'action approuvee, la **comptabilise** dans le budget et la fait passer.

### Consigne
1. Etends l'autonomy gate du jour : quand une action est `ESCALATE`, ajoute-la a une file `pending_approvals` (liste de dicts : `agent_id`, `action`, `cost`, `risk`).
2. Ecris `approve(gate, index, now)` : un humain approuve l'action a l'index donne. Elle est retiree de la file ; si le budget le permet a `now`, elle est `record`-ee et devient `ALLOW` ; sinon elle reste refusee (`DENY`, "budget still exceeded").
3. Ecris `reject(gate, index, reason)` : retire l'action de la file et la journalise en `DENY`.
4. Scenario : declenche une escalade (gros montant), approuve-la, verifie qu'elle est passee ET comptee dans le budget. Declenche une 2e escalade, **rejette**-la, verifie qu'elle n'est PAS comptee.

### Criteres de reussite
- [ ] Une action escaladee se retrouve dans `pending_approvals`.
- [ ] `approve` la fait passer en `ALLOW` **et** l'enregistre dans le budget (le cumul augmente).
- [ ] `reject` la sort de la file sans toucher le budget.
- [ ] Le journal d'audit contient une entree distincte pour l'escalade, l'approbation et le rejet.

---

## Exercice 2 : Reponse a incident orchestree (contain coupe l'agent)

### Objectif
Coupler la machine a etats d'incident avec le kill-switch et le budget : la phase `contain` doit **reellement** stopper l'agent et la phase `recover` le redemarrer en mode degrade (HITL force).

### Consigne
1. Reutilise l'autonomy gate, le kill-switch et la machine a etats d'incident.
2. Ecris un orchestrateur `handle_incident(gate, switch, agent_id, trigger)` qui :
   - **detect** : cree l'incident (`OPEN`) ;
   - **contain** : passe l'agent a `"killed"` dans le switch, avance l'incident en `CONTAINED`, et **verifie** qu'une action est desormais `DENY` ;
   - **eradicate** : avance en `ERADICATED` (simule un patch) ;
   - **recover** : remet l'agent `"active"` MAIS active un drapeau `hitl_forced[agent_id] = True` ; le gate, si ce drapeau est vrai, force `ESCALATE` sur toute action sensible (cost > 0) ; avance en `RECOVERED` ;
   - **close** : avance en `CLOSED`.
3. Verifie le comportement a chaque phase via une action de test.

### Criteres de reussite
- [ ] Pendant `CONTAINED`, toute action est `DENY` (agent killed).
- [ ] Apres `recover`, une action sensible est `ESCALATE` (HITL force), pas `ALLOW`.
- [ ] L'incident atteint `CLOSED` en passant par toutes les phases dans l'ordre.
- [ ] L'orchestrateur ne saute aucune phase et journalise chaque transition.

---

## Exercice 3 : Decommission gate (refuser tant que la checklist n'est pas verte)

### Objectif
Garantir qu'un agent ne peut atteindre `DECOMMISSIONED` que si tous les acces sont revoques, l'audit archive et l'owner clos — sinon, identite zombie.

### Consigne
1. Modelise le cycle de vie : `PROPOSED, APPROVED, ACTIVE, SUSPENDED, DECOMMISSIONED`.
2. Ecris `try_decommission(agent: dict) -> (bool, list[str])` ou `agent` porte les flags `revoked_access`, `audit_archived`, `owner_closed` et un champ `state`.
3. La transition vers `DECOMMISSIONED` n'est autorisee **que** depuis `SUSPENDED` ou `ACTIVE`, ET seulement si les 3 flags sont vrais. Sinon, retourne `(False, missing)` ou `missing` liste ce qui bloque.
4. Cas adversariaux a tester : (a) agent avec `revoked_access=False` -> refus + raison ; (b) agent en etat `PROPOSED` -> refus (mauvais etat de depart) ; (c) agent complet en `SUSPENDED` -> succes, `state` passe a `DECOMMISSIONED`.

### Criteres de reussite
- [ ] Un acces non revoque bloque la decommission, avec la raison "zombie".
- [ ] Une mauvaise transition d'etat (depuis `PROPOSED`) est refusee.
- [ ] Un agent complet en `SUSPENDED` est decommissionne et change d'etat.
- [ ] La fonction retourne toujours la liste des manques (jamais une exception non geree).
