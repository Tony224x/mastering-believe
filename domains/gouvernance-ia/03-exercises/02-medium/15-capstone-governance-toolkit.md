# J15 — Exercice moyen : brancher ENFORCE + LOG sur le pipeline

## Objectif

Etendre le squelette de l'exercice facile en ajoutant deux etages du toolkit : un **policy engine** (PDP/PEP) qui decide allow/deny/oblige sur les actions tentees par la flotte, et un **audit trail tamper-evident** qui journalise chaque decision. On obtient le cœur runnable du capstone : `ingest → enforce → log`.

## Consigne

1. Reprends `GovernedAgent`, `ingest`, `orphans`, `coverage` de l'exercice facile.
2. **Policy engine** — definis un `Verdict` (IntEnum `ALLOW < OBLIGE < DENY`), une dataclass `Action(tool, params, required_scope)`, et une dataclass `Decision(verdict, rule, reason)`. Ecris au moins **deux regles** pures `(action, agent, ctx) -> Decision | None` :
   - `rule_scope` : si `required_scope` n'est pas dans `agent.scopes` → `DENY`.
   - `rule_budget` : si `params["amount"] > ctx["auto_amount_limit"]` → `OBLIGE` (validation humaine).
3. Ecris `decide(rules, action, agent, ctx)` qui collecte les regles qui se declenchent et renvoie le verdict **le plus severe** (precedence de surete `deny > oblige > allow`). Si aucune ne se declenche → `ALLOW` par defaut.
4. **Audit trail** — ecris une classe `AuditTrail` chainee par hash (`hashlib.sha256`) : `record(...)` ajoute une entree (agent, tool, verdict, executed) chainee au hash precedent ; `verify()` recompute la chaine et renvoie `(ok, index_casse)`. Sers-toi de `json.dumps(..., sort_keys=True)` pour une serialisation deterministe.
5. Ecris une boucle `enforce_all(agents, attempts, ctx)` qui, pour chaque action tentee, decide, applique (executed True seulement si ALLOW, ou OBLIGE avec `human_approves`), et **journalise** la decision. Renvoie le nombre d'actions bloquees.
6. Smoke test : lance sur une flotte + une liste d'actions melant allow / deny (scope manquant) / oblige (montant eleve). Affiche le journal et verifie `verify() == (True, None)` sur la chaine intacte.

## Criteres de reussite

- [ ] `decide` renvoie `DENY` quand une regle deny et une regle oblige se declenchent simultanement (precedence de surete respectee).
- [ ] Une action dont le scope est manquant est bloquee (`DENY`, `rule_scope`) et `executed=False`.
- [ ] Une action au montant superieur au plafond renvoie `OBLIGE` ; elle ne s'execute que si `human_approves=True`.
- [ ] `AuditTrail.verify()` renvoie `(True, None)` sur une chaine intacte.
- [ ] Chaque action tentee (y compris les refus) figure dans le journal.
- [ ] Le script s'execute sans erreur (`python`, stdlib uniquement) et passe `python -m py_compile`.
