# Exercice (easy) — Une entree d'audit gouvernable

## Objectif
Savoir construire **une entree d'audit complete** pour une action d'agent : le quintuple `who / what / when / authorization / outcome`, et comprendre pourquoi le champ `authorization` est ce qui distingue un audit trail d'un simple log applicatif.

## Consigne
1. Definis une fonction `make_entry(agent_id, owner, action, params, scope, policy, decision, status)` qui retourne un `dict` representant une entree d'audit.
2. L'entree DOIT contenir au minimum ces cles : `who` (un sous-dict `{agent_id, owner}`), `what` (un sous-dict `{action, params}`), `when` (un timestamp UTC ISO-8601 genere automatiquement via `datetime.now(timezone.utc).isoformat()`), `authorization` (un sous-dict `{scope, policy, decision}`), `outcome` (un sous-dict `{status}`).
3. Ajoute une fonction `is_governable(entry)` qui renvoie `True` seulement si les cinq blocs sont presents ET si `authorization.scope`, `authorization.policy` et `authorization.decision` sont **non vides** (chaine non vide).
4. Dans un `if __name__ == "__main__":`, cree une entree pour un agent `finance-ops` (owner `a.dupont`) qui fait un `bank_transfer` de 40000 EUR, scope `payments:execute`, policy `auto<50k`, decision `ALLOW`, status `success`. Affiche l'entree puis le verdict de `is_governable`.
5. Cree une seconde entree **incomplete** (champ `authorization.scope` vide) et montre que `is_governable` renvoie `False`.

## Criteres de reussite
- [ ] `make_entry` produit un dict avec les 5 blocs `who / what / when / authorization / outcome`.
- [ ] Le timestamp est en UTC, format ISO-8601 (contient `T` et un offset/`+00:00`).
- [ ] `is_governable` renvoie `True` pour l'entree complete.
- [ ] `is_governable` renvoie `False` quand `scope` (ou `policy`/`decision`) est vide.
- [ ] Le script tourne en stdlib seule, sans erreur, et affiche les deux verdicts.
