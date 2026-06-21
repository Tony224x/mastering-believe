# J14 — Exercice facile : ecrire ta premiere regle de politique

## Objectif

Comprendre concretement ce qu'est une **regle de politique executable** (une fonction pure `(action, agent) -> verdict`) et la difference entre `allow`, `deny` et `oblige`. Tu vas ecrire **une seule regle** et l'evaluer sur quelques actions, sans moteur sophistique.

## Consigne

1. Definis deux petites structures de donnees en stdlib (un `dict` ou une `dataclass` suffit) :
   - un `agent` avec au minimum `id`, `scopes` (liste de chaines), `risk_tier` ;
   - une `action` avec au minimum `tool` (str) et `params` (dict).
2. Ecris une fonction `rule_scope(action, agent)` qui renvoie :
   - la chaine `"deny"` si l'action exige un scope que l'agent **n'a pas** (par exemple : l'outil `issue_refund` exige le scope `"refund:execute"`) ;
   - la chaine `"allow"` sinon.
3. Ecris une fonction `rule_refund_cap(action, agent)` qui renvoie `"oblige"` si `action.tool == "issue_refund"` **et** `params["amount"] > 1000`, sinon `"allow"`.
4. Cree une liste d'au moins **4 actions** de test (un petit remboursement, un gros remboursement, une action hors scope, une action neutre) et imprime, pour chacune, le verdict de chaque regle sous une forme lisible.
5. Ajoute un commentaire d'une ligne expliquant **pourquoi** `rule_refund_cap` renvoie `oblige` et non `deny` (indice : l'action est legitime, elle a juste besoin d'une validation humaine).

## Criteres de reussite

- [ ] Les deux regles sont des fonctions pures (aucun effet de bord, pas de variable globale modifiee).
- [ ] `rule_scope` renvoie bien `"deny"` pour une action hors scope et `"allow"` sinon.
- [ ] `rule_refund_cap` renvoie `"oblige"` uniquement pour `issue_refund` au-dela de 1000.
- [ ] La sortie imprimee montre clairement, pour chaque action de test, les verdicts obtenus.
- [ ] Le script s'execute sans erreur avec `python` (stdlib uniquement) et passe `python -m py_compile`.
- [ ] Le commentaire justifiant `oblige` vs `deny` est present.
