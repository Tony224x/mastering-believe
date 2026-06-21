# Exercice (easy) — Identité & moindre privilège d'un agent

## Objectif

Te faire manipuler les deux briques de base de l'IAM d'agents : une **identité machine** distincte et un jeu de **scopes au moindre privilège**. À la fin, tu sais répondre à « cet agent a-t-il le droit de faire ça ? » par une simple vérification de scope.

## Consigne

On reprend l'agent de réconciliation de factures du cours. Sa mission : lire les factures et écrire dans le grand livre — **rien d'autre**.

1. Définis une structure d'identité minimale (dict ou `dataclass`) avec au moins trois champs : `agent_id` (unique et stable), `owner` (un humain nommé), `active` (booléen, `True` par défaut).
2. Définis pour cet agent l'ensemble de scopes **strictement nécessaires** à sa mission, sous forme de chaînes type OAuth (`domaine:action`). N'inclus **pas** de scope de paiement.
3. Écris une fonction `can(token_scopes, requested_scope) -> bool` qui renvoie `True` si et seulement si le scope demandé est présent dans le jeu accordé.
4. Teste les trois requêtes suivantes et affiche pour chacune ALLOW ou DENY :
   - `invoices:read`
   - `ledger:write`
   - `payments:execute` (doit être refusée — c'est le moindre privilège qui protège)
5. En commentaire, écris une phrase expliquant pourquoi le refus de `payments:execute` est une *bonne* nouvelle du point de vue gouvernance.

## Critères de réussite

- [ ] L'identité a un `agent_id` unique, un `owner` humain et un champ `active`.
- [ ] Le jeu de scopes ne contient **que** ce qui sert la mission (pas de scope de paiement, pas de wildcard `finance:*`).
- [ ] `can()` renvoie le bon booléen pour les trois requêtes.
- [ ] La sortie affiche clairement ALLOW pour les deux scopes légitimes et DENY pour `payments:execute`.
- [ ] Le commentaire relie le refus au principe de **moindre privilège** (un agent détourné qui n'a pas le scope est inoffensif sur cette action).
