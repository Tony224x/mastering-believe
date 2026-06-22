# Exercice (medium) — Squelette de safety case et couverture

## Objectif

Passer de la *description* (agent card) à l'*argumentation* (safety case). Vous structurez un argument d'assurance en `claim → evidence → gaps`, vous le typez selon la taxonomie [Clymer et al., 2024], et vous calculez une métrique de maturité que pourrait lire un comité.

## Consigne

On reprend `refund-agent` (décide les remboursements < 200 €, peut déclencher un crédit). On veut un mini safety case.

1. Définissez une structure `Claim` (dataclass) avec : `statement` (str), `argument_type` (parmi `INABILITY`, `CONTROL`, `TRUSTWORTHINESS`, `DEFERENCE`), `evidence` (liste), `gaps` (liste).
2. Définissez une structure `SafetyCase` regroupant un `agent_id`, un `context`, et une liste de `Claim`.
3. Implémentez `coverage()` : la **fraction de claims ayant au moins une evidence**. Un claim sans evidence ne compte pas comme couvert.
4. Implémentez `open_gaps()` : la liste de tous les gaps déclarés, sous forme `(claim, gap)`.
5. Implémentez `weakest_arguments()` : trie les claims du **type d'argument le plus fragile au plus robuste**. Rappel de l'ordre de robustesse : `inability > control > trustworthiness > deference` (inability = le plus fort).
6. Construisez un safety case avec **au moins 3 claims**, dont **un volontairement sans evidence** (pour vérifier qu'il est compté comme non couvert) et **un de type `TRUSTWORTHINESS`** (pour vérifier qu'il remonte comme fragile).
7. Imprimez : la couverture en %, le nombre de gaps ouverts, et le claim au type d'argument le plus fragile.

## Critères de réussite

- [ ] `coverage()` renvoie bien `claims_avec_evidence / total_claims` (vérifié sur votre jeu : un claim sans evidence fait baisser le ratio).
- [ ] `weakest_arguments()[0]` renvoie un claim de type `TRUSTWORTHINESS` ou `DEFERENCE` quand il en existe un (jamais un `INABILITY` si un plus fragile existe).
- [ ] `open_gaps()` liste tous les gaps avec le claim associé.
- [ ] Le safety case contient ≥ 3 claims dont 1 sans evidence et 1 `TRUSTWORTHINESS`.
- [ ] Le script tourne en **stdlib pure** et imprime couverture %, nb de gaps, claim le plus fragile.
