# Exercice (moyen) — Mesurer la couverture de gouvernance

## Objectif

Aller au-delà du simple comptage : calculer un **taux de couverture de gouvernance** et isoler le quadrant le plus dangereux — les agents qui *agissent sur le monde réel* tout en étant *sans propriétaire*.

## Consigne

1. Repars de ta flotte d'agents (réutilise ou enrichis celle de l'exercice facile). Chaque agent a `agent_id`, `owner`, `tools` (liste) et un booléen `audit_enabled`.
2. Définis un ensemble `SIDE_EFFECT_TOOLS` contenant au moins : `send_payment`, `send_email`, `delete_record`, `issue_credit`, `post_social`.
3. Écris `acts_on_world(agent) -> bool` : vrai si l'agent dispose d'au moins un outil à effet de bord (présent dans `SIDE_EFFECT_TOOLS`).
4. Écris `is_governed(agent) -> bool` : vrai si l'agent a **à la fois** un owner non vide **et** `audit_enabled == True` (le seuil minimal du Jour 1 : redevable *et* prouvable).
5. Écris `governance_coverage(agents) -> float` : le pourcentage d'agents gouvernés. **Gère le cas de la flotte vide** (renvoie `100.0`, et surtout évite la division par zéro).
6. Écris `acting_orphans(agents) -> list` : la liste des agents qui *agissent* mais n'ont *pas* d'owner.
7. Dans `if __name__ == "__main__":`, affiche : le taux de couverture (arrondi à 1 décimale), le nombre d'agents « acting orphans », et leurs `agent_id`. Assure-toi qu'au moins un agent de ta flotte tombe dans ce quadrant.

## Critères de réussite

- [ ] `python -m py_compile` et `python 01-pourquoi-gouverner.py` passent sans erreur.
- [ ] `acts_on_world` détecte correctement les outils à effet de bord.
- [ ] `is_governed` exige **owner non vide ET audit activé** (les deux conditions).
- [ ] `governance_coverage` renvoie un pourcentage correct et ne plante pas sur une flotte vide (renvoie `100.0`).
- [ ] `acting_orphans` isole bien les agents qui agissent sans owner ; au moins un agent y figure.
- [ ] La sortie affiche le taux de couverture, le compte et les `agent_id` du quadrant à risque.
