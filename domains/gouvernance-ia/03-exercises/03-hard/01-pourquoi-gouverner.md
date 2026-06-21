# Exercice (difficile) — Un rapport board-ready avec score de risque

## Objectif

Produire un rapport de gouvernance exploitable par un comité de direction : non seulement *chiffrer* l'écart adoption / garde-fous, mais **prioriser** les agents par un score de risque défendable, et émettre la sortie dans **deux formats** (texte lisible + JSON pour outillage).

## Consigne

1. Repars de ta flotte enrichie. Ajoute à chaque agent un champ `tier` parmi `"minimal"`, `"limited"`, `"high"` (clin d'œil aux tiers de l'EU AI Act — pas besoin d'être exhaustif, juste cohérent).
2. Écris `risk_score(agent) -> int`, un score **borné de 0 à 100**, additionnant des pénalités explicites, par exemple :
   - +30 si l'agent est orphelin (sans owner),
   - +25 si l'agent agit sur le monde réel (`acts_on_world`),
   - +25 si `audit_enabled` est faux,
   - +20 si `tier == "high"` (ou +10 si `"limited"`).
   Le score doit être **plafonné à 100** (`min(100, ...)`).
3. Écris `build_report(agents) -> dict` qui retourne un dictionnaire structuré contenant au minimum : `total`, `coverage_pct`, `acting_orphans` (liste d'`agent_id`), et `agents_by_risk` (la liste des agents triée du score le plus élevé au plus bas, chaque entrée portant `agent_id` et `risk_score`).
4. Écris `render_text(report) -> str` (rendu humain) **et** sérialise le même `report` en JSON via le module standard `json` (`json.dumps(report, indent=2)`). Les deux doivent refléter exactement les mêmes données.
5. **Probe adversariale obligatoire** : ton code doit se comporter correctement sur une **flotte vide** (`coverage_pct == 100.0`, listes vides, aucune exception) **et** sur un agent dont `tools` est absent ou vide. Démontre-le dans le `__main__` (par ex. en appelant `build_report([])`).
6. Dans `if __name__ == "__main__":`, affiche le rapport texte, puis le rapport JSON, puis le résultat de la probe sur flotte vide.

## Critères de réussite

- [ ] `python -m py_compile` et `python 01-pourquoi-gouverner.py` passent sans erreur.
- [ ] `risk_score` est borné dans `[0, 100]` et reflète les pénalités décrites (orphelin, action, absence d'audit, tier).
- [ ] `agents_by_risk` est trié par score **décroissant**.
- [ ] La sortie produit **deux formats cohérents** : texte lisible **et** JSON valide (parsable par `json.loads`).
- [ ] La probe sur **flotte vide** ne lève aucune exception et renvoie `coverage_pct == 100.0`.
- [ ] Un agent sans clé/`tools` vide est géré sans erreur.
- [ ] Aucune dépendance externe (stdlib seule : `json`, `dataclasses` autorisés).
