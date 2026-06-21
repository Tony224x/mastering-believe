# Exercice (easy) — Normes & AIMS : classer obligatoire vs volontaire

## Objectif

Verifier que tu sais distinguer un referentiel **obligatoire** (loi) d'un referentiel **volontaire** (norme/cadre) et resumer la boucle PDCA d'un AIMS. C'est la base avant de construire un crosswalk.

## Consigne

1. Cree un fichier `solution.py` (dans ton `workspace/`) avec une liste de tuples `(nom_referentiel, mandatory: bool)` pour ces cinq referentiels : EU AI Act, RGPD, ISO/IEC 42001, NIST AI RMF, OECD AI Principles.
2. Ecris une fonction `is_mandatory(name: str) -> bool` qui renvoie `True` uniquement pour les referentiels **obligatoires** (les lois UE).
3. Ecris une fonction `pdca_steps() -> list[str]` qui renvoie les **4 etapes** de la boucle PDCA dans le bon ordre.
4. Dans un `if __name__ == "__main__":`, affiche pour chaque referentiel la mention `OBLIGATOIRE` ou `volontaire`, puis affiche la liste PDCA.
5. Verifie que `python solution.py` tourne sans erreur (stdlib seule).

## Criteres de reussite

- [ ] `is_mandatory("EU AI Act")` et `is_mandatory("RGPD")` renvoient `True`.
- [ ] `is_mandatory("ISO/IEC 42001")`, `is_mandatory("NIST AI RMF")`, `is_mandatory("OECD AI Principles")` renvoient `False`.
- [ ] `pdca_steps()` renvoie exactement `["Plan", "Do", "Check", "Act"]`.
- [ ] La sortie distingue clairement obligatoire et volontaire.
- [ ] Le script s'execute sans erreur en stdlib pure.
