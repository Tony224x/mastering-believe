# Exercice (medium) — Valider base légale & minimisation

## Objectif

Étendre le détecteur easy en un **validateur de licéité** : pour un agent qui traite des données personnelles, vérifier qu'une base légale (Art. 6) est déclarée et reconnue, exiger un test de mise en balance si la base est l'intérêt légitime (EDPB Opinion 28/2024), et faire respecter la **minimisation** (Art. 5(1)(c)) en signalant tout champ personnel non justifié par la finalité.

## Consigne

1. Repars de ton code easy (ou du `02-code/06-data-governance-rgpd.py` du module).
2. Définis un dictionnaire `LEGAL_BASES` avec au moins les 6 codes de l'Art. 6(1) (`consent`, `contract`, `legal_obligation`, `vital_interests`, `public_task`, `legitimate_interest`).
3. Écris `validate(declared_fields, fields_needed, legal_basis, has_li_balancing_test) -> dict` qui retourne une liste de `blockers` (chaînes) — vide si tout est conforme. Couvre ces règles :
   - **base légale manquante ou inconnue** → blocker « processing UNLAWFUL » ;
   - **`legitimate_interest` sans `has_li_balancing_test`** → blocker (test exigé, EDPB 28/2024) ;
   - **minimisation** : tout champ *personnel ou sensible* déclaré mais absent de `fields_needed` → blocker citant les champs en trop.
4. Ajoute une propriété/booléen `compliant = (len(blockers) == 0)`.
5. Teste : (a) un agent avec un champ `health` non nécessaire → doit être non conforme ; (b) un agent « contrat » à champs minimaux → conforme.

## Criteres de reussite

- [ ] Le script tourne avec `python <fichier>` sans erreur (stdlib seule).
- [ ] Une base légale `None` ou inconnue produit un blocker et `compliant == False`.
- [ ] `legitimate_interest` sans test de mise en balance produit un blocker.
- [ ] Un champ personnel non listé dans `fields_needed` déclenche un blocker de minimisation nommant ce champ.
- [ ] Un agent « contrat » avec uniquement des champs nécessaires renvoie `blockers == []` et `compliant == True`.
