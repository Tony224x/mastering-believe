# Solutions — Module 06 : Calibration & Vérification de l'Information

## Exercice 1 — Score de Brier

| # | p | o | (p−o)² |
|---|---|---|--------|
| 1 | 0,80 | 1 | 0,04 |
| 2 | 0,70 | 0 | 0,49 |
| 3 | 0,55 | 1 | 0,2025 |
| 4 | 0,90 | 1 | 0,01 |
| 5 | 0,40 | 0 | 0,16 |
| 6 | 0,20 | 0 | 0,04 |
| 7 | 0,30 | 1 | 0,49 |
| 8 | 0,15 | 0 | 0,0225 |

**Somme** : 0,04 + 0,49 + 0,2025 + 0,01 + 0,16 + 0,04 + 0,49 + 0,0225 = **1,455**

**Score de Brier** : 1,455 / 8 = **0,182**

**Comparaison baseline** : 0,182 < 0,25 → meilleur que le hasard.

**Prédictions les plus coûteuses** : #2 et #7 (0,49 chacune). #2 révèle une sur-confiance (70 % alors que le colis était en retard). #7 révèle une sous-confiance (30 % alors qu'on était en retard). Ces deux zones méritent recalibration.

---

## Exercice 2 — Vérification SIFT

**Signaux d'alerte dans l'affirmation** :
1. Claim extraordinaire : +15 points de QI en 3 mois est un effet massif, jamais répliqué en méta-analyse.
2. Source vague : "Harvard" sans auteur ni département précis.
3. Mesure contestable : le QI est stable sur l'âge adulte ; des gains de 15 points en 3 mois dépassent de loin les effets connus.

**SIFT appliqué** :
- **S** : pause avant de partager.
- **I** : chercher "Harvard meditation IQ 2022 Nature Medicine" → aucun résultat pertinent. Chercher "[auteur] retraction".
- **F** : méta-analyses "meditation cognitive performance" → effets modestes (d ≈ 0,2-0,4 sur l'attention, pas le QI brut).
- **T** : remonter à l'article original → introuvable sur PubMed/Google Scholar → probablement inventé ou déformé.

**Protocole LLM** : 1) Titre exact entre guillemets sur Google Scholar. 2) DOI sur doi.org. 3) Si introuvable → halluciné. Ne pas citer sans vérification primaire.

---

## Exercice 3 — Journal de prévisions (exemple)

| Date | Question | p (%) | Date résol. | Outcome | (p−o)² | Note |
|------|----------|--------|-------------|---------|--------|------|
| J+0 | Pluie ce week-end ? | 65 | J+2 | — | — | Météo France base |
| J+0 | Réunion maintenue vendredi ? | 80 | J+4 | — | — | Historique annulations |
| J+0 | Finir ce module avant jeudi ? | 55 | J+3 | — | — | Taux complétion MOOCs |
| J+1 | Colis livré demain ? | 70 | J+2 | — | — | Transporteur historique |
| J+1 | Temps de trajet < 30 min ? | 60 | J+1 | — | — | Moyenne semaine passée |

Classe de référence obligatoire pour chaque prédiction. Scorer dès résolution. Réviser après 10+ prédictions.
