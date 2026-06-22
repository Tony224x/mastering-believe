# Solutions — Module 03 : Spaced repetition

> Pour l'exercice 1, vérifiez vos calculs avec le script Python. Pour les exercices 2 et 3, ce corrigé donne des exemples de référence.

---

## Exercice 1 — Simulation SM-2 : tableau de référence

**Paramètres initiaux :** EF = 2.5, intervalle = 0, repetitions = 0
**Notes appliquées :** 4, 3, 5, 2, 4, 4

| Session | Note | EF avant | Calcul EF | EF après | Intv. préc. | Prochain intv. | Jours cumulés |
|---------|------|----------|-----------|----------|-------------|----------------|---------------|
| 1 | 4 | 2.50 | 2.50 + (0.1 − 1×(0.08+1×0.02)) = 2.50 + 0.00 = **2.50** | 2.50 | 0 | **1** (1re réussie) | J+1 |
| 2 | 3 | 2.50 | 2.50 + (0.1 − 2×(0.08+2×0.02)) = 2.50 − 0.14 = **2.36** | 2.36 | 1 | **6** (2e réussie) | J+7 |
| 3 | 5 | 2.36 | 2.36 + (0.1 − 0×…) = 2.36 + 0.10 = **2.46** | 2.46 | 6 | round(6×2.46) = **15** | J+22 |
| 4 | 2 | 2.46 | 2.46 + (0.1 − 3×(0.08+3×0.02)) = 2.46 − 0.32 = **2.14** | 2.14 | 15 | **1** (note < 3 → reset) | J+23 |
| 5 | 4 | 2.14 | 2.14 + 0.00 = **2.14** | 2.14 | 1 | **1** (1re réussie post-reset) | J+24 |
| 6 | 4 | 2.14 | 2.14 + 0.00 = **2.14** | 2.14 | 1 | **6** (2e réussie) | J+30 |

> **Détail important sessions 5 et 6 :** après le reset (note < 3 en session 4), le compteur de répétitions reussies repasse à 0. La session 5 est donc la *1re révision réussie* → intervalle fixe = 1 jour. La session 6 est la *2e révision réussie* → intervalle fixe = 6 jours. Ce n'est qu'à partir de la 3e révision réussie consécutive qu'on applique `round(intervalle × EF)`. Vérification : lancer le script Python donne exactement ces valeurs.

**Développement formule EF session 2 (note = 3) :**
```
EF = 2.50 + (0.1 - (5-3) × (0.08 + (5-3) × 0.02))
EF = 2.50 + (0.1 - 2 × (0.08 + 0.04))
EF = 2.50 + (0.1 - 2 × 0.12)
EF = 2.50 + (0.1 - 0.24) = 2.50 - 0.14 = 2.36
```

**Développement formule EF session 4 (note = 2) :**
```
EF = 2.46 + (0.1 - (5-2) × (0.08 + (5-2) × 0.02))
EF = 2.46 + (0.1 - 3 × (0.08 + 0.06))
EF = 2.46 + (0.1 - 3 × 0.14)
EF = 2.46 + (0.1 - 0.42) = 2.46 - 0.32 = 2.14
```

**Pourquoi la note < 3 remet à 1 et non à 0 :** L'intervalle 0 n'aurait pas de sens (révision le jour même en continu). SM-2 impose un minimum de 1 jour car l'espacement, même court, active mieux la consolidation qu'une révision immédiate. Le but n'est pas de punir l'échec mais de relancer le cycle d'espacement à partir du palier le plus bas.

---

## Exercice 2 — Calendrier de révision : exemple de référence

**Sujet exemple :** les 5 flash-cards du Module 01 | **Test final :** J+30

**Règle Cepeda (2008) :** gap optimal ≈ 10-20 % du délai → ici 3 à 6 jours pour le premier intervalle.

| Révision | Date | Intervalle depuis précédent | Technique | Durée |
|----------|------|-----------------------------|-----------|-------|
| 1 | J+1 | — | Blank-page recall | 10 min |
| 2 | J+4 | 3 jours | Flashcards (toutes les cartes) | 10 min |
| 3 | J+12 | 8 jours | Flashcards + auto-questionnement sur les lacunes | 15 min |
| 4 | J+28 | 16 jours | Retrieval complet (blank-page + flashcards) | 20 min |
| Test | J+30 | 2 jours | — | — |

**Règle "révision ratée" :** si la révision prévue est ratée, la faire le plus tôt possible dans les 2 jours suivants sans changer le reste du calendrier. Ne pas "compenser" en révisant deux fois le même jour — ça réintroduit la massed practice. Accepter un léger décalage de calendrier.

---

## Exercice 3 — Audit des habitudes : grille d'auto-évaluation

**Éléments d'un bon audit :**

**Question 4 (bachotage) :** Un bon audit inclut une distinction nette entre performance à l'examen et rétention 3 semaines après. Exemple de réponse honnête : "J'avais eu 17/20 mais je ne pourrais pas refaire le sujet aujourd'hui." C'est exactement la démonstration de la massed practice : pic de performance à court terme, courbe d'oubli rapide ensuite.

**Freins courants à la spaced repetition et réponses :**
- "C'est trop de travail de gérer les dates" → Anki automatise tout ; 10-15 min/jour suffisent
- "Je ne sais pas quoi mettre en flashcard" → une règle simple : une carte par concept que tu veux te rappeler dans 6 mois
- "Ça ne marche que pour la mémorisation, pas pour la compréhension" → faux : les cartes bien formulées testent la compréhension (mécanisme, nuance, application) pas juste la définition

**Exemples de 2 changements concrets (bonne granularité) :**
1. "Après chaque session de lecture de ce repo, je crée minimum 5 flashcards Anki avant de fermer le fichier." (déclencheur : fin de lecture ; action : 5 cartes ; outil : Anki)
2. "Je programme une alarme 'révision Anki' tous les matins à 8h pendant 15 min." (heure fixe, durée fixe)

**Changements trop vagues à éviter :**
- "Je vais faire plus d'Anki" (quand ? combien ? sur quoi ?)
- "Je vais arrêter de relire" (trop négatif, pas d'action de remplacement)

---

## Pour aller plus loin

Pour tester l'algorithme SM-2 directement :

```bash
python domains/vie/apprendre-a-apprendre/02-code/03-spaced-repetition.py
```

Le script simule une session complète avec 3 cartes exemples et affiche les intervalles calculés en temps réel. La sortie du tableau correspond exactement aux sessions 1-6 de l'exercice 1.
