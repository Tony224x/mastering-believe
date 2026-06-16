# Solutions — Module 03 : Spaced repetition

> Pour l'exercice 1, verifie tes calculs avec le script Python. Pour les exercices 2 et 3, ce corrige donne des exemples de reference.

---

## Exercice 1 — Simulation SM-2 : tableau de reference

**Parametres initiaux :** EF = 2.5, intervalle = 0 (premiere revision)
**Notes appliquees :** 4, 3, 5, 2, 4, 4

| Session | Note | EF avant | Calcul EF | EF apres | Intervalle precedent | Prochain intervalle | Jours cumules |
|---------|------|----------|-----------|----------|---------------------|---------------------|---------------|
| 1 | 4 | 2.50 | 2.50 + (0.1 - 1×(0.08 + 1×0.02)) = 2.50 + 0.0 = 2.50 | 2.50 | 0 | 1 (1re reussie) | J+1 |
| 2 | 3 | 2.50 | 2.50 + (0.1 - 2×(0.08 + 2×0.02)) = 2.50 + (0.1 - 2×0.12) = 2.50 - 0.14 = 2.36 | 2.36 | 1 | 6 (2e reussie) | J+7 |
| 3 | 5 | 2.36 | 2.36 + (0.1 - 0×...) = 2.36 + 0.10 = 2.46 | 2.46 | 6 | round(6 × 2.36) = 15 | J+22 |
| 4 | 2 | 2.46 | 2.46 + (0.1 - 3×(0.08 + 3×0.02)) = 2.46 + (0.1 - 3×0.14) = 2.46 - 0.32 = 2.14 | 2.14 | 15 | **1** (note < 3 → reset) | J+23 |
| 5 | 4 | 2.14 | 2.14 + 0.0 = 2.14 | 2.14 | 1 | round(1 × 2.14) = 2 | J+25 |
| 6 | 4 | 2.14 | 2.14 + 0.0 = 2.14 | 2.14 | 2 | round(2 × 2.14) = 4 | J+29 |

*Note : les intervalles 1re et 2e reussies sont fixes par SM-2 (1 et 6 jours). A partir de la 3e, c'est `intervalle × EF`. L'EF minimum est 1.3.*

**Formule EF pour la session 2 (note = 3) :**
- `EF = 2.50 + (0.1 - (5-3) × (0.08 + (5-3) × 0.02))`
- `EF = 2.50 + (0.1 - 2 × (0.08 + 0.04))`
- `EF = 2.50 + (0.1 - 2 × 0.12)`
- `EF = 2.50 + (0.1 - 0.24) = 2.50 - 0.14 = 2.36`

**Pourquoi la note < 3 remet a 1 et non a 0 :** L'intervalle 0 n'aurait pas de sens (revision le jour meme en continu). SM-2 impose un minimum de 1 jour car l'espacement, meme court, active mieux la consolidation qu'une revision immediate. Le but n'est pas de punir l'echec mais de relancer le cycle d'espacement a partir du palier le plus bas.

---

## Exercice 2 — Calendrier de revision : exemple de reference

**Sujet exemple :** les 5 flash-cards du Module 01 | **Test final :** J+30

**Regle Cepeda (2008) :** gap optimal ≈ 10-20 % du delai → ici 3 a 6 jours pour le premier intervalle.

| Revision | Date | Intervalle depuis precedent | Technique | Duree |
|----------|------|-----------------------------|-----------|-------|
| 1 | J+1 | — | Blank-page recall | 10 min |
| 2 | J+4 | 3 jours | Flashcards (toutes les cartes) | 10 min |
| 3 | J+12 | 8 jours | Flashcards + auto-questionnement sur les lacunes | 15 min |
| 4 | J+28 | 16 jours | Retrieval complet (blank-page + flashcards) | 20 min |
| Test | J+30 | 2 jours | — | — |

**Regle "revision ratee" :** si la revision prevue est ratee, la faire le plus tot possible dans les 2 jours suivants sans changer le reste du calendrier. Ne pas "compenser" en revisant deux fois le meme jour — ca reintroduit la massed practice. Accepter un leger decalage de calendrier.

---

## Exercice 3 — Audit des habitudes : grille d'auto-evaluation

**Elements d'un bon audit :**

**Question 4 (bachotage) :** Un bon audit inclut une distinction nette entre performance a l'examen et retention 3 semaines apres. Exemple de reponse honnete : "J'avais eu 17/20 mais je ne pourrais pas refaire le sujet aujourd'hui." C'est exactement la demonstration de la massed practice : pic de performance a court terme, courbe d'oubli rapide ensuite.

**Freins courants a la spaced repetition et reponses :**
- "C'est trop de travail de gerer les dates" → Anki automatise tout ; 10-15 min/jour suffisent
- "Je ne sais pas quoi mettre en flashcard" → une regle simple : une carte par concept que tu veux te rappeler dans 6 mois
- "Ca ne marche que pour la memorisation, pas pour la comprehension" → faux : les cartes bien formulees testent la comprehension (mecanisme, nuance, application) pas juste la definition

**Exemples de 2 changements concrets (bonne granularite) :**
1. "Apres chaque session de lecture de ce repo, je cree minimum 5 flashcards Anki avant de fermer le fichier." (declencheur : fin de lecture ; action : 5 cartes ; outil : Anki)
2. "Je programme une alarme 'revision Anki' tous les matins a 8h pendant 15 min." (heure fixe, duree fixe)

**Changements trop vagues a eviter :**
- "Je vais faire plus d'Anki" (quand ? combien ? sur quoi ?)
- "Je vais arreter de relire" (trop negatif, pas d'action de remplacement)

---

## Pour aller plus loin

Pour tester l'algorithme SM-2 directement :

```bash
python domains/apprendre-a-apprendre/02-code/03-spaced-repetition.py
```

Le script simule une session complete avec 3 cartes exemples et affiche les intervalles calcules en temps reel.
