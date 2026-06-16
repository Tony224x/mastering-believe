# Solutions (medium) — Module 03 : Spaced repetition

> Verifie l'exercice 1 avec le script Python. Les exercices 2 et 3 admettent plusieurs calendriers valides ; criteres ci-dessous.

---

## Exercice 1 — Deux cartes de difficulte differente

**Convention d'intervalle : `round(intervalle_precedent × EF apres mise a jour)`, comme dans `02-code/03-spaced-repetition.py` (style Anki).**

**Carte FACILE — notes 5, 5, 5, 5 :**

| Session | Note | EF apres | Intervalle | Jours cumules |
|---------|------|----------|-----------|---------------|
| 1 | 5 | 2.60 | 1 (1re reussie) | J+1 |
| 2 | 5 | 2.70 | 6 (2e reussie) | J+7 |
| 3 | 5 | 2.80 | round(6 × 2.80) = 17 | J+24 |
| 4 | 5 | 2.90 | round(17 × 2.90) = 49 | J+73 |

**Carte DIFFICILE — notes 3, 4, 3, 3 :**

| Session | Note | EF apres | Intervalle | Jours cumules |
|---------|------|----------|-----------|---------------|
| 1 | 3 | 2.36 | 1 (1re reussie) | J+1 |
| 2 | 4 | 2.36 | 6 (2e reussie) | J+7 |
| 3 | 3 | 2.22 | round(6 × 2.22) = 13 | J+20 |
| 4 | 3 | 2.08 | round(13 × 2.08) = 27 | J+47 |

**Comparaison apres 4 sessions :** la carte facile sera revue dans **49 jours** (a J+73), la difficile dans **27 jours** (a J+47). La facile s'espace presque deux fois plus vite.

**Pourquoi la difficile revient plus souvent :** chaque note moyenne fait *baisser* l'EF (2.50 -> 2.08), et l'intervalle = intervalle × EF. Un EF plus bas = un multiplicateur plus faible = des intervalles plus courts = des revisions plus frequentes. Tu n'as rien decide manuellement : l'algorithme concentre automatiquement l'effort sur ce qui resiste. C'est exactement le but de SM-2.

**Verification :** lance `simulate_sessions` avec `[5,5,5,5]` puis `[3,4,3,3]` ; les EF doivent concorder a ± 0.01.

---

## Exercice 2 — Regle Cepeda a 3 horizons

**Regle : intervalle optimal ≈ 10-20 % du delai avant le test.**

| Horizon | Delai | Premier intervalle (10-20 %) | Ordre de grandeur Cepeda (2008) |
|---------|-------|------------------------------|----------------------------------|
| A | 10 jours | ~1-2 jours | ~1 jour |
| B | 35 jours | ~4-7 jours | ~1 semaine |
| C | 350 jours | ~35-70 jours | ~3-4 semaines |

**Calendrier modele A (quiz J+10) :** J+1, J+3, J+6, (test J+10). Intervalles croissants ~1-2-3-4 j.

**Calendrier modele B (examen J+35) :** J+5, J+12, J+22, (test J+35). Intervalles croissants ~5-7-10 j.

**Pour C (1 an) :** un calendrier manuel devient impraticable — il faudrait suivre des dizaines de cartes avec des intervalles individuels evoluant a chaque revision. Un SRS (Anki) automatise : il calcule par carte le prochain intervalle (SM-2/FSRS), affiche chaque jour uniquement les cartes "dues", et adapte selon ta note. C'est exactement le probleme que SM-2 a ete cree pour resoudre.

**Reference :** Cepeda, N. J. et al. (2008). *Psychological Science*, 19(11), 1095–1102.

---

## Exercice 3 — Reparer le calendrier casse

**Problemes du calendrier d'origine :**
1. **Massed practice au debut** : R1/R2/R3 a J+0, J+1, J+2 sont trop rapprochees — la memoire est encore fraiche, rendement faible (viole la regle Cepeda : premier intervalle ~3-6 j pour un test a J+30).
2. **Bachotage final** : R4 = relecture complete la veille = massed practice de derniere minute, pic court terme qui s'effondre.
3. **Technique exclusivement passive** : 100 % relecture, utilite "faible" (Dunlosky 2013), aucune recuperation active.
4. **Trou de 27 jours** : rien entre J+2 et J+29 — l'oubli s'installe sans aucune revision intermediaire.
5. **Absence totale de retrieval** : aucune mesure objective de ce qui est su.

**Calendrier corrige (examen J+30) :**

| Revision | Jour | Intervalle | Technique |
|----------|------|-----------|-----------|
| R1 | J+1 | — | Blank-page recall |
| R2 | J+5 | 4 j | Flashcards (toutes les cartes) |
| R3 | J+13 | 8 j | Flashcards + auto-questionnement sur les lacunes |
| R4 | J+27 | 14 j | Retrieval complet (blank-page + quiz) |
| Test | J+30 | 3 j | — |

**Justifications :** intervalles croissants conformes a la regle 10-20 % (Cepeda 2008) ; relecture passive remplacee par du retrieval (Roediger & Karpicke 2006) ; espacement reel sur les 30 jours (pas de trou) ; derniere session = consolidation par recuperation, pas bachotage.
