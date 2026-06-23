# Exercices (medium) — Module 03 : Spaced repetition

> **Niveau** : Intermediaire | **Temps estime** : 45-55 min
> On approfondit l'algorithme SM-2 et l'application de la regle Cepeda a des cas non triviaux. Utilise le script `02-code/03-spaced-repetition.py` pour verifier.

---

## Exercice 1 — Simuler SM-2 sur deux cartes de difficulte differente

### Objectif
Comprendre comment l'EF differencie les cartes faciles des cartes difficiles, et l'impact sur la charge de revision.

### Consigne
Simule deux cartes en parallele, EF initial 2.5, intervalle 0 :
- **Carte FACILE** : notes `5, 5, 5, 5`.
- **Carte DIFFICILE** : notes `3, 4, 3, 3`.

1. Pour chaque carte, calcule a la main le tableau : Session | Note | EF apres | Intervalle | Jours cumules.
2. Compare : apres 4 sessions, dans combien de temps chaque carte sera-t-elle revue ?
3. Explique pourquoi la carte difficile revient bien plus souvent que la facile, alors qu'on n'a pas decide manuellement de la prioriser.
4. Verifie tes EF avec le script Python (`simulate_sessions`).

### Criteres de reussite
- [ ] Les deux tableaux sont complets pour les 4 sessions
- [ ] L'EF de la carte facile monte, celui de la carte difficile descend (sans passer sous 1.3)
- [ ] L'ecart d'intervalle final entre les deux cartes est calcule
- [ ] L'explication relie EF bas -> intervalle court -> revisions plus frequentes (le systeme concentre l'effort sur le difficile)
- [ ] Les EF sont verifies avec le script (ecart < 0.01)

---

## Exercice 2 — Adapter la regle Cepeda a 3 horizons de test

### Objectif
Appliquer la regle "intervalle optimal ≈ 10-20 % du delai" a des echeances differentes et en deduire des calendriers concrets.

### Consigne
Tu dois maitriser un meme contenu pour trois echeances differentes selon le contexte :
- (A) un quiz dans **10 jours**,
- (B) un examen dans **35 jours**,
- (C) une retention durable a **1 an** (~350 jours).

1. Pour chaque horizon, calcule le premier intervalle de revision selon la regle 10-20 %.
2. Donne un calendrier de 3-4 revisions a intervalles croissants pour A et B.
3. Pour C (1 an), explique pourquoi un calendrier manuel devient impraticable et ce qu'un SRS (Anki) apporte.
4. Relie tes chiffres aux ordres de grandeur donnes par Cepeda et al. (2008) (~1 jour pour 10 jours, ~1 semaine pour 35 jours, ~3-4 semaines pour 1 an).

### Criteres de reussite
- [ ] Les 3 premiers intervalles respectent la regle 10-20 % (≈ 1-2 j ; ≈ 4-7 j ; ≈ 35-70 j)
- [ ] Calendriers a intervalles croissants fournis pour A et B
- [ ] L'argument "manuel impraticable a 1 an -> SRS" est explicite
- [ ] Les chiffres sont coherents avec les ordres de grandeur de Cepeda et al. (2008)

---

## Exercice 3 — Reparer un calendrier de revision casse

### Objectif
Diagnostiquer les erreurs classiques d'un calendrier de revision et les corriger avec les principes du module.

### Consigne
Voici le calendrier d'un etudiant pour un examen a J+30 :

| Revision | Jour | Technique |
|----------|------|-----------|
| R1 | J+0 (jour de l'etude) | relire le cours |
| R2 | J+1 | relire le cours |
| R3 | J+2 | relire le cours |
| R4 | J+29 | relire le cours en entier (bachotage) |

Identifie **au moins 4 problemes** et propose un calendrier corrige (memes contraintes : examen a J+30, temps total comparable), en justifiant chaque correction par un principe (espacement, retrieval, difficulte desirable, regle Cepeda).

### Criteres de reussite
- [ ] Au moins 4 problemes identifies (massed practice au debut, gros bachotage final, technique exclusivement passive, premier intervalle trop court vs Cepeda, absence de retrieval, etc.)
- [ ] Le calendrier corrige a des intervalles croissants conformes a la regle 10-20 %
- [ ] La relecture passive est remplacee par du retrieval (flashcards / blank-page / quiz)
- [ ] Chaque correction est reliee a un principe nomme
- [ ] La derniere session n'est pas un bachotage massif mais une consolidation par retrieval

---

*Solutions et corrige dans `03-exercises/solutions/03-spaced-repetition-medium.md`*
*Script SM-2 : `02-code/03-spaced-repetition.py`*
