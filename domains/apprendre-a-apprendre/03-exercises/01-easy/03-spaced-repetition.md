# Exercices — Module 03 : Spaced repetition

> **Niveau** : Debutant | **Temps estime** : 40-50 min
> Ces exercices integrent la theorie SM-2 et l'utilisation du script `02-code/03-spaced-repetition.py`.

---

## Exercice 1 — Calculer des intervalles SM-2 a la main

### Objectif
Comprendre le mecanisme de l'algorithme SM-2 en simulant a la main les calculs d'intervalles.

### Consigne
Tu as une carte avec les parametres initiaux : **EF = 2.5**, **intervalle = 0** (premiere revision).

Simule 6 sessions de revision consecutives avec les notes suivantes : `4, 3, 5, 2, 4, 4`.

Pour chaque session, calcule :
1. Le nouvel EF avec la formule `EF_new = EF + (0.1 - (5 - note) × (0.08 + (5 - note) × 0.02))`
2. L'intervalle de la prochaine revision :
   - Note >= 3 : `intervalle_new = max(1, round(intervalle_precedent × EF))`
   - Note < 3 : `intervalle_new = 1` (recommencer)
   - Cas speciaux : premiere revision reussie → 1 jour ; deuxieme revision reussie → 6 jours
3. Le nombre total de jours ecoules depuis le debut

Presente les resultats dans un tableau : Session | Note | EF avant | EF apres | Intervalle precedent | Prochain intervalle | Jours cumules

Verifie ensuite tes calculs en lancant : `python domains/apprendre-a-apprendre/02-code/03-spaced-repetition.py`

### Criteres de reussite
- [ ] Tableau complete pour les 6 sessions
- [ ] La note 2 (session 4) remet correctement l'intervalle a 1 jour
- [ ] L'EF ne descend jamais sous 1.3
- [ ] Les resultats sont verifies avec le script Python (aucune difference > 0.01 sur les EF)
- [ ] Tu peux expliquer en 2 phrases pourquoi une note < 3 remet l'intervalle a 1 et non a 0

---

## Exercice 2 — Planifier un calendrier de revision espacee manuel

### Objectif
Construire un plan de revision espacee pour un sujet reel sans outil numerique, en appliquant la regle d'espacement de Cepeda (2008).

### Consigne
Choisis un sujet que tu dois maitriser dans **30 jours** (exemple : les flash-cards du module 01, un chapitre de cours, une liste de vocabulaire).

1. Applique la regle Cepeda : gap optimal ≈ 10-20 % du delai avant le test final.
   - Test final dans 30 jours → premier intervalle : ~3-6 jours.

2. Planifie 4 revisions espacees entre aujourd'hui et J+30, avec des intervalles croissants. Donne les dates exactes (ou J+N).

3. Pour chaque revision, specifie :
   - La technique de revision (retrieval practice / blank-page recall / flashcards / auto-questionnement)
   - La duree estimee
   - Ce que tu feras si tu trouves des lacunes pendant cette revision

4. Ecris une regle personnelle de "que faire si je rate une revision" — quand la rattraper, comment adapter le calendrier.

### Criteres de reussite
- [ ] Un sujet reel choisi
- [ ] 4 dates de revision planifiees entre J+0 et J+30
- [ ] Les intervalles sont croissants (pas le meme espacement a chaque fois)
- [ ] Le premier intervalle respecte la regle 10-20 % (3 a 6 jours si test a J+30)
- [ ] Chaque revision a une technique et une duree specifiees
- [ ] Une regle "rattrapage" ecrite

---

## Exercice 3 — Audit de ses habitudes de revision actuelles

### Objectif
Mesurer l'ecart entre ses pratiques actuelles et les pratiques evidence-based, et concevoir un plan correctif.

### Consigne
Reponds honnetement aux 6 questions suivantes sur tes habitudes d'apprentissage recentes :

1. Pour le dernier contenu que tu as etudie : quand as-tu refait une revision ? (jamais / le lendemain / 1 semaine / 1 mois)
2. Est-ce que tu relies du contenu pour "reviser" ? (oui systematiquement / parfois / non)
3. Est-ce que tu as un systeme de flashcards actif (Anki ou autre) ? Si oui, depuis quand ?
4. Est-ce que tu as deja "bachotee" pour un examen ? Decris le resultat 3 semaines apres.
5. Quelle est la strategie d'etude que tu utilises le plus souvent ?
6. Qu'est-ce qui t'empeche (si quelque chose t'empeche) d'utiliser la spaced repetition systematiquement ?

Puis :
- Identifie **2 changements concrets** que tu vas faire cette semaine pour integrer la spaced repetition
- Ecris un mini-plan : quoi changer, comment, quand

### Criteres de reussite
- [ ] 6 questions repondues honnetement (pas de reponse "ideale" — reponse reelle)
- [ ] La question 4 contient une reflexion sur l'oubli post-bachotage, pas juste la performance de l'examen
- [ ] 2 changements identifies avec un niveau de concretude suffisant (pas "je vais faire plus d'Anki" — "je vais creer 10 cartes Anki apres chaque session de lecture, tous les matins entre 8h et 9h")
- [ ] Le mini-plan est realisable dans cette semaine (pas "je vais refaire tout mon systeme d'etude")

---

*Solutions et corrige dans `03-exercises/solutions/03-spaced-repetition.md`*  
*Script SM-2 : `02-code/03-spaced-repetition.py`*
