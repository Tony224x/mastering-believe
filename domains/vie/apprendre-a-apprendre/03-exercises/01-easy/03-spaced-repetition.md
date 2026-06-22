# Exercices — Module 03 : Spaced repetition & Anki

> **Niveau** : Débutant → Intermédiaire | **Temps estimé** : 40-50 min
> Ces exercices couvrent la théorie SM-2, la règle d'espacement de Cepeda, et l'implémentation du script `02-code/03-spaced-repetition.py`.

---

## Exercice 1 — Calculer des intervalles SM-2 à la main

### Objectif
Comprendre le mécanisme de l'algorithme SM-2 en simulant à la main les calculs d'intervalles, puis valider avec le script Python.

### Consigne
Tu as une carte avec les paramètres initiaux : **EF = 2.5**, **intervalle = 0**, **répétitions réussies = 0** (première révision).

Simule 6 sessions de révision consécutives avec les notes suivantes : `4, 3, 5, 2, 4, 4`.

Pour chaque session, calcule :
1. Le nouvel EF avec la formule officielle SM-2 :
   `EF_new = EF + (0.1 - (5 - note) × (0.08 + (5 - note) × 0.02))`
   `EF_new = max(1.3, EF_new)`
2. L'intervalle de la prochaine révision :
   - Note >= 3 (succès) :
     - 1re révision réussie → intervalle = 1 jour
     - 2e révision réussie → intervalle = 6 jours
     - Suivantes → `intervalle_new = round(intervalle_précédent × EF_new)`
   - Note < 3 (échec) → intervalle = 1 jour, compteur remis à 0
3. Le nombre total de jours écoulés depuis le début

Présente les résultats dans un tableau : Session | Note | EF avant | EF après | Intv. précédent | Prochain intv. | Jours cumulés

Vérifie ensuite tes calculs en lançant :
```bash
python domains/vie/apprendre-a-apprendre/02-code/03-spaced-repetition.py
```

### Critères de réussite
- [ ] Tableau complété pour les 6 sessions
- [ ] La note 2 (session 4) remet correctement l'intervalle à 1 jour et le compteur à 0
- [ ] Les sessions 5 et 6 utilisent les intervalles fixes (1 puis 6 jours) — pas `round(intv × EF)` — car le compteur a été remis à 0
- [ ] L'EF ne descend jamais sous 1.3
- [ ] Les résultats sont vérifiés avec le script Python (aucune différence > 0.01 sur les EF)
- [ ] Tu peux expliquer en 2 phrases pourquoi une note < 3 remet l'intervalle à 1 et non à 0

---

## Exercice 2 — Planifier un calendrier de révision espacée manuel

### Objectif
Construire un plan de révision espacée pour un sujet réel sans outil numérique, en appliquant la règle d'espacement de Cepeda (2008).

### Consigne
Choisis un sujet que tu dois maîtriser dans **30 jours** (exemple : les flash-cards du module 01, un chapitre de cours, une liste de vocabulaire).

1. Applique la règle Cepeda : gap optimal ≈ 10-20 % du délai avant le test final.
   - Test final dans 30 jours → premier intervalle : ~3-6 jours.

2. Planifie 4 révisions espacées entre aujourd'hui et J+30, avec des intervalles croissants. Donne les dates exactes (ou J+N).

3. Pour chaque révision, spécifie :
   - La technique de révision (retrieval practice / blank-page recall / flashcards / auto-questionnement)
   - La durée estimée
   - Ce que tu feras si tu trouves des lacunes pendant cette révision

4. Écris une règle personnelle de "que faire si je rate une révision" — quand la rattraper, comment adapter le calendrier.

### Critères de réussite
- [ ] Un sujet réel choisi
- [ ] 4 dates de révision planifiées entre J+0 et J+30
- [ ] Les intervalles sont croissants (pas le même espacement à chaque fois)
- [ ] Le premier intervalle respecte la règle 10-20 % (3 à 6 jours si test à J+30)
- [ ] Chaque révision a une technique et une durée spécifiées
- [ ] Une règle "rattrapage" écrite

---

## Exercice 3 — Audit de ses habitudes de révision actuelles

### Objectif
Mesurer l'écart entre ses pratiques actuelles et les pratiques evidence-based, et concevoir un plan correctif concret.

### Consigne
Réponds honnêtement aux 6 questions suivantes sur tes habitudes d'apprentissage récentes :

1. Pour le dernier contenu que tu as étudié : quand as-tu refait une révision ? (jamais / le lendemain / 1 semaine / 1 mois)
2. Est-ce que tu relies du contenu pour "réviser" ? (oui systématiquement / parfois / non)
3. Est-ce que tu as un système de flashcards actif (Anki ou autre) ? Si oui, depuis quand ?
4. Est-ce que tu as déjà "bachoté" pour un examen ? Décris le résultat 3 semaines après (pas la note — la rétention).
5. Quelle est la stratégie d'étude que tu utilises le plus souvent ?
6. Qu'est-ce qui t'empêche (si quelque chose t'empêche) d'utiliser la spaced repetition systématiquement ?

Puis :
- Identifie **2 changements concrets** que tu vas faire cette semaine pour intégrer la spaced repetition
- Écris un mini-plan : quoi changer, comment, quand (déclencheur + action + outil + fréquence)

### Critères de réussite
- [ ] 6 questions répondues honnêtement (pas de réponse "idéale" — réponse réelle)
- [ ] La question 4 contient une réflexion sur l'oubli post-bachotage, pas juste la performance de l'examen
- [ ] 2 changements identifiés avec un niveau de concrétude suffisant (pas "je vais faire plus d'Anki" — "je vais créer 10 cartes Anki après chaque session de lecture, tous les matins entre 8h et 9h")
- [ ] Le mini-plan est réalisable dans cette semaine (pas "je vais refaire tout mon système d'étude")

---

*Solutions et corrigé dans `03-exercises/solutions/03-spaced-repetition.md`*
*Script SM-2 : `02-code/03-spaced-repetition.py`*
