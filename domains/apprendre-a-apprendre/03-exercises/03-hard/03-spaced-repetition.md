# Exercices (hard) — Module 03 : Spaced repetition

> **Niveau** : Avance | **Temps estime** : 70-85 min
> Calcul complet d'un planning SM-2, conception d'un systeme d'espacement personnel, et analyse critique d'un mythe sur l'espacement. Verifie tes calculs avec `02-code/03-spaced-repetition.py`.

---

## Exercice 1 — Calculer un planning de repetition espacee SM-2 complet

### Objectif
Maitriser l'algorithme SM-2 de bout en bout : appliquer les formules, gerer les cas particuliers (1re/2e revision, echec) et produire un planning de dates exploitable.

### Consigne
Carte initiale : **EF = 2.5**, **intervalle = 0**, **repetitions = 0**. Date de depart : aujourd'hui = J+0.

Applique la sequence de notes : **`5, 4, 3, 5, 5`**.

Rappels des formules (cf. module 03 et `02-code/03-spaced-repetition.py`) :
- `EF_new = EF + (0.1 - (5 - note) × (0.08 + (5 - note) × 0.02))`, plancher a **1.3**.
- Si note >= 3 : 1re reussite -> intervalle = 1 ; 2e reussite -> intervalle = 6 ; ensuite -> `round(intervalle_precedent × EF)` (avec l'EF **apres** mise a jour, comme dans le script / Anki).
- Si note < 3 : intervalle = 1, repetitions remis a 0.

1. Produis le tableau complet : Session | Note | EF avant | EF apres | Intervalle precedent | Prochain intervalle | Date de prochaine revision (J+cumul).
2. Pour la session 3 (note = 3), ecris le detail du calcul de l'EF, ligne par ligne.
3. Explique pourquoi, malgre une note moyenne (3) en session 3, l'intervalle continue d'augmenter au lieu de repartir a 1.
4. **Variante echec** : reprends la meme carte et remplace la note de la session 3 par **2**. Recalcule les sessions 3, 4, 5 et explique l'effet du reset sur le planning global.
5. Verifie l'ensemble avec `python domains/apprendre-a-apprendre/02-code/03-spaced-repetition.py` (adapte `test_notes`).

### Criteres de reussite
- [ ] Le tableau est complet pour les 5 sessions avec dates cumulees
- [ ] Le calcul EF de la session 3 est detaille et correct
- [ ] L'explication distingue "note >= 3 = succes (l'intervalle grandit)" de "note < 3 = reset"
- [ ] La variante echec (note 2) montre correctement le reset a 1 jour et le redemarrage du cycle
- [ ] Les resultats concordent avec le script (EF a ± 0.01)
- [ ] Tu peux expliquer pourquoi une seule mauvaise note coute beaucoup en repoussant la maitrise de plusieurs jours

---

## Exercice 2 — Concevoir ton systeme d'espacement personnel

### Objectif
Concevoir un systeme d'espacement reutilisable pour ton propre apprentissage, en choisissant les bons parametres et les bons garde-fous.

### Consigne
Concois (sur papier) un systeme d'espacement pour un corpus reel que tu dois retenir longtemps (vocabulaire d'une langue, faits d'un domaine, formules).

Documente :
1. **Choix de l'outil** : Anki (SM-2/FSRS) vs plan manuel — justifie selon le volume de cartes et l'horizon de retention.
2. **Politique de creation de cartes** : quelle granularite, quelle regle pour decider "ca merite une carte ou non", comment eviter la surcharge (300 nouvelles cartes/jour = abandon garanti).
3. **Politique de notation** : comment tu notes honnetement (resister a la tentation de se sur-noter, qui casse l'algorithme).
4. **Politique de retard** : que fais-tu quand tu accumules des cartes "en retard" apres une pause (ne pas tout bachoter le meme jour — pourquoi).
5. **Garde-fous anti-illusion** : comment t'assurer que les cartes testent la comprehension/application et pas seulement la reconnaissance.
6. **Critere d'arret** : quand une carte peut-elle "sortir" du systeme (maturite) ?

### Criteres de reussite
- [ ] Le choix d'outil est justifie par le volume et l'horizon
- [ ] La politique de creation limite explicitement le flux de nouvelles cartes
- [ ] La politique de notation adresse le risque de se sur-noter (et son effet : intervalles trop longs -> oubli)
- [ ] La politique de retard exclut le bachotage de rattrapage (re-massage de la pratique)
- [ ] Au moins un garde-fou anti-reconnaissance est present
- [ ] Un critere de maturite/sortie de carte est defini

---

## Exercice 3 — Analyser un mythe : "l'espacement, c'est juste pour ceux qui s'y prennent a la derniere minute en moins pire"

### Objectif
Demonter une mecomprehension repandue de l'espacement et la corriger avec le mecanisme et la preuve, sans le survendre.

### Consigne
Une croyance courante : *"L'espacement, c'est juste une astuce pour les gens desorganises. Si tu bosses serieusement et longtemps d'un coup, le bachotage intensif marche aussi bien — l'espacement n'apporte rien de plus a temps egal."*

Redige une analyse structuree :
1. **La part de vrai** : qu'est-ce qui rend le bachotage seduisant et parfois suffisant a court terme ?
2. **L'erreur de fond** : pourquoi, *a temps total egal*, l'espacement bat la pratique massee — quel est le mecanisme (un debut d'oubli rend la revision plus effortful, donc plus consolidante ; difficulte desirable) ?
3. **La preuve** : restitue ce qu'apporte Cepeda et al. (2006, meta-analyse : 839 mesures, 317 experiences) et la regle chiffree de Cepeda et al. (2008). Insiste sur "a temps egal".
4. **Honnetete sur la preuve** : dans quel cas precis le bachotage peut-il etre rationnel (test demain, contenu jetable) ? Ne pas pretendre que l'espacement est toujours superieur pour *tout* objectif.
5. **Regle pratique** en 2 phrases.

### Criteres de reussite
- [ ] La part de vrai (bachotage = pic court terme, parfois suffisant) est reconnue
- [ ] Le mecanisme "a temps egal, l'espacement gagne" est explique (oubli partiel -> effort -> consolidation)
- [ ] Cepeda et al. (2006) et (2008) sont cites correctement avec ce qu'ils etablissent
- [ ] L'honnetete sur la preuve identifie le cas legitime du bachotage (echeance immediate, contenu jetable)
- [ ] La regle finale est actionnable
- [ ] Aucune sur-vente : l'espacement n'est pas presente comme une solution universelle a tout

---

*Solutions et corrige dans `03-exercises/solutions/03-spaced-repetition-hard.md`*
*Script SM-2 : `02-code/03-spaced-repetition.py`*
