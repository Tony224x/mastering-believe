# Exercices — Module 07 : Décision sous incertitude

> **Prérequis** : avoir lu `01-theory/07-decision-incertitude.md`.
> **Durée estimée** : 30 à 40 min.
> **Consigne générale** : calculer à la main (calculatrice autorisée), justifier chaque étape.

---

## Exercice 1 — Espérance mathématique : comparer deux jeux (niveau facile)

### Objectif
Calculer et comparer des espérances pour choisir entre deux paris.

### Consigne

Un organisateur de kermesse propose deux jeux. Vous pouvez jouer une seule partie à l'un ou l'autre (mise unique, résultat immédiat).

**Jeu Alpha** : on tire une bille dans un sac de 10 billes (5 rouges, 3 bleues, 2 vertes).
- Rouge → vous gagnez 4 €
- Bleue → vous gagnez 1 €
- Verte → vous perdez 5 €

**Jeu Bêta** : on lance une pièce équilibrée.
- Pile → vous gagnez 3 €
- Face → vous perdez 2 €

Questions :
1. Calculez l'espérance mathématique E[Alpha] du Jeu Alpha.
2. Calculez l'espérance mathématique E[Bêta] du Jeu Bêta.
3. Quel jeu choisiriez-vous si vous voulez maximiser l'espérance ? Justifiez.
4. (Bonus) Si vous pouvez jouer 100 parties au même jeu, l'espérance change-t-elle par partie ? Qu'en est-il du gain total attendu ?

### Critères de réussite
- [ ] E[Alpha] calculé correctement avec les trois branches (rouge, bleue, verte)
- [ ] E[Bêta] calculé correctement (deux branches)
- [ ] Choix justifié par comparaison des deux espérances
- [ ] (Bonus) Distinction entre espérance par partie (constante) et gain total attendu (proportionnel au nombre de parties)

---

## Exercice 2 — Utilité espérée et aversion au risque (niveau intermédiaire)

### Objectif
Comprendre comment l'aversion au risque peut inverser une décision par rapport à l'espérance seule.

### Consigne

Un tournoi de jeu de plateau offre deux formules d'inscription :

**Formule Fixe** : vous payez 80 € et êtes assuré de récupérer 80 € en prix de participation (gain net = 0 €).

**Formule Risquée** : vous payez 80 €. À la fin du tournoi, tirage au sort :
- 40 % de probabilité : vous gagnez le jackpot et repartez avec +200 € net (après avoir récupéré votre mise)
- 60 % de probabilité : vous perdez votre mise, soit −80 € net

Questions :
1. Calculez E[Fixe] et E[Risquée] en termes de gain net.
2. Quelle formule maximise l'espérance monétaire ?
3. Calculez l'utilité espérée de chaque formule avec U(x) = √(x + 100) (décalage de +100 pour éviter les valeurs négatives sous la racine). Arrondissez à 3 décimales.
4. Avec cette fonction d'utilité, quelle formule est préférable ?
5. Que révèle la différence entre les réponses aux questions 2 et 4 sur l'attitude face au risque ?

### Critères de réussite
- [ ] E[Fixe] = 0 € et E[Risquée] calculé correctement
- [ ] Identification de la formule à espérance maximale
- [ ] EU calculé correctement pour les deux formules avec U(x) = √(x + 100)
- [ ] Conclusion correcte sur la préférence selon l'utilité espérée
- [ ] Explication de la notion de concavité / aversion au risque

---

## Exercice 3 — Arbre de décision : choisir un itinéraire (niveau difficile)

### Objectif
Construire et résoudre un arbre de décision à deux étapes pour choisir l'option à espérance maximale.

### Consigne

Un livreur doit rejoindre un entrepôt. Deux itinéraires s'offrent à lui.

**Route Nord** : durée de base 30 min.
- 20 % de chance de travaux → + 25 min de retard
- 80 % de chance : aucun retard

**Route Sud** : durée de base 40 min.
- 10 % de chance d'accident sur l'axe → + 40 min de retard
- 90 % de chance : aucun retard

Règles :
- Un retard de plus de 15 min entraîne une pénalité de 50 points de performance.
- Un retard nul ou inférieur à 15 min = 0 point de pénalité.
- Les points de pénalité se cumulent s'il y a plusieurs aléas (ici, un seul aléa par route).

Questions :
1. Calculez la durée totale attendue (espérance) pour chaque route.
2. Calculez l'espérance de pénalité (en points) pour chaque route.
3. Dessinez (ou décrivez) l'arbre de décision : nœuds de décision, nœuds de hasard, branches avec probabilités et valeurs de pénalité.
4. En minimisant l'espérance de pénalité, quelle route choisir ?
5. (Bonus) Si le livreur est très averse à la pénalité (il la ressent comme une perte grave), la conclusion pourrait-elle changer ? Expliquez sans recalculer formellement.

### Critères de réussite
- [ ] Durée attendue correcte pour les deux routes (Route Nord : 35 min, Route Sud : 44 min)
- [ ] Espérance de pénalité calculée correctement pour chaque route
- [ ] Arbre décrit ou dessiné avec nœuds, branches, probabilités et valeurs de pénalité
- [ ] Choix justifié par comparaison des espérances de pénalité
- [ ] (Bonus) Réflexion sur la distinction entre espérance monétaire et utilité espérée appliquée aux pénalités
