# Exercices — Module 04 : Pensée bayésienne

> **Prérequis** : Module 04 (théorie + script `02-code/04-pensee-bayesienne.py`).
> **Format** : 3 exercices gradués (facile → difficile). Calculatrice autorisée.
> **Solutions** : `03-exercises/solutions/04-pensee-bayesienne.md`.

---

## Exercice 1 — Application directe de Bayes (facile)

### Objectif
Appliquer la formule du théorème de Bayes dans un contexte d'urne, étape par étape, en identifiant prior, vraisemblance et posterior.

### Consigne

Une urne contient soit des billes à majorité bleue (type B : 20 % rouge, 80 % bleue), soit des billes à majorité rouge (type R : 80 % rouge, 20 % bleue). Vous ne savez pas quelle type d'urne est devant vous. Votre prior est uniforme : P(type R) = 0,50.

Vous tirez **une bille au hasard** et elle est **rouge**.

**Questions :**
1. Identifiez le prior P(R), la vraisemblance P(rouge|R) et la vraisemblance complémentaire P(rouge|B).
2. Calculez P(rouge) — la probabilité totale d'observer une bille rouge.
3. Appliquez le théorème de Bayes pour trouver le posterior P(R|rouge).
4. Exprimez le résultat en pourcentage et interprétez : la bille rouge vous a-t-elle appris quelque chose d'utile ?

### Critères de réussite
- [ ] Prior, vraisemblances et posterior sont clairement identifiés et nommés.
- [ ] P(rouge) est calculé correctement via la probabilité totale.
- [ ] Le posterior est correct à 0,5 % près.
- [ ] L'interprétation mentionne l'évolution de la confiance (avant/après).

---

## Exercice 2 — Mise à jour itérative (intermédiaire)

### Objectif
Enchaîner deux mises à jour bayésiennes en utilisant le posterior d'une étape comme prior de la suivante.

### Consigne

Une petite usine a **deux lignes de production** :
- **Ligne L1** : produit 70 % des pièces, taux de défaut 5 %.
- **Ligne L2** : produit 30 % des pièces, taux de défaut 15 %.

On prélève une pièce **sans savoir de quelle ligne elle vient**.

**Étape A** : On observe que la pièce est **défectueuse**.
1. Posez le prior P(L1) = 0,70 (proportionnel à la production de L1).
2. Calculez le posterior P(L1|défaut) après cette première observation.

**Étape B** : On prélève une **deuxième pièce de la même ligne inconnue** et elle est elle aussi **défectueuse** (les deux tirages sont indépendants).
3. Utilisez le posterior de l'étape A comme nouveau prior.
4. Calculez le posterior P(L1|défaut₁, défaut₂) après deux pièces défectueuses.

**Étape C** :
5. Calculez le rapport de vraisemblance LR = P(défaut|L1) / P(défaut|L2).
6. Expliquez en une phrase pourquoi LR < 1 fait baisser la confiance en L1 à chaque défaut observé.

### Critères de réussite
- [ ] Le prior de l'étape B est bien le posterior de l'étape A (pas le prior initial).
- [ ] Les deux posteriors sont calculés correctement à 0,5 % près.
- [ ] LR est calculé et correctement interprété (< 1 → preuve contre L1).
- [ ] L'explication de la mécanique itérative est claire en une phrase.

---

## Exercice 3 — Rapport de vraisemblance et sophisme (difficile)

### Objectif
Calculer et comparer des rapports de vraisemblance, puis identifier le sophisme du procureur dans un raisonnement donné.

### Consigne

Un contrôleur de stock évalue deux fournisseurs F1 et F2 pour un défaut d'emballage.

**Données :**
- P(emballage défectueux | F1) = 0,02
- P(emballage défectueux | F2) = 0,10
- P(emballage défectueux | F3, un troisième fournisseur) = 0,05

**Partie A — LR et interprétation :**
1. Calculez LR₁ = P(défaut|F1) / P(défaut|F2) pour la comparaison F1 vs F2.
2. Calculez LR₂ = P(défaut|F3) / P(défaut|F2) pour la comparaison F3 vs F2.
3. En supposant un prior uniforme P(F1) = P(F2) = P(F3) = 1/3, après avoir observé **un emballage défectueux**, classez F1, F2, F3 par probabilité décroissante d'être la source.

**Partie B — Sophisme du procureur :**

Le responsable logistique dit : « Le taux de défaut de F2 est 10 %. L'emballage est défectueux. Donc il y a 10 % de chance que F2 soit responsable. »

4. Identifiez l'erreur de raisonnement. Quel terme bayésien est mal utilisé ?
5. Recalculez correctement P(F2|défaut) en partant d'un prior uniforme.

### Critères de réussite
- [ ] LR₁ et LR₂ sont calculés correctement.
- [ ] Le classement des fournisseurs après observation est correct.
- [ ] Le sophisme est correctement nommé (confusion P(E|H) ↔ P(H|E)).
- [ ] P(F2|défaut) est calculé correctement à 0,5 % près.
- [ ] La distinction vraisemblance / posterior est expliquée clairement.
