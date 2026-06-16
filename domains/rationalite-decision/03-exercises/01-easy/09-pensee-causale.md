# Exercices — Module 09 : Pensée causale

> **Prérequis** : avoir lu `01-theory/09-pensee-causale.md` et (optionnel) lancé `02-code/09-pensee-causale.py`.
> **Niveau** : gradué du plus facile (Exercice 1) au plus difficile (Exercice 3).

---

## Exercice 1 — Identifier les confondants

### Objectif
Reconnaître qu'une corrélation observée peut être entièrement expliquée par une variable tierce (confondant), sans lien causal entre les deux variables observées.

### Consigne
Pour chacune des trois corrélations suivantes, **identifiez le confondant** le plus plausible et expliquez en une phrase pourquoi la corrélation X↔Y n'implique pas de causalité directe.

**Corrélation A** : dans une chaîne de magasins, les points de vente qui vendent le plus de parapluies ont aussi le plus fort chiffre d'affaires en bottes en caoutchouc.

**Corrélation B** : sur un réseau de livraison urbaine, les itinéraires parcourus par le plus grand nombre de véhicules présentent également le plus grand nombre de pannes mécaniques.

**Corrélation C** : dans une serre expérimentale, les plantes qui reçoivent le plus d'arrosage automatique montrent aussi la meilleure croissance foliaire.

### Critères de réussite
- [ ] Pour chaque corrélation, un confondant est nommé explicitement (une variable Z distincte de X et Y).
- [ ] L'explication montre comment Z cause (ou prédit) à la fois X et Y.
- [ ] On ne conclut pas à une causalité directe X→Y ou Y→X sans justification.
- [ ] Les réponses restent concises (3-4 phrases par item maximum).

---

## Exercice 2 — Contrefactuel et groupe contrôle

### Objectif
Comprendre ce qu'est un contrefactuel et pourquoi un groupe contrôle comparable est nécessaire pour estimer un effet causal.

### Consigne
Un responsable logistique teste un nouvel algorithme de planification des tournées (X) sur les entrepôts de la région Nord. Il observe que le coût par livraison (Y) baisse de 12 % en trois mois.

**Question 1** : Quel est le contrefactuel idéal pour cet entrepôt ?

**Question 2** : Le responsable compare ses résultats aux entrepôts de la région Sud, qui ont conservé l'ancien algorithme et dont le coût a baissé de 7 % sur la même période. Peut-il conclure que le nouvel algorithme a causé une baisse supplémentaire de 5 % ? Justifiez en identifiant au moins deux confondants possibles.

**Question 3** : Proposez un dispositif simple qui permettrait de mieux isoler l'effet causal de l'algorithme, sans nécessiter un essai clinique long et coûteux.

### Critères de réussite
- [ ] Le contrefactuel est défini correctement (même entrepôt, même période, sans le nouvel algorithme).
- [ ] Au moins deux confondants plausibles sont identifiés pour la comparaison Nord/Sud.
- [ ] La proposition du dispositif implique une forme d'assignation non biaisée (randomisation, tirage au sort entre entrepôts, etc.).
- [ ] L'argumentation distingue clairement corrélation observée et conclusion causale.

---

## Exercice 3 — Concevoir un A/B test

### Objectif
Appliquer le principe du RCT (essai contrôlé randomisé) à un contexte concret en identifiant les éléments clés : unité d'assignation, randomisation, mesure de l'effet, confondants neutralisés.

### Consigne
Une équipe produit veut savoir si ajouter un indicateur de stock restant ("Plus que 3 en stock !") sur une page produit augmente le taux d'achat.

Rédigez le **protocole d'un A/B test** en répondant à ces quatre points :

1. **Unité d'assignation** : qui ou quoi est randomisé ? Pourquoi ce choix ?
2. **Groupes** : que voit chaque groupe ? Combien de groupes ?
3. **Mesure** : quelle est la variable Y (résultat) ? Sur quelle durée la mesure-t-on ?
4. **Confondants neutralisés** : listez deux confondants que la randomisation neutralise automatiquement.

Puis répondez : si le taux d'achat passe de 3,5 % (contrôle) à 4,1 % (traitement), peut-on conclure à un effet causal ? Sous quelle condition ?

### Critères de réussite
- [ ] L'unité d'assignation est clairement définie (visiteur, session ou autre) et le choix est justifié.
- [ ] Les deux groupes sont décrits précisément (traitement vs contrôle, une seule différence entre eux).
- [ ] La variable Y est mesurable et bien définie (taux de conversion, pas une sensation vague).
- [ ] Deux confondants réels sont nommés et on explique pourquoi la randomisation les neutralise.
- [ ] La conclusion causale est conditionnée à la taille d'échantillon suffisante et à l'absence de contamination entre groupes.
