# Exercices Medium — Module 05 : Décision sous Incertitude

> **Niveau** : Medium | **Temps estimé** : ~35 min

---

## Exercice 1 — Arbre de décision en deux temps (prototype puis go/no-go)

### Objectif

Construire un arbre de décision à deux étages (un prototype, puis un lancement conditionnel), le résoudre par **remontée par espérance** (backward induction) et identifier la **politique optimale** ainsi que la valeur du prototype.

### Consigne

Un atelier envisage de commercialiser un nouvel accessoire. Deux décisions s'enchaînent (montants en **milliers d'euros, k€**).

**Décision 1 — Faire un prototype-test ?** Le test coûte **10 k€** et renvoie un signal :
- **Favorable** avec probabilité 0,60
- **Défavorable** avec probabilité 0,40

**Décision 2 — Après le signal, lancer ou abandonner ?** Le lancement aboutit à :
- un **succès** : +200 k€
- un **échec** : −150 k€

Les chances de succès dépendent du signal observé :
- P(succès | favorable) = 0,80
- P(succès | défavorable) = 0,30

Abandonner rapporte 0 k€ (aucun coût de lancement engagé).

Pour comparaison, **sans test**, on peut lancer directement : la probabilité de succès *a priori* vaut alors 0,60 × 0,80 + 0,40 × 0,30.

1. Dessinez l'arbre : nœud décision □ (test ?), nœud hasard ○ (signal), nœuds décision □ (lancer/abandonner), nœuds hasard ○ (succès/échec).
2. Remontez par espérance : calculez E[lancer | favorable] et E[lancer | défavorable], puis la décision optimale à chaque sous-branche.
3. Calculez la valeur du nœud « test » (avant et après déduction des 10 k€).
4. Calculez l'espérance du lancement direct (sans test) et comparez. Énoncez la politique optimale et la **valeur nette** apportée par le test.

### Critères de réussite

- [ ] E[lancer | favorable] = 0,80 × 200 + 0,20 × (−150) = **130 k€** (on lance)
- [ ] E[lancer | défavorable] = 0,30 × 200 + 0,70 × (−150) = **−45 k€** (on abandonne → 0)
- [ ] Valeur au nœud test = 0,60 × 130 + 0,40 × 0 = **78 k€** ; nette du coût = 78 − 10 = **68 k€**
- [ ] Lancement direct (sans test) : P(succès) = 0,60, E = 0,60 × 200 + 0,40 × (−150) = **60 k€**
- [ ] Politique optimale : **faire le test ; lancer si favorable, abandonner si défavorable** (68 k€ > 60 k€)
- [ ] La valeur de l'information est nommée : gain brut 78 − 60 = **18 k€**, dont **8 k€ nets** une fois le test payé

---

## Exercice 2 — Même pari, deux profils : neutre vs averse au risque

### Objectif

Montrer que, face au **même** pari, un agent **neutre au risque** (maximise l'espérance) et un agent **averse au risque** (maximise l'utilité espérée avec une fonction concave) peuvent choisir différemment.

### Consigne

On propose un choix unique :
- **Pari G** : 50 % de chances de 900 €, 50 % de chances de 100 €
- **Option sûre S** : 500 € garantis

L'agent averse au risque utilise une fonction d'utilité concave **U(x) = √x** (racine carrée du montant en euros). L'agent neutre compare directement les montants espérés.

1. Calculez E[G], l'espérance monétaire du pari, et comparez-la à S. Que choisit l'agent **neutre** ?
2. Calculez l'utilité espérée EU[G] = 0,5·√900 + 0,5·√100, puis EU[S] = √500. Que choisit l'agent **averse** ?
3. Expliquez en une phrase pourquoi l'aversion au risque est ici **rationnelle** (et non une « erreur »).

### Critères de réussite

- [ ] E[G] = 0,5 × 900 + 0,5 × 100 = **500 €** → l'agent neutre est **indifférent** (G et S valent 500 €)
- [ ] EU[G] = 0,5 × 30 + 0,5 × 10 = **20** ; EU[S] = √500 ≈ **22,36**
- [ ] L'agent averse choisit **S** (22,36 > 20), alors qu'il a la même espérance monétaire
- [ ] L'explication relie le choix à la **concavité de U** : un euro supplémentaire « vaut » moins quand on est déjà riche, donc la dispersion du pari réduit l'utilité espérée — c'est cohérent, pas irrationnel

---

## Exercice 3 — Équivalent-certain et prime de risque (assurance d'un objet)

### Objectif

Calculer l'**équivalent-certain** (CE) et la **prime de risque** d'un agent averse, et en déduire le prix maximal qu'il accepterait de payer pour s'assurer — puis le comparer au prix actuariellement juste.

### Consigne

Une personne possède un vélo dont la valeur de remplacement est **1 000 €**. Sur l'année :
- avec probabilité 0,10, le vélo est volé → valeur restante 0 €
- avec probabilité 0,90, rien ne se passe → valeur 1 000 €

Cette personne est averse au risque, avec **U(x) = √x**.

1. Calculez la valeur espérée du patrimoine-vélo E et la **perte espérée**.
2. Calculez l'utilité espérée **sans assurance** : EU = 0,90·√1000 + 0,10·√0.
3. Déduisez l'**équivalent-certain** CE = (EU)² : le patrimoine certain qui procure la même utilité.
4. Calculez la **prime de risque** (= E − CE) et le **prix maximal** d'une assurance couvrant tout le vol (= 1 000 − CE). Comparez ce prix maximal à la **prime juste** (= perte espérée). Que vous apprend l'écart ?

### Critères de réussite

- [ ] E = 0,90 × 1 000 + 0,10 × 0 = **900 €** ; perte espérée = **100 €**
- [ ] EU (sans assurance) = 0,90 × √1000 ≈ 0,90 × 31,623 ≈ **28,46**
- [ ] Équivalent-certain CE = (28,46)² = **810 €** (car (0,9·√1000)² = 0,81 × 1000)
- [ ] Prime de risque = E − CE = 900 − 810 = **90 €**
- [ ] Prix maximal d'assurance accepté = 1 000 − 810 = **190 €** ; prime juste (perte espérée) = **100 €**
- [ ] Lecture : l'agent averse paierait jusqu'à 190 € pour une couverture qui « coûte » 100 € en espérance ; l'écart de 90 € (la prime de risque) est précisément ce qui rend un marché de l'assurance viable
