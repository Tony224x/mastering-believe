# Exercices Hard — Module 05 : Décision sous Incertitude

> **Niveau** : Hard | **Temps estimé** : ~50 min

---

## Exercice 1 — Arbre complet : étude de marché puis lancement (valeur de l'information)

### Objectif

Résoudre un arbre de décision **multi-étages** (deux nœuds décision, plusieurs nœuds hasard) par **remontée par espérance**, calculer la **valeur de l'information** apportée par un test, puis commenter la **sensibilité** à une probabilité.

### Consigne

Un petit fabricant envisage de lancer un produit (montants en **k€**). Sans information supplémentaire :
- **Demande forte** *a priori* : P = 0,45 → lancement rapporte **+500 k€**
- **Demande faible** *a priori* : P = 0,55 → lancement rapporte **−200 k€**
- **Ne pas lancer** : 0 k€

Avant de décider, il peut payer une **étude de marché** coûtant **40 k€**, qui renvoie un rapport « positif » ou « négatif ». La fiabilité de l'étude :
- P(rapport positif | demande forte) = 0,80
- P(rapport positif | demande faible) = 0,30

L'enchaînement est : **Décision 1** — faire l'étude ? puis **Décision 2** — lancer ou non, en tenant compte du rapport (s'il y a eu étude) ou de l'*a priori* (sinon).

1. **Branche sans étude.** Calculez l'espérance du lancement direct et la meilleure décision sans information.
2. **Branche avec étude.** Calculez P(positif), P(négatif), puis les probabilités *a posteriori* P(forte | positif) et P(forte | négatif) (théorème de Bayes). Déduisez E[lancer | positif] et E[lancer | négatif], et la décision optimale après chaque rapport.
3. **Remontée.** Calculez la valeur du nœud « étude » (avant puis après déduction des 40 k€). Décidez s'il faut faire l'étude.
4. **Valeur de l'information.** Calculez l'écart entre la valeur du nœud-étude (brute) et la meilleure valeur sans étude : c'est la valeur attendue de l'information de l'étude (EVSI). Concluez sur l'achat du test.
5. **Sensibilité.** En dessous de quelle probabilité *a priori* de demande forte le lancement direct cesse-t-il d'être rentable ? Et que faudrait-il pour que l'étude devienne intéressante ?

### Critères de réussite

- [ ] Lancement direct : E = 0,45 × 500 + 0,55 × (−200) = **115 k€** → sans étude, on **lance** (115 > 0)
- [ ] P(positif) = 0,45 × 0,80 + 0,55 × 0,30 = **0,525** ; P(négatif) = **0,475**
- [ ] P(forte | positif) = (0,45 × 0,80)/0,525 = 0,36/0,525 ≈ **0,686** ; P(forte | négatif) = 0,09/0,475 ≈ **0,189**
- [ ] E[lancer | positif] = (0,36 × 500 + 0,165 × (−200))/0,525 = 147/0,525 = **280 k€** (on lance)
- [ ] E[lancer | négatif] = (0,09 × 500 + 0,385 × (−200))/0,475 = −32/0,475 ≈ **−67,4 k€** (on **n'lance pas** → 0)
- [ ] Valeur au nœud étude = 0,525 × 280 + 0,475 × 0 = **147 k€** ; nette du coût = 147 − 40 = **107 k€**
- [ ] **EVSI** = 147 − 115 = **32 k€** ; comme 32 k€ < 40 k€ (coût), l'étude n'est **pas rentable**
- [ ] Politique optimale : **ne pas faire l'étude, lancer directement** (115 k€ > 107 k€)
- [ ] Sensibilité : le lancement direct devient nul quand 700·p − 200 = 0, soit **p ≈ 0,286** ; l'étude deviendrait intéressante si son coût passait **sous 32 k€** (ou si la fiabilité augmentait l'EVSI)

---

## Exercice 2 — Paradoxe d'Ellsberg : ambiguïté, axiome violé, et choix d'un fournisseur

### Objectif

Reproduire la structure du paradoxe d'**Ellsberg** sur des nombres neutres frais, démontrer **précisément** la violation (aversion à l'ambiguïté), puis transférer le raisonnement à une décision réelle neutre (choisir un fournisseur « sûr » vs « de fiabilité inconnue ») en distinguant quand l'aversion à l'ambiguïté est **défendable** et quand elle est un **piège**.

### Consigne

Une urne contient **90 jetons** : exactement **30 rouges**, et **60 jetons** qui sont **noirs ou jaunes en proportion inconnue** (de 0 à 60 de chaque). On tire un jeton au hasard. Chaque pari gagnant paie **100 €**, sinon 0 €.

- **Pari A** : gagne si le jeton est **rouge**
- **Pari B** : gagne si le jeton est **noir**
- **Pari C** : gagne si le jeton est **rouge ou jaune**
- **Pari D** : gagne si le jeton est **noir ou jaune**

La plupart des gens choisissent **A plutôt que B** (situation 1) **et** **D plutôt que C** (situation 2).

1. Calculez l'espérance de A et de D, qui ne dépendent pas de la composition inconnue. Pourquoi sont-elles connues exactement ?
2. Notez P(noir) = b (inconnu). Exprimez E[B] et E[C] en fonction de b (sachant P(rouge) = 1/3 et P(noir ou jaune) = 2/3).
3. Démontrez que le couple de choix « A ≻ B et D ≻ C » est **impossible** pour tout agent qui attribue une probabilité unique b : explicitez les deux inégalités contradictoires sur b. Nommez l'axiome violé (principe de la chose-sûre / indépendance) et le phénomène (aversion à l'ambiguïté).
4. **Transfert.** Une responsable achats hésite entre un fournisseur **F1** au taux de défaut **connu = 1/3** et un fournisseur **F2** au taux de défaut **inconnu** (pourrait être meilleur ou pire). Donnez **un cas où préférer F1 est défendable** et **un cas où c'est un piège**, et **une action** qui réduit l'ambiguïté au lieu de la fuir.

### Critères de réussite

- [ ] E[A] = P(rouge) × 100 = (1/3) × 100 ≈ **33,33 €** ; E[D] = P(noir ou jaune) × 100 = (2/3) × 100 ≈ **66,67 €** — connues car 30/90 et 60/90 sont fixés, indépendamment du mélange noir/jaune
- [ ] E[B] = **100·b €** et E[C] = (1/3 + (2/3 − b)) × 100 = **(1 − b)·100 €** (en utilisant P(jaune) = 2/3 − b)
- [ ] A ≻ B ⟺ 1/3 > b ; D ≻ C ⟺ 2/3 > (1 − b) ⟺ **b > 1/3** : les deux exigent à la fois b < 1/3 **et** b > 1/3 → **contradiction**
- [ ] L'axiome violé est nommé (principe de la chose-sûre / indépendance) : retirer le « jaune » commun aux deux options ne devrait pas inverser la préférence, or il l'inverse → **aversion à l'ambiguïté**
- [ ] Cas **défendable** : F2 risque d'être sélectionné défavorablement (le vendeur en sait plus, sélection adverse) ou le pire scénario est catastrophique/irréversible → préférer le connu est prudent
- [ ] Cas **piège** : l'ambiguïté est symétrique/bénigne et F2 a une espérance supérieure ; s'accrocher au « connu » fait renoncer à de la valeur par simple inconfort
- [ ] **Action** proposée : réduire l'ambiguïté à bas coût (commande pilote, audit, période d'essai) plutôt que d'écarter F2 par principe
