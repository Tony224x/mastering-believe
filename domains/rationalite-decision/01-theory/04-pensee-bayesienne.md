# Module 04 — Pensée bayésienne

> **Temps estimé** : 45 min | **Prérequis** : Modules 01-03
> **Objectif** : Maîtriser la notation formelle du théorème de Bayes et la mise à jour itérative — le posterior d'une étape devient le prior de la suivante — pour accumuler des preuves de façon cohérente et quantitative.

---

## 1. Avant la formule : une intuition en trois actes

Vous êtes dans un appartement avec trois colocataires (A, B, C). Ce matin, la lumière du salon est allumée. Qui l'a oubliée ?

- **Acte 1 — Prior** : sans autre information, vous estimez P(A) = 1/3, P(B) = 1/3, P(C) = 1/3.
- **Acte 2 — Preuve 1** : A dort chez ses parents ce week-end. Il ne peut pas être l'auteur. Mise à jour : P(A) = 0, P(B) = 1/2, P(C) = 1/2.
- **Acte 3 — Preuve 2** : B était le dernier à regarder la télé hier soir (témoignage de C). Mise à jour : P(B) monte, P(C) descend.

Ce processus — *prior + preuve → posterior → nouveau prior + preuve → …* — est le **raisonnement bayésien itératif**. Le module 03 a posé les bases (prior/vraisemblance/posterior). Ce module ajoute la **notation rigoureuse** et la **dynamique de mise à jour en séquence**.

---

## 2. La notation formelle de Bayes

### Vocabulaire fixé

| Symbole | Nom | Signification |
|---------|-----|---------------|
| H | Hypothèse | Ce qu'on cherche à évaluer (ex. : "la lumière est l'œuvre de B") |
| E | Evidence (preuve) | Observation disponible (ex. : "B était le dernier à regarder la TV") |
| P(H) | Prior | Probabilité de H *avant* d'observer E |
| P(E\|H) | Vraisemblance | Probabilité de E *si* H est vraie |
| P(E\|¬H) | Vraisemblance complémentaire | Probabilité de E *si* H est fausse |
| P(H\|E) | Posterior | Probabilité de H *après* avoir observé E |

### La formule complète

$$P(H \mid E) = \frac{P(E \mid H) \cdot P(H)}{P(E)}$$

Le dénominateur P(E) est la **probabilité totale de la preuve** :

$$P(E) = P(E \mid H) \cdot P(H) + P(E \mid \lnot H) \cdot P(\lnot H)$$

Ce qui donne la forme développée, directement calculable :

$$\boxed{P(H \mid E) = \frac{P(E \mid H) \cdot P(H)}{P(E \mid H) \cdot P(H) + P(E \mid \lnot H) \cdot (1 - P(H))}}$$

> **À retenir** : le numérateur est *ce que Bayes favorise* ; le dénominateur normalise pour que le résultat soit bien une probabilité entre 0 et 1.

### La forme odds (souvent plus rapide)

Les **odds** (cotes) d'une hypothèse : Odds(H) = P(H) / P(¬H).

La mise à jour devient **multiplicative** :

$$\text{Odds}(H \mid E) = \text{Odds}(H) \times LR$$

avec le **rapport de vraisemblance** : LR = P(E|H) / P(E|¬H).

Avantage : pas de dénominateur à recalculer. Il suffit de multiplier, puis de reconvertir les odds en probabilité.

---

## 3. Exemple chiffré : l'urne et les billes

> Deux urnes. Urne A : 70 % rouge, 30 % bleue. Urne B : 40 % rouge, 60 % bleue. On choisit une urne au hasard (50/50) et on tire une bille. Elle est rouge.

**Première mise à jour (bille rouge) :**

| Étape | Calcul | Résultat |
|-------|--------|----------|
| Prior P(A) | — | 0,50 |
| P(rouge\|A) | — | 0,70 |
| P(rouge\|B) | — | 0,40 |
| P(rouge) | 0,70 × 0,50 + 0,40 × 0,50 | 0,55 |
| Posterior P(A\|rouge) | 0,70 × 0,50 / 0,55 | **0,636 ≈ 64 %** |

**Deuxième mise à jour (deuxième bille rouge, tirée de la même urne) :**

Le posterior 64 % *devient* le nouveau prior.

| Étape | Calcul | Résultat |
|-------|--------|----------|
| Prior P(A) | (posterior précédent) | 0,636 |
| P(rouge) | 0,70 × 0,636 + 0,40 × 0,364 | 0,590 |
| Posterior P(A\|2e rouge) | 0,70 × 0,636 / 0,590 | **0,754 ≈ 75 %** |

Après deux billes rouges, la confiance en l'urne A passe de 50 % → 64 % → 75 %. Chaque preuve *s'accumule* sans repartir de zéro.

---

## 4. Mise à jour itérative : le cœur du module

### Principe

```
Prior₀  →  [preuve₁]  →  Posterior₁
Posterior₁  →  [preuve₂]  →  Posterior₂
Posterior₂  →  [preuve₃]  →  Posterior₃
...
```

Le posterior encode **toute l'histoire des preuves** jusqu'ici. Quand une nouvelle preuve arrive, on n'a pas besoin de se souvenir de chaque observation individuelle : le prior courant suffit.

### Propriété d'ordre : les preuves indépendantes commutent

Si les preuves E₁ et E₂ sont **indépendantes** conditionnellement à H, leur ordre d'application ne change pas le posterior final. Observer E₁ puis E₂ donne le même résultat qu'observer E₂ puis E₁.

Ce n'est pas toujours vrai si les preuves sont corrélées — c'est une limite importante à garder en tête.

### Exemple : test médical répété

Maladie à 1 % de prévalence. Sensibilité 90 %, spécificité 95 %. Deux tests positifs indépendants.

| Étape | Prior | Preuve | Posterior |
|-------|-------|--------|-----------|
| Avant tout test | 1,0 % | — | — |
| Après test 1 (+) | 1,0 % | rouge (sens 90 %, 1-spec 5 %) | **15,4 %** |
| Après test 2 (+) | 15,4 % | même test | **76,6 %** |

De 1 % à 77 % en deux tests positifs. Sans la mise à jour itérative, on serait tenté de doubler le résultat du premier test — erreur majeure.

> **À retenir** : la mise à jour itérative est la procédure formelle pour "changer d'avis proportionnellement aux preuves". On ne repart jamais de zéro : toutes les preuves passées sont encodées dans le prior courant.

---

## 5. Rapport de vraisemblance : mesurer la force d'une preuve

$$LR = \frac{P(E \mid H)}{P(E \mid \lnot H)}$$

| LR | Interprétation |
|----|---------------|
| > 10 | Preuve forte pour H |
| 2–10 | Preuve modérée pour H |
| 1 | Preuve non informative |
| 0,1–0,5 | Preuve contre H |
| < 0,1 | Preuve forte contre H |

Exemple : test médical (sensibilité 90 %, spécificité 95 %).

```
LR(+) = 0,90 / 0,05 = 18   → preuve forte pour H
LR(−) = 0,10 / 0,95 ≈ 0,11 → preuve contre H (test négatif rassure)
```

Un LR de 18 ne dit pas que la maladie est probable — il dit que l'information apportée par le test est forte. Sa valeur réelle dépend du prior.

---

## 6. Pièges classiques

**Négliger le prior (base rate neglect)** : croire que LR = 18 suffit à affirmer que P(malade) ≈ 95 %, sans tenir compte d'un prior de 1 %. Le calcul corrige cette intuition.

**Confondre P(E|H) et P(H|E)** : "si le test est positif, je suis malade" (P(H|E)) n'est pas la même chose que "si je suis malade, le test sera positif" (P(E|H) = sensibilité). Cette confusion s'appelle le **sophisme du procureur** (*prosecutor's fallacy*).

**Mettre à jour avec des preuves corrélées** : si deux tests utilisent le même biomarqueur, leurs résultats ne sont pas indépendants — les appliquer comme deux preuves séparées surestime la confiance.

**Choisir un prior trop extrême** : un prior à 0 ou 1 est irrécupérable par Bayes — aucune preuve ne peut le modifier. En pratique, garder des valeurs entre 0,001 et 0,999.

---

## Flash-cards (5)

**Q1 : Écris la formule développée du théorème de Bayes.**
> R : P(H|E) = [P(E|H) × P(H)] / [P(E|H) × P(H) + P(E|¬H) × (1 − P(H))].

**Q2 : Qu'est-ce que le rapport de vraisemblance (LR) et que signifie LR = 1 ?**
> R : LR = P(E|H) / P(E|¬H). LR = 1 signifie que la preuve est également probable si H est vraie ou fausse : elle n'apporte aucune information.

**Q3 : Comment fonctionne la mise à jour itérative ?**
> R : Chaque posterior devient le prior de la mise à jour suivante. L'historique complet des preuves est encodé dans le prior courant — on n'a pas besoin de le rejouer.

**Q4 : Qu'est-ce que le sophisme du procureur ?**
> R : Confondre P(H|E) et P(E|H). Exemple : confondre "probabilité d'être malade sachant le test positif" avec "probabilité d'avoir un test positif si on est malade" (sensibilité).

**Q5 : Pourquoi un prior de 0 ou 1 pose-t-il problème en raisonnement bayésien ?**
> R : Un prior de 0 ou 1 ne peut jamais être modifié par une preuve — le numérateur ou le dénominateur de Bayes s'annule. En pratique, garder des valeurs intermédiaires.

---

## Points clés à retenir

- La formule de Bayes en forme développée : P(H|E) = [P(E|H) × P(H)] / [P(E|H) × P(H) + P(E|¬H) × (1−P(H))].
- La forme odds est plus rapide : Odds(H|E) = Odds(H) × LR.
- **Mise à jour itérative** : posterior → nouveau prior → prochain posterior. Les preuves s'accumulent sans repartir de zéro.
- Le rapport de vraisemblance LR mesure la *force informative* d'une preuve, indépendamment du prior.
- Piège majeur : confondre P(E|H) et P(H|E) — le sophisme du procureur.
- Deux preuves indépendantes peuvent s'appliquer dans n'importe quel ordre ; des preuves corrélées ne le peuvent pas.

---

## Pour aller plus loin

- **Peterson, M.** (2017). *An Introduction to Decision Theory* (2e éd.). Cambridge University Press. https://www.cambridge.org/core/books/an-introduction-to-decision-theory/B9EEB3DCE5D0CAFFB6F3F30B1D0A06A6 — Chapitres 3-4 : probabilité bayésienne, prior/posterior, mise à jour itérative.
- **Stanford Encyclopedia of Philosophy** — Normative Theories of Rational Choice. https://plato.stanford.edu/entries/rationality-normative-utility/ — Cadre bayésien de la décision, exposé rigoureux et libre d'accès.
- **Script interactif** : `02-code/04-pensee-bayesienne.py` — mise à jour séquentielle avec plusieurs scénarios (urne, lumière oubliée, contrôle qualité).
