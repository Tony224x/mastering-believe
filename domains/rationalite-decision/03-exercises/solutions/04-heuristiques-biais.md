# Solutions — Module 04 : Heuristiques & Biais Cognitifs

---

## Solution Exercice 1 — Ancrage numérique

**Étape 1 — Estimation indépendante** : sans le prix annoncé, l'apprenant doit noter son intuition brute. Toute valeur autour de 550 000-700 000 € basée sur le calcul est raisonnable selon les ajustements. L'important est que cette estimation soit faite *avant* le calcul.

**Étape 2 — Calcul objectif** :
```
Surface × prix médian × (1 − décote)
= 68 × 9 200 × (1 − 0,08)
= 68 × 9 200 × 0,92
= 625 760 × 0,92
= 575 699 €
≈ 576 000 €
```

**Étape 3 — Analyse de l'ancrage** :
Le prix annoncé (850 000 €) est supérieur de 47 % à la valeur calculée (≈ 576 000 €). L'ancrage fonctionne dans le sens de la surestimation : sans calcul rigoureux, une estimation faite "après" avoir vu 850 000 € tendra à être tirée vers le haut, même si l'acheteur croit avoir fait abstraction du chiffre.

**Étape 4 — Contre-mesure** :
- Calculer sa propre estimation *avant* de demander le prix ou d'ouvrir l'annonce.
- Rechercher des comparables indépendants (autres ventes dans le quartier à surface similaire) via une base notariale ou un agrégateur immobilier.
- Formuler la contre-offre basée sur le calcul objectif, pas sur un "pourcentage de réduction" par rapport au prix annoncé (ce qui resterait ancré dessus).

---

## Solution Exercice 2 — Disponibilité vs statistiques

**Étape 1 — Rapport de risque** :
```
Risque voiture / Risque avion = 4,2 / 0,007 = 600
```
La voiture est environ **600 fois plus mortelle par passager-km** que l'avion sur les routes et vols UE.

**Étape 2 — Mécanisme cognitif** :
L'ami subit le biais de disponibilité : la couverture médiatique intense de l'accident d'avion a rendu ce type d'événement très "disponible" mentalement (facile à évoquer, vivace, émotionnellement marquant). Comme notre cerveau estime la fréquence d'un événement par la facilité à en rappeler des exemples, la probabilité perçue de mourir en avion est massivement sur-estimée. À l'inverse, mourir en voiture est banal, rarement couvert, et donc sous-estimé.

**Étape 3 — Reformulation neutre** :
"D'après les données officielles de l'AESA, prendre la voiture sur 1 000 km est statistiquement environ 600 fois plus risqué que de faire le même trajet en avion. Si tu veux réduire ton risque de transport, la voiture est en fait le moins bon choix. Bien sûr, c'est une décision personnelle, mais les chiffres vont dans le sens inverse de ton intuition."

*Note* : ne pas moraliser ("tu as tort"), mais offrir les données pour que la personne raisonne elle-même.

---

## Solution Exercice 3 — Tâche de Wason

**Étape 1 — Cartes à retourner : A et 7**

- **A (voyelle)** : à retourner. Si l'autre côté est un chiffre *impair*, la règle est réfutée.
- **D (consonne)** : ne pas retourner. La règle dit "si voyelle → pair". D étant une consonne, peu importe ce qu'il y a derrière.
- **4 (pair)** : ne pas retourner. La règle dit "si voyelle → pair", *pas* "si pair → voyelle". Si derrière le 4 il y a une consonne, la règle n'est pas violée. Si c'est une voyelle, on confirme — mais la confirmation n'est pas le but d'un test logique.
- **7 (impair)** : à retourner. Si l'autre côté est une voyelle, alors la règle est réfutée (on a une voyelle avec un impair derrière, ce qui viole "si voyelle → pair").

**Étape 2 — Biais de confirmation**
La majorité choisit A et 4 car A semble "évidemment" lié à la règle (voyelle → on la vérifie) et 4 "confirme" la règle si on y trouve une voyelle. C'est le biais de confirmation : on cherche des cas qui valident la règle, pas des cas qui pourraient la réfuter. La logique correcte est *falsificationniste* : une règle ne se teste pas en cherchant à la confirmer, mais en cherchant l'exception qui la contredit.

**Étape 3 — Pourquoi le 4 est inutile**
La règle est conditionnelle dans un seul sens : "si voyelle → pair". Elle n'affirme pas "si pair → voyelle" (la réciproque). Retourner le 4 peut révéler une voyelle (confirmation) ou une consonne (pas d'information). Dans les deux cas, la règle n'est pas réfutée. La carte 4 ne peut jamais prouver que la règle est fausse — elle est donc inutile pour *tester* la règle.
