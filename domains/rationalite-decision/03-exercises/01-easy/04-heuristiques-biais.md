# Exercices — Module 04 : Heuristiques & Biais Cognitifs

---

## Exercice 1 — Ancrage numérique : l'estimation contaminée

### Objectif
Mesurer et corriger l'effet d'ancrage sur une estimation chiffrée.

### Consigne

Un vendeur annonce d'entrée de jeu que son appartement vaut **850 000 €**. Vous estimez ensuite sa valeur de marché à partir des éléments suivants :
- Surface : 68 m²
- Quartier : prix médian de 9 200 €/m² (source : base notariale publique).
- État général : quelques rénovations nécessaires (vous estimez une décote de 8 %).

**Questions** :
1. Estimez la valeur de l'appartement *sans regarder le prix annoncé* (couvrez-le ou imaginez qu'il n'a pas encore été annoncé). Notez votre estimation indépendante.
2. Calculez la valeur selon les données objectives : m² × prix médian × (1 − décote).
3. Comparez votre estimation indépendante (étape 1) et la valeur calculée (étape 2). Y a-t-il un écart avec le prix annoncé de 850 000 € ? Dans quelle direction ?
4. Proposez une contre-mesure concrète pour éviter l'ancrage dans une vraie négociation.

### Critères de réussite
- [ ] L'estimation indépendante (étape 1) est formulée *avant* de faire le calcul.
- [ ] Le calcul objectif (étape 2) est correct à ± 5 %.
- [ ] L'analyse de l'ancrage (étape 3) identifie la direction du biais (l'annonce tire vers le haut).
- [ ] La contre-mesure (étape 4) est concrète et applicable (ex. : calculer en aveugle, rechercher des comparables indépendants).

---

## Exercice 2 — Disponibilité vs statistiques : risques de transport

### Objectif
Distinguer la perception de risque (disponibilité) de la réalité statistique.

### Consigne

Après un accident d'avion couvert massivement dans les médias pendant 3 jours, un ami déclare : "Je ne prendrai plus jamais l'avion, c'est beaucoup trop dangereux. Je préfère prendre la voiture."

Les données officielles de l'Union européenne (AESA, 2023) indiquent :
- Mortalité en avion : **0,007 décès par milliard de passagers-km** (vols commerciaux UE).
- Mortalité en voiture : **4,2 décès par milliard de passagers-km** (routes UE).

**Questions** :
1. Calculez le rapport de risque (voiture / avion).
2. Expliquez en 2-3 phrases quel mécanisme cognitif explique la réaction de votre ami.
3. Proposez une façon de reformuler l'information pour aider votre ami à évaluer le risque plus précisément (sans le moraliser).

### Critères de réussite
- [ ] Le rapport de risque est calculé correctement (≈ 600× plus risqué en voiture par passager-km).
- [ ] Le mécanisme de disponibilité est identifié et expliqué sans jargon excessif.
- [ ] La reformulation proposée est neutre et utilise des chiffres comparatifs.

---

## Exercice 3 — Tâche de Wason : détecter le biais de confirmation

### Objectif
Identifier l'erreur de raisonnement dans la tâche de Wason (version abstraite) et comprendre la logique correcte.

### Consigne

On vous présente 4 cartes :

```
┌───┐  ┌───┐  ┌───┐  ┌───┐
│ A │  │ D │  │ 4 │  │ 7 │
└───┘  └───┘  └───┘  └───┘
```

Chaque carte a une lettre d'un côté et un chiffre de l'autre. La règle à tester : **"Si une carte a une voyelle d'un côté, elle a un chiffre pair de l'autre."**

**Questions** :
1. Quelles cartes faut-il retourner pour tester la règle ? Justifiez chaque carte (retourner / ne pas retourner, et pourquoi).
2. La majorité des gens choisissent A et 4. Pourquoi cette réponse révèle-t-elle un biais de confirmation ?
3. Expliquez pourquoi retourner le 4 n'apporte pas d'information utile pour tester la règle.

### Critères de réussite
- [ ] Les cartes correctes à retourner (A et 7) sont identifiées avec une justification logique.
- [ ] Le biais de confirmation est expliqué : on cherche à valider la règle plutôt qu'à la réfuter.
- [ ] L'explication du 4 est correcte : quelle que soit la lettre derrière (voyelle ou consonne), la règle n'est pas réfutée (elle dit "si voyelle → pair", pas "si pair → voyelle").
