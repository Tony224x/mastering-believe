# Solutions — Module 07 : Décision sous incertitude

> Corrigé chiffré modèle. Lire les exercices avant de consulter ce fichier.

---

## Exercice 1 — Espérance mathématique : comparer deux jeux

### Question 1 — E[Alpha]

Sac de 10 billes : 5 rouges (p = 0,5), 3 bleues (p = 0,3), 2 vertes (p = 0,2).

```
E[Alpha] = 0,5 × 4 + 0,3 × 1 + 0,2 × (−5)
         = 2,00 + 0,30 − 1,00
         = +1,30 €
```

### Question 2 — E[Bêta]

Pièce équilibrée : pile (p = 0,5) → +3 €, face (p = 0,5) → −2 €.

```
E[Bêta] = 0,5 × 3 + 0,5 × (−2)
        = 1,50 − 1,00
        = +0,50 €
```

### Question 3 — Choix

E[Alpha] = +1,30 € > E[Bêta] = +0,50 €.

Pour maximiser l'espérance, **choisir le Jeu Alpha**.

### Bonus — 100 parties

L'espérance *par partie* reste identique (1,30 € pour Alpha, 0,50 € pour Bêta) — elle ne dépend pas du nombre de répétitions.
Le gain total attendu = n × E[une partie] :
- Alpha sur 100 parties : 100 × 1,30 = **130 €**
- Bêta sur 100 parties  : 100 × 0,50 = **50 €**

La loi des grands nombres garantit que le gain réel se rapproche de cette valeur à mesure que n augmente.

---

## Exercice 2 — Utilité espérée et aversion au risque

### Question 1 — Espérances monétaires

**Formule Fixe** : gain net certain = 0 €.

```
E[Fixe] = 0 €
```

**Formule Risquée** : 40 % × (+200 €) + 60 % × (−80 €).

```
E[Risquée] = 0,4 × 200 + 0,6 × (−80)
           = 80 − 48
           = +32 €
```

### Question 2 — Formule à espérance maximale

E[Risquée] = +32 € > E[Fixe] = 0 €. La **Formule Risquée** maximise l'espérance monétaire.

### Question 3 — Utilité espérée avec U(x) = √(x + 100)

Décalage : on ajoute 100 à chaque valeur avant d'appliquer √.

**Formule Fixe** (gain net = 0 €) :

```
EU[Fixe] = 1,0 × √(0 + 100) = √100 = 10,000
```

**Formule Risquée** :

```
EU[Risquée] = 0,4 × √(200 + 100) + 0,6 × √(−80 + 100)
            = 0,4 × √300          + 0,6 × √20
            = 0,4 × 17,321        + 0,6 × 4,472
            = 6,928               + 2,683
            = 9,611
```

### Question 4 — Préférence selon l'utilité espérée

EU[Fixe] = 10,000 > EU[Risquée] = 9,611.

Avec cette fonction d'utilité concave, l'agent préfère la **Formule Fixe**.

### Question 5 — Lecture économique

La différence illustre l'**aversion au risque** : bien que la Formule Risquée ait une espérance monétaire supérieure (+32 € vs 0 €), la concavité de U(x) = √x fait que la douleur de perdre 80 € pèse plus que le plaisir symétrique de gagner 200 €. L'agent préfère la certitude — même à espérance inférieure — car la valeur marginale des euros supplémentaires décroît.

---

## Exercice 3 — Arbre de décision : choisir un itinéraire

### Question 1 — Durée attendue

**Route Nord** (base 30 min) :

```
E[durée Nord] = 0,20 × (30 + 25) + 0,80 × 30
              = 0,20 × 55 + 0,80 × 30
              = 11 + 24
              = 35 min
```

**Route Sud** (base 40 min) :

```
E[durée Sud] = 0,10 × (40 + 40) + 0,90 × 40
             = 0,10 × 80 + 0,90 × 40
             = 8 + 36
             = 44 min
```

### Question 2 — Espérance de pénalité

Un retard > 15 min → 50 points de pénalité.

**Route Nord** :
- Travaux (+25 min) : 25 > 15 → pénalité = 50 pts
- Pas de travaux (0 min) : 0 ≤ 15 → pénalité = 0 pts

```
E[pénalité Nord] = 0,20 × 50 + 0,80 × 0 = 10 points
```

**Route Sud** :
- Accident (+40 min) : 40 > 15 → pénalité = 50 pts
- Pas d'accident (0 min) : 0 ≤ 15 → pénalité = 0 pts

```
E[pénalité Sud] = 0,10 × 50 + 0,90 × 0 = 5 points
```

### Question 3 — Arbre de décision

```
□ DÉCISION (choisir une route)
│
├── Route Nord (durée base 30 min)
│   ○ Hasard
│   ├── Travaux    p=0,20 → durée 55 min → pénalité 50 pts
│   └── Libre      p=0,80 → durée 30 min → pénalité  0 pts
│   E[pénalité] = 10 pts
│
└── Route Sud (durée base 40 min)
    ○ Hasard
    ├── Accident   p=0,10 → durée 80 min → pénalité 50 pts
    └── Libre      p=0,90 → durée 40 min → pénalité  0 pts
    E[pénalité] = 5 pts
```

Remontée : au nœud de décision, on choisit la branche avec l'espérance de pénalité **minimale**.

### Question 4 — Choix optimal

E[pénalité Route Sud] = 5 pts < E[pénalité Route Nord] = 10 pts.

**Choisir la Route Sud** pour minimiser l'espérance de pénalité.

Note : la Route Sud est plus longue en espérance de durée (44 min vs 35 min), mais son risque de pénalité est deux fois plus faible. Si l'objectif est de minimiser les pénalités, la Route Sud est strictement préférable.

### Bonus — Aversion aux pénalités

Si le livreur est très averse aux pénalités (par exemple, une pénalité déclenche un avertissement grave), il peut utiliser une fonction d'utilité convexe sur les pénalités — la *douleur* d'une pénalité est bien supérieure à sa valeur nominale. Dans ce cas, **la Route Sud reste encore plus clairement préférable** : elle a non seulement une espérance de pénalité plus basse, mais aussi une probabilité de pénalité plus faible (10 % vs 20 %). L'aversion à la pénalité renforcerait le choix de la Route Sud plutôt que de l'inverser.

(Pour qu'une aversion au risque puisse *inverser* le choix, il faudrait une situation où la route à espérance inférieure a aussi une variance plus faible — ce n'est pas le cas ici, les deux routes ont la même amplitude de pénalité possible.)
