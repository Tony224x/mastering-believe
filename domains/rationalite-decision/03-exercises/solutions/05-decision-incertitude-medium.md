# Solutions Medium — Module 05 : Décision sous Incertitude

*Convention : montants en k€ pour l'exercice 1, en € pour les exercices 2 et 3. Les utilités √x sont arrondies à deux décimales ; les équivalents-certains tombent ici sur des valeurs exactes.*

---

## Exercice 1 — Arbre de décision en deux temps (prototype puis go/no-go)

**Structure de l'arbre** (de gauche à droite) :

```
□ Test ?
├── Oui (−10 k€)
│   └── ○ Signal
│       ├── Favorable (0,60) ── □ Lancer ? ── ○ succès (+200) / échec (−150)
│       └── Défavorable (0,40) ─ □ Lancer ? ── ○ succès (+200) / échec (−150)
└── Non
    └── □ Lancer directement ? ── ○ succès (+200) / échec (−150)
```

**Remontée par espérance — sous-décisions après le test :**

- E[lancer | favorable] = 0,80 × 200 + 0,20 × (−150) = 160 − 30 = **130 k€** → on lance (130 > 0).
- E[lancer | défavorable] = 0,30 × 200 + 0,70 × (−150) = 60 − 105 = **−45 k€** → on **abandonne** (0 > −45).

**Valeur du nœud hasard « signal » :**

```
Valeur = 0,60 × max(130, 0) + 0,40 × max(−45, 0)
       = 0,60 × 130 + 0,40 × 0
       = 78 k€
```

Nette du coût du test : 78 − 10 = **68 k€**.

**Branche sans test (lancement direct) :**

- P(succès) a priori = 0,60 × 0,80 + 0,40 × 0,30 = 0,48 + 0,12 = **0,60**.
- E[lancer direct] = 0,60 × 200 + 0,40 × (−150) = 120 − 60 = **60 k€** (> 0, donc on lancerait).

**Décision finale :** 68 k€ (avec test) > 60 k€ (sans test). **Politique optimale : faire le test ; lancer si favorable, abandonner si défavorable.**

**Valeur de l'information :** le nœud-test vaut **78 k€** brut contre **60 k€** sans test → l'information vaut **18 k€** brut. Le test coûte 10 k€, donc le **gain net** procuré par le test est 68 − 60 = **8 k€**. L'intérêt du test n'est pas de « gagner plus en moyenne » mais d'éviter les lancements dans les cas défavorables.

---

## Exercice 2 — Même pari, deux profils : neutre vs averse au risque

**Pari G** : 0,5 → 900 € ; 0,5 → 100 €. **Option sûre S** : 500 €.

**Agent neutre au risque** (compare les espérances monétaires) :

```
E[G] = 0,5 × 900 + 0,5 × 100 = 450 + 50 = 500 €
S    = 500 €
```

→ E[G] = S = 500 € : l'agent neutre est **indifférent**.

**Agent averse au risque**, U(x) = √x :

```
EU[G] = 0,5 × √900 + 0,5 × √100 = 0,5 × 30 + 0,5 × 10 = 15 + 5 = 20
EU[S] = √500 ≈ 22,36
```

→ EU[S] (22,36) > EU[G] (20) : l'agent averse **choisit l'option sûre S**, alors même que les deux ont la même espérance monétaire (500 €).

**Pourquoi c'est rationnel :** la fonction U est **concave**. Le gain de 400 € au-dessus de 500 (passer à 900) ajoute moins d'utilité que la perte de 400 € en dessous (descendre à 100) n'en retire. La dispersion du pari est donc pénalisée. Préférer le sûr n'est pas une erreur de calcul : c'est la conséquence cohérente d'une utilité marginale décroissante. (La maximisation de l'espérance monétaire n'est qu'un cas particulier — U linéaire ; le paradoxe de Saint-Pétersbourg montre d'ailleurs qu'elle peut donner des réponses absurdes.)

---

## Exercice 3 — Équivalent-certain et prime de risque (assurance d'un objet)

**Données :** vélo valant 1 000 € ; vol avec p = 0,10 (valeur 0), rien avec 0,90 (valeur 1 000). U(x) = √x.

**Étape 1 — espérance et perte espérée :**

```
E = 0,90 × 1 000 + 0,10 × 0 = 900 €
Perte espérée = 1 000 − 900 = 100 €   (= 0,10 × 1 000)
```

**Étape 2 — utilité espérée sans assurance :**

```
EU = 0,90 × √1000 + 0,10 × √0
   = 0,90 × 31,6228 + 0
   ≈ 28,46
```

**Étape 3 — équivalent-certain :** le patrimoine certain procurant la même utilité.

```
CE = (EU)² = (0,90 × √1000)² = 0,81 × 1000 = 810 €
```

**Étape 4 — prime de risque et prix d'assurance :**

```
Prime de risque         = E − CE      = 900 − 810 = 90 €
Prix max d'assurance     = 1 000 − CE  = 1 000 − 810 = 190 €
Prime actuariellement juste = perte espérée = 100 €
```

**Lecture :** l'agent averse accepte de payer jusqu'à **190 €** pour une couverture totale, alors que cette couverture ne « vaut » que **100 €** en espérance pour l'assureur. L'écart entre ce qu'il est prêt à payer et l'espérance de perte tient à sa **prime de risque** (90 €). C'est précisément cet écart qui crée une marge où l'assureur (neutre au risque, mutualisant beaucoup de polices) et l'assuré (averse) ont **tous deux** intérêt à signer : voilà pourquoi un marché de l'assurance existe.
