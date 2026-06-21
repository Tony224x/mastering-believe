# Solutions — Module 04 : Pensée bayésienne

> Corrigé chiffré complet. Lire APRÈS avoir tenté les exercices.

---

## Exercice 1 — Application directe de Bayes

### 1. Identification des termes

| Terme | Valeur |
|-------|--------|
| Prior P(R) | 0,50 (uniforme, on ne sait pas) |
| Vraisemblance P(rouge\|R) | 0,80 (urne type R = 80 % rouge) |
| Vraisemblance complémentaire P(rouge\|B) | 0,20 (urne type B = 20 % rouge) |

### 2. Probabilité totale de la preuve

$$P(\text{rouge}) = P(\text{rouge}|R) \times P(R) + P(\text{rouge}|B) \times P(B)$$
$$= 0{,}80 \times 0{,}50 + 0{,}20 \times 0{,}50 = 0{,}40 + 0{,}10 = 0{,}50$$

### 3. Posterior

$$P(R|\text{rouge}) = \frac{P(\text{rouge}|R) \times P(R)}{P(\text{rouge})} = \frac{0{,}80 \times 0{,}50}{0{,}50} = \frac{0{,}40}{0{,}50} = \mathbf{0{,}80 = 80\,\%}$$

### 4. Interprétation

La bille rouge fait passer la confiance en l'urne type R de **50 % à 80 %**. L'observation est informative : LR = 0,80/0,20 = 4. Une preuve avec LR = 4 est une preuve modérée. Elle ne donne pas la certitude, mais elle déplace clairement la croyance.

---

## Exercice 2 — Mise à jour itérative

### Étape A — Premier défaut

Prior : P(L1) = 0,70.

$$P(E) = 0{,}05 \times 0{,}70 + 0{,}15 \times 0{,}30 = 0{,}035 + 0{,}045 = 0{,}080$$

$$P(L1|\text{défaut}_1) = \frac{0{,}05 \times 0{,}70}{0{,}080} = \frac{0{,}035}{0{,}080} = \mathbf{0{,}4375 \approx 43{,}75\,\%}$$

### Étape B — Deuxième défaut (posterior → nouveau prior)

Nouveau prior : P(L1) = 0,4375.

$$P(E) = 0{,}05 \times 0{,}4375 + 0{,}15 \times 0{,}5625 = 0{,}021875 + 0{,}084375 = 0{,}10625$$

$$P(L1|\text{défaut}_1,\text{défaut}_2) = \frac{0{,}05 \times 0{,}4375}{0{,}10625} = \frac{0{,}021875}{0{,}10625} = \mathbf{0{,}2059 \approx 20{,}6\,\%}$$

### Étape C — LR et mécanique itérative

$$LR = \frac{P(\text{défaut}|L1)}{P(\text{défaut}|L2)} = \frac{0{,}05}{0{,}15} = \mathbf{0{,}333}$$

**Explication** : LR = 0,33 signifie qu'un défaut est 3 fois plus probable si la pièce vient de L2 que de L1 — chaque pièce défectueuse observée multiplie les odds(L1) par 0,33, les faisant baisser à chaque étape.

**Récapitulatif des états de croyance :**

| État | P(L1) |
|------|-------|
| Prior initial | 70,00 % |
| Après défaut 1 | 43,75 % |
| Après défaut 2 | 20,59 % |

---

## Exercice 3 — Rapport de vraisemblance et sophisme

### Partie A — LR et classement

**LR₁** (F1 vs F2 comme référence) :
$$LR_1 = \frac{P(\text{défaut}|F1)}{P(\text{défaut}|F2)} = \frac{0{,}02}{0{,}10} = \mathbf{0{,}20}$$

LR₁ < 1 : un défaut est moins probable si la source est F1 → défaut défavorise F1 par rapport à F2.

**LR₂** (F3 vs F2 comme référence) :
$$LR_2 = \frac{P(\text{défaut}|F3)}{P(\text{défaut}|F2)} = \frac{0{,}05}{0{,}10} = \mathbf{0{,}50}$$

LR₂ < 1 : un défaut est moins probable si la source est F3 → défaut défavorise F3 par rapport à F2.

**Classement après un défaut (prior uniforme 1/3) :**

$$P(E) = 0{,}02 \times \frac{1}{3} + 0{,}10 \times \frac{1}{3} + 0{,}05 \times \frac{1}{3} = \frac{0{,}17}{3} \approx 0{,}0567$$

| Fournisseur | Posterior |
|-------------|-----------|
| **F2** | 0,10 / (3 × 0,0567) ≈ **58,8 %** |
| F3 | 0,05 / (3 × 0,0567) ≈ **29,4 %** |
| F1 | 0,02 / (3 × 0,0567) ≈ **11,8 %** |

Le défaut désigne F2 comme source la plus probable (58,8 %), même si le prior était uniforme.

### Partie B — Sophisme du procureur

**L'erreur** : le responsable confond P(E|H) et P(H|E).

Il a utilisé **P(défaut|F2) = 10 %** (la vraisemblance — probabilité d'observer un défaut *si* F2 est la source) et l'a lu comme **P(F2|défaut)** (le posterior — probabilité que F2 soit la source *sachant* qu'il y a un défaut). C'est exactement le **sophisme du procureur** (*prosecutor's fallacy*).

**Calcul correct de P(F2|défaut) :**

Comme calculé ci-dessus : **P(F2|défaut) ≈ 58,8 %**, pas 10 %.

Le prior uniforme et la plus haute vraisemblance de F2 font monter le posterior à 58,8 %, soit presque 6 fois plus que l'affirmation incorrecte du responsable.

**La règle à retenir** : P(E|H) est ce qu'on observe dans les données de test ; P(H|E) est ce qu'on cherche à décider. Les confondre inverse le sens de la flèche causale.
