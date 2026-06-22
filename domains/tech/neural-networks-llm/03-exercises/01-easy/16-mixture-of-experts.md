# Exercices Faciles — Jour 16 : Mixture of Experts

---

## Exercice 1 : Top-k routing a la main

### Objectif

Calculer a la main la sortie d'un routeur MoE (softmax + top-k + renormalisation) pour internaliser la mecanique du gating.

### Consigne

Pour un token, le routeur a produit les logits suivants sur N=4 experts :

```
logits = [2.0, 1.0, 3.0, 0.0]
```

1. Calculer `softmax(logits)`. Donner les 4 probabilites arrondies a 4 decimales.
   Rappel : `softmax(z)_i = exp(z_i) / sum_j exp(z_j)`.

2. Faire un routing **top-2** : quels 2 experts sont selectionnes (indices) ?

3. **Renormaliser** les poids des 2 experts choisis pour qu'ils somment a 1. Donner les 2 poids renormalises.

4. La sortie du MoE est `output = w0 * expert_a(x) + w1 * expert_b(x)`. Avec `expert_2(x) = [10, 0]` et `expert_0(x) = [0, 10]` (les 2 experts choisis), calculer `output`.

5. Question conceptuelle : pourquoi renormalise-t-on les poids top-k au lieu de garder les probabilites softmax brutes ?

### Criteres de reussite

- [ ] softmax ≈ [0.2447, 0.0900, 0.6652, 0.0респ...] (verifier que la somme = 1 ; exp(2)=7.389, exp(1)=2.718, exp(3)=20.086, exp(0)=1 ; somme=31.19)
- [ ] top-2 = experts 2 et 0 (les 2 plus grands : 0.665 et 0.245)
- [ ] poids renormalises ≈ [0.731, 0.269] (0.6652/0.9099 et 0.2447/0.9099)
- [ ] output ≈ 0.731*[10,0] + 0.269*[0,10] = [7.31, 2.69]
- [ ] L'explication : sans renormalisation, la sortie serait attenuee (les poids ne somment pas a 1) ; renormaliser garde l'echelle de l'output stable quel que soit k

---

## Exercice 2 : Comptabilite Mixtral — params totaux vs actifs

### Objectif

Refaire le calcul "8x7B = 47B et pas 56B" et le ratio de sparsite, le coeur de l'argument MoE.

### Consigne

On considere une couche MoE simplifiee (FFN a 2 matrices up/down, on ignore SwiGLU) :

```
d_model = 4096
d_ff    = 14336
N       = 8 experts
k       = 2
layers  = 32
attn par layer = 4 * d_model * d_model   (Q, K, V, O)
ffn 1 expert   = 2 * d_model * d_ff
router         = d_model * N
```

1. Calculer les params d'**un** expert FFN (`ffn_1_expert`).

2. Calculer les params **totaux** d'une couche : `attn + N * ffn_1_expert + router`.

3. Calculer les params **actifs** d'une couche : `attn + k * ffn_1_expert + router` (seuls k experts s'activent par token).

4. Multiplier par `layers` pour les 2 (on ignore les embeddings pour cet exercice). Donner total et actif en milliards.

5. Calculer le **ratio de sparsite** = total / actif.

6. Question conceptuelle : pourquoi le nom commercial "8x7B" est-il trompeur, et pourquoi MoE n'economise-t-il PAS de VRAM ?

### Criteres de reussite

- [ ] ffn_1_expert = 2 * 4096 * 14336 ≈ 117.4 M
- [ ] total/layer = 4*4096^2 + 8*117.4M + 4096*8 ; actif/layer = 4*4096^2 + 2*117.4M + 4096*8
- [ ] total ≈ 32 G, actif ≈ 9.8 G (cohérent avec le NOTE du code : sous-estime le vrai Mixtral SwiGLU ~47B/13B)
- [ ] ratio de sparsite ≈ 3.3x (total / actif)
- [ ] L'explication : "8x7B" suggere 56B mais l'attention/embeddings/LayerNorm sont PARTAGES (≈47B reel) ; MoE economise les FLOPs (k/N), pas la VRAM (tous les experts doivent etre charges)

---

## Exercice 3 : Load balancing loss — uniforme vs collapse

### Objectif

Calculer la loss de Shazeer `L_aux = N * sum(f_i * P_i)` sur deux scenarios extremes pour comprendre ce qu'elle penalise.

### Consigne

N=4 experts. Top-1 routing, batch de 4 tokens.

**Scenario A (uniforme)** : chaque expert recoit exactement 1 token. `f = [1/4, 1/4, 1/4, 1/4]`. Le softmax moyen est aussi uniforme `P = [1/4, 1/4, 1/4, 1/4]`.

**Scenario B (collapse)** : les 4 tokens vont tous a l'expert 0. `f = [1, 0, 0, 0]`. Le softmax moyen est concentre `P = [0.85, 0.05, 0.05, 0.05]`.

1. Rappeler la definition de `f_i` (fraction de tokens routes vers l'expert i) et `P_i` (proba softmax moyenne attribuee a l'expert i).

2. Calculer `L_aux = N * sum_i (f_i * P_i)` pour le scenario A.

3. Calculer `L_aux` pour le scenario B.

4. Verifier : la loss est-elle minimale (=1) en uniforme et grande en collapse ? Quel est le max theorique de `L_aux` ?

5. Question conceptuelle : pourquoi le produit `f_i * P_i` et pas juste `f_i` ou juste `P_i` ? (Indice : differentiabilite.)

### Criteres de reussite

- [ ] Scenario A : L_aux = 4 * (4 * (1/4 * 1/4)) = 4 * 4 * 1/16 = 1.0
- [ ] Scenario B : L_aux = 4 * (1 * 0.85 + 0) = 4 * 0.85 = 3.4
- [ ] La loss = 1.0 en uniforme (minimum), grande en collapse ; max theorique = N = 4
- [ ] L'explication : f_i est non differentiable (passe par argmax/top-k), P_i seul ne contraint pas la charge reelle ; leur produit aligne le gradient de P (differentiable) sur la distribution dure f
