# Exercices Faciles — Jour 6 : Transformer Architecture

---

## Exercice 1 : Tracer les shapes a travers un bloc Transformer

### Objectif

Maitriser les transformations de shape a chaque etape d'un bloc Transformer.

### Consigne

Un bloc Transformer avec :
- `batch_size = 4`
- `seq_len = 10`
- `d_model = 128`
- `n_heads = 8`
- `d_ff = 512`

Donner la shape apres CHAQUE operation, en commencant par l'input.

1. Input : `(4, 10, 128)` — batch de 4 sequences de 10 tokens.

2. `LayerNorm(input)` → shape ?

3. `Linear(d_model → d_model)` (pour W_Q, W_K, W_V) → shape ?

4. Reshape en heads : `(batch, seq_len, n_heads, d_head)` → shape ? Que vaut `d_head` ?

5. Transpose pour passer a `(batch, n_heads, seq_len, d_head)` → shape ?

6. `Q @ K^T` (dans chaque tete, en parallele) → shape ? C'est la fameuse "attention matrix".

7. Apres softmax et multiplication par V : shape ?

8. Concat les heads : shape ?

9. Output projection W_O : shape ?

10. Residual + LayerNorm : shape ?

11. FFN : `Linear(d_model → d_ff) + GELU + Linear(d_ff → d_model)` → shapes intermediaires ?

12. Residual + LayerNorm final : shape ?

13. Question conceptuelle : pourquoi la shape finale est identique a l'input ? Pourquoi c'est indispensable ?

### Criteres de reussite

- [ ] Toutes les shapes sont correctes
- [ ] `d_head = d_model / n_heads = 16`
- [ ] La shape de la matrice d'attention est `(4, 8, 10, 10)` (batch, heads, seq, seq)
- [ ] La shape intermediate du FFN est `(4, 10, 512)` (le d_ff est plus grand)
- [ ] La shape finale = shape initiale = `(4, 10, 128)`
- [ ] L'explication : preserver la shape permet d'empiler N blocs identiques

---

## Exercice 2 : Residual connection — calcul manuel et intuition

### Objectif

Comprendre pourquoi les residuals resolvent le probleme du vanishing gradient dans un reseau profond.

### Consigne

Soit un "reseau" extremement simplifie, juste pour illustrer la mecanique du gradient :

```
Sans residual :
  y = f_3(f_2(f_1(x)))

Avec residual :
  y = x + f_3(x + f_2(x + f_1(x)))
```

Suppose que chaque `f_i` est une simple multiplication par un scalaire `0.5` (pour simplifier) :
- `f_1(x) = 0.5 * x`
- `f_2(x) = 0.5 * x`
- `f_3(x) = 0.5 * x`

1. **Sans residual** : calculer `y` en fonction de `x`. Que vaut `y / x` ? Que vaut `dy/dx` ?

2. **Sans residual, cas profond** : imagine 100 couches `f_i(x) = 0.5 * x`. Combien vaut `y / x` ? Combien vaut `dy/dx` ? Quel est le probleme ?

3. **Avec residual (1 couche)** : calculer `y = x + 0.5*x`. Que vaut `dy/dx` ?

4. **Avec residual, cas profond** : pour 100 couches de type `x_new = x + 0.5*x_old`, que devient `y/x` et `dy/dx` ? Compare avec le cas sans residual.

5. **Cas critique** : imagine que chaque `f_i` produise des valeurs tres petites (comme si le reseau apprenait une mini-correction). Exemple : `f_i(x) = 0.001 * x`.
   - Sans residual sur 100 couches : que vaut `y/x` ?
   - Avec residual sur 100 couches : que vaut `y/x` ?
   - Observation : les residuals permettent au modele d'apprendre "l'identite plus une petite correction" tres facilement.

6. Question conceptuelle : pourquoi on dit qu'avec les residuals, "la fonction par defaut d'une couche est l'identite" ? En quoi est-ce avantageux au debut de l'entrainement (quand les poids sont aleatoires) ?

### Criteres de reussite

- [ ] Sans residual, 3 couches : `y = 0.125 * x`, `dy/dx = 0.125`
- [ ] Sans residual, 100 couches : `y = 0.5^100 * x ≈ 8 * 10^-31 * x` (vanishing !)
- [ ] Avec residual, 100 couches : `y = 1.5^100 * x ≈ 4 * 10^17 * x` (explose, mais c'est un cas artificiel)
- [ ] Pour `f_i(x) = 0.001 * x` : sans residual on a ~0, avec residual on a ~1.1x (identite quasi preservee)
- [ ] L'explication : sans residual, meme des petites fonctions deviennent 0 apres de nombreuses couches. Avec residual, l'identite est le "point de depart" et le reseau apprend des corrections sur cette identite — bien plus facile a optimiser.

---

## Exercice 3 : LayerNorm vs BatchNorm — cas pratique

### Objectif

Voir concretement pourquoi LayerNorm est prefere dans les Transformers.

### Consigne

Soit un batch de 3 sequences, chacune avec 4 tokens, chaque token en dimension 3. Input tensor shape : `(3, 4, 3)`.

```
batch 0, seq 0 : [1.0, 2.0, 3.0]
batch 0, seq 1 : [4.0, 5.0, 6.0]
batch 0, seq 2 : [0.0, 0.0, 0.0]    (padding !)
batch 0, seq 3 : [0.0, 0.0, 0.0]    (padding !)

batch 1, seq 0 : [1.0, 1.0, 1.0]
batch 1, seq 1 : [2.0, 2.0, 2.0]
batch 1, seq 2 : [3.0, 3.0, 3.0]
batch 1, seq 3 : [4.0, 4.0, 4.0]

batch 2, seq 0 : [10.0, 20.0, 30.0]
batch 2, seq 1 : [0.0, 0.0, 0.0]    (padding !)
batch 2, seq 2 : [0.0, 0.0, 0.0]    (padding !)
batch 2, seq 3 : [0.0, 0.0, 0.0]    (padding !)
```

1. **LayerNorm** : la normalization se fait sur la dimension des features (axe 2, dim=3).
   - Pour le token (batch=0, seq=0) = `[1, 2, 3]`, calculer `mu`, `sigma`, et le vecteur normalise.
   - Pour le token (batch=2, seq=0) = `[10, 20, 30]`, meme calcul.
   - Observation : est-ce que les deux vecteurs normalises sont differents apres LayerNorm ?

2. **BatchNorm** : la normalization se fait sur la dimension batch (axe 0). Pour la position `(seq=0, dim=0)`, on normalise sur les 3 batches : `[1.0, 1.0, 10.0]`.
   - Calculer `mu` et `sigma` pour cette dimension.
   - Le resultat depend-il des autres batches ?
   - Question : que se passe-t-il si les paddings sont inclus dans le calcul (ex: pour seq=1, dim=0 avec valeurs `[4.0, 2.0, 0.0]`) ? Les paddings "contaminent" la statistique.

3. **Cas du batch=1 (inference autoregressive)** : si on genere un token a la fois, on a batch_size=1. Pour BatchNorm, `mu = x` et `sigma = 0` → division par zero. Comment BatchNorm gere ca habituellement ? (Indice : running statistics de l'entrainement).

4. **Le verdict** : lister 3 raisons concretes pour lesquelles LayerNorm est meilleur pour les Transformers.

### Criteres de reussite

- [ ] LayerNorm sur `[1, 2, 3]` : mu=2, sigma≈0.816, normalise ≈ `[-1.22, 0, 1.22]`
- [ ] LayerNorm sur `[10, 20, 30]` : mu=20, sigma≈8.16, normalise ≈ `[-1.22, 0, 1.22]`
- [ ] Observation : apres LayerNorm, les deux vecteurs sont IDENTIQUES (scale-invariant). C'est attendu et utile : LayerNorm normalise par instance.
- [ ] Les 3 raisons : (1) BatchNorm galere avec des longueurs variables et des paddings, (2) BatchNorm ne marche pas nativement avec batch=1 en inference, (3) BatchNorm melange des informations entre exemples (pas souhaitable).
