# Exercices Medium — Jour 6 : Transformer Architecture

---

## Exercice 4 : Bloc Transformer complet en NumPy (forward, pre-norm)

### Objectif

Assembler un bloc Transformer encoder complet (pre-norm, comme GPT-2+) en NumPy : MHA + residual + LayerNorm + FFN + residual + LayerNorm.

### Consigne

En t'appuyant sur `02-code/06-transformer-architecture.py` (mais en NumPy pur) :

1. Implementer `LayerNorm(x, gamma, beta, eps=1e-5)` qui normalise sur la derniere dimension (features).

2. Implementer `FeedForward(x, W1, b1, W2, b2)` : `Linear(d_model → d_ff) → GELU → Linear(d_ff → d_model)`. Utilise la GELU approximee (formule tanh du cours).

3. Assembler `transformer_block(x, params)` en variante **pre-norm** :
   ```
   x = x + MHA(LayerNorm(x))
   x = x + FFN(LayerNorm(x))
   ```
   (reutilise la MHA NumPy du jour 5, adaptee en batch `(B, T, d_model)` ou en `(T, d_model)`).

4. **Verifier la preservation de shape** : input `(seq, d_model)` → output `(seq, d_model)`. C'est ce qui permet d'empiler N blocs.

5. **Empiler N blocs** : appliquer 4 blocs d'affilee, verifier que la shape est preservee et que la norme de l'activation ne diverge pas (grace aux LayerNorm).

6. Question : pourquoi le pre-norm (LN avant le sous-module) est-il plus stable a entrainer que le post-norm (LN apres residual) pour les stacks profonds ?

### Criteres de reussite

- [ ] `LayerNorm` normalise correctement sur l'axe des features (mean=0, std=1 avant scale/shift)
- [ ] `FeedForward` est correct (expansion d_ff, GELU, contraction)
- [ ] Le bloc pre-norm est correct : `x = x + sublayer(LN(x))` (deux fois)
- [ ] La shape est preservee `(seq, d_model) → (seq, d_model)`
- [ ] Empiler 4 blocs preserve la shape et la norme reste bornee
- [ ] La reponse : en pre-norm, le chemin residuel reste "propre" (somme d'identites + corrections), le gradient circule sans etre re-normalise a chaque couche → stack profond entrainable sans warmup agressif

---

## Exercice 5 : LayerNorm backward + gradient check

### Objectif

Deriver et implementer le backward de LayerNorm (souvent mal compris) et le valider par gradient check.

### Consigne

LayerNorm sur un vecteur `x` de dimension `D` :
```
mu = mean(x)
var = mean((x - mu)^2)
x_hat = (x - mu) / sqrt(var + eps)
y = gamma * x_hat + beta
```

1. Implementer le forward avec cache.

2. **Deriver le backward**. Etant donne `dy`, calculer `dx`, `dgamma`, `dbeta`. Le point dur : `dx` doit tenir compte du fait que `mu` et `var` dependent de TOUS les `x_i`. La formule (par ligne, D = dim) :
   ```
   dx_hat = dy * gamma
   dvar = sum(dx_hat * (x - mu)) * (-0.5) * (var + eps)^(-3/2)
   dmu = sum(dx_hat * -1/sqrt(var+eps)) + dvar * mean(-2*(x - mu))
   dx = dx_hat / sqrt(var+eps) + dvar * 2*(x - mu)/D + dmu/D
   ```
   (ou utilise la forme compacte equivalente).

3. **Gradient check** : avec une loss `0.5*||y||^2`, compare `dx`, `dgamma`, `dbeta` aux gradients numeriques (eps=1e-5, erreur < 1e-5). Tester sur un batch `(B, D)`.

4. Question : pourquoi `dx` n'est-il PAS simplement `dy * gamma / sqrt(var+eps)` ? Quels termes manquent et pourquoi ?

### Criteres de reussite

- [ ] Le forward LayerNorm cache les bonnes valeurs (x_hat, mu, var, etc.)
- [ ] `dgamma = sum(dy * x_hat)`, `dbeta = sum(dy)` sont corrects (somme sur le batch)
- [ ] `dx` inclut bien les 3 termes (via x_hat, via var, via mu)
- [ ] Le gradient check passe pour dx, dgamma, dbeta (erreur < 1e-5)
- [ ] La reponse explique que mu et var sont fonctions de tous les x_i → le gradient doit "redescendre" a travers ces statistiques (termes dmu et dvar)

---

## Exercice 6 : Comptage exact des parametres d'un Transformer

### Objectif

Savoir compter exactement les parametres d'un Transformer et comprendre la repartition attention / FFN / embeddings.

### Consigne

Pour un Transformer GPT-style avec : `vocab_size=V`, `d_model=D`, `n_heads`, `d_ff=4D`, `n_layers=L`, positional embeddings appris de longueur `block_size=P`. (On suppose les Linear de l'attention SANS biais, les FFN AVEC biais, et le tied-embedding optionnel.)

1. **Par bloc**, compter :
   - Attention : `W_Q, W_K, W_V, W_O` (chacun `D x D`, sans biais) → `4 D^2`
   - FFN : `W1 (D x 4D) + b1 (4D) + W2 (4D x D) + b2 (D)` → `8 D^2 + 5D`
   - 2 LayerNorm : `2 * 2D = 4D` (gamma + beta)

2. **Total du stack** : `L * (params par bloc)`.

3. **Embeddings** : `token_emb (V x D) + pos_emb (P x D)` + LayerNorm finale `2D` + head de sortie `(D x V)` (si non tied).

4. **Appliquer a GPT-2 small** : `V=50257, D=768, n_heads=12, n_layers=12, P=1024`. Calculer le total et comparer aux ~124M reels.

5. **Ratio FFN/attention** : verifier que le FFN a ~2x plus de parametres que l'attention par bloc. Pourquoi `8 D^2` vs `4 D^2` ?

6. Question : pour GPT-2 small, quelle FRACTION des parametres est dans les embeddings ? Pourquoi cette fraction chute-t-elle quand le modele grandit (GPT-3 175B) ?

### Criteres de reussite

- [ ] Le comptage par bloc est exact : attention `4D^2`, FFN `8D^2 + 5D`, norms `4D`
- [ ] Le total GPT-2 small tombe proche de 124M (avec head tied : ~124M)
- [ ] Le ratio FFN/attention par bloc ≈ 2x est verifie et explique (d_ff = 4D → 2 matrices de 4D^2)
- [ ] La fraction embeddings de GPT-2 small est calculee (~30-40% avec token+pos, car V grand vs D petit)
- [ ] La reponse : les embeddings croissent en O(V*D) mais le stack en O(L*D^2) ; quand D et L grandissent, le stack domine
