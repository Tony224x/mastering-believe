# Exercices Medium — Jour 5 : Attention Mechanism

---

## Exercice 4 : Scaled dot-product attention causale from scratch

### Objectif

Implementer l'attention scaled dot-product avec masque causal en version batchee, et verifier ses proprietes mathematiques (la ou un bug se voit immediatement).

### Consigne

1. Implementer :

```python
def causal_attention(Q, K, V):
    """Q, K, V: (batch, n_heads, T, d_k). Returns (output, weights).
    output: (batch, n_heads, T, d_k), weights: (batch, n_heads, T, T)."""
```

   Etapes : scores = `Q @ K^T / sqrt(d_k)` → masque causal (positions futures a `-inf` AVANT le softmax) → softmax sur le **dernier axe** → `weights @ V`.

2. Verifier 3 proprietes sur des inputs aleatoires (batch=2, heads=2, T=6, d_k=8, seed fixe) :
   - **Normalisation** : chaque ligne de `weights` somme a 1 (tolerance 1e-9)
   - **Causalite** : `weights[..., i, j] == 0` exactement pour tout `j > i`
   - **Non-fuite** : modifier `V[..., 5, :]` (dernier token) ne change PAS `output[..., :5, :]` (tolerance 1e-12)

3. **Effet du scaling** : pour `d_k = 64` et des Q, K ~ N(0,1) (T=16), comparer l'entropie moyenne des lignes de softmax avec et sans division par `sqrt(d_k)`. Calculer `H = -sum(p * log(p))` par ligne.

### Criteres de reussite

- [ ] L'implementation est batchee (aucune boucle Python sur batch/heads/T)
- [ ] Les 3 proprietes (normalisation, causalite, non-fuite) passent avec les tolerances indiquees
- [ ] Le masque est applique AVANT le softmax avec `-inf` (pas une multiplication par 0 apres)
- [ ] L'entropie sans scaling est nettement plus faible (softmax sature) — ecart > 1 nat — et l'explication (variance des scores = d_k) est en commentaire
- [ ] Les shapes de sortie sont verifiees par assert

---

## Exercice 5 : Multi-head — gymnastique de shapes split/merge

### Objectif

Maitriser le decoupage en tetes (la manipulation de tenseurs la plus piegeuse du Transformer) : reshape + transpose, et savoir predire chaque shape intermediaire.

### Consigne

1. **Predire sur papier** les shapes pour `batch=2, T=10, d_model=512, n_heads=8` :
   - apres projection `X @ W_Q` ; apres `split_heads` ; les scores d'attention ; apres `merge_heads`

2. Implementer :

```python
def split_heads(x, n_heads):
    """(batch, T, d_model) -> (batch, n_heads, T, d_head)"""

def merge_heads(x):
    """(batch, n_heads, T, d_head) -> (batch, T, d_model)"""
```

   Attention au piege : `split_heads` est `reshape` PUIS `transpose` — pas l'inverse.

3. Verifier :
   - `merge_heads(split_heads(x)) == x` **exactement** (round-trip parfait)
   - La version fausse `x.reshape(batch, n_heads, T, d_head)` (sans transpose) donne un resultat DIFFERENT — montrer qu'elle melange les timesteps en comparant `split_heads(x)[0, 0]` aux bonnes colonnes de `x`
   - Chaque tete recoit bien des colonnes contigues de `x` : `split_heads(x)[b, h, t] == x[b, t, h*d_head:(h+1)*d_head]`

4. Assembler un forward multi-head complet (projections Q, K, V aleatoires, attention causale de l'exercice 4, merge, projection de sortie W_O) et verifier la shape finale `(2, 10, 512)`.

### Criteres de reussite

- [ ] Les 4 shapes predites sur papier sont correctes
- [ ] Round-trip `merge(split(x)) == x` exact (difference nulle, pas juste < epsilon)
- [ ] Le test montre explicitement que reshape-sans-transpose est faux (les valeurs different)
- [ ] Le test de contiguite par tete passe pour toutes les tetes
- [ ] Le forward multi-head complet sort la bonne shape et chaque etape a un assert de shape

---

## Exercice 6 : Debugger une attention cassee

### Objectif

Identifier 3 bugs subtils dans une implementation d'attention — des bugs qui ne crashent pas mais produisent des resultats silencieusement faux.

### Consigne

Le code suivant contient **3 bugs**. Les trouver, expliquer le symptome de chacun, corriger :

```python
def attention_buggy(Q, K, V, causal=True):
    T, d_k = Q.shape
    scores = Q @ K.T / d_k                      # BUG ?
    weights = softmax(scores, axis=0)           # BUG ?
    if causal:
        mask = np.tril(np.ones((T, T)))
        weights = weights * mask                # BUG ?
    return weights @ V
```

1. Pour chaque bug, ecrire en commentaire : ce qui est faux, le symptome observable, le test qui le detecte
2. Ecrire `attention_fixed` et la comparer a une implementation de reference ecrite independamment
3. Construire 3 tests cibles, chacun ne detectant qu'UN bug :
   - test scaling : avec d_k=64 et Q=K, la diagonale doit dominer raisonnablement (comparer aux valeurs de reference < 1e-8)
   - test axe softmax : les LIGNES doivent sommer a 1, pas les colonnes
   - test masque : apres correction, les lignes restent normalisees (somme = 1) MEME avec le masque — ce qui est impossible si on masque apres le softmax sans renormaliser

### Criteres de reussite

- [ ] Les 3 bugs sont identifies : division par `d_k` au lieu de `sqrt(d_k)`, softmax sur l'axe 0 (colonnes) au lieu de -1 (lignes), masque applique APRES softmax (les poids ne somment plus a 1)
- [ ] `attention_fixed` correspond a la reference avec une difference max < 1e-8
- [ ] Chaque test cible echoue sur la version buggee et passe sur la version corrigee
- [ ] Le symptome de chaque bug est documente (ex : masque apres softmax → le modele "perd de la masse de probabilite" et l'output est sous-dimensionne)
- [ ] La causalite de `attention_fixed` est verifiee (token i insensible aux tokens > i)
