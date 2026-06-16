# Exercices Hard — Jour 7 : Mini-Transformer (Capstone Week 1)

---

## Exercice 7 : KV-cache from scratch — equivalence prefill vs decode incrementale

### Objectif

Implementer le KV-cache d'un decoder-only (attention causale) en NumPy, prouver que la generation incrementale token-par-token (avec cache) produit EXACTEMENT les memes logits que le forward complet sur toute la sequence, et quantifier le gain de calcul.

### Consigne

1. Reprendre la `causal_self_attention` NumPy (Ex. 4 medium) et en faire une version **incrementale** :
   - `attn_step(x_t, cache, params, n_head)` ou `x_t` est le SEUL token courant `(d_model,)`.
   - Le cache stocke les `K` et `V` de tous les tokens precedents : on calcule `q_t, k_t, v_t` pour `x_t`, on APPEND `k_t, v_t` au cache, puis l'attention de `q_t` porte sur TOUT le cache (positions `0..t`).
   - Pas besoin de masque : par construction, le token `t` ne voit que `0..t` (les futurs ne sont pas encore dans le cache).

2. **Equivalence prefill vs decode** — le coeur de l'exercice :
   - Prefill : faire le forward complet sur une sequence `X (T, d_model)` avec attention causale masquee -> `O_full (T, d_model)`.
   - Decode : initialiser un cache vide, passer les tokens un par un via `attn_step`, accumuler les outputs -> `O_inc (T, d_model)`.
   - Verifier `max |O_full - O_inc| < 1e-10`. Le KV-cache est une optimisation EXACTE, pas une approximation.

3. **Au niveau du modele complet** : etendre au mini-GPT (embeddings + N blocs + head). Generer 20 tokens de deux facons — (a) re-forward de toute la sequence a chaque pas (naif), (b) avec KV-cache par bloc — et verifier que la suite de tokens (greedy) est identique.

4. **Cout de calcul** : pour generer `T` tokens a partir d'un prompt vide :
   - Sans cache : a l'etape `t` on recalcule l'attention sur `t` tokens -> total `O(T^2)` calculs d'attention (somme `1+2+...+T`).
   - Avec cache : a l'etape `t` on ne calcule que `q_t` contre `t` keys -> aussi `O(T^2)` pour l'attention MAIS on economise tout le recalcul des projections Q/K/V des tokens passes (qui devient `O(T)` au lieu de `O(T^2)`). Compter les multiplications Q/K/V projetees dans les deux cas et afficher le ratio.

5. Question : le KV-cache transforme la generation de "compute-bound" en "memory-bound". Pourquoi ? (Indice : a chaque pas de decode on ne fait qu'un tout petit matmul `(1, d) x (d, 3d)` mais on doit relire tout le cache `(t, d)` depuis la memoire.)

### Criteres de reussite

- [ ] `attn_step` met a jour le cache et fait l'attention de `q_t` sur tout `0..t`
- [ ] Equivalence prefill vs decode au niveau attention : `< 1e-10`
- [ ] Equivalence au niveau modele complet : meme suite de tokens greedy
- [ ] Le decompte des projections Q/K/V montre `O(T)` (cache) vs `O(T^2)` (naif)
- [ ] La reponse explique le passage compute-bound -> memory-bound (petit matmul, gros read du cache)

---

## Exercice 8 : Backprop du bloc Transformer + gradient check, et weight tying

### Objectif

Deriver et implementer le backward complet d'un mini-bloc (un seul Linear + LayerNorm + residual suffit pour la rigueur), valider par gradient check, puis mesurer l'effet du **weight tying** (embedding partage avec la tete de sortie) sur le compte de parametres.

### Consigne

1. **Backward de la cross-entropy + softmax** (deja vu J1/J7) : pour `p = softmax(z)` et target `y`, montrer que `dL/dz = p - onehot(y)`. Implementer `softmax_ce_backward(logits, target)`.

2. **Backward du LayerNorm** : pour `y = (x - mu) / sqrt(var + eps) * gamma + beta` sur le dernier axe de dimension `D`, deriver `dx` etant donne `dy`. La formule (par ligne) :
   ```
   x_hat = (x - mu) / sqrt(var + eps)
   dx_hat = dy * gamma
   dx = (1/sqrt(var+eps)) * (dx_hat - mean(dx_hat) - x_hat * mean(dx_hat * x_hat))
   ```
   Implementer `layernorm_backward(dy, cache)` et le verifier par difference finie (< 1e-5). Donner aussi `dgamma`, `dbeta`.

3. **Mini-reseau bout-en-bout** : `z = LayerNorm(x) @ W + b` ; `loss = cross_entropy(softmax(z), y)`. Backprop complet vers `dx, dW, db, dgamma, dbeta`. Gradient check de TOUS les parametres (< 1e-5).

4. **Residual** : ajouter un residual `out = x + LayerNorm(x) @ W` et montrer que `dx` recoit DEUX contributions (le chemin direct `+x` et le chemin a travers la couche). Verifier par gradient check. Expliquer pourquoi le residual fait que le gradient "ne s'eteint jamais completement" (au pire il vaut l'identite).

5. **Weight tying** : dans GPT-2, la table d'embedding token `(vocab, d_model)` et la matrice de la tete `(d_model, vocab)` partagent les memes poids (transposees). 
   - Calculer le nombre de parametres economises pour `vocab=50257, d_model=768`.
   - Quand les poids sont partages, le gradient total est la SOMME des gradients des deux usages (input embedding + output projection). Le verifier numeriquement sur un mini exemple (un meme `W` utilise comme embedding ET comme tete -> `dW = dW_embed + dW_head`).

### Criteres de reussite

- [ ] `dL/dz = p - onehot(y)` implemente et verifie
- [ ] `layernorm_backward` correct (formule a 3 termes), `dgamma/dbeta` corrects, gradient check < 1e-5
- [ ] Gradient check de `dx, dW, db, dgamma, dbeta` du mini-reseau (< 1e-5)
- [ ] Le residual donne 2 contributions a `dx`, verifie par gradient check
- [ ] Weight tying : params economises calcules (`vocab*d_model`), et `dW = dW_embed + dW_head` verifie numeriquement
