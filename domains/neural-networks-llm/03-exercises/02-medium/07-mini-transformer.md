# Exercices Medium — Jour 7 : Mini-Transformer (Capstone Week 1)

---

## Exercice 4 : Bloc Transformer complet en NumPy (attention causale + FFN + pre-norm)

### Objectif

Reimplementer en pur NumPy un bloc decoder-only complet (LayerNorm pre-norm + multi-head causal attention + FFN GELU + residuals) et verifier ses proprietes, en miroir de la classe `Block` de `02-code/07-mini-transformer.py`.

### Consigne

En te basant sur `02-code/07-mini-transformer.py` (architecture PyTorch a transposer en NumPy) :

1. Implementer les briques NumPy :
   - `layer_norm(x, gamma, beta, eps=1e-5)` sur le dernier axe (moyenne + variance population, `ddof=0`).
   - `gelu(x)` (approximation tanh de GPT-2 : `0.5 x (1 + tanh(sqrt(2/pi)(x + 0.044715 x^3)))`).
   - `causal_self_attention(X, W_qkv, W_o, n_head)` : une grosse projection `c_attn` produisant Q, K, V concatenes (shape `d_model -> 3*d_model`), split en tetes, scaled dot-product **masque causal**, concat, projection `W_o`.
   - `feed_forward(x, W1, b1, W2, b2)` : `Linear -> GELU -> Linear` avec `d_ff = 4 * d_model`.

2. Assembler `transformer_block(x, params)` en **pre-norm** (variante GPT-2+) :
   ```
   x = x + attn(ln1(x))
   x = x + ffn(ln2(x))
   ```

3. **Verifier les shapes** : pour `d_model=48, n_head=4, seq=10`, tracer la shape apres chaque etape (ln1, qkv, split en `(n_head, seq, head_dim)`, attention, concat, w_o, residual, ln2, ffn). Confirmer `head_dim = 12`.

4. **Verifier la causalite end-to-end** : modifier le token en position `t` ne doit PAS changer l'output des positions `< t`. Construire `X`, calculer l'output, perturber `X[t]`, recalculer, et verifier que `out[:t]` est inchange (ecart < 1e-10) alors que `out[t:]` change.

5. **Effet du residual** : montrer numeriquement qu'avec des poids initialises a ~0.02 (petite std), le bloc est proche de l'identite au depart (`||block(x) - x|| / ||x||` reste petit). Pourquoi le residual + petite init facilite-t-il l'entrainement d'un reseau profond ?

### Criteres de reussite

- [ ] `layer_norm`, `gelu`, `causal_self_attention`, `feed_forward` sont corrects
- [ ] Le bloc est bien pre-norm (`x + attn(ln(x))`, `x + ffn(ln(x))`)
- [ ] `head_dim = 12` et toutes les shapes intermediaires sont tracees
- [ ] La causalite end-to-end est verifiee : perturber `X[t]` laisse `out[:t]` inchange (< 1e-10)
- [ ] Le bloc est proche de l'identite a l'init (petite norme du delta), justifie par le residual

---

## Exercice 5 : Forward complet du mini-GPT + loss a l'initialisation

### Objectif

Empiler embeddings (token + position) + N blocs + projection finale en NumPy, calculer la cross-entropy next-token et verifier la valeur attendue de la loss a l'initialisation aleatoire.

### Consigne

1. Construire un mini-GPT NumPy : table d'embedding token `(vocab, d_model)`, embeddings positionnels appris `(block_size, d_model)`, `n_layer` blocs (ex. 4), LayerNorm final, `head` de projection `(d_model, vocab)`.

2. `forward(idx)` : `idx` est `(seq,)` d'ids ; renvoyer `logits (seq, vocab)`. Etapes : `tok_emb[idx] + pos_emb[:seq]`, passage dans les blocs, ln finale, `@ W_head`.

3. **Cross-entropy next-token** : implementer `cross_entropy(logits, targets)` (log-softmax stable) ou `targets[i]` est le token suivant `idx[i+1]`. Moyenner sur les positions.

4. **Loss attendue a l'init** : avec des poids aleatoires, le modele predit ~uniforme. Verifier que la loss initiale est proche de `log(vocab_size)` (l'entropie d'une distribution uniforme sur `vocab` classes). C'est le sanity check #1 de tout entrainement de LM.

5. **Sanity check #2 — overfit d'un batch** : prendre UNE seule sequence et faire 50-100 steps de descente de gradient (gradient numerique ou analytique simplifie sur `W_head` + embeddings) ; montrer que la loss sur cette sequence descend bien en-dessous de `log(vocab)`. Pourquoi "overfitter un batch" est le test minimal qui prouve que la boucle d'entrainement fonctionne ?

6. Question : si la loss initiale etait tres SUPERIEURE a `log(vocab)` (ex. 50), que pourrait-on en deduire sur l'initialisation des poids ?

### Criteres de reussite

- [ ] Le forward empile embeddings + blocs + ln + head et renvoie `(seq, vocab)`
- [ ] `cross_entropy` est stable (log-softmax) et correcte sur les positions next-token
- [ ] La loss a l'init est proche de `log(vocab_size)` (baseline uniforme)
- [ ] L'overfit d'un seul batch fait chuter la loss bien sous `log(vocab)`
- [ ] La reponse : loss >> log(vocab) a l'init = poids mal initialises (logits non bornes -> softmax sature)

---

## Exercice 6 : Sampling — temperature, top-k et top-p (nucleus)

### Objectif

Implementer les 3 strategies de sampling principales d'un LLM autoregressif et analyser leur effet sur la distribution de generation.

### Consigne

1. A partir de logits `(vocab,)`, implementer :
   - `softmax_temperature(logits, T)` : `softmax(logits / T)`. Gerer le cas limite `T -> 0` (greedy = one-hot sur l'argmax).
   - `top_k_filter(logits, k)` : garder seulement les `k` plus grands logits, mettre les autres a `-inf` avant softmax.
   - `top_p_filter(logits, p)` : nucleus sampling — trier par proba decroissante, garder le plus petit ensemble dont la masse cumulee depasse `p`, masquer le reste.

2. **Verifier** sur `logits = [2.0, 1.0, 0.5, 0.0, -1.0]` :
   - `T=1.0` : probas ~`[0.57, 0.21, 0.13, 0.08, 0.03]`.
   - `T=0.5` : distribution plus piquee ; `T=2.0` : plus plate. Mesurer l'entropie `H = -sum(p log p)` dans les 3 cas et verifier `H(T=0.5) < H(T=1) < H(T=2)`.
   - `top_k=2` : seuls les 2 premiers tokens ont une proba non nulle (somme = 1).
   - `top_p=0.9` : verifier quels tokens sont retenus et que leur masse depasse 0.9.

3. **Distribution empirique** : echantillonner 10000 tokens avec `np.random.choice` selon chaque distribution filtree et verifier que les frequences empiriques collent aux probas theoriques (ecart < 0.02 sur le token majoritaire).

4. **Interaction temperature + top-p** : appliquer top-p APRES la temperature. Montrer qu'augmenter `T` puis appliquer top-p=0.9 garde plus de tokens (la distribution est plus plate, donc plus de tokens passent le seuil).

5. Question : pourquoi top-p est-il souvent prefere a top-k ? (Indice : adaptativite au "pic" de la distribution — top-k garde toujours `k` tokens meme quand le modele est tres sur, top-p s'adapte.)

### Criteres de reussite

- [ ] `softmax_temperature` (avec cas `T->0` greedy), `top_k_filter`, `top_p_filter` sont corrects
- [ ] L'entropie decroit avec la temperature : `H(0.5) < H(1) < H(2)`
- [ ] top-k=2 laisse exactement 2 tokens, top-p=0.9 retient la nucleus correcte
- [ ] Les frequences empiriques (10000 tirages) collent aux probas theoriques
- [ ] La reponse : top-p est adaptatif (taille de nucleus variable selon la confiance du modele), top-k est fixe
