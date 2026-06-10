# Exercices Hard — Jour 7 : Mini-Transformer (Capstone)

---

## Exercice 7 : Forward pass complet d'un mini-GPT en NumPy + loss a l'init

### Objectif

Assembler le forward pass COMPLET d'un mini-GPT (tokens → logits → loss) en NumPy pur, et le valider avec le test le plus important du deep learning : la loss a l'initialisation.

### Consigne

1. Implementer un mini-GPT forward-only :
   - token embeddings `(vocab, d_model)` + positional embeddings `(block_size, d_model)`
   - 1 bloc decoder Pre-LN : LayerNorm → attention causale multi-tetes → residual → LayerNorm → FFN GELU → residual
   - LayerNorm final + lm_head **tie** avec la matrice d'embedding (logits = `x @ E.T`)
   - Config : `vocab=20, d_model=32, n_heads=4, d_ff=128, block_size=16`
   - Init : tous les poids `randn * 0.02` (la convention GPT-2), bias a 0

2. Implementer la cross-entropy autoregressive : pour un batch `(B, T)` de tokens, loss moyenne de `-log p(target)` sur toutes les positions, targets = inputs decales de 1.

3. **Test de la loss a l'init** : sur des tokens aleatoires, la loss doit valoir ~`ln(vocab) = ln(20) ≈ 3.0` (tolerance ±2%). Expliquer en commentaire pourquoi : a l'init les logits sont quasi nuls → distribution quasi uniforme. C'est le sanity check n°1 avant tout entrainement.

4. **Test de causalite de bout en bout** : changer le token d'entree en position t ne doit changer les logits qu'aux positions >= t (difference < 1e-12 avant t).

5. **Test de l'embedding tying** : verifier que monter artificiellement l'embedding du token k (E[k] += delta aligne avec le residual stream final) monte son logit partout — ou plus simple : verifier que `logits = x_final @ E.T` donne shape `(B, T, vocab)` et que le nombre total de parametres est reduit de `vocab * d_model` par rapport a une lm_head separee (compter les deux variantes).

### Criteres de reussite

- [ ] Le forward complet est en NumPy pur, chaque etape a un assert de shape
- [ ] Loss a l'init dans [0.98, 1.02] * ln(20)
- [ ] Causalite verifiee pour plusieurs positions t
- [ ] L'embedding tying est implemente et l'economie de parametres est chiffree
- [ ] La cross-entropy utilise le log-sum-exp stable (pas de `log(softmax)` naif) — verifie sur des logits extremes (±100) sans NaN
- [ ] Execution < 10 s

---

## Exercice 8 : Generation avec KV cache — equivalence exacte et comptage des FLOPs

### Objectif

Implementer la generation incrementale avec KV cache et prouver les deux affirmations cles : (1) le resultat est IDENTIQUE a la generation naive, (2) le cout par token passe de O(T²) a O(T).

### Consigne

En reutilisant le mini-GPT de l'exercice 7 (ou une version a 1 tete pour simplifier) :

1. `generate_naive(prompt, n_new)` : a chaque nouveau token, refaire le forward COMPLET sur toute la sequence et prendre les logits de la derniere position. Greedy decoding.

2. `generate_cached(prompt, n_new)` : maintenir un cache `K_cache, V_cache` par couche :
   - phase prefill : forward du prompt entier, stocker tous les K, V
   - phase decode : pour chaque nouveau token, calculer SEULEMENT q, k, v du dernier token, appender k, v au cache, attention du seul q contre tout le cache (pas de masque causal necessaire — pourquoi ? repondre en commentaire)
   - attention au positional embedding : le nouveau token est a la position `len(sequence)`, pas 0

3. **Test d'equivalence** : pour 5 prompts differents et n_new=20, les sequences generees par les deux methodes doivent etre IDENTIQUES token par token, et les logits du dernier step identiques a 1e-9 pres.

4. **Comptage des FLOPs** : instrumenter les deux versions avec un compteur de multiplications-additions des matmuls (compter `2*m*n*k` par matmul `(m,k)@(k,n)`).
   - Tableau : `n_new ∈ {10, 20, 40}` | FLOPs naive | FLOPs cached | ratio
   - Verifier que le ratio croit avec n_new (la naive est O(T²) par token, la cached O(T))

5. **Memoire du cache** : calculer la taille du cache en elements (`n_layers * 2 * T * d_model`) et l'afficher pour la config jouet, puis extrapoler la formule a un 7B (32 couches, d=4096, T=4096, fp16) pour retrouver ~2 GB.

### Criteres de reussite

- [ ] Equivalence exacte des tokens generes sur les 5 prompts (et logits a 1e-9)
- [ ] Le positional embedding est correctement indexe en mode decode (LE piege classique — un test le verifie)
- [ ] La reponse "pas de masque causal en decode" est correcte : le cache ne contient QUE du passe, le masque est implicite
- [ ] Le compteur de FLOPs montre un ratio croissant avec n_new, coherent avec O(T²) vs O(T)
- [ ] L'extrapolation memoire du cache 7B retombe sur ~2 GB (±20%)
- [ ] Execution < 30 s
