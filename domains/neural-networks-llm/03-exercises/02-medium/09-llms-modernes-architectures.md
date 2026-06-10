# Exercices Medium — Jour 9 : Architectures modernes des LLMs

---

## Exercice 4 : RoPE from scratch — la propriete de position relative

### Objectif

Implementer RoPE (Rotary Positional Embedding) et prouver numeriquement SA propriete fondatrice : le produit scalaire q·k ne depend que de la position RELATIVE.

### Consigne

1. Implementer :

```python
def apply_rope(x, pos, theta_base=10000.0):
    """x: (d,) avec d pair. Tourne chaque paire (x[2i], x[2i+1])
    d'un angle pos * theta_i, avec theta_i = theta_base^(-2i/d)."""
```

   Rotation 2D par paire : `x'[2i] = x[2i]*cos(a) - x[2i+1]*sin(a)`, `x'[2i+1] = x[2i]*sin(a) + x[2i+1]*cos(a)` avec `a = pos * theta_i`.

2. Verifier les proprietes (d=8, vecteurs q, k aleatoires, seed fixe) :
   - **Conservation de la norme** : `||apply_rope(x, pos)|| == ||x||` pour tout pos (tolerance 1e-10) — c'est une rotation
   - **Position relative** : `apply_rope(q, m) . apply_rope(k, n)` ne depend que de `n - m`. Verifier pour l'offset 3 : les paires (m, n) ∈ {(0,3), (5,8), (11,14), (40,43)} donnent le MEME produit scalaire (tolerance 1e-8)
   - **Identite a la position 0** : `apply_rope(x, 0) == x`

3. Version batchee `apply_rope_batch(X)` pour `X: (T, d)` ou la ligne t est tournee a la position t (vectorisee, pas de boucle sur d).

4. Calculer la matrice `A[m, n] = rope(q, m) . rope(k, n)` pour T=16 avec q, k FIXES, et verifier que A est (quasi) constante sur chaque diagonale (std par diagonale < 1e-8).

5. Question (commentaire) : pourquoi applique-t-on RoPE a Q et K mais PAS a V ? (Le score d'attention doit dependre de la position relative ; la valeur transportee, non.)

### Criteres de reussite

- [ ] Les 3 proprietes (norme, position relative, identite) passent avec les tolerances
- [ ] La constance par diagonale de A est verifiee (std < 1e-8 par diagonale)
- [ ] La version batchee correspond exactement a la version 1-vecteur
- [ ] Les frequences theta_i sont correctes (geometriques de 1 a 1/theta_base)
- [ ] La reponse Q/K-mais-pas-V est correcte

---

## Exercice 5 : RMSNorm vs LayerNorm — les invariances qui les distinguent

### Objectif

Implementer RMSNorm et identifier par l'experience CE qui change vraiment par rapport a LayerNorm (et pourquoi LLaMA s'en contente).

### Consigne

1. Implementer les deux from scratch :
   - `layernorm(x, gamma, beta)` : `gamma * (x - mean) / sqrt(var + eps) + beta`
   - `rmsnorm(x, gamma)` : `gamma * x / sqrt(mean(x**2) + eps)` — PAS de centrage, PAS de beta

2. Tester les invariances sur des vecteurs aleatoires (d=16, gamma=1, beta=0) :
   - **Echelle** (x → 2x) : LayerNorm invariant ET RMSNorm invariant (tolerance 1e-9)
   - **Translation** (x → x + 3) : LayerNorm invariant, RMSNorm **NON** invariant — mesurer l'ecart
   - Conclure : la seule chose que RMSNorm abandonne est l'invariance par translation (le centrage)

3. Compter parametres et operations pour d=4096 :
   - parametres : LN = 2d, RMS = d (moitie)
   - operations element-wise (compter mentalement puis verifier) : LN calcule mean ET var (2 reductions), RMS une seule reduction
4. Verifier que pour une entree DEJA centree (mean=0), `rmsnorm(x, 1) == layernorm(x, 1, 0)` a 1e-9 pres — d'ou l'intuition : dans un grand reseau les activations sont ~centrees, le centrage est un cout sans benefice.

5. Micro-benchmark (`time.perf_counter`, 1000 appels sur (64, 4096)) : RMSNorm doit etre mesurablement plus rapide (rapport > 1.1x ; afficher le rapport).

### Criteres de reussite

- [ ] Les 2 implementations sont correctes (verifiees contre un calcul manuel sur un vecteur de 4 valeurs)
- [ ] Le tableau des invariances est etabli experimentalement (echelle: oui/oui, translation: oui/non)
- [ ] L'egalite LN==RMS sur entree centree est verifiee
- [ ] Le compte de parametres (2d vs d) et de reductions (2 vs 1) est explicite
- [ ] Le benchmark montre RMSNorm plus rapide et le resultat est affiche

---

## Exercice 6 : GQA from scratch — du MHA au MQA

### Objectif

Implementer la Grouped Query Attention generique (qui couvre MHA, GQA et MQA) et chiffrer ce qu'elle fait gagner sur le KV cache.

### Consigne

1. Implementer :

```python
def gqa_attention(Q, K, V, n_heads, n_kv_heads):
    """Q: (T, n_heads * d_head), K, V: (T, n_kv_heads * d_head).
    Chaque groupe de n_heads/n_kv_heads tetes de query partage
    la meme tete K/V (repetition des K/V, pas des Q). Causale."""
```

   Implementer la repetition par `repeat` des tetes K/V (chaque tete KV servie a `n_heads // n_kv_heads` tetes Q consecutives).

2. Tests d'equivalence (T=6, d_head=4, seed fixe) :
   - `n_kv_heads == n_heads` (MHA) : le resultat doit etre IDENTIQUE a une MHA standard ecrite separement (tolerance 1e-12)
   - `n_kv_heads == 1` (MQA) : toutes les tetes Q doivent recevoir les memes K/V — verifier que les poids d'attention de 2 tetes Q different uniquement a cause de leurs Q (forcer Q identiques sur 2 tetes → poids identiques)
   - cas invalide `n_heads % n_kv_heads != 0` → exception propre

3. Calculateur de KV cache : `kv_cache_bytes(n_layers, n_kv_heads, d_head, T, batch, bytes_per)` puis tableau pour un modele type LLaMA-2-70B (`n_layers=80, d_head=128, T=4096, batch=8, fp16`) :
   - MHA (64 tetes KV), GQA (8), MQA (1) — verifier le facteur exactement 8x entre MHA et GQA-8, 64x pour MQA

4. Question (commentaire) : pourquoi reduire les tetes KV plutot que les tetes Q ? (Le cache ne stocke QUE K et V ; la qualite vient surtout de la diversite des queries.)

### Criteres de reussite

- [ ] GQA(n_kv=n_heads) == MHA a 1e-12 (le test le plus important)
- [ ] Le partage des K/V par groupe est correct (teste via le cas Q-identiques en MQA)
- [ ] Le cas non divisible leve une erreur explicite
- [ ] Le tableau 70B donne ~2.7 GB (GQA-8) vs ~21.5 GB (MHA) — facteurs 8x/64x exacts
- [ ] La reponse "pourquoi KV et pas Q" est correcte
