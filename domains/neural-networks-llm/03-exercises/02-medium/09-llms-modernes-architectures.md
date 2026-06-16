# Exercices Medium — Jour 9 : LLMs modernes (RoPE, RMSNorm, SwiGLU, GQA)

---

## Exercice 4 : RoPE — prouver l'invariance par translation du produit scalaire

### Objectif

Reimplementer RoPE depuis `02-code/09-llms-modernes-architectures.py` et PROUVER numeriquement sa propriete fondamentale : `<RoPE(q, m), RoPE(k, n)>` ne depend que de la distance relative `(n - m)`, pas des positions absolues.

### Consigne

En te basant sur `02-code/09-llms-modernes-architectures.py` :

1. Reprendre `precompute_rope_frequencies(head_dim, max_seq_len, base)` et `apply_rope(x, cos, sin)`.

2. **Invariance par translation** : pour des vecteurs `q` et `k` FIXES, calculer `score(m, n) = <RoPE(q, m), RoPE(k, n)>` pour plusieurs couples de meme distance `d = n - m` (ex. (2,5), (7,10), (0,3) ont tous `d=3`). Verifier que tous donnent le MEME score (ecart < 1e-9).

3. **Decroissance avec la distance** : pour un `q == k` fixe, montrer que le score d'auto-attention tend (en moyenne sur des vecteurs aleatoires) a decroitre quand la distance augmente. C'est le "biais de localite" implicite de RoPE.

4. **Equivalence avec la rotation 2D** : verifier que pour une paire `(x_0, x_1)`, `apply_rope` est EXACTEMENT une rotation 2D d'angle `theta = m * freq_0`. Construire la matrice de rotation `[[cos,-sin],[sin,cos]]`, l'appliquer a la paire, et comparer (ecart < 1e-12).

5. **Effet de la base** : la base (10000 par defaut) controle la "longueur d'onde" des frequences. Calculer la periode de la frequence la plus lente pour `base ∈ {10000, 500000}`. Pourquoi augmenter la base (comme LLaMA-3 a 500000) aide-t-il a etendre le contexte ?

### Criteres de reussite

- [ ] `apply_rope` correct (rotation des paires interleaved)
- [ ] Invariance par translation verifiee : meme distance -> meme score (< 1e-9)
- [ ] La decroissance moyenne du score avec la distance est montree
- [ ] Equivalence avec la rotation 2D explicite verifiee (< 1e-12)
- [ ] L'effet de la base sur la longueur d'onde est calcule et explique (base plus grande = contexte plus long)

---

## Exercice 5 : RMSNorm vs LayerNorm — equivalence, cout et gradient

### Objectif

Comparer RMSNorm et LayerNorm en profondeur : statistiques de sortie, compte d'operations, et derivee de RMSNorm.

### Consigne

1. Reprendre `layer_norm` et `rms_norm` du code. Implementer la version avec scale apprise `gamma` pour les deux.

2. **Statistiques de sortie** : sur un input decentre (`mean != 0`), verifier que :
   - LayerNorm : sortie de moyenne 0 et std 1 par ligne.
   - RMSNorm : sortie de RMS 1 par ligne, mais moyenne potentiellement NON nulle.
   - Conclure : RMSNorm normalise l'amplitude mais ne centre pas.

3. **Cout en operations** : compter (approximativement) les operations par ligne de dimension `D`. LayerNorm calcule moyenne PUIS variance (2 passes) ; RMSNorm calcule seulement la moyenne des carres (1 passe, pas de soustraction). Quel pourcentage d'operations RMSNorm economise-t-il ?

4. **Gradient de RMSNorm** : deriver et implementer le backward de `y = x / rms(x)` avec `rms(x) = sqrt(mean(x^2) + eps)`. La formule (par ligne, dimension `D`) :
   ```
   y = x / r,  r = sqrt(mean(x^2) + eps)
   dx = (1/r) * (dy - (x / (D * r^2)) * sum(dy * x))
   ```
   Verifier par difference finie (< 1e-5).

5. Question : pourquoi tous les LLMs modernes (LLaMA, Mistral, Qwen) sont-ils passes de LayerNorm a RMSNorm ? (Indice : meme qualite empirique, moins de calcul, plus stable en float16/bf16.)

### Criteres de reussite

- [ ] LayerNorm donne mean=0, std=1 ; RMSNorm donne RMS=1 mais mean != 0
- [ ] Le cout reduit de RMSNorm est quantifie (pas de centrage, 1 passe)
- [ ] Le backward de RMSNorm est implemente et passe le gradient check (< 1e-5)
- [ ] La justification du passage a RMSNorm est correcte

---

## Exercice 6 : KV-cache GQA/MQA — calcul de taille memoire et facteur de reduction

### Objectif

Calculer la taille exacte d'un KV-cache et quantifier l'economie memoire de GQA (Grouped Query Attention) et MQA (Multi-Query Attention) par rapport a la MHA classique.

### Consigne

D'apres le cours (`01-theory/09-...`, exercice 2 KV cache) :

1. **Formule de taille du KV-cache** : pour un modele, la taille du cache (en octets) est :
   ```
   taille = 2 * n_layers * n_kv_heads * head_dim * seq_len * batch * bytes_per_elem
   ```
   (le `2` = K et V). Implementer `kv_cache_bytes(...)`.

2. **MHA vs GQA vs MQA** : pour une config type LLaMA-2-70B (`n_layers=80, n_heads=64, head_dim=128, d_model=8192`) :
   - MHA : `n_kv_heads = n_heads = 64`.
   - GQA : `n_kv_heads = 8` (groupes de 8 queries par K/V).
   - MQA : `n_kv_heads = 1`.
   Calculer la taille du cache (en GB) pour `seq_len=4096, batch=1, fp16` dans les 3 cas. Donner le facteur de reduction GQA et MQA vs MHA.

3. **Croissance avec le contexte** : pour GQA, tracer la taille du cache pour `seq_len ∈ {4096, 32768, 131072}`. Verifier la croissance LINEAIRE en `seq_len`. Pourquoi le KV-cache devient-il le goulot d'etranglement memoire en long contexte (et pas les poids du modele) ?

4. **Equivalence GQA <-> MHA** : implementer le "repeat" des K/V (chaque K/V head sert `n_heads / n_kv_heads` queries via `repeat_interleave`) et verifier que GQA avec `n_kv_heads == n_heads` redonne EXACTEMENT la MHA (ecart < 1e-12 sur l'output d'attention).

5. Question : quel est le compromis de GQA ? (Indice : moins de K/V distinctes = moins de cache et plus de vitesse, mais legere perte de capacite vs MHA. MQA est l'extreme.)

### Criteres de reussite

- [ ] `kv_cache_bytes` implemente la formule correcte (facteur 2 pour K+V)
- [ ] Les tailles MHA/GQA/MQA pour LLaMA-2-70B sont calculees ; reductions 8x (GQA) et 64x (MQA)
- [ ] La croissance lineaire du cache avec seq_len est verifiee
- [ ] L'equivalence GQA(n_kv=n_heads) == MHA est verifiee (< 1e-12)
- [ ] Le compromis GQA (memoire/vitesse vs capacite) est correctement explique
