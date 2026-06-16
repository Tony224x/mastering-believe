# Exercices Hard — Jour 11 : Inference optimisee

---

## Exercice 7 : Flash Attention — softmax online (streaming) from scratch

### Objectif

Implementer le coeur algorithmique de Flash Attention : un softmax "online" qui traite les cles/valeurs par blocs sans jamais materialiser la matrice de scores `(n, n)`. Prouver l'equivalence numerique avec l'attention standard.

### Consigne

1. **Softmax online 1D** (echauffement) : implementer un softmax qui traite un vecteur de scores par blocs, en maintenant le running max `m` et la running sum `l`. La regle de mise a jour quand un nouveau bloc de max local `m_block` arrive :
   ```
   m_new = max(m, m_block)
   l_new = l * exp(m - m_new) + l_block * exp(m_block - m_new)
   ```
   Verifier que le resultat est identique au softmax classique (ecart < 1e-10).

2. **Flash Attention (un seul head, causal)** : implementer `flash_attention(Q, K, V, block_size)` qui :
   - Iterer sur les blocs de K/V
   - Maintenir pour chaque ligne de Q : le running max `m`, la running sum `l`, et l'accumulateur de sortie `O` (rescalé a chaque bloc)
   - Mise a jour de l'accumulateur :
     ```
     O = O * exp(m_old - m_new) + (exp(scores_block - m_new) @ V_block)
     ```
   - Appliquer le masque causal **par bloc** (un token de Q n'attend que les K a sa position ou avant)
   - A la fin : `O = O / l`

3. **Equivalence** : comparer `flash_attention(Q, K, V)` a une attention standard (calcul de la matrice complete + softmax + matmul). L'ecart doit etre < 1e-9 pour plusieurs tailles de bloc (1, 4, 16, n).

4. **Empreinte memoire** : compter la memoire de pic pour l'attention standard (`O(n^2)` pour la matrice de scores) vs Flash (`O(block_size * head_dim)` + accumulateurs `O(n)`). Tracer la memoire en fonction de n pour les deux.

5. Analyser :
   - Pourquoi le rescaling `exp(m_old - m_new)` est-il numeriquement stable (pas d'overflow) ?
   - Flash Attention fait-il MOINS de FLOPs que l'attention standard, ou le meme nombre ? Pourquoi est-il quand meme plus rapide sur GPU ? (indice : IO HBM vs SRAM)

### Criteres de reussite

- [ ] Le softmax online 1D matche le softmax classique (ecart < 1e-10)
- [ ] Flash Attention causale matche l'attention standard pour plusieurs block_size (ecart < 1e-9)
- [ ] La gestion du running max + rescaling de l'accumulateur est correcte
- [ ] La memoire de pic Flash est O(n) au lieu de O(n^2) — chiffree et tracee
- [ ] L'analyse FLOPs (memes FLOPs) + IO-bound est correcte

---

## Exercice 8 : Roofline de l'inference LLM + KV cache quantize

### Objectif

Construire un modele "roofline" (compute-bound vs memory-bound) de l'inference d'un LLM, l'utiliser pour predire le throughput en prefill et en decode, puis quantifier l'apport de GQA, du batching et de la quantization du KV cache.

### Consigne

On modelise une GPU par deux nombres : `peak_flops` (FLOPS) et `mem_bw` (bytes/s). Pour H100 : `peak_flops = 900e12` (fp16), `mem_bw = 3e12`.

1. **Intensite arithmetique** : implementer `arithmetic_intensity = flops / bytes_moved`. La regle roofline :
   - Si `intensity > peak_flops/mem_bw` (le "ridge point") → compute-bound
   - Sinon → memory-bound

2. **Prefill** (traiter un prompt de `P` tokens) :
   - FLOPs ≈ `2 * N_params * P` (forward, facteur ~2 par MAC)
   - Bytes ≈ poids lus une fois = `N_params * bytes_per_param`
   - Calculer l'intensite arithmetique pour P=512 et montrer que le prefill est COMPUTE-bound
   - Throughput prefill = `peak_flops / (2 * N_params)` tokens/s

3. **Decode** (generer 1 token, batch=1) :
   - Bytes ≈ poids (`N_params * bpp`) + KV cache lu (`kv_bytes`)
   - FLOPs ≈ `2 * N_params` (un seul token)
   - Montrer que l'intensite est ~1-2 → MEMORY-bound
   - Throughput decode = `mem_bw / (N_params*bpp + kv_bytes)` tokens/s

4. Appliquer a **LLaMA 2 7B fp16** (N=7e9, bpp=2) :
   - prefill tokens/s vs decode tokens/s : verifier que le decode est ~50-100x plus lent par token
   - KV cache pour seq=4096 (formule : `2 * n_layers * n_kv_heads * head_dim * seq * batch * bpp`)

5. **Leviers** (recalculer le throughput decode dans chaque cas) :
   - **GQA** : passer de MHA (n_kv_heads=32) a GQA (n_kv_heads=8) → cache /4. Gain reel sur un 7B (ou les poids dominent) vs sur un 70B (ou le cache compte plus) ?
   - **KV cache int8** : diviser `kv_bytes` par 2. Gain ?
   - **Batching** : avec batch B, les poids sont lus 1 fois pour B tokens generes → throughput total ≈ `B * mem_bw / (N*bpp + B*kv_bytes)`. Tracer le throughput total vs B et montrer la saturation quand `B*kv_bytes` devient comparable a `N*bpp`.

6. Analyser : pourquoi le batching est le levier #1 pour le throughput SERVEUR mais ne change rien a la latence d'UNE requete ? Pourquoi GQA aide surtout les gros contextes / gros modeles ?

### Criteres de reussite

- [ ] Le modele roofline (intensite vs ridge point) est correct
- [ ] Le prefill est correctement classe compute-bound, le decode memory-bound
- [ ] Le throughput LLaMA 7B est coherent (prefill milliers tok/s, decode ~100-200 tok/s)
- [ ] L'effet de GQA est analyse differemment pour 7B vs 70B
- [ ] La courbe throughput vs batch montre la saturation, avec l'explication latence vs throughput
