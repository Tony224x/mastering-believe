# Exercices Faciles — Jour 9 : Architectures modernes des LLMs

---

## Exercice 1 : RoPE — comprendre la rotation

### Objectif

Verifier a la main que le produit scalaire de deux vecteurs tournes depend seulement de leur difference angulaire.

### Consigne

Soit `q = [1, 0]` et `k = [1, 0]` (vecteurs 2D simples).

1. Calculer `q . k` sans rotation. Quel est le resultat ?

2. Tourner `q` d'un angle `alpha = π/4` (45 degres) et `k` d'un angle `beta = π/4`. La matrice de rotation 2D est :
   ```
   R(theta) = [cos(theta), -sin(theta)]
              [sin(theta),  cos(theta)]
   ```
   Calculer `q' = R(alpha) @ q` et `k' = R(beta) @ k`. Puis `q' . k'`.

3. Maintenant, tourner `q` d'un angle `alpha = π/3` (60 degres) et `k` d'un angle `beta = 2π/3` (120 degres). La difference `beta - alpha = π/3` (meme que dans le cas 2).
   Calculer `q'' = R(alpha) @ q` et `k'' = R(beta) @ k`. Puis `q'' . k''`.

4. Comparer `q' . k'` et `q'' . k''`. Que constate-t-on ? Expliquer avec la formule `(R(α) q) . (R(β) k) = q · R(β - α) · k`.

5. **Analogie avec RoPE** : dans RoPE, `alpha = m * theta_j` et `beta = n * theta_j` ou `m` et `n` sont les positions. Pourquoi l'attention capture donc la position relative (`n - m`) et non absolue ?

### Criteres de reussite

- [ ] Etape 1 : q . k = 1
- [ ] Etape 2 : q' . k' = 1 (meme resultat, car alpha = beta, difference = 0)
- [ ] Etape 3 : q'' . k'' depend seulement de (beta - alpha) = π/3, donc cos(π/3) = 0.5
- [ ] Etape 4 : les deux produits scalaires dependent de la DIFFERENCE d'angle uniquement
- [ ] Etape 5 : comprehension de RoPE = encodage relatif via rotation

---

## Exercice 2 : Calculer la taille d'un KV cache

### Objectif

Savoir estimer la memoire d'inference d'un LLM et comprendre pourquoi GQA est necessaire.

### Consigne

Pour chaque modele ci-dessous, calculer la taille du KV cache en Go pour :
- Une sequence de 4096 tokens
- Un batch de 1 et un batch de 8
- En bf16 (2 bytes par element)

Formule :
```
KV cache (bytes) = 2 * n_layers * n_kv_heads * head_dim * seq_len * batch_size * bytes_per_elem

(facteur 2 pour K et V separes)
```

1. **GPT-2 small** (MHA) : n_layers=12, n_heads=12, head_dim=64, n_kv_heads=12 (MHA)

2. **LLaMA 2 7B** (MHA) : n_layers=32, n_heads=32, head_dim=128, n_kv_heads=32 (MHA)

3. **LLaMA 2 70B** (GQA) : n_layers=80, n_heads=64, head_dim=128, n_kv_heads=8 (GQA)

4. **LLaMA 3 70B** (GQA) : n_layers=80, n_heads=64, head_dim=128, n_kv_heads=8 (GQA)

5. **Bonus** : pour LLaMA 2 70B, calculer combien le cache serait plus gros si on utilisait MHA (n_kv_heads=64) au lieu de GQA (n_kv_heads=8). Ratio ?

6. Si une GPU a 80 Go de VRAM, et que les poids de LLaMA 2 70B font 140 Go (en bf16), est-ce que le modele peut tourner sur UNE GPU ? Et sur 2 ? Comment le KV cache influence-t-il ce choix ?

### Criteres de reussite

- [ ] GPT-2 small, batch=1 : ~150 MB (tres petit, c'est pour ca que GPT-2 ne posait pas de probleme)
- [ ] LLaMA 2 7B, batch=1 : ~2.15 GB. batch=8 : ~17 GB (deja probleme en inference)
- [ ] LLaMA 2 70B GQA, batch=1 : ~1.34 GB. Sans GQA : ~10.7 GB (ratio 8x)
- [ ] LLaMA 2 70B en bf16 (140 GB de poids) + cache ne rentre pas sur 1 GPU 80 GB. Besoin d'au moins 2 GPUs.
- [ ] Comprehension : GQA etait strictement necessaire pour servir les 70B+ en production

---

## Exercice 3 : SwiGLU vs GeLU — parametres matches

### Objectif

Comprendre pourquoi LLaMA utilise `d_ff = 8/3 * d_model` au lieu du `4 * d_model` classique.

### Consigne

Soit `d_model = 4096` (LLaMA 7B).

1. **FFN classique (2 matrices)** : avec `d_ff = 4 * d_model`, calculer le nombre de parametres d'un FFN classique (GeLU) :
   - W1 : (d_model, d_ff)
   - W2 : (d_ff, d_model)
   - Total ?

2. **FFN SwiGLU (3 matrices)** : avec `d_ff = 4 * d_model` (meme valeur), calculer le nombre de parametres :
   - W_gate : (d_model, d_ff)
   - W_up : (d_model, d_ff)
   - W_down : (d_ff, d_model)
   - Total ?

3. Ratio SwiGLU / GeLU : combien de parametres en plus ?

4. **Correction** : si on veut que SwiGLU ait le meme nombre de parametres que GeLU, quelle valeur de `d_ff` faut-il ? (resoudre : `3 * d_model * d_ff_swiglu = 2 * d_model * d_ff_gelu` avec `d_ff_gelu = 4 * d_model`)

5. Verifier que pour LLaMA 7B avec d_model=4096, d_ff=11008 (valeur reelle de LLaMA). Est-ce que 11008 correspond a 8/3 * 4096 ? Pourquoi n'est-ce pas exactement cette valeur ?

6. Calculer le nombre total de parametres d'une couche LLaMA (attention + FFN + 2 norms) pour LLaMA 7B (d_model=4096, n_heads=32, d_ff=11008, GQA n_kv_heads=32 ici car le 7B utilise MHA). Multiplier par n_layers=32 pour avoir le total du stack.

### Criteres de reussite

- [ ] FFN GeLU : 2 * 4096 * 16384 = 134 217 728 params (134M par couche)
- [ ] FFN SwiGLU naif (d_ff = 4*d) : 3 * 4096 * 16384 = 201 326 592 (201M, 50% de plus)
- [ ] Pour matcher : d_ff_swiglu = 8/3 * d_model ≈ 10923
- [ ] LLaMA utilise 11008 (arrondi au multiple de 256 pour GPU efficiency)
- [ ] Couche LLaMA 7B : ~202M params. Stack 32 couches : ~6.47B params (proche des 7B totaux avec embeddings)
