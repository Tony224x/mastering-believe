# Exercices Faciles — Jour 11 : Inference optimisee

---

## Exercice 1 : Speedup du KV cache

### Objectif

Comprendre mathematiquement pourquoi le KV cache donne un speedup qui grandit avec la longueur de la sequence.

### Consigne

Soit un modele avec :
- `d_model = 4096`
- `n_layers = 32`
- Attention MHA standard

1. **Compute sans cache** : combien d'operations (ordre de grandeur) pour generer le token `t+1` ?
   - On doit recalculer Q, K, V pour TOUS les tokens 0..t+1
   - L'attention est une matmul `(t+1, d) @ (d, t+1)` = `(t+1, t+1)` → ~(t+1)² × d FLOPs
   - Multiplier par `n_layers` pour obtenir le total par token

2. **Compute avec cache** : combien d'operations pour le token `t+1` ?
   - On calcule seulement q, k, v pour le nouveau token : ~d² FLOPs
   - L'attention est `(1, d) @ (d, t+1)` = `(1, t+1)` → (t+1) × d FLOPs
   - Multiplier par `n_layers`

3. **Ratio** : quel est le ratio naive/cached par token pour des sequences de 100, 1000, 10 000 tokens ?

4. **Cout cumule** : pour generer une sequence de 1000 tokens :
   - Sans cache : somme de `O(t²)` pour t de 1 a 1000 → `O(n³)` total
   - Avec cache : somme de `O(t)` pour t de 1 a 1000 → `O(n²)` total
   - Compute le ratio exact.

5. **Memoire vs compute** : le KV cache economise du compute mais coute de la memoire. Pour LLaMA 7B seq=4k, le cache fait 2 GB. Est-ce un bon trade-off ?

### Criteres de reussite

- [ ] Naive : ~O(n² × d × n_layers) FLOPs par token
- [ ] Cached : ~O(n × d × n_layers) FLOPs par token
- [ ] Ratio par token : n (pour n=100, speedup ~100x ; pour n=1000, ~1000x)
- [ ] Cout cumule 1000 tokens : naive = n³/6 ≈ 1.6e8 × d, cached = n²/2 ≈ 5e5 × d, ratio ~330x
- [ ] Le cache est presque toujours rentable : le compute sauve vaut 1000x la memoire prise

---

## Exercice 2 : Quantization — calcul a la main

### Objectif

Savoir quantizer un petit vecteur en int8 et mesurer l'erreur introduite.

### Consigne

Soit le vecteur de poids : `W = [0.1, -0.3, 1.5, -2.0, 0.05, 0.8]`

1. **Symmetric quantization** :
   - Trouver `max_abs = max(|W|)`
   - Calculer `scale = max_abs / 127`
   - Pour chaque element : `q = round(W / scale)`, clip a [-127, 127]
   - Donner les valeurs int8 quantizees

2. **Dequantization** :
   - Pour chaque int8 quantize : `W_dequant = q * scale`
   - Donner le vecteur dequantize

3. **Erreur** :
   - Pour chaque element, calculer `error = |W - W_dequant|`
   - Quelle est l'erreur moyenne ? L'erreur max ?

4. **Observer le probleme** : le plus petit element du vecteur est `0.05`. Est-il bien preserve ? Pourquoi les petites valeurs sont plus affectees que les grandes ?

5. **Per-channel** : si on quantizait **chaque element separement** (extreme), l'erreur serait 0. Si on quantize **le tenseur entier**, l'erreur depend du max. Le compromis est de quantizer **par canal** (une scale par ligne/colonne).

6. **Int4** : refaire la question 1 mais en int4 (range [-7, 7]). Quelle est la nouvelle erreur ? Pourquoi int4 est beaucoup plus bruite qu'int8 ?

### Criteres de reussite

- [ ] max_abs = 2.0, scale = 2.0/127 ≈ 0.01575
- [ ] q = [6, -19, 95, -127, 3, 51]
- [ ] Erreur moyenne autour de 0.003-0.008
- [ ] Le 0.05 devient 3 * scale ≈ 0.047, erreur relative 6% (tres eleve)
- [ ] Int4 : scale = 2/7 ≈ 0.286, erreurs beaucoup plus grandes (10-20% sur petites valeurs)

---

## Exercice 3 : Estimer le temps d'inference

### Objectif

Savoir estimer a combien de tokens/seconde un LLM peut tourner sur une GPU donnee.

### Consigne

GPU : NVIDIA H100 avec 3 TB/s de bande passante memoire.

Modele : LLaMA 2 7B en fp16.
- Poids : 14 GB
- KV cache pour seq=2048 : ~1 GB
- Activations par token : negligeable

1. **Prefill** (traiter le prompt) :
   - La phase prefill est compute-bound, pas memory-bound
   - L'H100 fait ~900 TFLOPS en fp16
   - Par token de prompt, environ `6 * N_params` FLOPs (facteur 6 pour transformer)
   - Temps par token de prompt : `(6 * 7e9) / 900e12` = ? secondes
   - Combien de tokens de prompt par seconde ?

2. **Decode** (generer tokens) :
   - Le decode est memory-bound. Chaque token demande de lire tout le modele + le cache
   - Par token : lire 14 GB (poids) + 1 GB (cache) = 15 GB
   - Temps par token : `15 / 3000` = ? ms
   - Combien de tokens par seconde en decode ?

3. **Difference** : compare ta reponse de 1) et 2). Le prefill est-il 10x ou 100x plus rapide que le decode ?

4. **Avec GQA** : LLaMA 2 70B avec GQA (ratio 8) a un cache 8x plus petit. Si on avait MHA, combien de tokens/sec pour un 70B ? Avec GQA ? Combien d'amelioration ?

5. **Batching** : si on fait un batch de 32 requetes simultanees, le temps de lecture des poids est amorti sur 32 tokens generes. Quelle amelioration en tokens/sec total ?

### Criteres de reussite

- [ ] Prefill : ~46 ns/token en compute pur, soit ~20 000 tokens/sec theorique (mais en pratique ~2000-5000 a cause d'autres overhead)
- [ ] Decode : 15 / 3000 = 5 ms/token, soit ~200 tokens/sec
- [ ] Le decode est 100x plus lent que le prefill par token
- [ ] 70B MHA : cache ~10 GB + poids 140 GB = 150 GB → 50 ms/token → 20 tok/s
- [ ] 70B GQA : cache ~1.25 GB + 140 GB = 141 GB → 47 ms/token → 21 tok/s (pas tant de difference sur un 70B car le cache est petit devant les poids)
- [ ] Batching 32 : ~6400 tokens/sec total (32x plus, car les poids sont lus une seule fois)
