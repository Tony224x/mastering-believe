# Exercices Hard — Jour 18 : Long context (Flash Attention, RoPE scaling, ring attention)

---

## Exercice 7 : YaRN end-to-end + propriete de position relative

### Objectif

Implementer YaRN de bout en bout, **roter reellement** des vecteurs Q/K avec les frequences scalees, et verifier la propriete fondamentale de RoPE : `q_m . k_n` ne depend que de `(m - n)`. Montrer que YaRN preserve mieux cette propriete hors-plage qu'une extrapolation naive.

### Consigne

Rappel : RoPE rote chaque paire `(x_2i, x_2i+1)` d'un angle `pos * theta_i`. La propriete cle est que le produit scalaire `rotate(q, m) . rotate(k, n)` ne depend que de `m - n` (invariance par translation), tant que les angles restent **dans la distribution vue a l'entrainement**.

1. **apply_rope(x, pos, inv_freq)** : pour un vecteur `x` de dim `d` et une position `pos`, calculer `angles = pos * inv_freq` (un par paire), puis roter chaque paire `(x[2i], x[2i+1])` par cet angle (`cos`/`sin`). Vectoriser proprement.

2. **Frequences** : reutiliser `rope_frequencies` (base 10000) et `rope_yarn_frequencies` (du `02-code/18`). `d = 64`, `L_train = 4096`, `scale = 8`.

3. **Propriete relative IN-RANGE** : pour un meme `q` et `k` fixes (aleatoires), et des positions `m, n` **dans la plage** d'entrainement, montrer que `apply_rope(q, m) . apply_rope(k, n)` ne depend (quasi) que de `m - n` :
   - Prendre plusieurs couples `(m, n)` avec le **meme** `delta = m - n` -> les produits scalaires doivent etre quasi egaux.
   - Asserter que l'ecart-type des produits scalaires a delta fixe est `< 1e-3` (in-range, base RoPE).

4. **OUT-OF-RANGE : naif vs YaRN** : pour des positions **bien au-dela** de `L_train` (ex `m ~ 30000`), comparer la stabilite de la propriete relative :
   - **Naif** (extrapolation : frequences originales a des positions enormes) -> la propriete relative se degrade (variance a delta fixe plus grande).
   - **YaRN** (frequences scalees) -> degradation moindre.
   - Mesurer et afficher la variance a delta fixe des deux cas ; montrer que `var_yarn < var_naif` out-of-range.

5. **Analyse** : expliquer pourquoi l'extrapolation naive casse (angles `m*theta_i` hors distribution pour les basses freqs) et pourquoi YaRN limite la casse (basses freqs comprimees -> angles restent dans la plage vue).

### Criteres de reussite

- [ ] `apply_rope` rote correctement par paires (verifiable : norme preservee)
- [ ] In-range : la propriete `q_m . k_n` depend ~uniquement de `m - n` (std a delta fixe < 1e-3)
- [ ] Out-of-range : variance a delta fixe mesuree pour naif ET YaRN
- [ ] `assert var_yarn < var_naif` out-of-range (YaRN degrade moins)
- [ ] L'analyse relie la casse aux angles hors-distribution des basses frequences
- [ ] Code commente avec le POURQUOI

---

## Exercice 8 : Ring attention / sequence parallelism

### Objectif

Simuler le sequence parallelism de Ring Attention : decouper la sequence en chunks "GPU", faire circuler les K/V autour de l'anneau, accumuler l'attention en online softmax par chunk, et asserter que le resultat egale **exactement** l'attention complete (full).

### Consigne

Ring Attention (Liu et al., 2023) repartit la sequence sur P "GPU", chacun detenant un chunk `(Q_p, K_p, V_p)`. A chaque step, chaque GPU passe ses `K, V` au voisin (anneau). Apres `P` steps, chaque GPU a vu tous les `K, V` et a accumule son attention via online softmax (exactement comme Flash, mais reparti).

1. **Reference full** : `full_attention(Q, K, V)` (non causale ici pour la simplicite : full softmax sur toute la sequence). Materialise `S` et calcule `O`.

2. **Decoupage** : `N = 128`, `d = 32`, `P = 4` chunks de taille `N/P = 32`. Decouper `Q, K, V` en `P` chunks contigus.

3. **ring_attention(chunks_Q, chunks_K, chunks_V)** : pour chaque chunk de Q (chaque "GPU" `p`) :
   - Initialiser online softmax local (`O_p = 0`, `m_p = -inf`, `l_p = 0`).
   - Boucler sur les `P` steps de l'anneau : a chaque step, le GPU `p` "recoit" le chunk `K, V` du GPU `(p - step) mod P` (la rotation de l'anneau), calcule `S = Q_p @ K_recu^T / sqrt(d)`, et fusionne via online softmax (running max/sum + rescale de l'accumulateur).
   - Normaliser `O_p /= l_p` a la fin.
   - Recoller les `O_p` dans l'ordre.

4. **Verification** : asserter `max|O_full - O_ring| < 1e-5`. Le sequence parallelism ne change PAS le resultat, seulement la repartition memoire/compute.

5. **Comptage** : afficher combien de chunks K/V chaque "GPU" garde en memoire a un instant donne (1, le chunk courant) vs full (tous). C'est le gain : memoire par GPU `O(N/P)` au lieu de `O(N)`.

### Criteres de reussite

- [ ] `full_attention` (reference) et `ring_attention` implementees en numpy
- [ ] La rotation de l'anneau parcourt bien les `P` chunks K/V pour chaque GPU
- [ ] L'online softmax fusionne correctement les chunks (rescale du max entre steps)
- [ ] `assert max|O_full - O_ring| < 1e-5` PASSE
- [ ] Le gain memoire par GPU (`O(N/P)`) est explicite
- [ ] Code commente avec le POURQUOI de l'overlap communication/compute

---

## Exercice 9 : Attention sinks (StreamingLLM)

### Objectif

Comparer full-causal vs sliding-only vs sliding+sinks sur une sequence synthetique, mesurer la derive L2 de la sortie du dernier token vs l'oracle full-causal, et montrer (ou nuancer honnetement, comme le `02-code`) que les sinks aident. Asserter seulement ce qui est numeriquement exact (masquage hors-fenetre = 0).

### Consigne

StreamingLLM (Xiao et al., 2023) observe que retirer les **premiers** tokens d'un KV cache fait collapse le modele : le softmax (somme = 1) a besoin de "sinks" pour drainer l'attention residuelle. La fix : garder ~4 tokens sinks + une fenetre glissante.

1. **Sequence synthetique** : `N = 256`, `d = 32`. Generer `Q, K, V`. Pour rendre les 4 premiers tokens "sink-like", aligner leurs vecteurs `K` sur une direction commune `sink_dir`, et faire pencher tous les `Q` vers `sink_dir` (les tokens "ennuyeux" drainent leur attention vers les sinks). Reutiliser la construction du `02-code/18` (part 4).

2. **Trois strategies** (toutes causales) :
   - **A) Full causal** (oracle) : masque triangulaire inferieur complet.
   - **B) Sliding only** : fenetre glissante `W = 32`, premiers tokens jetes.
   - **C) Sliding + sinks** : sliding `W = 32` + les `n_sinks = 4` premiers tokens toujours visibles (re-multiplier par le masque causal pour rester causal).

3. **Masquage exact** : asserter que, pour le dernier token en strategie B (sliding only), l'attention vers les tokens hors-fenetre est **exactement 0** (`< 1e-7`). C'est un claim numeriquement exact -> assertion legitime.

4. **Derive L2 de la sortie** : calculer `O = P @ V` pour chaque strategie, puis la norme L2 de la difference `O[last]` vs l'oracle full-causal. Afficher `drift_sliding` et `drift_sinks`.

5. **Masse d'attention sur les sinks** : afficher la masse d'attention que le dernier token depose sur les `n_sinks` premiers tokens (en full-causal et en sliding+sinks).

6. **Caveat honnete** : comme le `02-code/18`, NE PAS sur-asserter que les sinks reproduisent l'oracle. Dans ce setup jouet (single-pass, single-layer), les deux strategies derivent. Le phenomene reel de StreamingLLM se manifeste sur des generations autoregressives multi-couches sur des millions de tokens. Imprimer ce caveat explicitement (renvoyer a Xiao et al. 2023, figure 4).

### Criteres de reussite

- [ ] Les 4 premiers tokens sont construits comme des sinks emergents (K alignes, Q penchant)
- [ ] Les 3 masques (full / sliding / sliding+sinks) sont corrects et causaux
- [ ] `assert` que l'attention hors-fenetre du dernier token (sliding only) est nulle (< 1e-7)
- [ ] La derive L2 du dernier token vs oracle est mesuree pour B et C
- [ ] La masse d'attention sur les sinks est affichee (full vs sliding+sinks)
- [ ] Le caveat du setup jouet est imprime honnetement (pas de sur-assertion sur le collapse)
