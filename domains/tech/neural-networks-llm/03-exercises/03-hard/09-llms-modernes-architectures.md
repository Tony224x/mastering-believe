# Exercices Hard — Jour 9 : LLMs modernes (RoPE, RMSNorm, SwiGLU, GQA)

---

## Exercice 7 : RoPE — preuve formelle de la propriete relative + extension de contexte (NTK / interpolation)

### Objectif

Demontrer NUMERIQUEMENT la propriete relative de RoPE par le calcul complexe equivalent, puis implementer les deux techniques d'extension de contexte (position interpolation et NTK-aware scaling) utilisees pour passer un modele de 4k a 32k+ sans re-entrainer.

### Consigne

1. **Formulation complexe de RoPE** : RoPE peut s'ecrire en nombres complexes. Pour une paire `(x_{2j}, x_{2j+1})` vue comme `z_j = x_{2j} + i x_{2j+1}`, RoPE a la position `m` est `z_j * e^{i m theta_j}`. Implementer `apply_rope_complex(x, pos, freqs)` via `np.complex` et verifier qu'elle donne EXACTEMENT le meme resultat que la version reelle `apply_rope` (< 1e-12).

2. **Preuve de la propriete relative** : montrer analytiquement (en commentaire) ET numeriquement que `Re(<RoPE(q,m), conj(RoPE(k,n))>)` ne depend que de `(m - n)`, car les phases se combinent en `e^{i(m-n)theta_j}`.

3. **Position Interpolation (PI)** : pour etendre de `L_train` a `L_new`, PI compresse les positions : `pos' = pos * L_train / L_new`. Implementer et montrer que les angles restent dans la plage vue au training (donc le modele extrapole moins). Calculer le facteur de compression pour `4096 -> 32768`.

4. **NTK-aware scaling** : au lieu de compresser uniformement, NTK ajuste la BASE : `base' = base * (L_new / L_train)^(d/(d-2))`. Implementer, et comparer les longueurs d'onde des hautes vs basses frequences entre PI et NTK. Montrer que NTK preserve mieux les hautes frequences (details locaux) tout en etendant les basses (longue portee).

5. Question : pourquoi RoPE permet-il l'extension de contexte la ou les embeddings positionnels APPRIS (GPT-2) ne le permettent pas du tout ? (Indice : RoPE est une fonction continue de la position, les embeddings appris sont une table finie.)

### Criteres de reussite

- [ ] `apply_rope_complex` == `apply_rope` reel (< 1e-12)
- [ ] La propriete relative est prouvee via la combinaison des phases `e^{i(m-n)theta}`
- [ ] Position Interpolation implementee, facteur de compression calcule
- [ ] NTK-aware scaling implemente, comparaison des longueurs d'onde PI vs NTK
- [ ] La reponse explique pourquoi RoPE (continu) extrapole la ou une table apprise ne peut pas

---

## Exercice 8 : Mini-bloc LLaMA complet (RMSNorm + RoPE + GQA + SwiGLU) en NumPy

### Objectif

Assembler un bloc LLaMA COMPLET en pur NumPy — RMSNorm + GQA causale avec RoPE + SwiGLU + residuals — et verifier sa coherence : shapes, causalite, et compte de parametres vs un bloc GPT-2 classique a iso-`d_model`.

### Consigne

En t'inspirant de `02-code/14-capstone.py` (LLaMA en PyTorch, a transposer en NumPy) :

1. **Briques NumPy** :
   - `rms_norm(x, gamma)` (J9).
   - `precompute_rope` + `apply_rope` (J9).
   - `gqa_attention(x, params, n_heads, n_kv_heads)` : projections `W_q (d, n_heads*hd)`, `W_k/W_v (d, n_kv_heads*hd)`, RoPE sur q et k, `repeat_interleave` des K/V pour matcher les queries, attention causale masquee, `W_o`.
   - `swiglu_ffn(x, W_gate, W_up, W_down)` : `SiLU(x@W_gate) * (x@W_up) @ W_down`, avec `d_ff = round_to_multiple(8/3 * d_model)`.

2. **Bloc complet** (pre-norm) :
   ```
   x = x + gqa_attention(rms_norm(x, g1))
   x = x + swiglu_ffn(rms_norm(x, g2))
   ```

3. **Verifications** :
   - Shapes : pour `d_model=64, n_heads=8, n_kv_heads=2, seq=12`, tracer toutes les shapes intermediaires (q, k, v avant/apres repeat, scores, output).
   - Causalite end-to-end : perturber `x[t]` ne change pas `out[:t]` (< 1e-10).
   - GQA grouping : verifier que `n_rep = n_heads / n_kv_heads = 4` et que le repeat duplique bien chaque K/V head 4 fois.

4. **Compte de parametres LLaMA-block vs GPT-2-block** (meme `d_model`) :
   - LLaMA : `W_q + W_k + W_v + W_o` (GQA : K/V plus petits) + SwiGLU (3 matrices) + 2 RMSNorm (scale seulement).
   - GPT-2 : `c_attn (3*d^2) + c_proj (d^2)` + FFN (2 matrices, `d_ff=4d`) + 2 LayerNorm (scale + bias).
   - Comparer les totaux et expliquer ou LLaMA economise (GQA) et ou il depense (SwiGLU a 3 matrices).

5. Question : LLaMA combine 4 ameliorations (RMSNorm, RoPE, GQA, SwiGLU) par rapport au Transformer original. Laquelle a le plus d'impact sur (a) la qualite, (b) la vitesse d'inference, (c) la memoire ?

### Criteres de reussite

- [ ] Toutes les briques NumPy (rms_norm, RoPE, gqa_attention, swiglu_ffn) sont correctes
- [ ] Le bloc pre-norm complet tourne ; toutes les shapes sont tracees, `n_rep=4`
- [ ] La causalite end-to-end est verifiee (< 1e-10)
- [ ] Le compte de parametres LLaMA-block vs GPT-2-block est correct et commente
- [ ] La reponse hierarchise correctement l'impact des 4 ameliorations

---

## Exercice 9 : SiTU-GLU — borner une activation pour survivre au FP4 (Kimi K3, 2026)

### Objectif

Comprendre pourquoi un modele entraine en precision tres basse ne peut pas garder SwiGLU tel quel, et pourquoi la solution retenue est un *soft cap* (tanh) et non un `clip`.

### Consigne

Rappel des deux formules (`d` = dimension du modele, `x` un vecteur d'activations) :

```
SwiGLU   : y = Swish(W_g x)             * (W_u x)
SiTU-GLU : y = [b1*tanh(W_g x / b1) * Sigmoid(W_g x)] * [b2*tanh(W_u x / b2)]
           avec b1 = 4, b2 = 25   (valeurs Kimi K3)
```

1. **Borne theorique.** `tanh` est borne par 1 et `Sigmoid` par 1. En deduire la borne dure de `|y|` pour SiTU-GLU avec `b1 = 4, b2 = 25`. Comparer a SwiGLU, dont la sortie n'est bornee par rien.

2. **Implementer** `situ_glu(gate_pre, up_pre, b1=4, b2=25)` et `swiglu(gate_pre, up_pre)` en NumPy (attention a la stabilite numerique du `Sigmoid` : `np.clip` de l'argument de `exp`).

3. **Regime normal.** Tirer `gate_pre, up_pre ~ N(0, 1)` de shape `(512, 1024)`. Mesurer l'ecart relatif median entre les deux sorties. Conclusion attendue : dans le regime ou vivent 99 % des activations, SiTU-GLU est une *reparametrisation quasi transparente* de SwiGLU. Pourquoi est-ce une condition necessaire pour que le remplacement soit acceptable ?

4. **Regime outlier.** Reprendre le meme tirage, puis multiplier 20 canaux choisis au hasard par 60 (les "massive activations" observees dans les LLMs reels). Recalculer `max|y|` pour les deux fonctions. Comparer chacune au maximum representable de FP8 E4M3, soit **448**. Laquelle deborde ?

5. **Soft cap vs hard clamp.** On pourrait obtenir la meme borne avec `y = np.clip(y, -100, 100)`. Calculer numeriquement la derivee des deux options en `x = 100`, par difference finie **a droite** (`(f(x+h) - f(x)) / h`, `h = 1e-4`). Attention : le `clip` a un point anguleux exactement en 100, donc une difference *centree* l'enjamberait et renverrait 0.5 — une pente moyenne qui n'existe nulle part. C'est la derivee a droite qui decrit ce que vit un neurone sature.
   - `d/dx [b2 * tanh(x / b2)]` avec `b2 = 25`
   - `d/dx [clip(x, -100, 100)]`

   Que se passe-t-il pour un neurone sature si l'on choisit le `clip` ? Relier a la notion de gradient mort.

6. **Question de synthese.** Le format numerique choisi pour l'inference a impose une modification de la fonction d'activation, donc de la *fonction* que le reseau peut representer. Donner un autre exemple, vu dans le cours, ou une contrainte materielle a remonte jusqu'au design mathematique du modele. (Indice : jour 17, le gate de decroissance de KDA.)

### Criteres de reussite

- [ ] Borne calculee : `|y| <= b1 * b2 = 4 * 25 = 100` (le facteur `Sigmoid <= 1` ne l'augmente pas)
- [ ] Les deux fonctions sont implementees et `Sigmoid` est numeriquement stable
- [ ] Regime normal : ecart relatif median de l'ordre de 1 %, donc reparametrisation quasi transparente
- [ ] Regime outlier : `max|SwiGLU|` depasse largement 448 (debordement FP8 E4M3), `max|SiTU-GLU|` reste sous 100
- [ ] Derivee a droite en x=100 : ~1.3e-3 pour le soft cap, exactement 0 pour le `clip` (et le piege de la difference centree est identifie) -> le clip tue le gradient, un neurone sature ne peut plus jamais se corriger
- [ ] La synthese cite un second cas de co-design algorithme/materiel (gate borne de KDA, ou le format MX du jour 19)
