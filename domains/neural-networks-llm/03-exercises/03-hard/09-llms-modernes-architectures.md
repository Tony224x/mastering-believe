# Exercices Hard — Jour 9 : Architectures modernes des LLMs

---

## Exercice 7 : Bloc LLaMA complet from scratch

### Objectif

Assembler un bloc decoder type LLaMA (RMSNorm + RoPE + GQA + SwiGLU) en NumPy pur — l'architecture exacte de la quasi-totalite des LLMs open-source actuels.

### Consigne

1. Implementer le bloc :

```
h = x + GQA_Attention(RMSNorm(x))     # RoPE applique a Q et K, masque causal
out = h + SwiGLU_FFN(RMSNorm(h))
```

   - `SwiGLU_FFN(x) = (silu(x @ W_gate) * (x @ W_up)) @ W_down` avec `silu(z) = z * sigmoid(z)` — 3 matrices, pas de bias
   - RMSNorm sans bias, RoPE par paire de dimensions, GQA avec `n_heads=4, n_kv_heads=2`
   - Config : `d_model=32, d_head=8, T=8, batch=2`

2. **Parameter matching SwiGLU** : calculer `d_ff` pour qu'un FFN SwiGLU (3 matrices `d*d_ff`) ait le meme nombre de parametres qu'un FFN GELU classique (2 matrices `d*4d`) : `d_ff = 8d/3`, arrondi au multiple de 32 superieur. Verifier l'ecart de parametres < 5%. Comparer avec le vrai LLaMA-7B (`d=4096, d_ff=11008`) : verifier que 11008 ≈ 8*4096/3 arrondi au multiple de 256.

3. Tests de proprietes :
   - shapes a chaque etape (assert)
   - causalite de bout en bout (perturber le futur ne change pas le passe, 1e-12)
   - **dependance a la position** : contrairement a un bloc SANS positional encoding, permuter les tokens d'entree ne doit PAS permuter les sorties (RoPE casse l'equivariance) — verifier les deux affirmations en activant/desactivant RoPE
   - zero-init de `W_o` et `W_down` → bloc == identite

4. Compter les parametres du bloc et verifier la formule `4*d*d_head*n_heads_equiv... ` — plus simplement : `params_attn = d*(n_heads*d_head) + 2*d*(n_kv_heads*d_head) + (n_heads*d_head)*d` et `params_ffn = 3*d*d_ff`, verification par sommation directe des tailles des matrices.

### Criteres de reussite

- [ ] Le bloc complet tourne et tous les asserts de shapes passent
- [ ] SwiGLU est correct (3 matrices, gating multiplicatif, silu — verifie sur un calcul manuel 2x2)
- [ ] Le parameter matching 8d/3 est calcule et le cas LLaMA-7B (11008) est retrouve
- [ ] Causalite (1e-12) et test d'identite (exact) passent
- [ ] Le test RoPE-casse-la-permutation passe dans les deux sens (avec et sans RoPE)
- [ ] Le compte de parametres par formule == compte par sommation des shapes

---

## Exercice 8 : Couche Mixture-of-Experts from scratch

### Objectif

Implementer une couche MoE avec routage top-2 et loss d'equilibrage de charge — comprendre comment decoupler nombre de parametres et FLOPs par token.

### Consigne

1. Implementer :

```python
class MoELayer:
    """n_experts FFN (2 couches, GELU ou silu) + router lineaire.
    forward(X): X (n_tokens, d) ->
      logits_router = X @ W_router            # (n_tokens, n_experts)
      top-2 par token, poids = softmax sur les 2 logits retenus
      output[t] = somme ponderee des 2 experts appliques a X[t]"""
```

   Config : `d=16, d_ff=32, n_experts=4, top_k=2`, 64 tokens.

2. Tests de base :
   - shape de sortie == shape d'entree
   - chaque token utilise exactement 2 experts et ses 2 poids somment a 1 (1e-9)
   - **coherence** : si tous les experts ont les MEMES poids, la sortie MoE == sortie d'un seul expert (1e-10), quel que soit le routage

3. **Load balancing loss** (style Switch/Mixtral) : `L_aux = n_experts * sum_i(f_i * P_i)` ou `f_i` = fraction de tokens dont l'expert i est dans le top-2 (normalisee pour sommer a 1) et `P_i` = moyenne sur les tokens de la proba softmax du router pour l'expert i.
   - Verifier : routage parfaitement uniforme (forcer logits identiques + tie-break tournant) → `L_aux ≈ 1.0` (le minimum)
   - Routage effondre (un biais enorme sur l'expert 0 → tous les tokens vont aux experts 0 et 1) → `L_aux` nettement > 1 (calculer la valeur attendue : f = [.5,.5,0,0] a cause du top-2)

4. **Compte params vs FLOPs actifs** : calculer (a) le total des parametres FFN de la couche (n_experts * params_expert), (b) les FLOPs par token (seulement top_k experts). Verifier le ratio params/FLOPs-actifs = n_experts/top_k = 2. Extrapoler en commentaire a Mixtral 8x7B (8 experts top-2 : ~47B params, ~13B actifs).

5. Statistiques de routage : afficher l'histogramme d'utilisation des experts sur 64 tokens aleatoires (router aleatoire) et verifier qu'aucun expert n'est mort avec une init raisonnable + tokens varies (sinon, commenter pourquoi ca peut arriver et pourquoi L_aux existe).

### Criteres de reussite

- [ ] Le routage top-2 est correct (2 experts/token, poids normalises sur les 2 retenus)
- [ ] Le test "experts identiques → MoE == 1 expert" passe a 1e-10
- [ ] `L_aux ≈ 1.0` (± 1e-6) en routage uniforme et > 1.5 en routage effondre, valeurs affichees
- [ ] Le ratio params totaux / FLOPs actifs == n_experts/top_k est demontre numeriquement
- [ ] L'histogramme d'utilisation est affiche et interprete
- [ ] Execution < 10 s
