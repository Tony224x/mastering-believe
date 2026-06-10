# Exercices Hard — Jour 5 : Attention Mechanism

---

## Exercice 7 : Multi-Head Attention forward + backward avec gradient check

### Objectif

Implementer le backward pass complet d'une Multi-Head Attention (le morceau de backprop le plus exigeant du cursus) et le valider par gradient check.

### Consigne

1. Implementer une classe `MultiHeadAttention` (NumPy pur) avec parametres `W_q, W_k, W_v, W_o` (shape `(d_model, d_model)` chacun, sans bias) :
   - `forward(X)` : projections → split heads → attention causale scaled → merge → W_o. Stocker TOUS les intermediaires (Q, K, V par tete, scores, weights) pour le backward
   - `backward(dout)` : retourner `dW_q, dW_k, dW_v, dW_o` et `dX`

2. Le backward du softmax (par ligne) : si `p = softmax(s)` et `dp` est le gradient entrant, alors
   `ds = p * (dp - sum(dp * p, axis=-1, keepdims=True))`
   Les positions masquees (-inf) donnent p=0 → ds=0 automatiquement. Verifier ce point.

3. Gradient check par differences finies centrees (eps=1e-5) avec une loss scalaire `L = sum(forward(X) * G)` ou `G` est une matrice fixe aleatoire :
   - verifier TOUS les elements de `dW_q, dW_k, dW_v, dW_o` et `dX`
   - erreur relative `|a - n| / (|a| + |n| + 1e-8) < 1e-5`

4. Dimensions de test : `T=4, d_model=8, n_heads=2`, seed fixe, init `randn * 0.3`.

5. Bonus : verifier que `dX[t]` recoit des contributions des positions futures via K et V (le gradient REMONTE le masque causal : la position 0 influence la loss aux positions 1..T-1, donc son gradient en depend).

### Criteres de reussite

- [ ] Le forward est strictement equivalent a une implementation de reference simple (difference < 1e-12)
- [ ] Le backward traverse : W_o → merge → attention (weights @ V) → softmax → scaling → projections
- [ ] Le softmax backward utilise la formule jacobienne complete (pas une approximation diagonale)
- [ ] Gradient check : erreur relative max < 1e-5 sur les 4 matrices de poids ET sur dX (tous les elements)
- [ ] Les gradients des positions masquees dans les scores sont exactement 0 (verifie)
- [ ] Execution < 20 s

---

## Exercice 8 : Cross-attention + masques de padding

### Objectif

Implementer les deux variantes d'attention manquantes en pratique reelle : la cross-attention (decoder → encoder) et la gestion correcte du padding dans un batch de sequences de longueurs differentes.

### Consigne

1. Implementer :

```python
def cross_attention(Q_dec, K_enc, V_enc, src_pad_mask):
    """Q_dec: (T_dec, d), K_enc/V_enc: (T_enc, d),
    src_pad_mask: (T_enc,) bool — True = vrai token, False = padding.
    Les positions paddees doivent recevoir un poids d'attention EXACTEMENT 0."""
```

   Noter que T_dec != T_enc en general, et qu'il n'y a PAS de masque causal en cross-attention (le decoder peut regarder tout l'encoder).

2. **Test de retrieval construit** : fabriquer `K_enc` avec des vecteurs quasi-orthogonaux (ex : base canonique * 10) et `Q_dec` = copie bruitee d'une des cles. Verifier que la query attend a > 0.95 sur la position correspondante, et que l'output est proche de la valeur associee (cosine similarity > 0.99 avec `V_enc[cible]`).

3. **Padding** : pour un batch de 2 sequences encoder de longueurs reelles {3, 5} paddees a T_enc=5 :
   - construire les masques de padding
   - verifier : poids sur les positions paddees == 0, lignes toujours normalisees a 1
   - verifier la propriete cle : **changer les valeurs des tokens de padding (K et V) ne change pas l'output** (difference < 1e-12)

4. **Self-attention causale + padding combines** : implementer la combinaison des deux masques (ET logique avant le -inf) pour un decoder avec padding, et verifier les deux proprietes simultanement. Gerer le cas piege : une ligne entierement masquee ne doit pas produire de NaN (choisir et documenter une strategie, ex : ligne du masque forcee sur soi-meme ou output mis a zero).

### Criteres de reussite

- [ ] La cross-attention gere T_dec != T_enc et les shapes sont correctes
- [ ] Le test de retrieval passe (poids > 0.95 sur la bonne cle, cosine > 0.99 sur l'output)
- [ ] Les positions paddees ont un poids exactement 0 et les lignes somment a 1
- [ ] Le test d'invariance au contenu du padding passe a 1e-12
- [ ] Le masque combine causal+padding est correct et le cas "ligne entierement masquee" est gere sans NaN avec la strategie documentee
- [ ] Aucune fuite : un decoder token n'attend jamais sur du padding NI sur le futur
