# Exercices Hard — Jour 12 : Multimodalite

---

## Exercice 7 : Entrainer un mini-CLIP dual-encoder from scratch

### Objectif

Entrainer un vrai modele contrastif a deux encodeurs (gradients manuels, NumPy) jusqu'a un retrieval quasi parfait sur des donnees synthetiques — le coeur de CLIP en miniature.

### Consigne

1. **Donnees synthetiques** (seed fixe) : 8 "concepts". Chaque concept k a un prototype image `mu_img[k] ∈ R^12` et un prototype texte `mu_txt[k] ∈ R^10` (tous ~N(0,1), independants entre modalites — l'alignement doit etre APPRIS, il n'existe pas au depart).
   Echantillons : `x_img = mu_img[k] + 0.1*bruit`, `x_txt = mu_txt[k] + 0.1*bruit`. Train : 256 paires ; test : 64 paires.

2. **Modele** : deux projections lineaires `W_img (12 -> 4)`, `W_txt (10 -> 4)`, similarite = produit scalaire des projections normalisees L2, temperature fixe 0.1.

3. **Gradients manuels** : pour l'InfoNCE symetrique avec logits `S = Z_img @ Z_txt.T / tau` :
   - `dL/dS = (P_row - I)/(2B) + (P_col - I).T/(2B)` ou P_row = softmax par ligne, P_col = softmax par colonne de S.T (deriver et commenter)
   - retropropager a travers la normalisation L2 : pour `z = u/||u||`, `du = (dz - z*(z.du... ))` — formule : `du = (I - z z^T)/||u|| @ dz` (l'implementer par ligne)
   - puis `dW = X.T @ dU`
   - **verifier les gradients de W_img et W_txt par differences finies (< 1e-5)** avant d'entrainer — obligatoire

4. **Entrainement** : batchs de 32, 300 steps, lr=1.0 (ajuster si besoin). Suivre la loss (elle part de ~ln(32) ≈ 3.47).

5. **Evaluation retrieval sur le test** :
   - R@1 image→texte et texte→image >= 95% (avec 64 distracteurs)
   - la similarite moyenne des paires correctes > similarite moyenne des paires incorrectes + 3 ecarts-types
   - **zero-shot "classification"** : construire 8 "class embeddings" texte (projections des prototypes) et classifier les images de test par similarite max → accuracy >= 95%. C'est exactement le mecanisme zero-shot de CLIP.

### Criteres de reussite

- [ ] Les gradients manuels (InfoNCE + normalisation L2 + projections) passent le check < 1e-5
- [ ] La loss initiale ≈ ln(batch) et decroit jusqu'a < 0.5
- [ ] R@1 >= 95% dans les DEUX directions sur le test
- [ ] La classification zero-shot >= 95% et le parallele avec CLIP est explique
- [ ] Execution < 30 s

---

## Exercice 8 : Cross-attention multimodale — le texte interroge l'image

### Objectif

Implementer le module de fusion par cross-attention (queries texte, keys/values image) et le valider sur un scenario construit ou l'on SAIT quel patch doit etre regarde.

### Consigne

1. Implementer :

```python
def cross_modal_attention(txt_tokens, img_patches, W_q, W_k, W_v, W_o):
    """txt_tokens: (T_txt, d_txt), img_patches: (N_patch, d_img).
    Q depuis le texte, K et V depuis l'image. Pas de masque causal.
    Retourne (output (T_txt, d_txt via W_o), weights (T_txt, N_patch))."""
```

2. **Scenario construit** (le coeur de l'exercice) : 4 concepts a 4 positions.
   - patchs image : `img_patches[i] = concept_emb[c_i] + petite signature de position`, ou les `concept_emb` sont quasi orthogonaux (base canonique scaled)
   - tokens texte : chaque token "parle" d'un concept : `txt_token[t] = W_align @ concept_emb[c_t] + bruit 0.01` avec un W_align fixe connu
   - choisir `W_q` et `W_k` tels que `Q @ K.T` retombe sur la similarite de concepts (par construction : W_q = inverse/pseudo-inverse de W_align, W_k = I — a justifier en commentaire)
   - **verification** : pour chaque token texte, le poids d'attention sur le patch du MEME concept est > 0.9, et l'output (avant W_o, avec V=identite) a une cosine similarity > 0.99 avec la valeur du bon patch

3. **Carte d'attention** : afficher la matrice weights (T_txt x N_patch) en ASCII (valeurs arrondies) — elle doit etre quasi diagonale par blocs de concepts.

4. **Test d'ablation** : remplacer les K par du bruit (l'image ne porte plus l'information de concept) → les poids deviennent ~uniformes (entropie > 90% de ln(N_patch)) et l'output ne ressemble plus au bon patch (cosine < 0.7). Cela prouve que c'est bien le CONTENU des keys qui pilote l'attention, pas un artefact.

5. **Asymetrie des roles** (commentaire + mini-test) : inverser les roles (queries image, keys/values texte) et expliquer dans quel cas reel chaque direction est utilisee (Flamingo : queries texte sur keys image pour generer du texte ; certains decodeurs d'image font l'inverse). Verifier que les shapes de sortie changent en consequence : `(N_patch, ...)` au lieu de `(T_txt, ...)`.

### Criteres de reussite

- [ ] La cross-attention gere des dimensions texte/image DIFFERENTES (d_txt != d_img)
- [ ] Le scenario construit atteint poids > 0.9 sur le bon patch pour TOUS les tokens et cosine > 0.99 sur l'output
- [ ] La carte d'attention ASCII est affichee et coherente
- [ ] L'ablation des keys casse le retrieval (entropie quasi max, cosine < 0.7)
- [ ] L'explication des deux directions de cross-attention est correcte et le test de shapes passe
