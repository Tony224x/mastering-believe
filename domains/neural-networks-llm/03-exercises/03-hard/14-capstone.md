# Exercices Hard — Jour 14 : Capstone (extensions de mini-LLaMA)

> Ces exercices ETENDENT le mini-LLaMA de `02-code/14-capstone.py`. Le code de
> reference est en PyTorch ; les solutions fournies implementent les composants
> en NumPy (forward ET backward a la main) pour etre runnable sans framework.

---

## Exercice 7 : RoPE — implementer, deriver la propriete cle et la verifier

### Objectif

Implementer RoPE (Rotary Positional Embedding) en NumPy, PROUVER sa propriete fondamentale (l'attention ne depend que de la position RELATIVE), et verifier la stabilite de l'extrapolation.

### Consigne

1. **Implementer RoPE** (convention interleaved-pairs, comme `02-code/14-capstone.py`) :
   - `precompute_rope(head_dim, max_seq_len, base=10000)` → cos, sin de shape `(seq, head_dim/2)`
   - `apply_rope(x, cos, sin)` qui rote les paires (0,1), (2,3), ...

2. **Propriete cle — invariance par translation** : RoPE garantit que le produit scalaire `<RoPE(q, m), RoPE(k, n)>` ne depend que de `m - n` (position relative), PAS de `m` et `n` absolus.
   - Verifier numeriquement : pour un meme `q`, `k` et un meme decalage `delta = m - n`, le produit scalaire `<RoPE(q,m), RoPE(k,n)>` est constant quand on fait glisser `(m, n)` (ex: (5,3) et (10,8) donnent le meme score).
   - Le verifier pour plusieurs delta et plusieurs paires (q, k).

3. **Derivation (a ecrire en commentaire)** : pour une paire de dimensions (rotation 2D d'angle `theta`), montrer pourquoi `R(m theta)^T R(n theta) = R((n-m) theta)`, ce qui prouve l'invariance.

4. **Frequences** : afficher les angles `theta_j = base^(-2j/head_dim)`. Montrer que les premieres dimensions tournent vite (haute frequence, position locale) et les dernieres lentement (basse frequence, position globale). Lien avec les embeddings sinusoidaux.

5. **Extrapolation** : appliquer RoPE a des positions AU-DELA de `max_seq_len` (ex: entrainer la table jusqu'a 128, tester a 256). Montrer que les angles restent bien definis (RoPE extrapole "mecaniquement") mais que les scores deviennent imprevisibles hors distribution → motivation du NTK-aware / YaRN scaling (cours J11).

### Criteres de reussite

- [ ] RoPE (precompute + apply, convention interleaved) est correct
- [ ] L'invariance par translation est verifiee numeriquement (meme delta → meme score)
- [ ] La derivation R(mθ)^T R(nθ) = R((n-m)θ) est ecrite et correcte
- [ ] Le spectre de frequences (rapide → lent) est affiche et explique
- [ ] L'extrapolation au-dela de max_seq_len est testee et discutee (lien NTK/YaRN)

---

## Exercice 8 : Mini-LLaMA NumPy entrainable — bloc complet + ablations

### Objectif

Implementer un bloc de mini-LLaMA simplifie EN NUMPY avec forward ET backward complets (RMSNorm + attention causale + MLP + residuals), l'entrainer sur un corpus char-level, et mener des ablations qui reproduisent les conclusions du paper LLaMA.

### Consigne

1. **Composants avec backward** (NumPy, gradient a la main) :
   - `RMSNorm` : `x / sqrt(mean(x^2) + eps) * gamma` + son gradient
   - Attention causale (mono ou multi-head, sans RoPE pour simplifier le backward, ou avec RoPE en bonus) + backward du softmax + matmuls
   - MLP (au choix : ReLU classique pour un backward simple, ou SwiGLU en bonus) + backward
   - Connexions residuelles pre-norm : `x + attn(norm(x))`, `x + mlp(norm(x))`
   - Verifier le gradient de bout en bout par difference finie sur un petit modele (ecart < 1e-4)

2. **Modele complet** : token embedding + N blocs + final norm + lm_head (tie avec l'embedding). Forward + backward + un optimiseur (SGD ou Adam NumPy).

3. **Entrainement** : entrainer sur un petit corpus char-level. Tracer la loss et la perplexite. Generer un echantillon apres entrainement.

4. **Ablations** (reproduire l'esprit des ablations du paper LLaMA) :
   - **RMSNorm vs LayerNorm** : remplacer RMSNorm par LayerNorm. Comparer la vitesse de convergence et la loss finale. (Le paper trouve RMSNorm ~equivalent mais plus rapide.)
   - **Pre-norm vs post-norm** : `x + f(norm(x))` vs `norm(x + f(x))`. Montrer que pre-norm est plus stable (la loss ne diverge pas) sur un modele profond.
   - **Avec vs sans connexions residuelles** : retirer les residuals et montrer que l'entrainement se degrade fortement (vanishing/exploding du gradient).

5. **Tableau de synthese** : pour chaque ablation, donner loss finale + observation, et relier a la decision architecturale de LLaMA.

### Criteres de reussite

- [ ] RMSNorm, attention causale, MLP, residuals ont un forward ET un backward corrects
- [ ] Le gradient de bout en bout passe le test de difference finie (< 1e-4)
- [ ] Le modele s'entraine (loss + perplexite descendent) sur un corpus char-level
- [ ] Les 3 ablations (RMSNorm/LayerNorm, pre/post-norm, avec/sans residual) sont menees
- [ ] Le tableau de synthese relie chaque resultat a un choix de LLaMA
