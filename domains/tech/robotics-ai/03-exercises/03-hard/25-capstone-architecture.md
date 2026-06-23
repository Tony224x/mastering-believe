# Exercice hard J25 — Transformer denoiser ou Classifier-Free Guidance

## Objectif

Aller au-dela du UNet 1D fourni et explorer une variante d'architecture **non-triviale**. Choisir UNE des deux pistes :

- **Piste A (architecture)** : remplacer le UNet 1D par un **Transformer denoiser**.
- **Piste B (sampling)** : ajouter du **Classifier-Free Guidance** (CFG) et mesurer l'effet sur des actions generees.

## Piste A — Transformer denoiser

### Consigne

1. Creer `TransformerDenoiser` qui respecte le meme contrat que `ConditionalUNet1D` :

    ```python
    forward(action: (B, T_act, A), timestep: (B,), global_cond: (B, C)) -> (B, T_act, A)
    ```

2. Architecture :
   - Project `action` -> `(B, T_act, d_model)` via une `Linear`.
   - Ajouter un **positional embedding** apprenable de longueur `T_act`.
   - Concat ou cross-attention avec `global_cond + time_emb` (deux options legitimes).
   - 4-6 couches de `nn.TransformerEncoderLayer` avec `d_model = 256`, `nhead = 4`.
   - Final `Linear` -> `(B, T_act, A)`.
3. Plug dans la `DiffusionPolicy` (juste swap `self.denoiser`).
4. Verifier que `predict_action` tourne, et comparer le nombre de parametres avec le UNet 1D par defaut.

### Criteres de reussite

- Forward sans erreur, shapes coherentes, `compute_loss` retourne un scalaire fini.
- Discussion ecrite (5-10 phrases) :
  - Quand un Transformer est-il **meilleur** que le UNet 1D ?
  - Quel est le **cout d'attention** en `O(T_act^2 * d_model)` et pourquoi ce n'est PAS un probleme avec `T_act = 16` ?
  - Pourquoi Diffusion Policy choisit le UNet 1D **par defaut** sur PushT image-based ?

## Piste B — Classifier-Free Guidance

### Consigne

1. Modifier l'entrainement (sans le lancer pour de vrai) pour qu'avec une probabilite `p_uncond = 0.1`, le `global_cond` soit remplace par un **token "null"** (par ex. un buffer `nn.Parameter` de shape `cond_dim` initialise a zero, appris en parallele).
2. A l'inference, sampler en deux passes : une avec cond, une avec null. Combiner :

    ```
    eps_guided = eps_uncond + w * (eps_cond - eps_uncond)
    ```

   avec `w` (`guidance_scale`) typiquement 1.0 - 3.0 (`w = 0` desactive le guidance, `w = 1` pas de boost, `w > 1` accentue la condition).

3. Generer des actions avec `w in {0.0, 1.0, 2.0, 5.0}` sur un meme batch et **mesurer la dispersion** : variance des actions generees a `cond` fixe sur 32 echantillons. CFG fort -> moins de diversite mais plus d'adherence au cond.

### Criteres de reussite

- L'entrainement-jouet (1 epoch sur le fake batch suffit pour debugger) tourne sans NaN.
- Le sampling avec guidance produit des shapes correctes.
- Plot ou tableau qui montre la variance decroissante quand `w` augmente.
- Discussion : pourquoi CFG aide en imitation learning, et quel est le **risque** de sur-conditioning ?

## Sources

- Ho & Salimans 2022, "Classifier-Free Diffusion Guidance" (arxiv 2207.12598).
- Vaswani et al. 2017, "Attention Is All You Need" pour le transformer block.
- Repo `real-stanford/diffusion_policy/diffusion_policy/model/diffusion/transformer_for_diffusion.py` pour reference.
