# Exercice J16 — Medium : BC vs Diffusion Policy sur distribution bimodale

## Objectif

Reproduire empiriquement la **thèse centrale** du papier Chi 2023 (REFERENCES.md #19) : sur une distribution d'actions multimodale, Behavior Cloning par MSE s'effondre vers la moyenne des modes, tandis que Diffusion Policy préserve la multimodalité.

## Consigne

1. Réutilise `make_toy_dataset` du fichier `02-code/16-diffusion-policy.py` (dataset bimodal "stratégie A vs B").
2. Implémente un **BC baseline** : un MLP `f_θ : obs → action_chunk` (2 couches cachées, 128 unités, Mish), entraîné en MSE sur `(obs, actions)`.
3. Entraîne BC pendant 20 epochs, puis pour `obs = 0` génère 64 prédictions (déterministes, donc identiques) — note la trajectoire moyenne.
4. Entraîne Diffusion Policy en réutilisant `train()` du fichier code (20 epochs aussi).
5. À l'évaluation, pour `obs = 0`, échantillonne 64 séquences avec `ddpm_sample` et plot :
   - les 64 trajectoires Diffusion Policy en gris transparent,
   - la trajectoire BC unique en rouge épais,
   - les deux modes "ground truth" (A et B) en vert pointillé.
6. Calcule la variance des `argmax` temporels sur l'axe x pour BC vs Diffusion Policy. Commente.

## Criteres de reussite

- BC produit **une seule trajectoire** au milieu des deux modes (pas une cardioïde, plutôt un mélange aplati).
- Diffusion Policy produit un nuage couvrant **les deux modes** (visuellement on distingue 2 groupes).
- Quantifié : la variance de l'argmax pour Diffusion Policy est au moins 10× celle de BC (qui est ≈ 0 puisque les 64 prédictions BC sont identiques). Tu imprimes les deux valeurs.
- Test de bimodalité : en classant les 64 samples Diffusion Policy par mode le plus proche (distance L2 aux deux ground truths A et B), chaque mode reçoit entre 20% et 80% des samples.
- En 3 phrases écrites en commentaire de fin de script, tu expliques pourquoi BC échoue (perte MSE = log-vraisemblance d'une gaussienne unimodale → prédiction = moyenne des modes) et pourquoi DDPM ne rencontre pas ce problème (apprend à débruiter un échantillon du mélange, pas à régresser sa moyenne).
