# Exercice J15 — Medium : DDIM sampling depuis un DDPM entraîné

## Objectif

Implémenter le **DDIM sampler déterministe** (eta=0) à partir d'un modèle DDPM déjà entraîné, et comparer la qualité visuelle / numérique vs ancestral DDPM en moins de steps (50 vs 1000).

## Consigne

Reprends le code de `02-code/15-diffusion-flow-matching.py` (ou le tien). Tu dois :

1. **Entraîner un DDPM** sur la distribution two-moons avec T = 200 (ou réutiliser un modèle déjà entraîné).
2. Implémenter une fonction :

```python
@torch.no_grad()
def my_ddim_sample(model, schedule, n_samples: int, n_steps: int) -> torch.Tensor:
    """
    DDIM deterministic sampler (eta=0).
    Sélectionne n_steps timesteps régulièrement espacés dans [0, T-1].
    Itère :
        x0_pred = (x - sqrt(1 - alpha_bar_t) * eps_pred) / sqrt(alpha_bar_t)
        x_{t_next} = sqrt(alpha_bar_t_next) * x0_pred + sqrt(1 - alpha_bar_t_next) * eps_pred
    """
    ...
```

3. Génère 1000 échantillons avec `n_steps = 10`, `50`, `200` (= ancestral équivalent).
4. Compare numériquement :
   - Mean & std des échantillons générés vs distribution réelle.
   - Pourcentage de samples dans la "bonne" lune (top vs bottom — utiliser la coordonnée y comme heuristique).

## Critères de réussite

- À 50 steps, la qualité (mean/std proches du réel à ±10%) doit être comparable à l'ancestral 200-step.
- À 10 steps, la dégradation doit être **observable** (mean/std s'écartent davantage). Documente-le.
- Le sampler tourne en `torch.no_grad()` mode, pas de fuite mémoire grad.
- Reproductibilité : même seed → mêmes échantillons (DDIM est déterministe).
