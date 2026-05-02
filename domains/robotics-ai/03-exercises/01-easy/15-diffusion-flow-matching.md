# Exercice J15 — Easy : forward noising d'un point

## Objectif

Vérifier ta compréhension de la formulation DDPM en codant **le forward process** (`q_sample`) à la main, sans toucher au reste de la pipeline.

## Consigne

Écris une fonction Python pure :

```python
def forward_diffuse(x0: np.ndarray, t: int, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
    """
    Renvoie x_t (échantillon du forward process DDPM linear schedule)
    pour un point x0 donné, au timestep t.
    Doit aussi renvoyer le bruit eps utilisé.
    """
    ...
```

Détails :
1. Construis la schedule beta linéaire : `betas = np.linspace(beta_start, beta_end, T)`.
2. Calcule `alpha = 1 - beta`, puis `alpha_bar = np.cumprod(alpha)`.
3. Tire un bruit `eps ~ N(0, I)` de la même shape que `x0`.
4. Renvoie `x_t = sqrt(alpha_bar[t]) * x0 + sqrt(1 - alpha_bar[t]) * eps` et `eps`.

Teste-la sur un point 2D `x0 = np.array([1.0, 0.5])` aux timesteps `t = 0, 100, 500, 999`.

## Critères de réussite

- À `t = 0`, `x_t` doit être quasi égal à `x0` (alpha_bar ~ 1, eps multiplié par ~ 0).
- À `t = 999`, la norme de `x_t` doit être proche de `sqrt(2) ~ 1.41` (variance unitaire en 2D, donc x_t ~ N(0, I)).
- Numpy uniquement, pas de torch.
- Couvre 0 < beta_start < beta_end < 1 sans NaN.
