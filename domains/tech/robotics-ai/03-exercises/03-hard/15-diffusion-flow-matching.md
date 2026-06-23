# Exercice J15 — Hard : Classifier-Free Guidance sur 2 distributions

## Objectif

Implémenter **Classifier-Free Guidance (CFG)** sur un modèle DDPM 2D conditionné par une variable de classe binaire `c ∈ {0, 1}` (deux distributions différentes : two-moons et anneau circulaire), et observer l'effet de la guidance scale `w` sur la précision vs diversité.

## Consigne

1. **Construis un dataset mixte** :
   - Classe `c=0` : two-moons (réutilise `make_two_moons` du 02-code).
   - Classe `c=1` : anneau circulaire (rayon 1, bruit gaussien 0.05).
   - 4000 points par classe, total 8000.

2. **Modifie le denoiser** pour qu'il accepte une condition de classe `c`. Embed `c` via `nn.Embedding(num_classes=3, ...)` où l'index 2 est réservé au token "null" (inconditionnel).

3. **Training avec dropout de la condition** :
   - À chaque batch, avec probabilité `p_drop = 0.15`, remplace `c` par le token null (= apprentissage conjoint conditionnel et inconditionnel).
   - Loss : `MSE(eps_pred, eps_target)` standard.

4. **Sampling avec CFG** :

```python
def sample_cfg(model, schedule, n, c_target: int, w: float):
    """
    À chaque step :
        eps_cond   = model(x, t, c=c_target)
        eps_uncond = model(x, t, c=null)
        eps_guided = (1 + w) * eps_cond - w * eps_uncond
    Puis update DDPM standard avec eps_guided.
    """
    ...
```

5. **Évaluation** : génère 500 samples conditionnés sur `c=0` (moons) et 500 sur `c=1` (anneau), pour `w ∈ {0, 1, 3, 7}`. Mesure :
   - Précision : % de samples dans la "bonne" zone (anneau de rayon ~1 ou two-moons distinctes — définis un test géométrique simple).
   - Diversité : variance totale des samples par classe.

## Critères de réussite

- Précision croît avec `w` (de ~50% à `w=0`, vers >85% à `w=3`).
- Diversité décroît avec `w` (variance baisse).
- À `w=7`, on observe du **mode collapse** : les samples se concentrent sur une partie de la distribution. Documente.
- Le code doit gérer proprement les 3 cas de `c` (0, 1, null) sans branche if/else dans le forward (utilise un seul `nn.Embedding(3, ...)`).
- Bonus : reproduis l'effet *guidance trick* — montre une figure 2x4 (mode x w) qui rend l'effet visible.
