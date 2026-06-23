# Exercice J8 — Easy : configuration space et collision check

## Objectif

Comprendre concretement la notion de **configuration space** et coder la
primitive `is_collision_free(q)` qui sert de fondation a tout planner
sampling-based.

## Consigne

Tu disposes d'un robot ponctuel 2D evoluant dans une boite `[0, 10] x [0, 10]`
et de trois obstacles rectangulaires axes-aligned :

```python
obstacles = [
    (2.0, 2.0, 4.0, 5.0),   # (x_min, y_min, x_max, y_max)
    (6.0, 1.0, 8.0, 4.0),
    (4.5, 6.0, 7.5, 8.0),
]
```

Ecris une fonction :

```python
def in_collision(q: tuple[float, float], obstacles: list[tuple]) -> bool:
    """Renvoie True si q est a l'interieur d'au moins un obstacle."""
```

Puis :

1. Genere 1000 points uniformement dans `[0, 10]^2` (numpy seed = 0).
2. Compte combien sont dans `C_free` et combien dans `C_obs`.
3. Compare le ratio `|C_free| / |C|` empirique avec le ratio analytique
   (`1 - somme_aires / 100`).
4. Trace les points en deux couleurs (bleu = libre, rouge = collision) avec
   matplotlib. Affiche aussi les rectangles d'obstacles.

## Criteres de reussite

- La fonction `in_collision` renvoie le bon booleen sur 4-5 cas que tu te
  fixes (point clairement dans un obstacle, point clairement libre, point sur
  la frontiere — comportement documente).
- Le ratio empirique est proche du ratio analytique a 2-3% pres avec 1000
  samples.
- La visualisation distingue clairement les deux populations.

## Indices

- Tester `q in rect` revient a 4 comparaisons : `xmin <= x <= xmax` ET
  `ymin <= y <= ymax`.
- Pour le ratio : aire totale `100`, aire de chaque obstacle `(xmax-xmin) *
  (ymax-ymin)`.
