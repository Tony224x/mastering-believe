# Exercice EASY — Cinematique directe

## Objectif

Coder la FK d'un bras planaire 3-DOF (3 segments en serie dans le plan XY) et calculer la position de l'effecteur pour une configuration donnee.

## Consigne

Soit un bras planaire avec 3 segments revolutes en serie :
- Longueurs : `L1 = 0.5`, `L2 = 0.4`, `L3 = 0.3`
- Angles relatifs (chaque `θᵢ` est l'angle du segment `i` par rapport au precedent) : `θ₁ = 45°`, `θ₂ = -30°`, `θ₃ = 60°`

Implemente une fonction Python :

```python
def fk_3dof_planar(theta1: float, theta2: float, theta3: float,
                   L1: float = 0.5, L2: float = 0.4, L3: float = 0.3) -> tuple[float, float, float]:
    """Retourne (x, y, phi) ou phi est l'orientation absolue du dernier segment."""
    ...
```

Calcule et affiche la pose de l'effecteur (x, y) ainsi que l'orientation `φ = θ₁ + θ₂ + θ₃` pour la configuration ci-dessus.

## Criteres de reussite

- La fonction prend les angles en radians.
- Tu utilises `numpy` (cos/sin sur scalaires).
- Tu obtiens trois valeurs numeriques pour `(x, y, φ)`.
- Tu verifies a la main (calculatrice ou stylo) au moins l'une des coordonnees, en suivant la formule generalisee :

  ```
  x = L1·cos(θ₁) + L2·cos(θ₁+θ₂) + L3·cos(θ₁+θ₂+θ₃)
  y = L1·sin(θ₁) + L2·sin(θ₁+θ₂) + L3·sin(θ₁+θ₂+θ₃)
  ```

## Indice

Comme dans la theorie (J3 §1), les angles relatifs s'**additionnent** pour donner les angles absolus. Donc `α₃ = θ₁ + θ₂ + θ₃` est l'orientation absolue du dernier segment dans le plan XY.
