# Exercice J7 — Easy : projection pinhole et back-projection

## Objectif

Manipuler la matrice intrinseque K et verifier que le passage 3D -> pixel
puis pixel + depth -> 3D est bien reciproque.

## Consigne

1. Construis une matrice intrinseque pour une camera 640x480 avec une
   focale equivalente 500 px (`fx = fy = 500`) et le principal point au
   centre de l'image.

2. Cree manuellement 5 points 3D dans le repere camera (a toi de choisir,
   mais leurs Z doivent tous etre > 0, par exemple `Z` entre 1.0 m et 3.0 m).

3. Projette ces 5 points en pixels avec la formule pinhole :
   `u = fx * X/Z + cx`, `v = fy * Y/Z + cy`. Imprime `(u, v, depth)`.

4. Reconstruis les 5 points 3D a partir de `(u, v, depth)` via la formule
   inverse :
   ```
   X = (u - cx) * depth / fx
   Y = (v - cy) * depth / fy
   Z = depth
   ```

5. Compare numpy `np.allclose(reconstructed, original, atol=1e-6)`. Tu
   devrais obtenir `True` (a la precision flottante pres).

## Criteres de reussite

- Le script tourne sans erreur.
- L'assertion `np.allclose` est `True`.
- Tu sais expliquer pourquoi un point a `(X=0, Y=0, Z=2)` se projette
  exactement au principal point `(cx, cy)`.

## Bonus

Que se passe-t-il si tu mets `Z = 0` pour un point ? Quel pixel obtiens-tu ?
Pourquoi le code de production filtre `Z > epsilon` avant de projeter ?
