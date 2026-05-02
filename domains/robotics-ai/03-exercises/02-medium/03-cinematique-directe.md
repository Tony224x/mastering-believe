# Exercice MEDIUM — FK 3D avec PoE

## Objectif

Implementer la FK Product of Exponentials (Lynch) pour un bras 3D **RRR spherique** (3 joints revolutes dont les axes ne sont pas tous paralleles), et verifier la coherence sur deux configurations.

## Consigne

On considere un bras 3D simplifie inspire d'une « tete spherique » :
- Joint 1 : revolute autour de l'axe `z` mondial, ancre en `p₁ = (0, 0, 0)`
- Joint 2 : revolute autour de l'axe `y` mondial, ancre en `p₂ = (0, 0, 0.5)` (apres un offset vertical de 0.5 sur z)
- Joint 3 : revolute autour de l'axe `y` mondial (parallele au joint 2 quand `q=0`), ancre en `p₃ = (0, 0, 0.9)`
- Pose home `M` de l'effecteur en `q = 0` :
  - rotation = identite
  - translation = `(0, 0, 1.2)` (l'effecteur est aligne sur l'axe z)

1. Construis les 3 screws spatiaux `Sᵢ = (ωᵢ, -ωᵢ × pᵢ)`.
2. Implemente (ou reutilise depuis le code du jour) :
   - `expm_so3(omega, theta)` (Rodrigues)
   - `expm_se3(screw, theta)`
   - `fk_poe(M, screws, q)`
3. Calcule `T(q)` pour :
   - `q_a = (0, 0, 0)` → tu dois retrouver `M` exactement.
   - `q_b = (π/4, π/6, -π/3)` → affiche la pose 4×4.
4. **Test de coherence** : verifie que pour `q = (π, 0, 0)` (rotation pi du joint 1 autour de z), la position de l'effecteur reste `(0, 0, 1.2)` (car l'effecteur est sur l'axe de rotation du joint 1).

## Criteres de reussite

- Les fonctions prennent des arguments en radians et numpy arrays.
- `T(q_a)` est exactement `M` (a 1e-12 pres).
- Pour `q = (π, 0, 0)`, la translation de `T` est `(0, 0, 1.2)` a 1e-9 pres.
- Tu commentes pourquoi `vᵢ = -ωᵢ × pᵢ` (et pas `+`).

## Indice

Si la pose pour `q_a = 0` n'est pas `M`, tu as probablement compose `M` au mauvais bout. La formule PoE est :

```
T(q) = e^([S₁]q₁) · ... · e^([Sₙ]qₙ) · M
```

`M` est tout a droite, et chaque `e^([Sᵢ]·0) = I`, donc `T(0) = I · ... · I · M = M`.
