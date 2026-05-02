# Exercice J5 — Easy : pendule numerique vs analytique

## Objectif

Verifier en pratique que la solution analytique petit-angle d'un pendule simple ne tient que pour... les petits angles. Tu vas integrer numeriquement l'equation exacte `θ̈ = -(g/L) sin(θ)` et la comparer a l'approximation lineaire `θ(t) = θ_0 cos(ωt)`.

## Consigne

Ecris un script Python (`numpy` seul, pas de `mujoco`) qui :

1. Definit la dynamique exacte d'un pendule simple : masse `m=1.0 kg`, longueur `L=1.0 m`, `g=9.81 m/s²`, sans frottement.
2. Integre l'equation exacte pendant 5 secondes avec un pas `dt = 1e-3` en utilisant un schema **Euler semi-implicite** :
   ```
   θ̇ ← θ̇ + dt · (-(g/L) sin(θ))
   θ  ← θ  + dt · θ̇
   ```
3. Calcule en parallele la solution analytique petit-angle `θ_analytique(t) = θ_0 · cos(√(g/L) · t)`.
4. Pour deux conditions initiales — `θ_0 = 0.1 rad` et `θ_0 = 1.5 rad` — affiche dans la console pour `t ∈ {0, 1, 2, 3, 4, 5} s` :
   - `θ_numerique(t)`
   - `θ_analytique(t)`
   - `|écart|`
5. Calcule a chaque pas l'energie totale `E = ½ m L² θ̇² + m g L (1 − cos(θ))` et affiche la derive relative `(E_final − E_initial) / E_initial` a la fin.

## Criteres de reussite

- A `θ_0 = 0.1 rad`, l'ecart numerique-analytique reste < `5e-3 rad` sur les 5 s.
- A `θ_0 = 1.5 rad`, l'ecart depasse `0.3 rad` apres quelques periodes (preuve que l'approximation petit-angle craque).
- La derive d'energie est < 1 % dans les deux cas (preuve que le schema semi-implicite est conservatif).
- Aucun import autre que `numpy` et `math`.

## Indices

- L'equation exacte n'a pas de solution analytique elementaire (elle utilise des integrales elliptiques) — c'est bien pourquoi on simule.
- Pour Euler explicite tu ferais `θ ← θ + dt·θ̇` puis `θ̇ ← θ̇ + dt·θ̈(θ)` ; le semi-implicite **utilise la nouvelle vitesse** pour mettre a jour θ — l'ordre compte.
- Si l'energie diverge, c'est que tu as melange l'ordre des updates ou que `dt` est trop grand.
