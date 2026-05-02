# Exercice J9 — Medium : Implementer Value Iteration sur GridWorld 5x5

## Objectif

Coder **value iteration from scratch** sur un GridWorld 5x5 nouveau, avec une configuration differente (plusieurs lavas, recompense terminale variable). Comprendre l'effet de `gamma` et de la `living_penalty` sur la politique optimale.

## Consigne

### Setup

GridWorld 5x5 avec :

```
+---+---+---+---+---+
| S | . | . | . | . |
+---+---+---+---+---+
| . | X | . | X | . |
+---+---+---+---+---+
| . | . | . | . | . |
+---+---+---+---+---+
| . | X | . | . | . |
+---+---+---+---+---+
| . | . | . | . | G |
+---+---+---+---+---+
```

- `S = (0, 0)` (start)
- `G = (4, 4)` (goal, R = +1)
- 3 lavas en `(1, 1)`, `(1, 3)`, `(3, 1)` (R = -1 chacun)
- transitions stochastiques 80/10/10 (comme dans le cours)
- 4 actions
- terminal sur goal et lavas

### Travail

1. **Implementer** `build_mdp_5x5()` qui retourne `P` et `R` au format `(N_STATES, N_ACTIONS, N_STATES)`.
2. **Implementer** `value_iteration(P, R, gamma, eps)` (tu peux reutiliser le code du cours).
3. **Faire tourner** value iteration pour trois configurations :
   - (a) `gamma = 0.9`, `living_penalty = 0`
   - (b) `gamma = 0.9`, `living_penalty = -0.04`
   - (c) `gamma = 0.5`, `living_penalty = -0.04`
4. **Afficher** la politique optimale pour chacune et **decrire en 2-3 phrases** comment elle change.

### Questions a repondre dans un commentaire ou markdown final

- Q1 : pourquoi avec `living_penalty = 0` la politique a-t-elle parfois tendance a "errer" ?
- Q2 : qu'est-ce qui se passe en (c) avec `gamma = 0.5` ? Le robot prend-il plus de risques pres du goal ou en est-il dissuade ?
- Q3 : combien d'iterations faut-il pour converger a `eps = 1e-6` dans chaque config ? Justifie le lien avec `gamma`.

## Criteres de reussite

- `build_mdp_5x5` produit un `P` qui satisfait `P.sum(axis=2) == 1` partout.
- Pour `gamma = 0.9`, `living_penalty = -0.04`, la politique optimale evite clairement les 3 lavas et converge en moins de 50 iterations.
- Pour `gamma = 0.5`, le robot a une politique plus "myope" et certaines cases pres des lavas peuvent avoir une politique surprenante (pas directement vers le goal).
- Tu rapportes les nombres d'iterations et tu fais le lien avec le taux de contraction `gamma`.
