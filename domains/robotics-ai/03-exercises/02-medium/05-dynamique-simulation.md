# Exercice J5 — Medium : matrice d'inertie d'un bras 2-DOF en MuJoCo

## Objectif

Manipuler `M(q)` directement : extraire la matrice d'inertie d'un bras 2-DOF a chaque pas, verifier ses proprietes structurelles (symetrie, definite positivite), et observer comment elle change avec la configuration `q`.

## Consigne

1. Ecris un MJCF inline pour un bras planaire 2-DOF (deux liens en serie, hinges autour de `y`, longueurs et masses de ton choix mais documentees en commentaire). Active `<flag energy="enable"/>`.
2. Charge le modele avec `mujoco.MjModel.from_xml_string(...)`.
3. Ecris une fonction `inertia_matrix(model, data) -> np.ndarray` qui retourne `M(q) ∈ ℝ^(nv × nv)` en utilisant `mujoco.mj_fullM`.
4. Pour `q = [0.0, 0.0]`, `q = [π/4, π/4]`, `q = [π/2, π/2]`, et `q = [0, π/2]` :
   - Set `data.qpos[:] = q`, appelle `mj_forward` (sinon les positions Cartesiennes utilisees pour calculer M ne sont pas a jour).
   - Affiche `M(q)`.
   - Verifie numeriquement que `M = Mᵀ` (`np.allclose(M, M.T)`).
   - Verifie que `M` est definie positive (toutes les valeurs propres > 0 via `np.linalg.eigvalsh`).
5. Pour la configuration ou `M` a la plus grande condition number, commente en une ligne pourquoi cette configuration est "plus difficile a controler" qu'une autre.

## Criteres de reussite

- `M` symetrique et definie positive dans toutes les configurations testees.
- `M` change effectivement avec `q` (les diagonales et les off-diagonales bougent — sinon c'est que tes liens sont mal connectes).
- Le print final identifie la config la plus mal conditionnee et l'explique en termes de couplage entre articulations.

## Indices

- `mj_fullM` attend une matrice pre-allouee `np.zeros((nv, nv))`. Ne reutilise pas la meme entre appels sans la remettre a zero.
- Sur un bras 2-DOF planaire au sol-vertical, le terme diagonal `M[0,0]` augmente quand le second lien s'eloigne du joint 1 — l'intuition mecanique : plus le levier est long, plus l'inertie reflechie est grande.
- Condition number = `λ_max / λ_min`. Une `M` mal conditionnee = un controleur a gains fixes performera tres differemment selon la direction articulaire.
