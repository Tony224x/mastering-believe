# Exercice J9 — Easy : Bellman a la main + petit MDP 3 etats

## Objectif

Verifier que tu maitrises la **definition d'un MDP** et que tu sais appliquer **l'equation de Bellman expected** sur un cas a la main.

## Consigne

### Partie A — Definition

On considere un mini-MDP **deterministe** a 3 etats : `S = {A, B, C}`, 2 actions : `a1` (gauche), `a2` (droite). `C` est terminal.

Transitions deterministes :
- depuis `A` : `a1 -> A` (boucle), `a2 -> B`
- depuis `B` : `a1 -> A`, `a2 -> C`
- depuis `C` : terminal (self-loop, recompense 0)

Recompenses :
- `R(B, a2, C) = +10`
- toute autre transition : `R = 0`

`gamma = 0.5`.

**Question A.1** Ecris explicitement les 5 elements du tuple `(S, A, P, R, gamma)` pour ce MDP.

### Partie B — Evaluation d'une politique

Soit la politique deterministe `pi` :
- `pi(A) = a2` (aller a droite depuis A)
- `pi(B) = a2` (aller a droite depuis B)
- `pi(C) = a1` (peu importe, terminal)

**Question B.1** Ecris l'equation de Bellman *expected* pour `V_pi(A)` et `V_pi(B)` (sans simplifier).

**Question B.2** Resous le systeme. Donne `V_pi(A)`, `V_pi(B)`, `V_pi(C)`.

### Partie C — Politique alternative

Soit la politique `pi'` :
- `pi'(A) = a1` (boucler sur A)
- `pi'(B) = a2`
- `pi'(C) = a1`

**Question C.1** Que vaut `V_{pi'}(A)` ? Justifie.

**Question C.2** Quelle politique vaut mieux entre `pi` et `pi'` ? Pourquoi cela illustre-t-il l'utilite de `gamma < 1` ?

## Criteres de reussite

- Le tuple MDP est complet : transitions et recompenses listees pour **toutes** les paires `(s, a)`.
- L'equation de Bellman est ecrite sous la forme `V(s) = R + gamma * V(s')` (deterministe ici, donc pas de somme).
- `V_pi(A) = 2.5`, `V_pi(B) = 5.0`, `V_pi(C) = 0`.
- Tu identifies que `V_{pi'}(A) = 0` (boucle infinie sans recompense, somme actualisee = 0).
- Tu argumentes que `pi` est strictement meilleure et que sans `gamma < 1`, la sommation diverge ou perd son sens.
