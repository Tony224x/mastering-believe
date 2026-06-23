# Exercice 06 (medium) - LQR sur masse-ressort-amortisseur instable

## Objectif

Calculer un gain LQR pour un systeme masse-ressort instable (raideur negative), comparer a un PID Ziegler-Nichols, et explorer l'effet du choix de Q et R.

## Consigne

Soit le systeme :

```
m x_ddot = -k x - c x_dot + u
m = 1 kg, k = -2 N/m (raideur NEGATIVE => repulsif), c = 0.1 N.s/m
```

Le ressort "pousse" hors de l'equilibre x = 0 : systeme instable a stabiliser.

1. Ecrire la forme d'etat `x_dot = A x + B u` avec `x = [x, x_dot]^T`. Verifier que `A` a une valeur propre a partie reelle > 0.
2. Calculer trois gains LQR avec `scipy.linalg.solve_continuous_are` :
   - **(L1)** `Q = diag(1, 1), R = 1` (effort cher)
   - **(L2)** `Q = diag(100, 1), R = 1` (etat cher)
   - **(L3)** `Q = diag(10, 1), R = 0.01` (action quasi-gratuite)
   Pour chacun, afficher `K` et les valeurs propres boucle fermee.
3. Simuler les 3 LQR et un PID `(Kp=20, Ki=0, Kd=5)` depuis `x0 = [0.5, 0]^T`, horizon 5 s, dt = 0.005 s.
4. Comparer dans un tableau : settling time (seuil 0.02), overshoot, RMSE, couple max (`max |u|`).
5. Repondre par ecrit (5-10 lignes) :
   - Quel LQR est le plus rapide ? Pourquoi ?
   - Quel LQR utilise le moins d'effort ? Pourquoi ?
   - Le PID gagne-t-il sur un critere ? (transparence, ancrage industriel, modele non requis...)

## Criteres de reussite

- Code tourne en < 5 s.
- Les 3 LQR stabilisent (toutes valeurs propres boucle fermee a partie reelle negative).
- Tableau metrique correct : (L2) plus rapide que (L1), (L3) couple max > (L1).
- Reponse ecrite identifie correctement le compromis Q/R et reconnait l'avantage pratique du PID (pas de modele requis).

## Indices

- L'heuristique de Bryson : `Q_ii = 1/x_i_max^2`, `R_jj = 1/u_j_max^2`. Plus `Q/R` grand, plus le LQR est agressif.
- Le PID Ziegler-Nichols ici n'est pas optimal — il y a oscillation residuelle car le systeme n'est ni du second ordre standard, ni passif.
- Reference : [Tedrake, ch. 7], [Siciliano et al., 2009, ch. 8.5].
