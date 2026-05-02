# Exercice 06 (easy) - PID sur double integrateur

## Objectif

Implementer un PID articulaire sur un double integrateur (`x_ddot = u`) et comprendre les effets respectifs de Kp, Ki, Kd sur la reponse.

## Consigne

1. Modeliser le systeme `x_ddot = u`, etat `[x, x_dot]^T`. Integration semi-implicit Euler `dt = 0.01 s`, horizon `T = 5 s`. Condition initiale `x = 1.0, x_dot = 0`.
2. Coder un controleur PID a 3 gains `(Kp, Ki, Kd)` qui regule `x` vers 0. Integrateur classique (pas d'anti-windup pour cet exercice).
3. Faire 4 simulations en faisant varier les gains, comparer les courbes :
   - **(a)** `Kp=10, Ki=0, Kd=0` (P seul) — observer oscillation entretenue.
   - **(b)** `Kp=10, Ki=0, Kd=2` (PD) — observer amortissement.
   - **(c)** `Kp=10, Ki=5, Kd=2` (PID complet).
   - **(d)** `Kp=100, Ki=0, Kd=2` (Kp eleve) — montrer overshoot.
4. Calculer pour chaque config : settling time (|x| < 0.02), overshoot max, RMSE.
5. Repondre par ecrit : pourquoi un controleur P seul oscille indefiniment sur un double integrateur sans frottement ?

## Criteres de reussite

- Le code tourne en moins de 2 s.
- Les 4 courbes sont produites (matplotlib ou print structure).
- La config (b) PD converge sans overshoot.
- La config (a) P-seul oscille (RMSE > 0.5 sur tout l'horizon).
- La reponse ecrite mentionne que le double integrateur est un systeme conservatif : sans terme dissipatif (Kd ou frottement), l'energie injectee par P se conserve sous forme oscillatoire.

## Indices

- Le double integrateur a deux poles a l'origine. Avec un retour P pur, les poles boucle fermee deviennent `±j sqrt(Kp)`, purement imaginaires => oscillation pure.
- Avec PD, les poles deviennent `(-Kd ± sqrt(Kd^2 - 4 Kp))/2`. Pour `Kd^2 < 4 Kp`, oscillation amortie ; au-dela, regime pseudo-aperiodique.
- Reference : [Siciliano et al., 2009, ch. 8.3].
