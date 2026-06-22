# Exercice 06 (hard) - Computed torque control sur bras 2-DOF

## Objectif

Implementer un controleur computed torque pour faire tracker une trajectoire sinusoidale a un bras planaire 2-DOF, et le comparer a un PID independant par axe.

## Contexte

On reprend le bras 2-DOF planaire vu en J5, dynamique de Lagrange :

```
M(q) q_ddot + C(q, q_dot) q_dot + g(q) = tau
```

Parametres : `m1 = m2 = 1 kg`, `l1 = l2 = 1 m`, `lc1 = lc2 = 0.5 m` (centres de masse au milieu), `I1 = I2 = m l^2 / 12` (cylindres minces), `g = 9.81 m/s^2`. Le bras est vertical : la gravite agit sur les segments.

## Consigne

1. Ecrire `M(q)`, `C(q, q_dot)`, `g(q)` pour ce bras 2-DOF (formules dans Siciliano ch. 7 ou Lynch ch. 8). Verifier que `M` est symetrique definie positive en quelques configurations.
2. Generer une trajectoire de reference sur 4 secondes :
   ```
   q1_d(t) = 0.5 sin(2 pi t / 4)
   q2_d(t) = 0.3 cos(2 pi t / 4)
   ```
   Calculer analytiquement `q_dot_d(t)` et `q_ddot_d(t)`.
3. Implementer **deux controleurs** :
   - **(A) PID independant par axe** : `tau_i = Kp (q_d - q) + Ki integrale(e) + Kd (q_dot_d - q_dot)`. Choisir `Kp = 100, Ki = 10, Kd = 20` par axe.
   - **(B) Computed torque** : `tau = M(q) [q_ddot_d + Kp e + Kd e_dot] + C(q, q_dot) q_dot + g(q)`, `Kp = 100 I, Kd = 2 sqrt(Kp) I` (amortissement critique).
4. Simuler les deux (RK4, dt = 0.001 s) depuis `q0 = [0, 0]^T, q_dot_0 = 0`. Tracer `q1(t), q2(t)` vs `q1_d(t), q2_d(t)`.
5. Calculer la RMSE de tracking sur les 2 articulations pour chaque controleur.
6. **Robustesse** : refaire l'experience avec **20% d'erreur de modele** (les masses utilisees dans le controleur valent `1.2 kg` au lieu de `1 kg`, mais la simu utilise les vraies). Comparer la degradation pour PID vs computed torque.
7. Repondre par ecrit (10-15 lignes) :
   - Pourquoi le computed torque tracke mieux que le PID en l'absence d'erreur de modele ?
   - Pourquoi se degrade-t-il avec l'erreur de modele ? Quelle structure restaurerait la robustesse ?

## Criteres de reussite

- Code tourne en < 30 s.
- Sans erreur de modele : RMSE computed torque < RMSE PID d'au moins 50%.
- Avec 20% d'erreur de modele : la RMSE du computed torque augmente significativement, et reste meilleure que le PID OU s'en rapproche (selon les gains).
- Reponse ecrite mentionne : decouplage exact en lineaire double-integrateur sans erreur de modele (CTC) ; sensibilite parametrique ; mention possible d'**adaptive control (Slotine-Li)** ou de **robust H-inf** comme remede.

## Indices

- `M(q)` du 2-DOF planaire :
  ```
  M11 = m1 lc1^2 + m2 (l1^2 + lc2^2 + 2 l1 lc2 cos q2) + I1 + I2
  M12 = m2 (lc2^2 + l1 lc2 cos q2) + I2
  M22 = m2 lc2^2 + I2
  ```
- `C(q, q_dot) q_dot` peut s'ecrire via les symboles de Christoffel ; en pratique, `h = -m2 l1 lc2 sin q2`, et le vecteur Coriolis s'ecrit `[h q_dot2 (2 q_dot1 + q_dot2), -h q_dot1^2]^T`.
- `g(q) = [m1 g lc1 cos q1 + m2 g (l1 cos q1 + lc2 cos(q1+q2)), m2 g lc2 cos(q1+q2)]^T` (axes adapte selon ta convention).
- Reference : [Siciliano et al., 2009, ch. 7-8], [Lynch & Park, 2017, ch. 8].
