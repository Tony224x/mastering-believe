# Exercice J12 (medium) — MPC avec modele bruite (introduction au model bias)

## Objectif

Mesurer la sensibilite de MPC a une erreur de modele. C'est l'angle mort majeur du model-based RL.

## Consigne

Le pendule de `02-code/12-sac-mpc-model-based.py` utilise des parametres `PendulumParams` (g, L, m, b). On va simuler que **le modele du planner est mauvais** par rapport a l'environnement reel.

1. Cree deux instances :
   - `p_real = PendulumParams()` (le "vrai" monde)
   - `p_model = PendulumParams(L=1.3, m=1.5)` (le modele appris a 30%/50% pres sur la longueur et la masse)

2. Modifie `run_mpc` (ou ecris une copie) pour que :
   - Les rollouts internes du planner CEM utilisent `p_model`.
   - Le step reel de l'env utilise `p_real`.

3. Compare le reward cumule sur 100 steps avec :
   - Modele parfait : `p_model = p_real`.
   - Modele biaise : `p_model = PendulumParams(L=1.3, m=1.5)`.
   - Modele tres biaise : `p_model = PendulumParams(L=2.0, m=2.5, g=8.0)`.

4. Trace (ou affiche en texte) la trajectoire `theta(t)` pour les 3 cas.

Reponds aux questions :
- A partir de quel niveau de bias le pendule ne se stabilise plus ?
- Pourquoi le re-planning a chaque step **attenue** mais ne **supprime pas** l'erreur de modele ?
- Comment MBPO (cf. theorie) gere-t-il ce probleme ?

## Criteres de reussite

- Script reproductible avec les 3 conditions et leurs rewards.
- Tu peux exprimer l'idee suivante : "le receding horizon corrige une derive si l'erreur par step est petite, mais pas si chaque step pousse fortement l'etat dans la mauvaise direction".
- Tu cites au moins UNE technique pour reduire le model bias en MBRL (ensembles, rollouts courts, model uncertainty bonus, etc.).

## Indice

Pour le tres biaise (L=2.0, m=2.5), le planner croit qu'il faut peu de couple alors qu'en realite il en faut beaucoup. Le pendule n'arrive plus a remonter. C'est le scenario que MBPO gere via ensembles + horizon court.
