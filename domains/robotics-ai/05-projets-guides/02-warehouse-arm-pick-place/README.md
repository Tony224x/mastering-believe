# Projet 02 — Bras de picking : FK/IK + planning de trajectoire

## Contexte metier

Les postes de picking FleetSim utilisent des bras planaires 3 axes montes en bord de
convoyeur : ils saisissent des bacs sur la ligne et les deposent dans une goulotte de tri,
en passant PAR-DESSUS le muret separateur qui protege la zone operateur. Le module bras de
FleetSim doit convertir une cible cartesienne ("bac a (0.55, 0.25), prise verticale") en
angles moteurs, puis generer une trajectoire articulaire lisse (les moteurs ont une vitesse
max) qui ne touche jamais le muret.

Bug classique en production : l'IK retourne la solution "coude bas" qui balaie le muret.
Le planner doit choisir la branche ET verifier la collision sur toute la trajectoire,
pas seulement aux points de depart/arrivee.

## Objectif technique

Implementer en numpy pur, pour un bras planaire RRR (3 rotoides, longueurs `L = [0.40, 0.35, 0.15]` m) :
1. La **cinematique directe** : `fk(q) -> (x, y, phi)` (position + orientation de l'outil)
   et les positions de toutes les articulations (pour la collision).
2. L'**IK analytique** : reduction 3R -> 2R via le point poignet, deux branches coude
   haut / coude bas, et l'**IK numerique** par moindres carres amortis (DLS) sur le jacobien.
3. Un **planner articulaire** : interpolation quintique entre configurations avec point de
   passage "bras leve" au-dessus du muret, duree calee sur la vitesse moteur max, et
   verification de collision par echantillonnage des segments du bras.

## Consigne

```python
def fk(q: np.ndarray) -> tuple[np.ndarray, float]:
    """q (3,) -> (position outil (2,), orientation phi). phi = q1 + q2 + q3."""

def joint_positions(q: np.ndarray) -> np.ndarray:
    """(4, 2) : base, coude, poignet, outil — pour le check collision."""

def ik_analytic(target_xy: np.ndarray, target_phi: float, elbow_up: bool) -> np.ndarray | None:
    """IK fermee. None si la cible est hors d'atteinte (pas d'exception)."""

def ik_dls(target_xy: np.ndarray, target_phi: float, q0: np.ndarray,
           max_iters: int = 200, tol: float = 1e-6) -> np.ndarray | None:
    """IK iterative par damped least squares sur la tache (x, y, phi)."""

def plan_pick_place(q_start, q_pick, q_place, obstacle) -> np.ndarray | None:
    """Trajectoire (N, 3) lisse start -> pick -> via -> place, sans collision."""
```

Contraintes :
- IK analytique : `fk(ik_analytic(p, phi)) == (p, phi)` a `1e-9` pres pour toute cible atteignable.
- DLS : jacobien 3x3 ANALYTIQUE (pas de differences finies), amortissement ADAPTATIF
  `lambda = clip(0.5 * ||err||, 1e-3, 0.1)` (un lambda fixe converge trop lentement
  pres des singularites — teste-le, c'est l'etape 5).
- Vitesse articulaire max `QDOT_MAX = 1.5 rad/s` — la duree de chaque segment en decoule.
- Profil quintique : vitesse ET acceleration nulles aux extremites de chaque segment.
- Muret = rectangle axis-aligned `[-0.24, -0.14] x [0.0, 0.34]` (entre la base et la
  goulotte) ; collision testee en echantillonnant chaque segment du bras tous les ~2 cm,
  a chaque pas de la trajectoire.
- Cibles hors d'atteinte ou trajectoire impossible : retourner `None`, jamais d'exception.

## Etapes guidees

1. **FK** — accumule les angles : `phi_k = q1 + ... + qk`, chaque articulation ajoute
   `L_k * [cos(phi_k), sin(phi_k)]`. Verifie a la main `fk([0,0,0]) = (0.90, 0, 0)`.
2. **Reduction au poignet** — l'orientation `phi` de l'outil est imposee : le poignet est
   donc a `target - L3 * [cos(phi), sin(phi)]`. Il reste un 2R classique pour `(q1, q2)`,
   puis `q3 = phi - q1 - q2`.
3. **2R ferme** — `cos(q2) = (r^2 - L1^2 - L2^2) / (2 L1 L2)`. Si `|cos(q2)| > 1`, hors
   d'atteinte -> `None`. Les deux signes de `q2` donnent coude haut / coude bas.
4. **Jacobien analytique** — derive les colonnes : la colonne j est
   `[-sum_{k>=j} L_k sin(phi_k), sum_{k>=j} L_k cos(phi_k), 1]`. Verifie contre des
   differences finies (erreur < 1e-6) avant de t'en servir.
5. **DLS** — `dq = J^T (J J^T + lambda^2 I)^{-1} err`. Pourquoi l'amortissement : pres
   d'une singularite (bras tendu), `J J^T` est mal conditionnee et la pseudo-inverse pure
   explose. Essaie d'abord `lambda = 0.1` fixe : certaines cibles demandent > 1000
   iterations. Passe ensuite a `lambda ~ ||err||` (adaptatif) et observe la convergence
   quasi-quadratique (< 20 iterations) — c'est l'idee de Levenberg-Marquardt.
6. **Quintique** — `s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5` ; vitesse max au milieu :
   `1.875 * |dq| / T`. Choisis `T = max(1.875 * max|dq| / QDOT_MAX, T_min)`.
7. **Via point + branche** — entre pick et place, insere une config "bras leve" (outil a
   `(0.0, 0.62)`, oriente vers le haut) : c'est la SOP LogiSim "transfert en hauteur".
   Verifie la collision sur TOUTE la trajectoire echantillonnee. Compare ensuite les deux
   branches IK a la depose : coude haut passe, coude bas balaie le muret — c'est le bug.

## Criteres de reussite

- FK de reference : `fk([0, 0, 0]) = ((0.90, 0.0), 0.0)` exactement.
- Roundtrip IK analytique : sur 200 cibles atteignables aleatoires (seed fixe), erreur
  position `< 1e-9` m et orientation `< 1e-9` rad, pour les DEUX branches coude.
- Jacobien analytique vs differences finies : erreur max `< 1e-5` sur 50 configs aleatoires.
- DLS converge (`tol 1e-6`) en `< 200` iterations sur 50 cibles atteignables (seed fixe),
  et retourne `None` (sans exception) sur une cible a 1.2 m (hors d'atteinte, reach 0.90 m).
- La trajectoire home -> pick -> via -> place complete : erreur outil aux waypoints
  `< 1e-6`, vitesse articulaire max `<= QDOT_MAX * 1.001`, vitesses nulles aux
  extremites (`< 1e-9`), et **zero** echantillon du bras dans le muret (~300 pas x 3 segments).
- Le bug de production est reproduit : la config de depose **coude bas** touche le muret
  (`arm_hits_obstacle == True`) et la trajectoire pick -> place en coude bas collisionne —
  l'assertion verifie que ton planner DOIT choisir la branche coude haut.

## Solution

Voir `solution/arm_pick_place.py` — correction commentee, executable telle quelle
(`python solution/arm_pick_place.py`, < 10 s, numpy seul). Les criteres ci-dessus y sont
verifies par des assertions.

## Pour aller plus loin

- **Espace cartesien vs articulaire** — interpole l'outil en ligne droite cartesienne
  (IK a chaque pas) et compare les profils articulaires avec le planning articulaire
- **RRT articulaire** — remplace le via point fixe par un RRT dans l'espace des configs
  (cf. J8) quand le muret devient un vrai labyrinthe
- **Singularites** — trace le determinant de `J J^T` le long de la trajectoire et relie
  les pics de vitesse articulaire aux passages pres de la singularite bras tendu
- **Redondance** — passe a 4 axes et utilise la null-space projection pour garder le
  coude loin du muret pendant que l'outil suit sa cible
