# Projet 01 — AGV differentiel : cinematique + suivi de trajectoire

## Contexte metier

Dans FleetSim, le pathfinder (cf. projet `algorithmie-python/05-projets-guides/01`) sort une
liste de cases a traverser. Mais un AGV n'est pas un point qui se teleporte de case en case :
c'est une base **differentielle** (deux roues motrices independantes) qui doit transformer
"suis ce chemin" en vitesses de roues gauche/droite, sans couper les coins dans les allees
etroites ni osciller autour de la ligne.

Le module cinematique de FleetSim doit etre **certifiable** : modele documente, deterministe,
et erreur de suivi bornee — un client audite ISO 9001 demande la preuve que l'AGV reste dans
son couloir de 30 cm autour du trajet nominal.

## Objectif technique

Implementer en numpy pur :
1. La **cinematique differentielle** : `(v_left, v_right) -> (v, omega)` et l'integration
   exacte de la pose `(x, y, theta)` sur un pas de temps (modele unicycle, arcs de cercle).
2. Un **suivi de trajectoire pure pursuit** : a chaque tick, viser un point d'anticipation
   (lookahead) sur le chemin et en deduire `(v, omega)` puis les vitesses de roues.
3. Une **simulation de mini-flotte** : 3 AGV decales sur la meme boucle d'entrepot, avec
   verification de la separation minimale inter-AGV.

## Consigne

```python
def wheels_to_body(v_left: float, v_right: float, wheel_base: float) -> tuple[float, float]:
    """(vitesses roues) -> (v lineaire, omega angulaire) du chassis."""

def body_to_wheels(v: float, omega: float, wheel_base: float) -> tuple[float, float]:
    """Inverse : (v, omega) -> (v_left, v_right). Doit verifier wheels_to_body(body_to_wheels(...)) == identite."""

def integrate_pose(pose: np.ndarray, v: float, omega: float, dt: float) -> np.ndarray:
    """Pose (x, y, theta) apres dt. Integration EXACTE (arc de cercle), pas Euler."""

def pure_pursuit_step(pose: np.ndarray, path: np.ndarray, lookahead: float, v_nominal: float) -> tuple[float, float]:
    """Retourne (v, omega) pour viser le point du chemin a distance `lookahead` devant l'AGV."""
```

Contraintes :
- `integrate_pose` gere le cas `omega ~ 0` sans division par zero (ligne droite limite).
- Vitesses de roues saturees a `V_WHEEL_MAX` (les moteurs ont une limite physique).
- Deterministe : pas de randomness dans le controleur, seed fixe pour le reste.
- Chemin de test : boucle rectangulaire d'allees d'entrepot avec coins droits.

## Etapes guidees

1. **Cinematique** — pose le modele : `v = (v_r + v_l) / 2`, `omega = (v_r - v_l) / L`.
   Ecris `body_to_wheels` en inversant, et teste l'aller-retour sur 100 valeurs aleatoires.
2. **Integration exacte** — pour `omega != 0`, l'AGV decrit un arc de rayon `R = v / omega`
   autour de l'ICC (Instantaneous Center of Curvature). Derive les formules fermees.
   Compare a Euler sur un cercle complet : Euler derive, l'exacte revient au point de depart.
3. **Reechantillonnage du chemin** — interpole les waypoints en points espacés de ~5 cm :
   le lookahead se cherche bien plus facilement sur un chemin dense.
4. **Pure pursuit** — trouve le point du chemin a distance `lookahead` DEVANT le point le plus
   proche (sinon l'AGV revient en arriere). Transforme-le dans le repere de l'AGV, puis
   courbure `kappa = 2 * y_local / lookahead**2`, `omega = v * kappa`.
5. **Ralentissement en virage** — module `v` par la courbure (`v = v_nominal / (1 + k*|kappa|)`)
   pour ne pas saturer la roue exterieure dans les coins.
6. **Flotte** — fais partir 3 AGV avec un offset de depart le long de la boucle, simule
   60 s, et calcule la distance minimale entre toute paire d'AGV a chaque tick.

## Criteres de reussite

- `body_to_wheels` puis `wheels_to_body` rend `(v, omega)` a `1e-12` pres sur 100 cas.
- Apres un cercle complet en integration exacte (`omega` constant), la pose finale est
  egale a la pose initiale a `1e-6` pres ; la meme manoeuvre en Euler (dt=0.05) derive de `> 1 cm`.
- Sur la boucle d'entrepot, **l'erreur laterale max** (cross-track error) reste `< 0.30 m`
  et l'erreur moyenne `< 0.10 m` (le couloir certifiable du cahier des charges).
- Chaque AGV termine sa boucle : distance au point de depart `< 0.5 m` apres un tour.
- Avec 3 AGV decales, la separation minimale inter-AGV reste `> 1.0 m` sur les 60 s.
- Deux executions consecutives produisent exactement les memes metriques (determinisme).

## Solution

Voir `solution/agv_tracking.py` — correction commentee, executable telle quelle
(`python solution/agv_tracking.py`, < 10 s, numpy seul). Les criteres ci-dessus y sont
verifies par des assertions.

## Pour aller plus loin

- **Controleur Stanley** (front-axle feedback) — compare l'erreur laterale a pure pursuit
- **Lookahead adaptatif** — proportionnel a la vitesse, comme sur les AGV industriels
- **Dynamique** — ajoute une constante de temps moteur (les roues n'atteignent pas
  la consigne instantanement) et observe l'effet sur les coins
- **Evitement reactif** — si la separation inter-AGV passe sous 1.5 m, l'AGV suiveur ralentit
  (regle Safety Policy de LogiSim)
