# Jour 5 — Dynamique des robots & simulation MuJoCo hands-on

> Acquis visé fin de journée : un robot Franka qui tombe sous gravité dans MuJoCo, et tu sais calculer son énergie cinétique + potentielle pour vérifier la conservation.

---

## 1. Pourquoi la dynamique ?

J3-J4 répondaient à *« où est l'effecteur ? »* (cinématique). Aujourd'hui on répond à *« quel couple `τ` faut-il appliquer aux moteurs pour produire un mouvement donné ? »* — c'est la **dynamique**.

Trois problèmes selon ce qu'on cherche :

| Problème | Entrée | Sortie | Cas d'usage |
|----------|--------|--------|-------------|
| **Dynamique inverse** | `q, q̇, q̈` désirés | `τ` requis | Computed torque control (J6) |
| **Dynamique directe** | `τ, q, q̇` | `q̈` | Simulation MuJoCo |
| **Détection d'effort** | `τ, q, q̇, q̈` | force externe | Force/admittance control |

La dynamique directe est ce que MuJoCo résout 1000 fois par seconde dans `mj_step`.

---

## 2. Concret AVANT abstrait — le pendule simple

Avant de toucher Lagrange ou Newton-Euler, on dérive l'équation d'un pendule simple à la main, on simule numériquement, et on compare à la solution analytique petit-angle.

### Géométrie

- Une masse ponctuelle `m` au bout d'une tige rigide sans masse de longueur `L`.
- Angle `θ` mesuré depuis la verticale basse (`θ = 0` ⇒ pendule en bas).
- Gravité `g` vers le bas.

### Newton sur l'arc

La masse est contrainte à un cercle. On projette `F = ma` sur la **direction tangentielle** (la radiale est encaissée par la tige). Force tangentielle = composante de la gravité projetée sur la tangente : `F_t = -mg sin(θ)` (le signe `-` rappelle qu'elle ramène vers θ=0).

Accélération tangentielle = `L θ̈` (pour un mouvement circulaire à rayon constant, l'arc parcouru vaut `L·θ`, donc `s̈ = L θ̈`).

D'où :

```
m L θ̈ = -m g sin(θ)
θ̈ = -(g/L) sin(θ)
```

C'est la version **non-linéaire**. Pour de petits angles, `sin(θ) ≈ θ`, et on retrouve le pendule harmonique linéaire `θ̈ = -(g/L) θ` de pulsation `ω = √(g/L)`.

### Simulation numérique vs analytique

Avec `θ(0) = 0.1 rad`, `θ̇(0) = 0`, `g = 9.81`, `L = 1.0` :

- **Solution analytique petit-angle** : `θ(t) = 0.1·cos(ω t)`, `ω ≈ 3.13 rad/s`, période `T ≈ 2.007 s`.
- **Solution numérique** : on intègre `θ̈ = -(g/L) sin(θ)` avec un schéma. À petit angle les deux courbes se superposent ; à `θ(0) = 1.5 rad`, l'analytique petit-angle se déphase visiblement de la numérique exacte (l'amplitude grandit la période réelle — pendule non-isochrone).

### Schémas d'intégration — pourquoi MuJoCo n'utilise pas Euler explicite

Trois schémas sur le même pas `dt` :

| Schéma | Update | Conserve l'énergie sur pendule conservatif ? |
|--------|--------|----------------------------------------------|
| Euler explicite | `θ ← θ + dt·θ̇`, `θ̇ ← θ̇ + dt·θ̈(θ)` | Non, énergie diverge (instable) |
| Euler semi-implicite (= symplectique) | `θ̇ ← θ̇ + dt·θ̈(θ)` puis `θ ← θ + dt·θ̇` | Oui, énergie oscille mais reste bornée |
| RK4 | 4 évaluations pondérées | Oui en pratique sur dt raisonnable |

**MuJoCo utilise par défaut Euler semi-implicite** (option `integrator="Euler"` dans MJCF, qui est en réalité le schéma semi-implicite ; `RK4` est aussi disponible mais plus coûteux). C'est le choix standard pour la dynamique de robots à contacts : peu coûteux, stable, et symplectique sur la partie sans dissipation.

```
[Lynch & Park, 2017, ch. 8] — formulations dynamique des bras articulés
[MuJoCo docs, computation/integrator] — schémas et options
```

---

## 3. Formulation Lagrange — la forme canonique

Pour un robot articulé à `n` degrés de liberté, la dynamique s'écrit toujours sous la **même forme matricielle** :

```
M(q) q̈ + C(q, q̇) q̇ + g(q) = τ + J(q)ᵀ F_ext
```

| Terme | Dimension | Signification |
|-------|-----------|---------------|
| `q ∈ ℝⁿ` | n×1 | configuration articulaire |
| `M(q)` | n×n | **matrice d'inertie** symétrique définie positive |
| `C(q, q̇)` | n×n | termes de Coriolis et centrifuges |
| `g(q)` | n×1 | vecteur de gravité |
| `τ` | n×1 | couples articulaires (commande) |
| `J(q)ᵀ F_ext` | n×1 | couples induits par forces extérieures à l'effecteur |

### D'où ça vient — Euler-Lagrange

Soit `T(q, q̇)` l'énergie cinétique totale (somme des `½ vᵀ M_link v + ½ ωᵀ I ω` de chaque lien) et `V(q)` l'énergie potentielle (typiquement gravitaire). Le Lagrangien `L = T − V`. Les équations d'Euler-Lagrange donnent pour chaque coordonnée généralisée `qᵢ` :

```
d/dt (∂L/∂q̇ᵢ) − ∂L/∂qᵢ = τᵢ
```

Comme `T = ½ q̇ᵀ M(q) q̇`, on dérive et on regroupe : on tombe exactement sur la forme `M(q)q̈ + C(q,q̇)q̇ + g(q) = τ`.

### Propriétés-clés à retenir

1. **`M(q)` est symétrique définie positive** ⇒ inversible, `q̈ = M⁻¹(τ − Cq̇ − g)`.
2. **`Ṁ − 2C` est antisymétrique** (avec un choix particulier de `C`) ⇒ propriété clé pour les preuves de stabilité de contrôleurs Lyapunov (J6).
3. **Linéarité en les paramètres dynamiques** ⇒ on peut écrire `M(q)q̈ + Cq̇ + g = Y(q,q̇,q̈) Θ` avec `Θ` les paramètres (masses, inerties, longueurs) ⇒ identification dynamique par moindres carrés.
4. **`g(q) = ∂V/∂q`** — le vecteur gravité est le gradient de l'énergie potentielle.

```
[Lynch & Park, 2017, ch. 8.1] — dérivation Lagrange standard
[Siciliano et al., 2009, ch. 7] — propriétés structurelles + linéarité paramétrique
```

---

## 4. Newton-Euler récursif — la formulation efficace

Lagrange est élégant mais coûteux : pour un robot 6-DOF, calculer symboliquement `M, C, g` produit des centaines de termes. **Newton-Euler récursif** (algorithme de Featherstone) calcule la dynamique inverse en `O(n)` :

### Forward sweep — base vers effecteur

Propage les vélocités et accélérations linéaires/angulaires lien après lien :

```
Pour i = 1 … n :
  ω_i = ω_{i-1} + q̇_i · ẑ_i               # vitesse angulaire
  α_i = α_{i-1} + q̈_i · ẑ_i + ω_{i-1} × (q̇_i · ẑ_i)
  a_i = a_{i-1} + α_i × r_{i,i-1} + ω_i × (ω_i × r_{i,i-1})
```

(ẑ_i = axe de l'articulation `i`, r = vecteurs de position).

### Backward sweep — effecteur vers base

Calcule les forces et couples de réaction de l'effecteur vers la base :

```
Pour i = n … 1 :
  f_i = f_{i+1} + m_i · a_{c,i}                            # Newton sur le centre de masse
  n_i = n_{i+1} − f_i × r_{c,i} + f_{i+1} × r_{c,i+1}
       + I_i α_i + ω_i × (I_i ω_i)                          # Euler sur l'inertie
  τ_i = n_iᵀ ẑ_i                                            # projection sur l'axe
```

Coût total : `O(n)` opérations vectorielles. **C'est ce que MuJoCo utilise en interne** pour calculer `M(q)`, `C(q,q̇)q̇`, `g(q)` — tu n'as jamais à les écrire à la main.

```
[Lynch & Park, 2017, ch. 8.3] — Newton-Euler récursif détaillé
[Siciliano et al., 2009, ch. 7.5] — formulation alternative en cadres mobiles
```

---

## 5. Inertie, masse, frottements, contacts

### Tenseur d'inertie

Pour un corps rigide, le tenseur d'inertie `I ∈ ℝ³ˣ³` au centre de masse est symétrique défini positif. Diagonale = moments principaux ; off-diagonale = produits d'inertie.

En MJCF (MuJoCo XML) :

```xml
<body name="link1">
  <inertial pos="0 0 0.1" mass="2.5"
            diaginertia="0.05 0.05 0.01"/>
  <joint type="hinge" axis="0 0 1"/>
  <geom type="cylinder" size="0.04 0.1"/>
</body>
```

`diaginertia` suffit si les axes principaux sont alignés avec le frame du body. Sinon utilise `fullinertia="Ixx Iyy Izz Ixy Ixz Iyz"`.

### Frottements articulaires

Trois sources, modélisables séparément :

- **Visqueux** (proportionnel à la vitesse) : `τ_friction = -d · q̇`. En MJCF : `<joint damping="0.1"/>`.
- **Coulomb** (sec, signe-dépendant) : `τ_friction = -μ_c · sign(q̇)`. En MJCF : `<joint frictionloss="0.05"/>`.
- **Stiction** (statique seuil) : MuJoCo le modélise via le solveur de contraintes — pas un terme additif simple.

### Contacts

MuJoCo utilise un **solveur de contraintes convexes** (Todorov 2014) qui calcule les forces de contact à chaque pas de manière à empêcher la pénétration tout en respectant le cône de friction de Coulomb. C'est ce qui rend MuJoCo précis sur les contacts riches (mains, pieds, manipulation) là où Bullet ou ODE se déchirent.

Cône de friction par contact : `<geom friction="μ_t μ_r μ_p"/>` (translation tangentielle, rolling, pivoting).

```
[MuJoCo docs, computation/solver] — pyramide vs cône, paramètres solver
[MuJoCo docs, modeling/inertial] — spécification inertie
```

---

## 6. Charger un robot Menagerie et simuler

### Pipeline minimal

```python
import mujoco
import numpy as np

# 1. Charge le modèle MJCF (XML)
model = mujoco.MjModel.from_xml_path("franka_emika_panda/panda.xml")
data  = mujoco.MjData(model)

# 2. Appelle mj_forward pour synchroniser xpos/xmat avec qpos
mujoco.mj_forward(model, data)

# 3. Boucle de simulation
dt = model.opt.timestep   # typiquement 0.002 s
for step in range(1000):
    # τ vit dans data.ctrl si le modèle a des actuateurs ; sinon dans data.qfrc_applied
    data.ctrl[:] = 0.0
    mujoco.mj_step(model, data)
    # data.qpos, data.qvel sont mis à jour
```

### Ce que MuJoCo fait dans `mj_step`

1. Calcule `M(q)` via Composite Rigid Body Algorithm (CRBA) — `O(n²)`.
2. Calcule biais `b(q,q̇) = C(q,q̇)q̇ + g(q) − τ_ext` via Recursive Newton-Euler — `O(n)`.
3. Résout `M q̈ = τ − b + Jᵀλ` avec `λ` les multiplicateurs de Lagrange du solveur de contacts (problème quadratique convexe).
4. Intègre `q̇ ← q̇ + dt q̈` puis `q ← q + dt q̇` (Euler semi-implicite).

### Lire l'état

| Quantité | Champ MuJoCo | Forme |
|----------|--------------|-------|
| Configuration | `data.qpos` | `(nq,)` — peut différer de `nv` si quaternions présents |
| Vitesse | `data.qvel` | `(nv,)` |
| Accélération | `data.qacc` | `(nv,)` |
| Forces appliquées | `data.qfrc_applied` | `(nv,)` |
| Energie cinétique | calculer via `0.5 · qvelᵀ · M · qvel` après `mj_fullM` | scalaire |
| Energie potentielle | `data.energy[1]` (si `enableflags` inclut `energy`) | scalaire |

Pour activer le calcul d'énergie auto par MuJoCo dans le MJCF :

```xml
<option>
    <flag energy="enable"/>
</option>
```

Ensuite `data.energy = [kinetic, potential]`.

```
[MuJoCo docs, programming/simulation] — pipeline mj_step
[MuJoCo docs, XMLreference] — option/flag energy
[MuJoCo Menagerie] — Franka Panda MJCF prêt à l'emploi
```

---

## 7. Conservation de l'énergie — le test sanity

Sans actuateurs, sans frottement, sans contact dissipatif, un système sous gravité conserve l'énergie totale `E = T + V`. En pratique avec MuJoCo et Euler semi-implicite :

- **Sans dissipation déclarée** : `E` oscille à haute fréquence avec amplitude `O(dt)` mais la moyenne reste constante sur de longues durées (caractère symplectique du schéma).
- **Avec damping ou frictionloss** : `E` décroît monotonement.
- **Sur un robot avec contacts** : `E` peut chuter brutalement lors des impacts (le solveur de contacts dissipe par défaut un peu, paramétrable via `solref`/`solimp`).

Une chute de pendule de 1 m avec `m=1, g=9.81` libère `mgh ≈ 9.81 J` qu'on retrouve en énergie cinétique au point bas — si la simu est correcte. C'est exactement ce que le code du jour vérifie sur Franka qui tombe.

---

## 8. Récap — les pièges à éviter

- **`nq ≠ nv`** quand il y a des bodies free joints (quaternion = 4 composantes en `qpos`, vitesse angulaire = 3 en `qvel`). Toujours utiliser `model.nq`/`model.nv`, jamais hardcoder.
- **`mj_step` modifie `data` in-place** ; pour rejouer, restaure `qpos`/`qvel` ou utilise `mj_resetData`.
- **`data.ctrl` n'est utilisé que si le modèle a des `<actuator>`**. Sinon les couples passent par `data.qfrc_applied`.
- **Timestep trop grand** = simu instable. `dt = 0.002` (500 Hz) est le standard pour bras manipulateur ; descends à `0.0005` pour mains avec contacts riches.
- **Énergie qui explose** = soit timestep trop grand, soit forces externes oubliées, soit modèle inertie aberrant.

---

## 9. Flash-cards (spaced repetition)

1. **Q** : Sous quelle forme matricielle s'écrit la dynamique d'un bras à n DDL ?
   **R** : `M(q)q̈ + C(q,q̇)q̇ + g(q) = τ` — `M` symétrique définie positive, `C` Coriolis, `g` gravité.

2. **Q** : Pourquoi MuJoCo utilise-t-il Euler semi-implicite plutôt qu'explicite ?
   **R** : Schéma symplectique → énergie bornée sur systèmes conservatifs ; coût `O(n²)` par pas, stable sur grands `dt` que l'explicite ferait diverger.

3. **Q** : Quelle est la complexité de Newton-Euler récursif vs Lagrange explicite pour la dynamique inverse ?
   **R** : `O(n)` pour Newton-Euler récursif (Featherstone), contre `O(n³)` voire pire pour Lagrange dérivé symboliquement.

4. **Q** : `nq` et `nv` peuvent-ils différer dans MuJoCo ? Donne un exemple.
   **R** : Oui. Un free joint a 7 composantes `qpos` (3 position + 4 quaternion) mais 6 composantes `qvel` (3 linéaire + 3 angulaire).

5. **Q** : Comment vérifie-t-on qu'une simu MuJoCo est physiquement plausible ?
   **R** : Test conservation d'énergie : sans actuateur ni dissipation, `E_total = T + V` doit rester constante (modulo bruit `O(dt)`). Si elle dérive, le timestep est trop grand ou le modèle a un bug d'inertie.

---

## Ressources

- Lynch & Park 2017, *Modern Robotics*, ch. 8 — http://hades.mech.northwestern.edu/index.php/Modern_Robotics
- Siciliano et al. 2009, *Robotics: Modelling, Planning and Control*, ch. 7
- MuJoCo Documentation 3.x — https://mujoco.readthedocs.io/ (sections : Computation, XML reference, Programming)
- MuJoCo Menagerie (Franka Panda MJCF) — https://github.com/google-deepmind/mujoco_menagerie
