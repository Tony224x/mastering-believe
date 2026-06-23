# Jour 3 — Cinematique directe (Forward Kinematics)

> **Objectif du jour** : passer d'une configuration articulaire `q = [θ₁, θ₂, ..., θₙ]` a la pose 4×4 de l'effecteur. Coder la FK from scratch en numpy pour un bras planaire 2-DOF, puis verifier avec `mj_forward` sur le Franka Panda 7-DOF de MuJoCo Menagerie.

## 1. Exemple concret AVANT abstrait — bras planaire 2-DOF a la main

On a un bras planaire (dans le plan XY) avec deux segments rigides :

- segment 1 : longueur `L₁ = 1.0`, articulation 1 a l'origine, angle `θ₁ = 30° = π/6`
- segment 2 : longueur `L₂ = 0.5`, articulation 2 a l'extremite du segment 1, angle `θ₂ = 45° = π/4` **relatif** au segment 1

**Question** : ou est l'effecteur (extremite du segment 2) ?

L'angle absolu du segment 2 dans le repere monde est `θ₁ + θ₂` (angles relatifs s'additionnent dans le plan). On compose deux deplacements :

```
x = L₁ · cos(θ₁) + L₂ · cos(θ₁ + θ₂)
y = L₁ · sin(θ₁) + L₂ · sin(θ₁ + θ₂)
```

Numeriquement :

- `θ₁ = π/6` → `cos = 0.8660`, `sin = 0.5000`
- `θ₁ + θ₂ = π/6 + π/4 = 5π/12 ≈ 1.3090 rad` → `cos = 0.2588`, `sin = 0.9659`
- `x = 1.0 × 0.8660 + 0.5 × 0.2588 = 0.9954`
- `y = 1.0 × 0.5000 + 0.5 × 0.9659 = 0.9830`

Effecteur en `(0.995, 0.983)`. Verification dans `02-code/03-cinematique-directe.py`.

**Generalisation immediate** : pour un bras planaire a `n` joints revolutes en serie, la FK est la somme cumulative `Σ Lᵢ · [cos(Σ θⱼ), sin(Σ θⱼ)]`. Mais des qu'on sort du plan (rotations 3D, axes non paralleles, offsets), il faut une machinerie : DH ou PoE.

## 2. Definition formelle de la FK

> **FK** : application `f : Q → SE(3)` qui a une configuration articulaire `q ∈ Q` (espace des configurations) associe la pose 4×4 `T_se = f(q)` du repere effecteur (`s` = space/world, `e` = end-effector) [Lynch & Park, 2017, ch. 4].

Pour un robot serie a `n` joints (revolute ou prismatic) :
- `q = (θ₁, ..., θₙ)` ∈ ℝⁿ (radians pour revolute, metres pour prismatic)
- `T_se(q)` ∈ SE(3) : matrice 4×4 homogene (rotation 3×3 + translation 3×1)

La FK est **toujours** soluble en closed-form (contrairement a l'IK), c'est juste une chaine de multiplications matricielles. Le choix de **representation** (DH classique, DH modifiee, PoE) change la lisibilite, pas le resultat.

## 3. Trois representations canoniques

### 3.1. Denavit-Hartenberg classique (DH 1955)

Convention historique pour parametrer un bras serie avec **4 parametres par joint** : `(αᵢ₋₁, aᵢ₋₁, dᵢ, θᵢ)`.

- `αᵢ₋₁` : twist angle entre axes `zᵢ₋₁` et `zᵢ`
- `aᵢ₋₁` : link length (distance entre axes)
- `dᵢ` : link offset (le long de `zᵢ`)
- `θᵢ` : joint angle (revolute) ou variable (prismatic)

La transformation entre repere `i-1` et `i` s'ecrit comme produit de 4 matrices elementaires (`Rotz · Transz · Transx · Rotx`). Au final :

```
       | cos θᵢ           -sin θᵢ           0           aᵢ₋₁          |
Tᵢ⁻¹ᵢ = | sin θᵢ cos αᵢ₋₁  cos θᵢ cos αᵢ₋₁  -sin αᵢ₋₁  -dᵢ sin αᵢ₋₁  |
       | sin θᵢ sin αᵢ₋₁  cos θᵢ sin αᵢ₋₁   cos αᵢ₋₁   dᵢ cos αᵢ₋₁  |
       | 0                 0                  0           1             |
```

Et `T_se = T₀₁ · T₁₂ · ... · Tₙ₋₁ₙ`.

**Limites de DH** :
- placement des reperes ambigu (deux conventions valides → bugs frequents)
- la convention « modifiee » (Craig) place `αᵢ` et `aᵢ` au joint `i` au lieu de `i-1` → encore une source de confusion
- mauvaise pour les chaines paralleles ou les robots avec axes coincidants
- difficulte a inserer des outils ou capteurs intermediaires sans recalculer

C'est encore le standard en industrie (URDF, ROS), mais **Lynch et toutes les nouvelles refs prennent PoE** pour la pedagogie.

### 3.2. Denavit-Hartenberg modifiee (Craig 1986)

Variante : `Tᵢ⁻¹ᵢ = Rotx(αᵢ₋₁) · Transx(aᵢ₋₁) · Rotz(θᵢ) · Transz(dᵢ)`. Identique en puissance, juste une autre convention. **A retenir** : si tu lis un papier ou un URDF, demande-toi laquelle.

### 3.3. Product of Exponentials — PoE (Brockett 1984, popularise par Lynch)

Formulation moderne, geometriquement plus claire. Idee : chaque joint `i` est decrit par sa **screw axis** dans le repere espace (world) `Sᵢ ∈ ℝ⁶`, et la FK est :

```
T_se(q) = e^([S₁]q₁) · e^([S₂]q₂) · ... · e^([Sₙ]qₙ) · M
```

ou :
- `M ∈ SE(3)` est la pose **home** de l'effecteur (configuration q = 0)
- `Sᵢ = (ωᵢ, vᵢ)` est le screw axis du joint `i` exprime dans le repere espace
  - `ωᵢ ∈ ℝ³` : axe de rotation unitaire (revolute) ou zero (prismatic)
  - `vᵢ = -ωᵢ × pᵢ` (revolute, `pᵢ` point sur l'axe) ou direction (prismatic)
- `[Sᵢ]` : matrice 4×4 « twist » (algebra `se(3)`)
- `e^([Sᵢ]qᵢ)` : matrice exponentielle = transformation rigide « visse de `qᵢ` autour de `Sᵢ` »

**Pourquoi PoE est plus propre** [Lynch & Park, 2017, ch. 4.1] :
1. **Pas de reperes intermediaires arbitraires** : on choisit juste un repere espace et un repere effecteur. Tout le reste est geometrique.
2. **Insensible aux offsets** : ajouter un capteur ne demande pas de recalculer toute la chaine, juste mettre a jour `M`.
3. **Formules fermees pour `e^([S]θ)`** via Rodrigues etendue (pas besoin de developpement en serie numerique).
4. **Generalise naturellement aux Jacobiens spatiaux** (J4 demain).

**Formule de Rodrigues pour SE(3)** (revolute, `‖ω‖ = 1`) :

```
e^([S]θ) = | e^([ω]θ)              G(θ) v |
           | 0                      1     |

avec  e^([ω]θ) = I + sin(θ) [ω] + (1 - cos(θ)) [ω]²       (Rodrigues SO(3))
      G(θ)    = I θ + (1 - cos(θ)) [ω] + (θ - sin(θ)) [ω]²
```

et `[ω]` = matrice antisymetrique (skew) de `ω`.

### Quand prendre quoi ?

| Situation | Choix |
|-----------|-------|
| Lire un URDF, parler a ROS / industrie | DH (souvent classique) |
| Implementer FK propre depuis zero | **PoE** |
| Charger un robot MuJoCo (Menagerie) | MJCF te donne la chaine cinematique → `mj_forward` fait la FK pour toi |
| Calculer Jacobiens analytiquement | PoE (J4) |

## 4. FK dans MuJoCo : `mj_forward`

MuJoCo stocke la cinematique dans le `model` (`mjModel`, parse depuis MJCF/URDF) et l'etat dans `data` (`mjData`). Apres avoir mis `data.qpos[:] = q`, un appel a :

```python
mujoco.mj_forward(model, data)
```

execute la propagation cinematique (et autres calculs forward dynamics non utilises ici). Les poses des bodies sont alors disponibles dans `data.xpos` (positions 3) et `data.xmat` (matrices rotation 3×3 aplaties en 9), ou via `data.body(name).xpos` / `data.body(name).xmat`.

Pour le Franka Panda (`mujoco_menagerie/franka_emika_panda/panda.xml`), l'effecteur typique est le body `hand` ou un site `attachment_site`. Verification croisee : implementer la FK PoE a la main pour Panda et comparer a `mj_forward`. C'est exactement ce que fait `02-code/03-cinematique-directe.py`.

## 5. Franka Panda — la chaine 7-DOF

Le Panda a 7 joints revolutes (donc une **redondance** : 7 DOF pour 6 DOF dans SE(3) → infinite d'IK pour une pose donnee, sujet du J4). Sa pose home `M` et ses screw axes `Sᵢ` sont publies (Lynch & Park ch. 4 fournit l'equivalent pour le UR5 ; pour Panda, on les extrait du MJCF).

**Strategie pratique** :
1. Charger Panda dans MuJoCo, lire les positions et axes des joints en config home (`q = 0`) via `model.jnt_axis` et `data.xanchor`.
2. Construire les screws `Sᵢ = (ωᵢ, -ωᵢ × pᵢ)` dans le repere espace.
3. Lire la pose home `M = T_se(q=0)` directement de `data.body('hand').xpos / xmat`.
4. Implementer FK PoE a la main et comparer `np.allclose(T_poe, T_mujoco)` pour `q` aleatoires.

Ecart attendu : `< 1e-6`. Si plus, c'est un bug d'axe ou de signe.

## 6. Erreurs classiques

- **Signe de `θ`** : DH inverse parfois le sens de rotation selon la convention.
- **Twist `Sᵢ` mal exprime** : oublier que `vᵢ = -ωᵢ × pᵢ` (et pas `ωᵢ × pᵢ`). Le « moins » vient de l'identite screw : un point `p` vu comme tourne autour de l'axe se deplace a vitesse `-ω × p` dans la convention spatiale.
- **Ordre des produits exponentiels** : PoE en repere espace = `e^[S₁] · e^[S₂] · ... · M`. En repere body c'est l'ordre **inverse** avec les `Bᵢ`. Memoriser une seule convention.
- **Confondre `xmat` MuJoCo (3×3 row-major aplati en 9) avec une matrice 4×4** : il faut reconstruire la 4×4 a partir de `xpos` et `xmat`.
- **Oublier `mj_forward`** apres `data.qpos = q` : `data.xpos` n'est pas mis a jour automatiquement.

## 7. Cle a retenir

> **FK = produit de transformations rigides parametrees par `q`. Que tu choisisses DH ou PoE, tu finis avec une matrice 4×4. PoE (Lynch) est plus propre geometriquement et se prolonge proprement vers les Jacobiens. MuJoCo te donne `mj_forward` qui le fait pour toi — utilise-le comme oracle pour valider ton implementation manuelle.** [Lynch & Park, 2017, ch. 4]

## 8. Flash cards (spaced repetition)

1. **Q** : Quelle est la signature de la FK ?
   **R** : `f : Q → SE(3)`, configuration articulaire `q ∈ ℝⁿ` → pose 4×4 de l'effecteur. Toujours closed-form pour un robot serie.

2. **Q** : Quel est l'ordre du produit dans la formule PoE en repere espace ?
   **R** : `T(q) = e^([S₁]q₁) · e^([S₂]q₂) · ... · e^([Sₙ]qₙ) · M`. `M` (pose home) **a droite**, screws en repere espace.

3. **Q** : Pour un joint revolute d'axe `ω` passant par le point `p`, quel est le screw `S` ?
   **R** : `S = (ω, -ω × p)` ∈ ℝ⁶. Le moins est crucial.

4. **Q** : Pourquoi Lynch prefere PoE a DH ?
   **R** : Pas de reperes intermediaires arbitraires, formule directement geometrique, generalise aux Jacobiens, insensible aux offsets de capteurs.

5. **Q** : Comment recuperer la pose d'un body MuJoCo apres avoir change `qpos` ?
   **R** : `mujoco.mj_forward(model, data)` puis lire `data.body(name).xpos` (3,) et `data.body(name).xmat.reshape(3,3)`.

6. **Q** : Combien de DOF a le Franka Panda et qu'implique cette redondance ?
   **R** : 7 DOF revolutes. Redondance par rapport aux 6 DOF de SE(3) → famille a 1 parametre d'IK pour une pose donnee (sujet J4).

## References

- [Lynch & Park, 2017, ch. 4] — Modern Robotics, formulation PoE. http://hades.mech.northwestern.edu/index.php/Modern_Robotics
- [CS223A — Khatib, L4] — Forward kinematics, DH parameters. https://www.youtube.com/playlist?list=PL65CC0384A1798ADF
- [MuJoCo Menagerie] — modeles MJCF curatés (Franka Panda inclus). https://github.com/google-deepmind/mujoco_menagerie
