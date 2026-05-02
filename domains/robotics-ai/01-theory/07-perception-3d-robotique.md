# J7 — Perception 3D pour robotique

## Le scenario qui structure le jour

> Un bras robot doit attraper une bouteille posee sur une table. Il a une camera RGB-D montee sur sa tete (Intel RealSense, ou capteur ToF integre). Sur l'image RGB il voit la bouteille. Sur la depth map il voit, pour chaque pixel, la distance au capteur. Question : comment passe-t-il de "j'ai un tableau de pixels avec une distance" a "voici la pose 6D (x, y, z, roll, pitch, yaw) de la bouteille dans le repere du robot, je peux y envoyer mon end-effector" ?

C'est exactement le pipeline de perception 3D moderne :

1. Depth map (par pixel) -> point cloud (par point 3D dans le repere camera).
2. Point cloud camera -> point cloud monde (transformation extrinseque).
3. Point cloud monde -> alignement avec un modele de bouteille connu (ICP).
4. Resultat : pose 6D de l'objet -> IK (J4) -> commande de bras (J6).

Ce module construit pas-a-pas les briques de ce pipeline. La toute fin pointe vers les approches 2025 (NeRF, 3DGS, FoundationPose) qui changent la donne quand on n'a pas de modele 3D prealable.

---

## 1. Le modele pinhole — du monde 3D a un pixel

Le capteur le plus simple, et celui dont herite tout capteur RGB-D, est la camera pinhole. Imagine une boite avec un trou (pinhole) au centre d'une face et une plaque sensible sur la face opposee. Un point 3D du monde envoie un rayon a travers le trou ; le rayon frappe la plaque ; on lit un pixel.

Mathematiquement : un point `X = (X, Y, Z)` exprime dans le repere camera (axe Z pointant vers la scene) se projette sur le plan image en :

```
u = fx * (X / Z) + cx
v = fy * (Y / Z) + cy
```

avec :

- `fx`, `fy` : focales en pixels (egales si pixels carres),
- `cx`, `cy` : coordonnees du centre optique (principal point), souvent proche de (largeur/2, hauteur/2),
- `Z` : profondeur du point (= distance le long de l'axe optique).

Sous forme matricielle compacte, on note la **matrice intrinseque** :

```
K = [ fx   0   cx ]
    [  0  fy   cy ]
    [  0   0    1 ]
```

et `s * [u, v, 1]^T = K * [X, Y, Z]^T` ou `s = Z`.

### Calibration

`K` n'est pas connue a la fabrication avec assez de precision. On la **calibre** : photo d'un damier de dimensions connues sous plusieurs poses, on resout un probleme d'optimisation pour retrouver `K` (et les coefficients de distorsion). C'est le `cv2.calibrateCamera` d'OpenCV ou les routines Open3D.

### Extrinseque — passer du repere camera au repere monde/robot

`K` te place dans le repere camera. Pour fusionner avec ce que sait le robot (cinematique du bras, J3), il te faut la transformation `T_world_camera` (souvent appelee `T_cw`) — une matrice SE(3) 4x4 (cf. J2) :

```
[X_world]   [ R   t ] [X_camera]
[Y_world] = [       ] [Y_camera]
[Z_world]   [ 0   1 ] [Z_camera]
[   1   ]             [   1    ]
```

Si la camera est montee sur l'effecteur, `T_world_camera = T_world_effector @ T_effector_camera`. La calibration `T_effector_camera` est le fameux **hand-eye calibration** (Tsai-Lenz, 1989), souvent un cauchemar en pratique mais fait une fois pour toutes.

### Modele inverse — pixel + depth -> point 3D

Le pipeline RGB-D fait l'**inverse** de la projection. Pour chaque pixel `(u, v)` de la depth map ou la profondeur lue est `d = depth[v, u]` :

```
X = (u - cx) * d / fx
Y = (v - cy) * d / fy
Z = d
```

On obtient un point 3D dans le repere camera. Empile-les tous : c'est ton point cloud. C'est exactement la recette qu'utilise `open3d.geometry.PointCloud.create_from_depth_image` [Open3D docs].

> Encadre cle : depth map -> point cloud, c'est juste l'inversion de `K` pixel par pixel. Pas de magie.

---

## 2. Capteurs de profondeur — comment on obtient la depth map

Un capteur RGB seul ne donne pas la profondeur (l'echelle est ambigue). Trois technologies dominent en 2025-2026 :

### Stereo

Deux cameras alignees, separees d'une **baseline** `b`. Un point du monde apparait a deux positions `u_L` et `u_R` dans les deux images. La **disparite** `Δu = u_L - u_R` te donne la profondeur :

```
Z = (fx * b) / Δu
```

Avantages : passif (pas de projection IR), fonctionne en exterieur. Inconvenients : echoue sur surfaces sans texture (mur blanc) et exige une calibration stereo precise.

### Time-of-Flight (ToF)

Le capteur emet un pulse infrarouge et mesure le temps de retour (ou la phase). Distance directe par pixel.

Avantages : precis a courte/moyenne portee, fonctionne dans le noir, pas besoin de texture. Inconvenients : interferences entre capteurs, multipath, ambiguite de phase.

### Structured light

Le capteur projette un motif IR connu (Kinect v1, RealSense). La distorsion du motif sur la scene te donne la profondeur via triangulation.

Avantages : tres precis a courte portee. Inconvenients : capote en exterieur (lumiere du soleil sature l'IR).

> Encadre : la majorite des capteurs RGB-D consumer (RealSense, Kinect, iPhone LiDAR) combinent IR active + stereo pour de la robustesse. Le pipeline Open3D s'en fiche : il prend une depth map et se debrouille.

---

## 3. Point clouds — la representation canonique

Un point cloud, c'est un ensemble non ordonne `{p_i ∈ R^3}` (parfois augmente de couleur, normale, intensite). L'objet `open3d.geometry.PointCloud` encapsule ca [Open3D docs].

Trois operations essentielles :

### Downsampling — voxelisation

Un capteur Realsense te crache 300k a 1M points par frame. Trop pour ICP en temps reel. La **voxelisation** discretise l'espace en cubes de taille `v` (ex. 5 mm) et garde un point par cube. `pcd.voxel_down_sample(voxel_size=0.005)`.

C'est un downsampling **deterministe et uniforme** dans l'espace. Apres voxelisation un nuage typique fait quelques milliers de points — gerable.

### Estimation des normales

Pour chaque point on cherche ses k voisins (KD-tree), on calcule la matrice de covariance locale, et la normale est le vecteur propre associe a la plus petite valeur propre. Critique pour les variantes ICP point-to-plane.

### Outlier removal

Statistical outlier removal : pour chaque point, distance moyenne aux k voisins ; rejette les points dont la distance est a plus de N ecart-types de la moyenne. Indispensable apres une depth map bruitee.

---

## 4. ICP — Iterative Closest Point

Tu as deux nuages :

- `source` : ce que la camera voit maintenant (la bouteille observee, partiellement),
- `target` : un modele 3D connu (CAD de la bouteille) **ou** une frame precedente.

Tu cherches la transformation rigide `T ∈ SE(3)` qui aligne `source` sur `target`. C'est de l'**enregistrement** (registration).

### L'algorithme (point-to-point, Besl & McKay 1992)

```
T <- T_init  (souvent identite, ou la pose precedente comme warm start)
boucle :
    1. pour chaque p_i de source, trouver le voisin le plus proche q_i dans target (avec T courant applique)
    2. resoudre min_T sum_i || T(p_i) - q_i ||^2  (probleme de Procrustes — closed-form via SVD)
    3. mettre a jour T
    arreter si le delta de T est sous un seuil
```

Le point critique : l'**initialisation**. ICP est local. Si T_init est trop loin, l'algo converge dans un mauvais bassin et te donne une pose absurde. Deux strategies pratiques :

- **Tracking** : la pose de la frame precedente est un excellent warm start (l'objet a peu bouge entre deux frames a 30 Hz).
- **Global registration** d'abord (RANSAC sur features FPFH) puis ICP pour raffiner — c'est le pipeline `register_via_global_then_local` standard d'Open3D [Open3D docs].

### Variante point-to-plane (Chen & Medioni 1991)

Au lieu de minimiser la distance point-a-point, on minimise la distance d'un point source au **plan tangent** local du target (defini par sa normale). Convergence beaucoup plus rapide quand on a des normales correctes. C'est ce qu'utilise Open3D par defaut quand tu lui passes des nuages avec normales.

> Encadre : ICP marche tres bien quand l'overlap entre source et target est grand (>30%) ET quand l'init est proche. Pour des scenes ou les deux conditions ne tiennent pas, il faut passer aux methodes apprises (FoundationPose section 6).

---

## 5. NeRF & 3D Gaussian Splatting — la revolution 2023-2025

Jusqu'ici on a parle de point clouds (representations explicites discretes). Depuis 2020, deux nouvelles classes de representations dominent la recherche :

### NeRF (Neural Radiance Fields, Mildenhall 2020)

Une scene 3D est encodee dans un MLP `f(x, y, z, θ, φ) -> (couleur, densite)`. Pour rendre une image, on tire des rayons et on integre le long de chaque rayon (volume rendering). 20-100 photos suffisent pour reconstruire une scene photorealiste.

Pourquoi ca compte en robotique : un robot peut se construire un modele 3D de son environnement a partir de quelques photos, sans capteur de profondeur. Probleme : le rendu et l'entrainement sont lents (minutes a heures).

### 3D Gaussian Splatting (Kerbl 2023)

Au lieu d'un MLP, la scene est un **nuage explicite de gaussiennes 3D** (chacune avec position, covariance, couleur, opacite). Le rendu est explicite (rasterization) et tres rapide (>100 FPS sur GPU consumer). Train en ~30 minutes, rendu temps reel.

Pour la robotique : 3DGS rapproche NeRF du temps reel. Combine avec une politique visuelle (J16+), un robot peut planifier en utilisant directement la geometrie reconstitute.

### V-JEPA 2 — la voie LeCun

[V-JEPA 2, Meta 2025] (REFERENCES.md #21) ne **reconstruit pas** explicitement la 3D. Au lieu de ca, il apprend a predire dans un **espace latent** ce que la suite d'une video va montrer. Ce latent encode implicitement la geometrie 3D et la dynamique. Le robot peut zero-shot planifier un pick-and-place en imaginant les frames futures dans ce latent. Position fondamentale : pour LeCun, reconstruire pixel-par-pixel (NeRF) ou point-par-point est un gaspillage ; ce qui importe, c'est la representation utile a la decision.

> Encadre : ce module reste centre sur les point clouds (canonique, robuste, peu de prerequis). NeRF/3DGS et JEPA sont l'horizon : a connaitre, mais on les manipule plus tard.

---

## 6. Du point cloud a la pose d'objet — pipelines modernes

L'objectif final : transformer un point cloud bruite et partiel en **pose 6D d'un objet**, idealement avec un score de confiance.

Trois approches selon ce dont on dispose :

### Approche modele-based (ICP brut)

On a un CAD de l'objet. Segmentation (Mask R-CNN, SAM) + ICP global puis local. Marche bien sur objets industriels avec CAD propre. C'est ce qu'on code aujourd'hui.

### Approche apprise — PoseCNN (Xiang 2018)

Reseau qui prend l'image RGB(-D) et regresse directement la pose 6D pour des objets connus a l'entrainement. Robuste aux occlusions partielles. Limite : il faut entrainer sur chaque objet.

### Approche zero-shot — FoundationPose (Wen 2024)

State-of-the-art 2024. Tu donnes une image RGB-D + un modele 3D **inconnu a l'entrainement**, et FoundationPose te donne la pose 6D. Combinaison de generation de templates synthetiques + matching neuronal + raffinement ICP. C'est la direction ou va le champ : moins d'entrainement par objet, plus de generalisation.

L'observation cle : meme les approches deep finissent par un **raffinement ICP** au bout. ICP n'est pas mort, il est devenu la derniere brique d'un pipeline plus large.

---

## Acquis fin de jour

- Modele pinhole : projection 3D->pixel et inverse pixel+depth->3D.
- Distinction intrinseques (`K`) vs extrinseques (`T_world_camera`).
- Pourquoi un capteur RGB-D (stereo, ToF, structured light) suffit a generer un point cloud.
- Voxelisation, estimation de normales, outlier removal sur point cloud.
- ICP point-to-point et point-to-plane : algo, conditions de convergence, role de l'initialisation.
- Vue d'ensemble NeRF / 3DGS / V-JEPA 2 : ce qui est explicite, ce qui est latent.
- Pipeline complet RGB-D -> pose 6D pour pick-and-place.

---

## Spaced repetition — flashcards

1. Q : Soit un pixel `(u=320, v=240)` avec depth `d=1.5 m`, et `K` de focales `fx=fy=500`, principal point `(cx=320, cy=240)`. Quel est le point 3D dans le repere camera ?
   R : `X = (320-320)*1.5/500 = 0`, `Y = (240-240)*1.5/500 = 0`, `Z = 1.5`. C'est sur l'axe optique a 1.5 m.

2. Q : Pourquoi ICP echoue si la pose initiale est trop loin de la pose vraie ?
   R : ICP est un descente locale base sur le voisin le plus proche. Si T_init est mauvais, l'association point-voisin est fausse, le minimum local atteint est arbitraire. D'ou l'usage de global registration (RANSAC + FPFH) en pre-traitement.

3. Q : Qu'apporte point-to-plane par rapport a point-to-point ICP ?
   R : Convergence plus rapide quand les normales sont fiables. La fonction objectif penalise moins le glissement le long du plan tangent (qui est un degre de liberte legitime quand on aligne deux surfaces).

4. Q : Pourquoi voxeliser un point cloud avant ICP ?
   R : Reduire le cout (de 500k a ~5k points), uniformiser la densite (le capteur sur-echantillonne les surfaces proches), reduire le bruit local par moyennage.

5. Q : 3D Gaussian Splatting vs NeRF — la difference principale ?
   R : NeRF stocke la scene dans un MLP (implicite, rendu lent par integration). 3DGS stocke la scene comme un nuage explicite de gaussiennes 3D, rasterizable (rendu temps reel >100 FPS).

6. Q : Quelle est la position de [V-JEPA 2, Meta 2025] sur la reconstruction 3D pixel-par-pixel ?
   R : C'est un gaspillage. Mieux vaut apprendre une representation latente predictive de la dynamique, dans laquelle la 3D est encodee implicitement. Le robot peut planifier dans ce latent sans jamais reconstruire la scene visuellement.
