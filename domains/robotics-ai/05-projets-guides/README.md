# 05 — Projets guides (Robotics & AI)

> Voir `shared/logistics-context.md` pour le contexte metier de LogiSim.

Trois projets qui appliquent la theorie du domaine (cinematique, planning, diffusion policies)
au terrain de jeu de FleetSim : la flotte robotisee d'un entrepot (AGV de transport, bras de
picking, postes de tri sur convoyeur). Tout est en **Python pur numpy/stdlib** : pas de MuJoCo,
pas de torch — chaque solution tourne en moins de 60 secondes sur CPU.

## Projets

| # | Projet | Theorie mobilisee | Difficulte |
|---|---|---|---|
| 01 | **AGV differentiel : cinematique + suivi de trajectoire** | J2-J3 (transformations, FK), J6 (controle) | medium |
| 02 | **Bras de picking : FK/IK + planning de trajectoire** | J3-J4 (FK/IK, jacobiens), J8 (planning) | medium |
| 03 | **Mini Diffusion Policy sur poste de tri 2D** ⭐ | J13 (BC), J15-J16 (diffusion, Diffusion Policy) | hard |

## Methodologie

Pour chaque projet :
1. Lire le contexte metier et la consigne du `README.md`
2. Coder ta version a partir des etapes guidees (sans regarder la solution)
3. Verifier les criteres de reussite (ils sont testables : assertions numeriques, tolerances)
4. Confronter a la correction commentee dans `solution/`
5. Faire tourner la solution et comparer les sorties

## Pourquoi ces trois projets ?

Ils couvrent les trois couches d'autonomie d'un robot FleetSim :
- **01** : la couche *mouvement* — comment une base mobile transforme des commandes
  roues en deplacement dans l'entrepot, et comment on la fait suivre un trajet
  calcule par le pathfinder (cf. projet `algorithmie-python/05-projets-guides/01`).
- **02** : la couche *manipulation* — comment un bras passe d'une cible cartesienne
  ("attrape le bac sur le convoyeur") a des angles moteurs, avec une trajectoire lisse.
- **03** : la couche *politique apprise* — pourquoi la regression BC echoue sur des
  demonstrations multimodales et comment une diffusion policy resout le probleme.
  C'est la version "pur numpy" du capstone J24-J28 (qui lui utilise torch + MuJoCo).

## Requirements

```bash
pip install numpy   # seule dependance, stdlib sinon
python <projet>/solution/<script>.py
```

Toutes les solutions sont deterministes (seed fixe) — contrainte LogiSim : meme seed,
meme resultat, pour la reconstitution d'incident et la certification.
