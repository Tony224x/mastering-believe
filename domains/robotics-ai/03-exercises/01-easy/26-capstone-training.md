# Exercice EASY — J26 : explorer le learning rate

## Objectif

Comprendre l'impact du learning rate sur la dynamique d'entrainement d'une Diffusion Policy. Voir concretement ce qui se passe en LR-trop-bas, LR-bien-calibre, et LR-trop-haut.

## Consigne

Pars du script `domains/robotics-ai/02-code/26-capstone-training.py`. Lance le entrainement avec **trois valeurs de LR differentes** :

1. `lr_max = 1e-5` (trop bas)
2. `lr_max = 1e-3` (la valeur par defaut, supposee bien calibree)
3. `lr_max = 1e-1` (clairement trop haut)

Pour chaque run :
- Sauvegarde la liste `losses` retournee par `train(...)` dans un fichier `.npy` distinct.
- Garde 1500 steps comme dans le script de base.

Trace ensuite les **trois courbes de loss sur le meme graphe matplotlib** (axe Y en log pour bien voir les ecarts), avec une legende explicite.

## Criteres de reussite

- Le script tourne sans erreur pour les trois LR.
- Le graphe genere montre clairement :
  - LR trop bas : loss qui descend tres lentement, plateau bien au-dessus de 0.5.
  - LR bien calibre : loss qui descend rapidement vers ~0.1 puis se stabilise.
  - LR trop haut : loss instable, oscillations marquees, voire divergence (NaN).
- Tu peux expliquer en une phrase pourquoi le warmup lineaire (100 premiers steps) attenue partiellement le probleme du LR trop haut au demarrage.

## Indice

Pour la divergence (LR=1e-1), tu vas probablement voir des `NaN` dans les loss apres quelques dizaines de steps. C'est attendu. Filtre-les avec `np.isnan` avant de plotter, ou affiche jusqu'au premier NaN.
