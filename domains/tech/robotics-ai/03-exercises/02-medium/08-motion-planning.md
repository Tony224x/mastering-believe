# Exercice J8 — Medium : RRT-Connect bidirectionnel

## Objectif

Etendre le RRT du cours en **RRT-Connect** : deux arbres qui poussent en
parallele depuis `q_start` et `q_goal`, jusqu'a ce qu'ils se rencontrent au
milieu. C'est le standard de fait pour les bras industriels (MoveIt, OMPL).

## Consigne

A partir du squelette de l'exercice (ou de `02-code/08-motion-planning.py`) :

1. **Deux arbres** `T_a` (enracine en `q_start`) et `T_b` (enracine en
   `q_goal`).
2. A chaque iteration :
   - Sample un `q_rand` dans `C` (sans goal-bias cette fois — l'autre arbre
     joue le role du biais).
   - **Extend** : faire grandir `T_a` d'un pas vers `q_rand`, obtenir `q_new`.
   - **Connect** : depuis `T_b`, avancer **autant que possible** par pas de
     `eps` vers `q_new`. La sequence avance tant que `is_free_segment` reussit ;
     si elle atteint exactement `q_new`, les deux arbres se sont rencontres.
   - Sinon swap `T_a` et `T_b` et continuer.
3. Si rencontre : reconstruire le chemin global en concatenant
   `path_T_a(q_start -> q_meet)` puis `inverse(path_T_b(q_goal -> q_meet))`.
4. **Compare empiriquement** sur la meme scene que le cours :
   - Nombre de noeuds total (`T_a + T_b`) vs RRT classique
   - Nombre d'iterations
   - Sur 10 seeds differentes
5. Visualise les deux arbres en deux couleurs differentes + le chemin solution.

## Criteres de reussite

- RRT-Connect resout la scene avec **strictement moins** de noeuds totaux que
  RRT classique sur au moins 8 des 10 seeds.
- Le chemin reconstruit est continu (pas de discontinuite a la jointure entre
  les deux arbres).
- Tu sais expliquer pourquoi le swap `T_a <-> T_b` chaque iteration est
  important (sinon un seul arbre fait tout le boulot).

## Indices

- L'operation **Connect** est une boucle `while`, pas un seul `extend`.
  Garde une condition de sortie : `q_new` atteint, ou collision rencontree.
- Pour reconstruire le chemin : remonte les parents dans chaque arbre
  separement, puis inverse l'un des deux et concatene.
- Le critere "rencontre" : la derniere extension de `T_b` produit un noeud
  egal a `q_new` (a tolerance numerique pres).
