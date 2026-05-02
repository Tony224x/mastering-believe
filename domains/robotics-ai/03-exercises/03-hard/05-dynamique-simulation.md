# Exercice J5 — Hard : compensation de gravite + analyse de stabilite

## Objectif

Combiner dynamique inverse et simulation : sur un bras 2-DOF qui tomberait sous gravite, calculer le couple de compensation `τ_g(q) = g(q)` exact via MuJoCo, l'appliquer en boucle ouverte, et observer le comportement reel — incluant les sources d'instabilite numerique.

## Consigne

1. Reprends le bras 2-DOF de l'exercice medium (ou un bras 3-DOF si tu as envie de te challenger).
2. **Phase A — chute libre** :
   - Pose initiale arbitraire (par exemple `q = [π/3, -π/4]`).
   - Simule 2 secondes sans aucun couple. Logge `q(t)` toutes les 100 ms et l'energie totale.
   - Verifie que l'energie est conservee a < 1 % (pas de damping/frictionloss).
3. **Phase B — compensation gravite parfaite** :
   - A chaque pas, calcule `g(q)` en utilisant le truc `mj_rne` :
     - Mets `data.qvel[:] = 0` et `data.qacc[:] = 0` temporairement (sauvegarde-les avant).
     - Appelle `mujoco.mj_rne(model, data, 0, data.qfrc_inverse)` — ca calcule `qfrc_inverse = M·0 + C·0 + g(q) = g(q)`.
     - Applique `data.qfrc_applied[:] = data.qfrc_inverse[:]` (oppose ? non, `qfrc_inverse` est *deja* le couple necessaire pour maintenir l'equilibre statique — donc on l'applique tel quel).
     - Restaure `qvel`, `qacc`.
   - Simule 5 secondes depuis la meme pose initiale. Logge `q(t)` et l'energie.
4. **Analyse comparee** : produit un tableau qui montre, a `t = 0, 1, 2, 5 s` :
   - Phase A : `q(t)` (les angles partent en chute)
   - Phase B : `q(t)` (les angles devraient rester stables au bruit numerique pres)
   - Variation cumulee `||q(t) − q(0)||` dans les deux cas
5. **Adversariale** : refais Phase B mais en perturbant le modele utilise pour calculer `g(q)` : multiplie les masses internes par `1.1` (donc compensation imparfaite). Que se passe-t-il ? Donne 2 lignes d'analyse.

## Criteres de reussite

- Phase A : derive d'energie < 1 %, le bras tombe vers une configuration d'energie potentielle minimale.
- Phase B : `||q(5s) − q(0)||` < 1e-3 rad (le bras tient en l'air sans bouger).
- Phase B avec masses fausses : le bras derive lentement mais ne reste pas a `q(0)` — preuve que la compensation gravite *sans feedback* n'est pas robuste a l'incertitude des parametres.

## Indices

- `mj_rne` (Recursive Newton-Euler) calcule `qfrc_inverse = M(q)q̈ + C(q,q̇)q̇ + g(q)` pour le `(q, q̇, q̈)` courant, sans modifier qvel/qacc effectifs. Avec `q̇=q̈=0`, tu obtiens `g(q)` propre.
- L'astuce "mettre qvel et qacc a 0 temporairement" est standard pour isoler le terme gravite. Pense a sauvegarder-restaurer pour ne pas casser ta simu.
- `data.qfrc_applied` est ajoute aux forces calculees par MuJoCo a chaque pas (independant de `data.ctrl`).
- Adversariale : pour modifier les masses du modele en cours d'execution, tu peux faire `model.body_mass[i] *= 1.1` — c'est en place et MuJoCo recalculera M, C, g correctement aux pas suivants. Restaure-les apres ton test.
- Lecon profonde : la compensation gravite open-loop est utile (elle linearise approximativement le systeme) mais **insuffisante** sans terme PD en feedback — c'est ce qu'on attaque le J6 (computed torque control).
