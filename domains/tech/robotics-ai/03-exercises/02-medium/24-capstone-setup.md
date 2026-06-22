# J24 — Exercice MEDIUM : analyser la distribution des actions et détecter la multimodalité

## Objectif

Démontrer **quantitativement** que le dataset PushT contient des actions **multimodales** étant donné un état d'observation similaire — exactement le phénomène qui justifie Diffusion Policy.

C'est l'exercice qui matérialise la promesse de la théorie (key takeaway #2 du module) : si on ne voit pas la multimodalité dans les données, le capstone n'a pas de raison d'être.

## Consigne

1. **Charge** le dataset comme dans l'EASY (`obs`, `action`, `ep_start`, `ep_length`).

2. **Statistiques globales sur les actions** :
   - histogramme de `action[:, 0]` (delta_x) et `action[:, 1]` (delta_y) sur **toutes** les transitions,
   - moyenne, std, min, max par dimension.

3. **Mise en évidence de la multimodalité conditionnelle** :
   - définis un **bucket d'observation** : par exemple "block dans le quart inférieur-gauche du plan ET agent au centre" (à toi de fixer un critère raisonnable),
   - récupère **toutes les transitions** dont l'observation tombe dans ce bucket,
   - plotte le scatter `(action[:, 0], action[:, 1])` pour ces transitions.
   - **Tu dois voir au moins 2 clusters distincts** — c'est la signature visuelle de la multimodalité.

4. **Score de multimodalité** : implémente une mesure simple. Suggestion : applique un **KMeans à k=2** sur le scatter du bucket (sklearn ou implé manuelle), calcule la **distance entre les deux centroïdes**, et compare-la à l'écart-type intra-cluster. Si `dist_centroids / mean_intra_std > 2.5`, on déclare le bucket "multimodal". Tu peux aussi simplement comparer la log-vraisemblance d'un mélange à 2 gaussiennes vs 1 (BIC).

5. **Variante (recommandée)** : refais l'analyse en restreignant aux transitions de **phase d'alignement** (les premiers ~20% de chaque épisode) vs **phase de poussée**. La multimodalité doit être **plus prononcée** en phase d'alignement (l'expert décide là du côté de contournement) qu'en phase de poussée (où la direction est forcée par la géométrie).

6. **Rendu** : un seul script `solution_medium.py` qui imprime un mini-rapport et sauvegarde 2 figures (`exo_medium_global.png`, `exo_medium_bucket.png`).

## Étapes suggérées

1. Lecture du dataset (cf. EASY).
2. Histogrammes globaux avec `plt.hist(...)`, 50 bins suffisent.
3. Filtrer les transitions par bucket : `mask = (obs[:, 2] < 256) & (obs[:, 3] < 256) & ...`.
4. KMeans k=2 sur les actions du bucket. Si tu n'as pas sklearn, code-le en 15 lignes (initialisation random + 10 itérations Lloyd).
5. Pour différencier alignement / poussée : pour chaque épisode, ses ~20% premiers steps sont alignement, le reste poussée. Construis 2 indices booléens.

## Critères de réussite

- [ ] Le script tourne en < 5 secondes.
- [ ] Histogrammes globaux affichés ; tu commentes 1 ligne sur la forme (souvent : symétrique en x autour de 0, biais positif en y vu que la target est en haut, ou similaire selon la seed).
- [ ] Au moins 1 bucket exhibe **clairement** au moins 2 clusters dans le scatter (visible à l'œil).
- [ ] Le score de multimodalité (BIC ou dist/std) confirme que le bucket d'alignement est plus multimodal que celui de poussée.
- [ ] Tu rédiges en 3-4 phrases l'interprétation : "voilà pourquoi BC unimodal échouera et pourquoi diffusion va aider".

## Pièges classiques

- **Bucket trop large** : si tu ne filtres pas assez, tu mélanges des transitions de plein de contextes différents, le scatter devient un nuage uniforme et tu ne vois pas la multimodalité. Sois sélectif (ex. "block dans une zone de 100×100 px, agent dans une zone de 80×80 px").
- **Bucket trop étroit** : tu finis avec 5 transitions, KMeans n'a pas de signal. Vise 50-200 points dans ton bucket.
- **Confondre multimodalité dans les *trajectoires* et dans les *transitions*** : ce qu'on regarde ici c'est `p(action | obs)`. Pour bien voir la multimodalité dans la trajectoire complète (action chunk), il faudrait grouper par épisode et comparer les chunks — mais c'est plutôt l'objet du HARD.
- **Oublier de normaliser les axes** quand tu plottes deux variables d'échelles différentes (ici les deux dimensions de l'action sont en pixels donc même échelle, OK).

## Pour aller plus loin (bonus)

- Trace pour 5 buckets différents la **proportion** de transitions multimodales selon ton score. Voir si elle corrèle avec la position du block sur la table.
- Compare `p(action | obs)` brut vs après marginalisation sur le `side` (lisible dans `episodes.jsonl`). Le scatter doit perdre la multimodalité une fois conditionné sur le side — c'est cohérent : `side` *est* la variable latente que la diffusion doit modéliser implicitement.
