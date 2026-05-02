# J22 — Exercice MEDIUM : instrumenter le decouplage temporel

## Objectif

Manipuler concretement l'orchestrateur dual-system du module : mesurer le **gain de latence** que procure le decouplage System2/System1, et verifier que le goal embedding **change effectivement moins souvent** que les actions.

Tu utilises le code `02-code/22-frontier-humanoid.py` comme base.

## Consigne

A partir de `DualSystemController` du module :

1. **Compteurs**. Ajoute deux compteurs internes : `n_system2_calls` (incremente quand on (re)calcule l'embedding) et `n_system1_calls` (incremente a chaque appel a `act`).
2. **Latence simulee**. Mesure le temps reel passe dans `system2(...)` et `system1(...)` separement (`time.perf_counter`), accumule sur tout l'episode (1000 steps).
3. **Trois configurations**. Lance le meme episode avec `system2_period in {1, 4, 8, 16}` et imprime un tableau :
   - period
   - n_system2_calls
   - n_system1_calls
   - total_time_system2_ms
   - total_time_system1_ms
   - avg_distance (la metrique de l'env)
4. **Inspection embeddings**. Pour `system2_period=8`, capture l'embedding sortie System2 a chaque (re)calcul et imprime la difference L2 entre embeddings consecutifs : `||z_{k+1} - z_k||_2`. Tu dois observer que cette norme bouge nettement (le goal embedding ne reste pas constant).
5. **Une phrase d'analyse**. Sur la base de ton tableau, justifie pourquoi en pratique on choisit `period > 1` cote production : le compromis "qualite du suivi vs cout System2".

## Criteres de reussite

- Les compteurs `n_system2_calls` et `n_system1_calls` sont coherents : pour 1000 steps, `n_system1_calls == 1000` et `n_system2_calls ≈ 1000 / period`.
- Les temps cumules sont mesures en `ms` avec `time.perf_counter` (pas `time.time` pour la precision).
- Le tableau imprime montre que `total_time_system2_ms` decroit quand `period` augmente.
- Les diffs L2 sur les embeddings consecutifs sont strictement non nulles (sinon ton System2 ne digere rien).
- La phrase d'analyse mentionne explicitement le compromis frequence-de-replanning vs cout-VLM.

## Indices

- N'entraine pas les reseaux : reseaux non-entrainees, le but est l'instrumentation.
- Pour la precision, fais 3 runs et affiche la moyenne (warm-up : ignore le 1er run).
- Solution-style : un script ~80 lignes, classe heritee de `DualSystemController` qui ajoute les compteurs.

## Verification

`python -m py_compile mes-reponses-22-frontier-humanoid-medium.py` doit PASS, puis `python mes-reponses-22-frontier-humanoid-medium.py` produit le tableau.
