# J27 — Exercice HARD : ablation complète 4 conditions × 3 seeds + plot mean ± std + test statistique

## Objectif

Reproduire (à échelle réduite) le **tableau d'ablations** de la section §6 du papier Diffusion Policy : tu lances 4 conditions × 3 seeds, tu produis un plot `mean ± std`, et tu fais un test statistique simple pour distinguer "vrai gain" de "bruit".

C'est l'exercice qui matérialise la valeur d'un capstone : à la fin tu as un livrable que tu peux montrer (plot + tableau + commentaire).

## Consigne

1. Choisis **4 conditions** (configurations d'ablation) à comparer. Suggestions :
   - C1 : Diffusion Policy full (chunking H=16, EMA on, T_alpha=8, T_denoise=50),
   - C2 : sans chunking (H=1, EMA on, T_alpha=1),
   - C3 : sans EMA (H=16, EMA off, T_alpha=8),
   - C4 : BC baseline (MLP, T_alpha=1).

2. Pour chaque condition, **entraîne** la policy correspondante et **évalue** sur 3 seeds × N=15 rollouts. Tu peux réutiliser/copier le module `02-code/27-capstone-eval-ablations.py` comme base.

3. Calcule pour chaque condition :
   - `success_rate_per_seed` = liste de 3 floats (un par seed),
   - `mean = np.mean(...)`,
   - `std = np.std(..., ddof=0)`.

4. Produis un **plot matplotlib** de type bar chart avec :
   - 4 barres (une par condition),
   - hauteur = `mean`,
   - errbar = `std` (sur la moyenne entre seeds),
   - titre, ylabel, xticks lisibles,
   - sauvegarde en `j27_hard_ablation.png`.

5. Implémente un **test statistique simple** : pour les paires d'intérêt (ex : C1 vs C2, C1 vs C4), calcule un test t de Welch entre les deux distributions de 3 valeurs. Tu peux utiliser `scipy.stats.ttest_ind(a, b, equal_var=False)` ou faire à la main :

   ```
   t = (mean_a - mean_b) / sqrt(var_a/N_a + var_b/N_b)
   ```

   Avec N=3 c'est un test d'effet faible, donc tu interprètes :
   - p < 0.05 : effet probable (mais avec 3 seeds c'est un signal *faible*, à reproduire),
   - p > 0.20 : pas de signal détectable.

6. Écris un **commentaire de 5-8 phrases** dans la console qui interprète :
   - Quelle ablation a l'effet le plus fort ? (Tu devrais voir : sans chunking >> sans EMA.)
   - L'écart entre Diffusion Policy full et BC est-il statistiquement détectable avec 3 seeds ?
   - Quelle métrique secondaire (smoothness ou ep_len) confirme ou contredit le ranking en success rate ?

## Étapes suggérées

1. Pars du module `02-code/27-capstone-eval-ablations.py` : copie-le dans un fichier `solution_hard.py`.
2. Réduis `n_steps` d'entraînement à 400 pour itérer vite (la qualité absolue importe peu, c'est le RANKING qui compte).
3. Reporte les 4 résultats dans une dataclass `EvalResult` (déjà fournie par le module).
4. Plot avec matplotlib `plt.bar(..., yerr=stds)`.
5. Pour le test t : `from scipy.stats import ttest_ind ; t, p = ttest_ind(c1_seeds, c2_seeds, equal_var=False)`.
6. Print un commentaire structuré.

## Critères de réussite

- [ ] Le script tourne end-to-end en < 10 min sur CPU (4 trainings + 4 evals).
- [ ] Le plot est lisible : 4 barres, errbars visibles, labels non-tronqués.
- [ ] Le tableau de tests t est imprimé : au moins (C1 vs C2) et (C1 vs C4).
- [ ] Le ranking attendu est respecté : `C1 > C3 > C2 > C4` en success rate (avec petites variations selon le seed).
- [ ] Le commentaire identifie correctement que C2 (sans chunking) est l'ablation la plus coûteuse.
- [ ] Bonus : tu fais un 2e plot superposant `success_rate` et `smoothness` (deux y-axes) pour montrer la corrélation inverse.

## Pour aller plus loin (bonus)

- Augmenter à 5 seeds × 30 rollouts et relancer : la `std` doit DIMINUER, et le test t doit devenir plus net. Cette progression illustre directement la **loi des grands nombres** appliquée à l'eval.
- Ajouter une condition C5 : "Diffusion Policy avec T_denoise=10" (DDIM-like sans réentraîner). Tu devrais voir un *gros* gain de latence pour une perte minime de success rate. C'est l'optimisation principale pour passer de 10Hz à 30Hz contrôle.
- Plot une `Pareto frontier` `success_rate` (y) vs `latency_ms` (x), avec les 4-5 conditions comme points. Visualisation canonique pour parler de déployabilité.

## Pièges classiques

- **Seeds non-indépendants** : si tu fais `torch.manual_seed(0)` UNE fois en début de script et tu lances 3 trainings d'affilée, les 3 trainings ne sont PAS 3 seeds vraiment indépendants. Ils partagent l'état du RNG global qui dérive entre les runs. Pour des "vrais" seeds, refais `torch.manual_seed(s)` AU DÉBUT de chaque training.
- **Comparaison déloyale** : si BC s'entraîne sur `n_steps=400` et Diffusion Policy sur `n_steps=2000`, tu compares des budgets différents. Pour une comparaison juste, soit tu mets le même `n_steps`, soit tu reportes explicitement le coût d'entraînement.
- **Sur-interprétation du test t avec N=3** : 3 seeds c'est très peu. Un p < 0.05 est *suggestif*, pas une preuve. Toujours commenter le nombre de seeds quand tu rapportes un test stat. Un papier sérieux utilise N=10+ seeds.
- **Plot avec yerr trop grand** : si les errbars de C1 et C2 se chevauchent visuellement, ne pas conclure "il n'y a pas d'effet". Le test t ci-dessus quantifie réellement la séparation, pas le chevauchement visuel.

## Critère de fierté

À la fin tu as :
- un PNG (`j27_hard_ablation.png`) que tu pourrais coller tel quel dans un README portfolio,
- un tableau Markdown imprimé avec les 4 conditions,
- un commentaire de 5-8 phrases qui défend ton ranking statistiquement.

C'est exactement ce qu'on attend dans la partie "Experiments" d'un papier IL/RL moderne.
