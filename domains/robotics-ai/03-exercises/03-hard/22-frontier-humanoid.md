# J22 — Exercice HARD : entrainer le dual-system par imitation

## Objectif

Entrainer effectivement les deux reseaux du dual-system jouet sur la tache de poursuite 2D, en imitation d'un expert oracle. Ca te force a affronter l'**interface** entre System2 et System1 : comment passer un signal qui aide System1 sans que ce dernier doive re-comprendre le monde.

C'est l'analogue conceptuel de ce qui se passe pendant l'entrainement de GR00T N1 ou Helix : le VLM produit des features, l'action head apprend a les exploiter.

## Consigne

1. **Generer un dataset expert (5000+ transitions)**. Ecris un oracle qui, a chaque step, sort l'action optimale `a* = clamp(target - agent, [-0.3, 0.3])`. Roule cet oracle pendant N=20 episodes de 1000 steps. Pour chaque step, log dans un buffer :
   - `obs` (4 floats)
   - `target_history` (16 dernieres positions de la cible, 32 floats)
   - `a*` (2 floats)
   - **`true_strategy`** (label parmi `static`, `horizontal_oscillation`, `circle`)

2. **Pre-train System2 par auxiliary task**. System2 doit apprendre a inferer la strategie courante depuis l'historique. Ajoute une tete de classification (3 classes) sur la sortie System2 et entraine-la cross-entropy sur `true_strategy`. **Sans cette etape, System2 ne sait rien faire.** Mesure l'accuracy validation.

3. **Train System1 conditionne**. System1 prend `obs` + l'embedding System2 (couches profondes, en gelant System2). Entraine System1 par MSE sur `a*`. Compare la loss vs un baseline System1-only (zero embedding). **Discute** : sur cette tache jouet, l'action optimale est lineaire dans `obs`, donc le baseline peut deja bien la fitter ; le gain de System2 doit se voir si tu modifies l'env pour rendre la cible **partiellement observable** (par exemple en n'envoyant a System1 que la position agent, pas celle de la cible). C'est exactement la difference entre "OpenVLA monolithique sur tache simple" et "Helix sur tache requiring un goal langagier".

4. **Evaluation environnement**. Roule le controller dual-system entraine sur 10 episodes de 1000 steps, calcule l'`avg_distance`. Compare a :
   - oracle (optimum)
   - System1-only entraine (sans embedding)
   - dual-system entraine

5. **Ablation : `system2_period`**. Fais varier `system2_period in {1, 4, 16, 64}` au moment de l'evaluation (sans re-entrainer). Trace `avg_distance(period)` et identifie le point ou le decouplage casse (la frequence de raffraichissement du goal devient trop basse pour suivre le regime change toutes les 80 steps).

6. **Une page de retro (en commentaire de fin de fichier ou markdown joint)**. 5-8 lignes. Reponds : "Quand est-ce que le decouplage temporel commence a couter de la performance ? Comment le pattern industriel (Helix/GR00T) gere ca en pratique d'apres ce que tu as lu ?"

## Criteres de reussite

- Dataset expert construit avec ≥ 5000 transitions, **stratifie** sur les 3 strategies (au moins 25% de chaque).
- System2 atteint ≥ 75% accuracy de classification sur la strategie courante (validation).
- System1 conditionne **et** System1-only convergent tous les deux et tu **discutes** l'ecart de MSE (qui peut etre faible sur la version "fully observable"). Bonus si tu fais l'experience "partially observable" et montres que le gain de System2 devient strictement positif.
- L'eval environnement montre : `dual-system_trained` < `system1-only_trained` ≤ `random` en avg_distance.
- L'ablation `system2_period` montre une rupture quand `period` depasse `regime_change_every / 4` (intuitif : tu rates les bascules de regime).
- Le code passe `python -m py_compile` et roule en moins de 5 minutes sur CPU.

## Indices

- AdamW lr=1e-3 sur les deux reseaux, 50-100 epochs suffit.
- Stratification : echantillonne depuis l'env en forcant la strategie initiale (modifie `_pick_new_strategy` pour boucler).
- Pour System1 conditionne : pendant le train, **gele** System2 (`requires_grad_=False`) pour eviter qu'il oublie sa classification. C'est exactement ce que fait NVIDIA quand ils freeze le VLM Eagle pendant le post-training de la diffusion head GR00T N1.
- Le bug classique : oublier que System1 doit recevoir un embedding **fresh** a chaque step ou il triche en memorisant. Tu peux desactiver le cache pendant le training.

## Bonus (facultatif)

Lis le repo `NVIDIA/Isaac-GR00T` et identifie : (a) ou le VLM est appele, (b) ou la diffusion transformer head est appelee, (c) si elles tournent au meme rythme dans `eval` ou non. Note tes observations en 5 lignes.

## Verification

`python -m py_compile mes-reponses-22-frontier-humanoid-hard.py` doit PASS. Joint le tableau d'evaluation final dans un commentaire ou un .md adjacent.
