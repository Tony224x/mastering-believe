# J17 — Exercice hard : ablation des 4 ingrédients « single-config » de DreamerV3

## Objectif

Reproduire à petite échelle l'expérience d'ablation qui justifie le titre du papier Hafner 2023 : montrer que retirer un seul des 4 piliers (symlog, latents discrets, KL balancing, free-bits) dégrade la robustesse du training.

## Consigne

1. Pars du fichier `02-code/17-world-models-dreamer.py`. Crée 4 variantes de la classe `WorldModel` (ou utilise des flags dans `Config`) :
   - **A. No symlog** : reward et obs en MSE direct, pas de symlog.
   - **B. Continuous z** : remplace les latents catégoriels par un gaussien diagonal `N(μ, σ)` avec reparam trick.
   - **C. No KL balancing** : KL symétrique `KL(q || p)` sans les deux termes asymétriques.
   - **D. No free-bits** : retire le `torch.clamp(min=free_bits)`.
2. Pour chaque variante, lance le training avec **3 seeds** (`torch.manual_seed(s)` pour s ∈ {0, 1, 2}) sur 60 itérations. Trace la courbe de `recon_loss` moyennée sur les 3 seeds, avec écart-type en bandeau.
3. Compare au baseline (V3 complet). Quels variantes sont :
   - **moins stables** (variance haute entre seeds) ?
   - **plus lentes à converger** (recon_loss reste élevé) ?
   - **carrément cassées** (training diverge ou collapse) ?
4. Écris une page de notes (en markdown, dans un fichier libre, pas commitée si tu ne veux pas) résumant :
   - quel ingrédient était le plus critique sur cette toy task,
   - hypothèses sur ce qui change à plus grande échelle (Atari, MuJoCo).
5. Cite la table d'ablation de Hafner 2023 (REFERENCES.md #20, section 4) et compare qualitativement avec tes résultats.

## Critères de réussite

- Tu produis un plot 2D (recon_loss vs iteration) avec 5 courbes (baseline + 4 ablations) et bandeau d'écart-type.
- Au moins **une** des 4 ablations dégrade visiblement la stabilité ou la convergence (sinon, tu n'as probablement pas implémenté l'ablation correctement — debug).
- Ta page de notes identifie *avec les bons mots* (KL collapse, posterior collapse, exploding gradients, scale-invariance) ce qui se casse dans chaque variante.
- Tu cites explicitement REFERENCES.md #20 (Hafner 2023, *Mastering Diverse Domains through World Models*).

## Pour aller plus loin

- Lance le repo officiel `danijar/dreamerv3` sur DMC `walker_walk` (GPU recommandé) et confirme qualitativement les conclusions sur un benchmark sérieux.
