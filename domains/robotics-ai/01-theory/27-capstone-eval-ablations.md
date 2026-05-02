# J27 — Capstone : évaluation, ablations et baseline BC

> Durée d'étude : 45-60 min
> Prérequis : J16 (Diffusion Policy theorie), J24 (PushT + dataset), J25 (architecture ResNet18 + UNet1D + DDPM), J26 (training loop + EMA).
> Source principale : REFERENCES.md #19 — Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion", RSS 2023 / IJRR 2024 — section §6 (Experiments) https://diffusion-policy.cs.columbia.edu/

---

## 1. Hook — la loss descend, la policy échoue

Tu as passé J24-J26 à entraîner ton Diffusion Policy sur PushT. `wandb` montre une loss MSE qui descend joliment de 0.45 à 0.012 sur 200 epochs. Tu lances un rollout : le robot pousse le T pendant 80 steps, puis le bloc dérape et l'épisode timeout sans succès. Tu refais 3 rollouts : 0/3 succès. Mais tu sais que le papier annonce 91% de success rate sur PushT.

Ce qui s'est passé : **la loss d'entraînement n'est pas la métrique de la policy**. Une loss MSE de 0.012 dit "en moyenne, je prédis le bruit à 0.012 près sur le dataset de démos". Elle ne dit RIEN sur :
- est-ce que la trajectoire générée est *cohérente* sur 16 steps consécutifs ?
- est-ce que le robot *récupère* après une perturbation (distribution shift) ?
- est-ce que la policy contrôle bien la vraie dynamique ou juste l'imitation moyenne ?

Le capstone J27 répond à cette question : **comment évalue-t-on rigoureusement une policy d'imitation, et qu'est-ce qu'on apprend en la comparant à des baselines simples et à des versions ablatées ?**

Concret avant abstrait : voici le tableau qu'on veut produire à la fin du jour.

```
                                  success_rate   episode_len   action_smooth   latency_ms
Diffusion Policy (full)           0.84  ± 0.04        72.3          0.011           18.2
Diffusion Policy (no chunking)    0.61  ± 0.07       108.4          0.038           21.0
Diffusion Policy (no EMA)         0.72  ± 0.06        85.7          0.014           18.1
Behavior Cloning (MLP baseline)   0.42  ± 0.08       142.6          0.083            1.4
```

Chaque ligne est 50 rollouts, chaque chiffre dit quelque chose de précis sur le système. C'est ça qu'on va construire.

> **Key takeaway #1**
> Un module IL/RL **n'est pas terminé** quand la loss descend. Il est terminé quand tu as : (a) un protocole d'eval reproductible avec N≥50 rollouts, (b) au moins une baseline triviale (BC) pour contextualiser le score, (c) au moins 2 ablations qui isolent les choix de design importants.

---

## 2. Protocole d'évaluation — N rollouts, seeds, métriques

### 2.1 Pourquoi N rollouts et pas un seul

Une policy stochastique (DDPM échantillonne du bruit) sur un environnement stochastique (initial state aléatoire, parfois bruit de capteur) produit un **success rate** qui est une grandeur statistique. Avec N=1 rollout, ton estimateur a une variance énorme. Avec N=50, l'erreur standard d'une proportion vaut `σ ≈ √(p·(1-p) / N)` — pour `p=0.8`, `σ ≈ 0.057`. Pour passer sous 0.02 il faudrait N=400. Le compromis raisonnable du papier Diffusion Policy : **50 rollouts** par config, parfois 100 sur les benchmarks finaux (cf. Tab. 2 du papier).

Bonne pratique :
- **fixer K seeds** (typiquement K=3) sur lesquels tu tournes l'évaluation, pour obtenir mean ± std *across seeds* (la vraie incertitude qu'on veut reporter),
- **fixer N rollouts par seed** (typiquement N=50),
- la taille effective d'évaluation = K × N rollouts.

Petit piège : si tu utilises `np.random.seed(42)` puis tu génères 50 init states, c'est UN seul "seed". Pour avoir 3 seeds réels tu dois ré-instancier l'environnement avec 3 seeds parents distincts (`42`, `1337`, `2026`) et générer 50 rollouts par parent.

### 2.2 Trois métriques à toujours reporter

#### (a) Success rate

Définition d'un succès : dépend de l'environnement. Sur PushT, c'est `IoU(bloc_T, cible) > 0.95` (ou `0.9` selon le papier). Sur une tâche pick-and-place : `dist(objet, target) < 5cm` à la fin de l'épisode. C'est un **booléen** par rollout, agrégé en taux.

Reporting : `success_rate = mean ± std across seeds`.

#### (b) Episode length

Pour les épisodes qui réussissent (et seulement eux), combien de steps a-t-il fallu ? Une policy qui réussit en 50 steps est **meilleure** qu'une qui réussit en 180 steps (moins d'erreur cumulée, meilleur contrôle, moins d'énergie). Sur les épisodes qui échouent, on rapporte typiquement le timeout (souvent confondu avec horizon max de l'env, ex. 200 steps).

#### (c) Action smoothness

Métrique de qualité de contrôle, pas de réussite. Définition canonique :

```
smoothness = mean_t || a_{t+1} - a_t ||²
```

Une policy "qui tremble" produit des actions saccadées (smoothness élevée), même si elle réussit. Pour un bras réel, un score `smoothness > 0.5` (en unités d'action normalisées) signifie usure mécanique et instabilité. Diffusion Policy excelle sur cette métrique grâce à l'**action chunking** : la policy prédit 16 actions consécutives qui sont déjà cohérentes par construction (le UNet 1D voit toute la séquence).

#### (d) Latence (compatibilité contrôle temps-réel)

Combien de millisecondes prend `policy.predict(obs)` sur ton hardware cible ? PushT tourne à 10 Hz (100 ms par step). Si ta policy prend 250 ms, **elle ne peut pas contrôler le robot en temps réel** sans buffer.

Diffusion Policy a un coût d'inférence non-trivial : **T=100 steps de denoising DDPM**. Le papier annonce ~50ms sur GPU avec optimisation, mais sur CPU consumer tu verras 300-800ms — d'où l'astuce du **receding-horizon** (cf. §4) qui amortit le coût d'inférence sur plusieurs steps.

> **Key takeaway #2**
> Toujours reporter ≥3 métriques : success rate (réussite), episode length (qualité), smoothness (contrôle), latency (déployabilité). Une policy qui maximise success_rate en sacrifiant la latence est **inutilisable** sur un robot @ 30 Hz.

---

## 3. Baseline BC — pourquoi c'est OBLIGATOIRE

### 3.1 La baseline triviale qu'on n'a aucune excuse de pas faire

**Behavior Cloning** = un MLP qui prend l'observation `o_t` et prédit l'action `a_t`. Loss MSE supervisée sur les démos. C'est l'algo IL le plus bête possible (cf. J13).

Pourquoi c'est crucial dans un capstone Diffusion Policy :
- si BC fait déjà 80%, ta complexité DDPM n'apporte que 4% — peut-être pas justifiée,
- si BC fait 30%, tu démontres que ton problème EST multimodal (la moyenne ne suffit pas),
- BC te donne une **borne basse** : si Diffusion Policy fait pire que BC, t'as un bug.

Sur PushT spécifiquement, le papier (§6, Tab. 2) reporte :
- BC (MLP) : ~30% success rate,
- LSTM-GMM : ~70%,
- Diffusion Policy : ~91%.

L'écart BC vs Diffusion Policy sur PushT est large parce que la tâche a une **multimodalité d'action** : pour pousser le T, on peut aller à gauche ou à droite, deux trajectoires valides. Le MLP qui régresse la moyenne tombe entre les deux et échoue. Le DDPM échantillonne UNE des deux (mode-covering puis sampling).

### 3.2 Choix de l'architecture BC

Pour comparaison **équitable** :
- même **input** (image + état) que Diffusion Policy,
- même **encoder** vision si possible (ResNet18 frozen ou shared),
- output = action 1-step (pas de chunking — c'est ce qui distingue BC).

Choix de modèle BC raisonnable : `[image → ResNet18 → MLP 256-256 → action]`. Pas plus compliqué — sinon tu compares à une autre policy, pas à une *baseline*.

### 3.3 Le piège : "BC bien tuné" peut surprendre

Plusieurs papiers récents (cf. survey IL Zare 2024) montrent que **BC + augmentation des observations + LayerNorm + dropout** comble une bonne partie de l'écart. Ne pas sous-tuner volontairement BC pour faire briller Diffusion Policy. Ton capstone est plus crédible si la baseline est honnête.

---

## 4. Receding-horizon en eval — l'art de réutiliser un chunk

### 4.1 Rappel J16 : action chunking

À chaque step `t`, Diffusion Policy reçoit `o_t` et prédit un **chunk de H actions futures** : `[a_t, a_{t+1}, ..., a_{t+H-1}]`. Typiquement H=16.

Stratégie naïve d'exécution :
```
for t in range(T):
    a_chunk = policy(o_t)        # chunk de 16 actions
    env.step(a_chunk[0])         # on n'utilise que la première
```
Coût : 1 inférence DDPM (~T_denoise=100 steps) PAR step physique. À 100ms d'inférence et un horizon T=200, ça fait 20 secondes d'inférence par rollout. ×50 rollouts = 16 minutes juste à évaluer.

### 4.2 Receding horizon : exécuter Tα steps puis replanifier

Le papier propose : **exécuter Tα = 8 actions** (sur 16 prédites) avant de replanifier. Code :
```
t = 0
while t < T:
    a_chunk = policy(o_t)        # 1 inférence
    for k in range(T_alpha):     # exécuter Tα actions
        env.step(a_chunk[k])
        t += 1
```
Gain : **1 inférence pour Tα=8 steps physiques**, soit 8× moins d'inférences. Pour H=16 et Tα=8, on garde une "marge de sécurité" de 8 actions futures qu'on jette à chaque replanning — c'est le compromis.

Trade-off :
- `Tα = 1` (replan à chaque step) : **réactivité maximale** (la policy voit la dernière obs), latence rédhibitoire.
- `Tα = H = 16` (exécuter tout le chunk avant replan) : 16× moins cher, mais **aveugle** pendant 1.6s — fragile si l'environnement change.
- `Tα = 8` (compromis Diffusion Policy) : la moitié du chunk exécutée, l'autre moitié sécurisée et prête en cas de retard d'inférence.

Le papier (§6.3) montre que Tα ∈ [4, 8] est l'optimum sur PushT.

> **Key takeaway #3**
> Receding-horizon n'est pas une optimisation : c'est **partie intégrante du protocole d'eval**. Sans ça, Diffusion Policy semble trop lent ; avec ça, elle devient déployable. Toujours reporter `(H, Tα)` quand on parle de latence.

---

## 5. Ablations canoniques — qu'est-ce qui compte vraiment

Une ablation = on désactive UN composant, on re-mesure les métriques, on quantifie la perte. Les ablations qui méritent d'être dans un capstone Diffusion Policy :

### 5.1 Ablation 1 : sans action chunking (H=1)

Le modèle prédit 1 action future au lieu de 16. C'est le test critique : **est-ce que le chunking matter ?**

Effet attendu : le success rate chute (de 84% à 60% typiquement), et **action smoothness empire fortement** (×3-5). Pourquoi : sans chunking, le modèle voit chaque step indépendamment, les trajectoires perdent leur cohérence temporelle (le cross-attention entre actions voisines est ce qui rend les chunks lisses).

### 5.2 Ablation 2 : sans EMA

EMA (Exponential Moving Average) sur les poids — vu en J26 — stabilise les checkpoints en moyennant exponentiellement les poids du modèle pendant l'entraînement. C'est un trick standard des modèles de diffusion (Ho 2020).

Effet attendu : success rate baisse de 5-15%, et **variance entre seeds augmente**. EMA est ce qui rend l'entraînement reproductible : sans EMA, chaque seed peut diverger légèrement.

### 5.3 Ablation 3 : schedule alternative

Diffusion Policy par défaut utilise un schedule `square_cosine_cap_v2` (du papier). Ablation : remplacer par schedule **linéaire** (DDPM original) ou **cosine** simple.

Effet attendu : variation petite mais mesurable (±3% success rate). Le schedule contrôle la distribution de bruit pendant le training et le sampling — le cosine adapte mieux le bruit aux fréquences pertinentes pour les actions.

### 5.4 Ablation 4 : classifier-free guidance (CFG)

CFG (Ho & Salimans 2022) : pendant le sampling, on calcule la velocity à la fois avec et sans condition (l'observation), puis on extrapole `v_guided = v_uncond + s · (v_cond - v_uncond)` avec `s > 1`. Effet : amplifier l'influence de la condition, parfois utile en imitation learning quand la policy "oublie" l'observation.

Effet attendu sur PushT : marginal (la condition est déjà très forte). Plus utile sur les VLAs où la condition est langagière et faible.

> **Key takeaway #4**
> Hiérarchie d'impact des composants Diffusion Policy : **chunking >> EMA > schedule > CFG**. Si tu n'as le budget que pour 2 ablations, fais chunking et EMA — c'est là que se joue 90% de l'écart vs BC.

---

## 6. Latence — la métrique qu'on oublie systématiquement

Mesurer la latence proprement :

```python
import time
n_warmup, n_measure = 5, 50
for _ in range(n_warmup):                # warmup pour torch JIT/CUDA cache
    policy.predict(obs)
times = []
for _ in range(n_measure):
    t0 = time.perf_counter()
    policy.predict(obs)
    times.append(time.perf_counter() - t0)
mean_ms = 1000 * np.mean(times)
p99_ms  = 1000 * np.quantile(times, 0.99)
```

Ce qu'on rapporte :
- **mean** : pour comparer les modèles entre eux,
- **p99** : pour la déployabilité (si p99 > 100ms, ta policy peut ne pas tenir 10 Hz dans le pire cas).

Ordres de grandeur attendus (PushT, image 96×96) :
- BC MLP : ~1-3 ms (CPU),
- Diffusion Policy avec T=100 denoising steps : ~150-400 ms (CPU), ~30-80 ms (GPU consumer),
- Diffusion Policy avec DDIM 16 steps : ~25-60 ms (CPU) — DDIM (Song 2021) est l'autre levier d'optimisation, pas une ablation mais une optimisation.

Tableau récapitulatif :
- @ 10 Hz contrôle (100 ms budget) : BC OK, Diffusion Policy CPU limite, GPU OK,
- @ 30 Hz contrôle (33 ms budget) : BC OK, Diffusion Policy nécessite GPU + DDIM,
- @ 100 Hz contrôle (10 ms budget) : seul BC tient sans optim agressive.

C'est *ça* le contexte où Helix (J22) tourne à 200 Hz : ils ne font pas de DDPM à 100 steps, ils font de la flow matching @ 5 steps + cache backbone.

---

## 7. Mettre tout ensemble — le tableau final

Le livrable de ton capstone doit ressembler à :

| Method | Success ↑ | EpLen ↓ | Smooth ↓ | Latency (ms) ↓ |
|---|---|---|---|---|
| Diffusion Policy (full) | 0.84 ± 0.04 | 72.3 | 0.011 | 18.2 |
| − sans chunking (H=1) | 0.61 ± 0.07 | 108.4 | 0.038 | 21.0 |
| − sans EMA | 0.72 ± 0.06 | 85.7 | 0.014 | 18.1 |
| − schedule linéaire | 0.81 ± 0.05 | 76.1 | 0.012 | 18.3 |
| BC MLP (baseline) | 0.42 ± 0.08 | 142.6 | 0.083 | 1.4 |

Lecture critique de ce tableau (à toujours faire, sinon le tableau ne sert à rien) :
1. Diffusion Policy bat BC sur success rate (×2) et smoothness (÷7), pour 13× la latence,
2. l'ablation chunking est la plus coûteuse → confirme que c'est le composant clé,
3. EMA contribue 12% de success rate → ROI massif pour 0 paramètre supplémentaire,
4. la latence à 18ms est compatible 30 Hz — on peut déployer.

> **Key takeaway #5**
> Tu n'as pas terminé un capstone IL avant d'avoir produit le tableau eval × ablations × baseline. C'est le livrable qui te permet de dire "j'ai compris le système", pas le notebook de training.

---

## 8. Q&A — spaced repetition

**Q1 — Combien de rollouts minimum pour reporter un success rate raisonnable ?**
N=50 par seed × K=3 seeds = 150 rollouts. Pour un success rate de 0.8, ça donne σ ≈ 0.06 entre seeds, suffisant pour distinguer un écart de 10%.

**Q2 — Qu'est-ce qui motive le receding horizon en eval ?**
Coût d'inférence DDPM × horizon = rédhibitoire si on replanifie à chaque step. Tα = 8 (sur H=16) divise le coût par 8 tout en gardant assez de réactivité. C'est aussi prouvé empiriquement comme l'optimum sur PushT (§6.3 du papier).

**Q3 — Pourquoi BC est-elle obligatoire en baseline et pas un autre algo ?**
BC est l'algo IL le plus simple : si elle marche déjà bien, tu n'as pas besoin de Diffusion Policy. Si elle échoue, tu as confirmé que ton problème est *multimodal* — ce qui justifie la diffusion. C'est la borne basse honnête.

**Q4 — Quelle ablation tu fais en priorité si tu n'as le budget que pour une seule ?**
Sans action chunking (H=1). C'est le composant qui contribue le plus au succès de Diffusion Policy vs BC. Si retirer le chunking ne change presque rien, ton archi a un autre problème ; si ça écroule tout, tu as confirmé l'hypothèse du papier.

**Q5 — Comment lire un résultat où Diffusion Policy fait pire que BC ?**
Hypothèses dans l'ordre : (1) bug d'inférence (mauvais schedule de sampling), (2) checkpoint pas EMA (re-essayer avec EMA), (3) chunking mal câblé (Tα > H ou normalization incorrecte), (4) trop peu d'épochs (DDPM converge plus lentement que BC, parfois 5×). Si rien ne marche, ton dataset est probablement *unimodal* (BC est suffisant).

---

## 9. Pour aller plus loin

- §6 du papier Diffusion Policy (REFERENCES.md #19) — résultats détaillés sur 11 environnements, lecture obligatoire pour calibrer tes attentes.
- "A Survey of Imitation Learning" (REFERENCES.md #10) — tableau §5 qui compare BC, BC-RNN, Diffusion Policy, ACT sur les mêmes benchmarks.
- DDIM (Song et al. 2021) pour réduire le nombre de steps de denoising sans réentraîner — levier de latence indispensable en pratique.
- π0-FAST tokenizer (REFERENCES.md #14) — vu en J21, montre comment l'industrie 2025 attaque le problème latence à un autre niveau.
