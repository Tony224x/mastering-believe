# Projet 03 — Mini Diffusion Policy sur poste de tri 2D

## Contexte metier

Au poste de tri FleetSim, un petit robot deflecteur transporte chaque colis depuis la ligne
d'arrivee (en bas) jusqu'a la zone d'expedition (en haut), en contournant le **pilier
central** du poste. Les operateurs experts contournent indifferemment **par la gauche ou
par la droite** — les deux sont optimaux. L'equipe AutonomyAI a entraine un behavior
cloning (BC) sur 300 demonstrations... et le robot fonce droit dans le pilier : la
regression MSE moyenne les deux modes et produit le geste qu'AUCUN expert n'a jamais fait.

C'est le probleme central de l'imitation multimodale (cf. theorie J13 et J16) : la
solution moderne est une **diffusion policy** (Chi 2023) qui apprend a debruiter des
sequences d'actions au lieu de regresser leur moyenne.

## Objectif technique

Implementer en numpy pur (zero torch — c'est le point du projet) :
1. Un **generateur de demonstrations** multimodales : trajectoires expertes qui contournent
   le pilier par la gauche ou la droite (50/50), decoupees en paires
   `(obs (2,), action_chunk (H=8, 2))` — le chunking vient de Diffusion Policy.
2. Une **baseline BC** par regression a noyau (Nadaraya-Watson) : exactement la moyenne
   conditionnelle `E[chunk | obs]` vers laquelle converge un MLP entraine en MSE.
3. Un **DDPM conditionne sur l'observation** dans l'espace des chunks (16-dim) :
   schedule cosine, sampling ancestral K=50 pas, et le **debruiteur optimal en forme
   fermee** `E[a0 | ak, obs]` calcule directement sur le dataset (pas de reseau —
   on isole la mecanique de diffusion ; la version reseau UNet, c'est le capstone J24-J28).
4. Un **controleur receding horizon** : echantillonner un chunk, executer les 4 premieres
   actions, re-planifier — et comparer BC vs diffusion sur le taux de reussite.

## Consigne

```python
def make_demos(n_demos: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """-> (obs (N, 2), chunks (N, 16)) : paires extraites des demos expertes."""

def bc_policy(obs: np.ndarray, dataset) -> np.ndarray:
    """Moyenne conditionnelle des chunks ponderee par noyau gaussien sur l'obs."""

def denoise(a_k: np.ndarray, k: int, obs: np.ndarray, dataset) -> np.ndarray:
    """E[a0 | ak, obs] exact sous la distribution empirique du dataset :
    moyenne des chunks experts ponderee par N(ak; sqrt(ab_k)*a0_i, (1-ab_k)I) * K(obs, obs_i)."""

def ddpm_sample(obs: np.ndarray, dataset, rng) -> np.ndarray:
    """Sampling ancestral DDPM K=50 pas, conditionne sur obs. -> chunk (16,)."""

def rollout(policy_fn, env, rng) -> dict:
    """Receding horizon : execute 4 actions du chunk, replanifie. -> success, side, trace."""
```

Contraintes :
- Les poids du debruiteur se calculent en **log-espace** (logsumexp) — les vraisemblances
  gaussiennes a k petit font underflow en espace direct.
- Les chunks sont **normalises** (mean/std du dataset) avant diffusion, denormalises apres —
  comme dans le vrai Diffusion Policy (cf. J24, normalisation des actions).
- Le noyau d'observation se calcule **une fois par replanification** (l'obs ne change pas
  pendant les 50 pas de debruitage) — sinon le script depasse la minute.
- Schedule cosine (Nichol & Dhariwal) ; bruit final nul (`z = 0` quand `k = 0`).
- Deterministe a seed fixe ; runtime total < 60 s CPU.

## Etapes guidees

1. **Env + demos** — pilier disque centre `(0, 0.5)` rayon `0.15`. Demos = courbes de
   Bezier quadratiques start -> point de controle lateral (`x = ±0.55`) -> goal `(0, 1)`,
   30 pas, bruit gaussien leger. Verifie qu'aucune demo ne touche le pilier.
2. **Paires (obs, chunk)** — fenetre glissante : a chaque index `t`, obs = position courante,
   chunk = les 8 deltas suivants aplatis (16,). Padding par repetition du dernier delta
   en fin de trajectoire.
3. **BC** — `w_i = exp(-||obs - obs_i||^2 / (2 h^2))`, chunk predit = moyenne ponderee.
   Trace le chunk predit a l'obs de depart : il part tout droit vers le pilier. Mesure-le.
4. **Forward process** — schedule cosine `ab_k`, `a_k = sqrt(ab_k) a_0 + sqrt(1-ab_k) eps`.
   Sanity check : a `k=0`, `a_k ≈ a_0` ; a `k=K`, `a_k ~ N(0, I)` (sur chunks normalises).
5. **Debruiteur ferme** — c'est l'etape cle. La posterior `p(a0_i | a_k, obs)` sur un
   dataset fini est un softmax de log-vraisemblances : `log w_i = -||a_k - sqrt(ab_k) a0_i||^2
   / (2 (1-ab_k)) + log K(obs, obs_i)`. Implemente avec logsumexp, retourne `sum_i w_i a0_i`.
6. **Sampling ancestral** — depuis `a_K ~ N(0,I)` : `eps_hat = (a_k - sqrt(ab_k) a0_hat) /
   sqrt(1-ab_k)`, puis le pas DDPM standard. Echantillonne 100 chunks a l'obs de depart :
   tu dois voir DEUX faisceaux (gauche/droite), jamais le milieu.
7. **Receding horizon + eval** — 20 rollouts par policy. BC percute le pilier, la
   diffusion choisit un cote (pas toujours le meme) et passe.

## Criteres de reussite

- Dataset : 300 demos, ratio gauche/droite dans `[0.40, 0.60]`, **zero** demo expert
  en collision avec le pilier.
- Mode averaging mesure : a l'obs de depart, le chunk BC a un deplacement lateral
  `|dx| < 0.05` alors que le deplacement lateral moyen des chunks experts au depart
  est `> 0.15` — BC produit un geste qui n'existe pas dans les donnees.
- Multimodalite de la diffusion : sur 100 chunks echantillonnes a l'obs de depart, la
  fraction partant a gauche est dans `[0.25, 0.75]`, et **aucun** chunk n'est "moyen"
  (tous ont `|dx| > 0.05`).
- Success rate sur 20 rollouts : diffusion `>= 0.90`, BC `<= 0.20` (il percute le pilier).
- Les rollouts diffusion reussis utilisent **les deux cotes** (au moins 3 a gauche et
  3 a droite sur 20).
- Deux executions consecutives donnent exactement les memes metriques (seed fixe),
  et le script complet tourne en `< 60 s` CPU.

## Solution

Voir `solution/diffusion_sorting.py` — correction commentee, executable telle quelle
(`python solution/diffusion_sorting.py`, < 60 s, numpy seul). Les criteres ci-dessus y
sont verifies par des assertions.

## Pour aller plus loin

- **Debruiteur appris** — remplace le debruiteur ferme par un petit MLP entraine en
  numpy (backprop manuelle) ou passe au capstone J24-J28 (UNet 1D + torch sur PushT)
- **DDIM** — implemente le sampler deterministe et mesure le speedup a qualite egale
  (cf. exercice hard J16)
- **Guidance** — ajoute une condition discrete "side hint" (ordre OCC : "libere le
  couloir gauche") et observe le controle du mode via le conditionnement
- **Horizon** — fais varier `T_a` (actions executees par replanification) de 1 a 8 et
  mesure l'effet sur le success rate : tu retrouveras l'ablation du paper Chi 2023
