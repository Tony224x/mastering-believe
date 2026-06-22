# J15 — Diffusion + Flow Matching unifié

> **Pourquoi ce jour ?** Avant d'attaquer Diffusion Policy (J16) et le capstone, il faut maîtriser la mécanique mathématique sous-jacente : comment on apprend à générer des données en partant du bruit. Le tronc commun *score matching → DDPM → flow matching* est aujourd'hui la fondation des SOTA en robotique générative (Diffusion Policy, π0, GR00T N1).

---

## 1. Exemple concret avant tout : générer des points sur une spirale 2D

Imagine qu'on dispose d'un nuage de 10 000 points 2D formant une spirale. On veut un modèle qui, à partir de bruit gaussien pur, **régénère** des points qui ressemblent à cette spirale.

L'idée, contre-intuitive mais géniale (Sohl-Dickstein 2015, Ho 2020) :

1. **Forward** : on prend nos vrais points x_0 et on leur ajoute progressivement du bruit gaussien sur T étapes. Au bout de T = 1000 étapes, x_T est indiscernable d'un bruit gaussien standard N(0, I).
2. **Reverse** : on entraîne un réseau neuronal `eps_theta(x_t, t)` qui, étant donné un point bruité x_t et le timestep t, **prédit le bruit ajouté**. Si on sait débruiter, on sait inverser le processus : on part d'un bruit pur x_T ~ N(0, I), on applique le débruitage T fois, on retombe sur la spirale.

Concrètement, pour la spirale en 2D, on entraîne en quelques minutes sur CPU un MLP de 64 unités x 3 couches qui, après convergence, génère bien des spirales propres. C'est ce qu'on code en 02-code aujourd'hui.

Une fois ce mécanisme acquis, **on remplace 2D par "séquences d'actions robotiques"** et on a Diffusion Policy. Rien de plus.

> **Key takeaway** : un modèle de diffusion = un *denoiser* appris. On ne modélise jamais p(x) directement, on apprend à enlever du bruit pas à pas. La génération est juste l'inversion répétée du débruiteur.

---

## 2. Score matching : l'idée fondatrice

Le **score** d'une distribution p(x) est le gradient du log-densité :

```
s(x) = ∇_x log p(x)
```

Si on connaît s(x) en tout point, on peut faire de la **Langevin dynamics** : partir d'un x aléatoire et itérer

```
x ← x + (alpha/2) * s(x) + sqrt(alpha) * eta,    eta ~ N(0, I)
```

et on converge vers des échantillons de p. Génération sans densité explicite, juste avec le score.

Problème : on ne connaît pas p(x). On a juste des échantillons. Solution **denoising score matching** (Vincent 2011) : on bruite x_0 → x_t = x_0 + sigma*eps, et on montre que le score de la distribution bruitée vaut

```
∇_{x_t} log p_sigma(x_t) ≈ -(x_t - x_0) / sigma^2 = -eps / sigma
```

Donc apprendre le score = apprendre à prédire le bruit eps. C'est le pont avec DDPM.

---

## 3. DDPM (Ho et al., 2020) : la formulation discrète canonique

Le papier *Denoising Diffusion Probabilistic Models* [Ho et al., 2020] popularise la formulation discrète qu'on utilise en 02-code.

**Forward process** (fixé, pas de paramètres à apprendre) :

```
q(x_t | x_{t-1}) = N(x_t ; sqrt(1 - beta_t) * x_{t-1}, beta_t * I)
```

avec une schedule `beta_1 < beta_2 < ... < beta_T` (typiquement linear de 1e-4 à 0.02 sur T=1000, ou cosinus pour des résultats plus stables).

Propriété clé : on peut sauter directement à n'importe quel pas t avec

```
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps,    eps ~ N(0, I)
```

où `alpha_t = 1 - beta_t` et `alpha_bar_t = prod(alpha_s, s=1..t)`.

**Loss simplifiée** (le coup de génie pédagogique de Ho) :

```
L_simple = E_{t, x_0, eps}[ || eps - eps_theta(x_t, t) ||^2 ]
```

Le réseau apprend à prédire le bruit. Trois lignes de PyTorch dans la training loop. C'est tout.

**Reverse sampling** (DDPM ancestral sampler) :

```
x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * eps_theta(x_t, t)) + sigma_t * z
```

avec z ~ N(0, I) pour t > 1, z = 0 pour t = 1.

**Variantes**: DDIM (Song 2021) supprime le bruit stochastique au sampling et permet de générer en 50 steps au lieu de 1000.

---

## 4. Flow matching : la vue déterministe (Lipman et al., 2022)

Flow matching généralise la diffusion en remplaçant la SDE stochastique par une **ODE déterministe**.

L'idée : au lieu d'un processus stochastique entre x_0 (data) et x_T (bruit), on définit un **chemin déterministe** psi_t(x_0, x_1) qui interpole entre x_1 ~ N(0, I) (au temps 0) et x_0 ~ data (au temps 1). Le plus simple : interpolation linéaire.

```
psi_t(x_0, x_1) = (1 - t) * x_1 + t * x_0,    t in [0, 1]
```

On définit un **champ de vecteurs cible** u_t qui pousse les particules le long de ce chemin :

```
u_t(x | x_0, x_1) = x_0 - x_1
```

(c'est constant le long du chemin pour l'interpolation linéaire — d'où la simplicité).

**Loss conditional flow matching** :

```
L_CFM = E_{t, x_0, x_1}[ || v_theta(psi_t(x_0, x_1), t) - (x_0 - x_1) ||^2 ]
```

Le réseau v_theta apprend le champ de vecteurs. Au sampling, on intègre une ODE (Euler ou Runge-Kutta) :

```
dx/dt = v_theta(x, t)
```

avec x(0) ~ N(0, I), on récupère x(1) qui suit la distribution data.

**Avantages concrets** :
- Sampling déterministe → reproductible et rapide (10-50 steps NFE typiquement vs 1000 DDPM).
- Loss aussi simple que DDPM, pas de schedule beta à tuner.
- Adopté par **π0** [Black et al., 2024] pour la tête d'action robotique : 50ms d'inference sur GPU consumer.

---

## 5. Vue unifiée : DDPM est un cas particulier de flow matching

C'est le grand insight pédagogique du cours [Holderrieth & Erives, MIT 6.S184, 2025].

Tout modèle de diffusion / flow matching peut s'écrire comme :

```
x_t = alpha_t * x_0 + sigma_t * x_1,    x_1 ~ N(0, I)
```

avec deux fonctions du temps `alpha_t` et `sigma_t` qui varient continûment :

| Modèle | alpha_t | sigma_t |
|---|---|---|
| **DDPM (VP-SDE)** | sqrt(alpha_bar_t) | sqrt(1 - alpha_bar_t) |
| **Flow matching linéaire** | t | 1 - t |
| **VE-SDE (Song)** | 1 | sigma_max ^ t |

Le réseau apprend toujours soit le bruit (eps), soit le score (s = -eps/sigma_t), soit la vitesse (v = alpha_t' * x_0 + sigma_t' * x_1). Toutes les paramétrisations sont équivalentes par changement de variable.

**Conséquence pratique** : ne pas se laisser intimider par le "diffusion vs flow matching" dans les papers récents. C'est la même famille, juste un choix de schedule. Diffusion Policy utilise DDPM, π0 utilise flow matching linéaire — la différence d'archi est minime.

---

## 6. Classifier-Free Guidance (Ho & Salimans, 2022)

Pour conditionner la génération (texte, image, état du robot), on utilise la **classifier-free guidance** :

1. À l'entraînement, on apprend conjointement un modèle conditionnel `eps_theta(x_t, t, c)` et inconditionnel `eps_theta(x_t, t, null)`. On *drop* la condition c avec proba 10-20%.
2. Au sampling, on calcule

```
eps_guided = (1 + w) * eps_theta(x_t, t, c) - w * eps_theta(x_t, t, null)
```

avec w (guidance scale) typiquement entre 1 et 7. Plus w grand, plus la génération colle à la condition.

En robotique, c est l'observation visuelle + état proprioceptif. CFG augmente la précision du tracking au prix de la diversité — utile en manipulation.

---

## 7. Pourquoi c'est central pour la robotique

Trois raisons que tu valideras les jours suivants :

1. **Multi-modalité** : Diffusion modélise des distributions multi-modales naturellement. Une politique robotique pour "ramasser un objet" peut avoir plusieurs chemins valides (gauche ou droite). MSE / BC moyenne ces modes (action moyenne = dégueulasse). Diffusion préserve la multi-modalité.
2. **Action chunking** : On génère une **séquence** de k=8-16 actions en une passe. Plus stable que prédire pas-à-pas, et compatible avec la receding horizon control.
3. **Conditionnement riche** : CFG permet de conditionner sur observations visuelles (ResNet/ViT), goal-images, ou langage (VLA π0/GR00T).

---

## 8. Schedule cosinus (Nichol & Dhariwal, 2021)

Petit détail technique mais important : la schedule beta linéaire de DDPM ajoute trop de bruit trop vite. La **schedule cosinus** suivante donne de meilleurs résultats sans modifier l'algo :

```
alpha_bar_t = cos((t/T + s) / (1 + s) * pi/2) ^ 2,    s = 0.008
```

Diffusion Policy l'utilise par défaut. À retenir : si ta loss explose ou stagne, change la schedule avant de toucher à l'archi.

---

## 9. Récap visuel — l'arbre de décisions

```
Tu génères quoi ?
├── Distribution unimodale, latence critique → Flow Matching ODE (10 NFE)
├── Distribution multimodale, qualité max → DDPM/DDIM (50-1000 NFE)
├── Conditionnement langage → +CFG (w in [1, 7])
└── Cas robotique typique (Diffusion Policy)
    → DDPM cosinus + UNet 1D + EMA + 100 NFE (DDIM en eval)
```

---

## 10. Quiz spaced-repetition (3 jours / 1 semaine / 1 mois)

**Q1.** Pourquoi prédit-on le bruit eps plutôt que x_0 directement dans DDPM ?
> Parce que la loss MSE sur eps est équivalente (à un facteur près) au denoising score matching, qui a une justification théorique propre. Prédire x_0 marche aussi mais converge moins bien empiriquement (Ho 2020).

**Q2.** Quelle est la différence formelle entre DDPM et flow matching linéaire ?
> Choix des fonctions alpha_t et sigma_t. DDPM : alpha_t = sqrt(alpha_bar_t) avec schedule beta. Flow matching linéaire : alpha_t = t, sigma_t = 1 - t. Sampling DDPM = SDE stochastique, flow matching = ODE déterministe.

**Q3.** Pourquoi la diffusion est-elle particulièrement adaptée à l'imitation learning robotique ?
> Multi-modalité (préserve les modes des démos d'expert qui peuvent être contradictoires), action chunking (séquence en une passe = stable), conditionnement riche par CFG.

**Q4.** Si tu veux générer en moins de 50 steps, qu'utilises-tu ?
> DDIM sampler (deterministic, équivalent en quality avec 50 steps vs 1000) ou flow matching ODE avec un solver Runge-Kutta.

**Q5.** Quel est le rôle de la guidance scale w en CFG ?
> Trade-off précision-diversité. w=0 = inconditionnel. w=1 = conditionnel "vanilla". w>1 = amplifie le conditionnement (plus précis, moins divers). Typique : 1-7 en image, 1-3 en robotique.

---

## Sources

- [Ho et al., 2020] *Denoising Diffusion Probabilistic Models*, arXiv:2006.11239 — formulation DDPM canonique, loss simplifiée, sampling ancestral.
- [Holderrieth & Erives, MIT 6.S184, 2025] *Generative AI with Stochastic Differential Equations* — vue unifiée score matching → flow matching → CFG, notes PDF + vidéos (REFERENCES.md #23).
- [Lipman et al., 2022] *Flow Matching for Generative Modeling*, arXiv:2210.02747 — formulation ODE.
- [Ho & Salimans, 2022] *Classifier-Free Diffusion Guidance*, arXiv:2207.12598 — CFG.
- [Black et al., 2024] *π0: A Vision-Language-Action Flow Model* (REFERENCES.md #14) — application flow matching à la robotique.
