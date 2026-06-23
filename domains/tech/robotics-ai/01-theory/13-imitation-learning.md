# J13 — Imitation Learning : BC, DAgger, IRL/GAIL

> **Objectif du jour** : comprendre pourquoi cloner un expert n'est pas un simple problème de regression supervisee, et maitriser les trois familles canoniques d'imitation learning (BC, DAgger, IRL/GAIL) avec leurs trade-offs.
>
> **Pre-requis** : J9 (MDPs, Bellman), J10 (Q-learning), J11 (PPO).
>
> **Sources** : `[Zare et al., 2024, Survey IL]`, `[CS285 L2 — Levine]`, `[CS224R L2 — Finn]`.

---

## 1. Le probleme concret : "j'ai 100 demos, comment cloner sa politique ?"

Imaginons : tu as enregistre **100 trajets d'un humain conduisant** en ville. Pour chaque pas de temps, tu disposes de :

- l'observation `s_t` (image camera + vitesse + GPS),
- l'action `a_t` (volant, accelerateur, frein).

Tu veux entrainer une politique `π_θ(a | s)` qui conduit comme cet humain. Pas de fonction de reward (tu ne sais pas quoi recompenser : "rouler droit" ? "ne pas tuer un pieton" ? trop flou). Mais tu as des demos.

**Premier reflexe naif** : c'est de l'apprentissage supervise. Entraine un reseau a predire `a_t` a partir de `s_t`, comme une regression standard. C'est **Behavior Cloning (BC)**.

```
dataset D = {(s_1, a_1), (s_2, a_2), ..., (s_N, a_N)}  ← demos expert
loss(θ) = E_(s,a)~D [ -log π_θ(a | s) ]   # ou MSE si actions continues
```

C'est simple, rapide, et... **ca casse en production**. Pourquoi ? Parce qu'un trajet de conduite n'est PAS i.i.d. — c'est ici que naissent les trois familles d'IL.

---

## 2. Behavior Cloning (BC) : la baseline et son talon d'Achille

### 2.1 Formulation

BC reduit l'imitation a un probleme d'apprentissage supervise classique :

- **Input** : observation `s`
- **Target** : action expert `a*`
- **Loss** : `L(θ) = E_(s,a*)~D_expert [ ℓ(π_θ(s), a*) ]`
  - `ℓ` = cross-entropy (actions discretes) ou MSE (actions continues)
- **Optim** : SGD/Adam comme n'importe quel reseau de neurones.

Resultat : on obtient `π_BC` qui, sur les `s` du dataset, predit bien les `a*`.

### 2.2 Le probleme central : **distribution shift**

Voici le piege. L'apprentissage supervise suppose `train ≈ test` (meme distribution). En IL, c'est faux :

- **Au train** : on voit `s ~ p_expert(s)` (etats visites par l'expert).
- **Au deploiement** : on voit `s ~ p_πθ(s)` (etats visites par NOTRE politique apprise).

Et `p_πθ ≠ p_expert` des qu'on fait une petite erreur. Pire : les erreurs **se composent** dans le temps.

> **Resultat theorique cle** `[Ross & Bagnell, 2010]`, repris dans `[CS285 L2]` :
>
> Si la politique BC fait une erreur avec probabilite ≤ ε sous la distribution expert, alors le regret cumule sur un episode de longueur T peut atteindre **O(T² ε)** (au lieu du O(T ε) qu'on aurait en supervise i.i.d.).
>
> Erreur quadratique en T = catastrophe sur trajets longs.

### 2.3 Intuition visuelle (route de conduite)

```
Expert (centre route) :        ─────●─────●─────●─────
BC apres petite erreur :            └→●     ?
                                       └─→ jamais vu cet etat
                                          → action aleatoire
                                          → derive aggravee
                                          → crash
```

L'expert ne s'eloigne **jamais** du centre, donc le dataset ne contient **aucun exemple** "comment recuperer quand on derive". Des qu'on derive un peu, on est out-of-distribution, et c'est game over.

### 2.4 Quand BC marche quand meme

- **Beaucoup de donnees** couvrant les etats hors-trajectoire-optimale.
- **Episodes courts** (T petit, l'erreur quadratique reste tolerable).
- **Stochasticite expert** : si l'expert lui-meme bavote autour de l'optimal, le dataset couvre naturellement les etats voisins.
- **Action chunking + diffusion** (J16) : predire des sequences d'actions au lieu d'une seule reduit l'effet compositionnel — c'est pour ca que Diffusion Policy peut faire du BC quasi-pur a SOTA.

> **Encadre** : BC ≠ mort. C'est la baseline. Toutes les architectures modernes (Diffusion Policy, OpenVLA, π0) sont des BC sophistiques. Le hack pour battre le distribution shift est soit data scale + diversity, soit action chunking, soit les deux. `[Zare et al., 2024]`

---

## 3. DAgger : on-policy correction de BC

`[Ross, Gordon, Bagnell, 2011 — DAgger]` resout le distribution shift par une **boucle interactive**.

### 3.1 Algorithme

```
Initialiser π_0 = BC entraine sur D_0 = demos expert
Pour i = 1, 2, ..., N :
    1. Rouler π_i dans l'env  → trajectoires {(s_t, ?)}
    2. Demander a l'EXPERT : "qu'aurais-tu fait sur ces s_t ?"
       → recuperer a*_t pour chaque s_t visite par π_i
    3. D_i = D_{i-1} ∪ {(s_t, a*_t)}_t
    4. π_{i+1} = train BC sur D_i
Retourner π_N
```

L'idee : on ramene la **distribution d'entrainement** vers la **distribution de deploiement**, en ajoutant au dataset des etats visites par notre propre politique, etiquetes par l'expert.

### 3.2 Garantie theorique

DAgger atteint un regret **O(T ε)** au lieu de O(T² ε) pour BC `[Ross et al., 2011]`. Lineaire en T = exploitable sur episodes longs.

### 3.3 Le cout : il faut un expert interrogeable

DAgger suppose qu'on peut demander a l'expert "que ferais-tu ici ?" pour des etats arbitraires. C'est :

- **Facile en simulation** : l'expert est un autre algo (PPO, MPC, oracle), on l'interroge gratuitement.
- **Difficile en reel** : faut un humain disponible pour annoter, ou un teleoperateur — couteux.

C'est pourquoi DAgger est tres utilise en **sim-to-real** (entrainer en simu avec expert MPC, puis fine-tune) et beaucoup moins en robotique reelle pure.

### 3.4 Variantes notables

- **HG-DAgger** `[Kelly et al., 2019]` : l'expert humain ne corrige que quand il pense que la politique deraille (gating).
- **SafeDAgger** : ajoute un classifieur de "doit-on solliciter l'expert ?" pour reduire la charge d'annotation.

---

## 4. Quand BC/DAgger ne suffisent plus : l'envie de comprendre **pourquoi** l'expert agit

BC et DAgger imitent **le comportement** ("fais comme lui"). Ils ne capturent pas **l'intention** ("il fait ca pour atteindre l'objectif X").

Probleme : si l'environnement change (nouveaux obstacles, nouvelle disposition), une politique BC ne sait pas extrapoler. Elle a memorise des `(s, a)`, pas un objectif.

**Inverse Reinforcement Learning (IRL)** retourne le probleme : au lieu d'apprendre la politique, on essaie de **deviner la fonction de reward `R*` que l'expert optimise**. Une fois `R*` recupere, on peut faire du RL classique pour s'adapter a tout nouveau contexte.

---

## 5. IRL : recuperer la fonction de reward

### 5.1 Formulation

Etant donne des trajectoires expert `τ = (s_0, a_0, s_1, a_1, ...)`, trouver `R(s, a)` tel que la politique optimale pour `R` soit (proche de) la politique expert.

> **Probleme** : sous-determine. Beaucoup de rewards differents peuvent expliquer les memes demos. Reward `R = 0` partout rend toute politique optimale par exemple.

### 5.2 Maximum Entropy IRL `[Ziebart et al., 2008]`

Le principe MaxEnt impose une distribution unique : parmi toutes les politiques compatibles avec les demos, choisir celle de **plus grande entropie** (la moins biaisee).

Concretement :

- Probabilite d'une trajectoire : `p(τ) ∝ exp(R(τ))` ou `R(τ) = Σ_t R(s_t, a_t)`
- On parametrise `R_φ(s, a)` (lineaire ou MLP).
- On maximise la log-vraisemblance des demos sous cette distribution.

Difficulte : calculer la partition fonction `Z = ∫ exp(R(τ)) dτ` est intractable pour MDPs continus.

### 5.3 Pipeline IRL classique (couteux)

```
Boucle externe :
    1. Estimer R_φ qui rend les demos vraisemblables
    2. Resoudre le RL "forward" pour trouver π* sous R_φ
    3. Comparer π* aux demos, ajuster R_φ
    4. Repeter
```

Chaque iteration externe necessite un RL complet a l'interieur. Tres couteux. C'est pourquoi GAIL a ete une revolution.

### 5.4 Quand utiliser IRL

- On veut **transferer** le comportement vers un environnement different.
- On veut **expliquer** ce que l'expert optimise (interpretabilite, safety).
- On a un **expert humain non-quantifiable** (ex : preferences ethiques) — IRL plus RLHF.

---

## 6. GAIL : adversarial imitation learning

`[Ho & Ermon, 2016 — GAIL]` court-circuite l'extraction explicite du reward en utilisant un **discriminateur adversarial**, dans l'esprit GAN.

### 6.1 Idee

- **Discriminateur** `D_ψ(s, a)` : classifie si `(s, a)` vient des demos expert ou de la politique apprise.
- **Generateur** = la politique `π_θ` : entraine par RL (ex: PPO) avec **reward** = `-log(1 - D_ψ(s, a))` (donc forte recompense quand le discriminateur est trompe).

```
Boucle :
    1. Roll-out de π_θ → trajectoires "fake"
    2. Train D_ψ pour distinguer fake (π_θ) vs real (expert)
    3. Train π_θ par PPO avec reward issu de D_ψ
    4. Repeter jusqu'a ce que D_ψ ne sache plus distinguer
```

A convergence : la distribution `(s, a)` de `π_θ` matche celle de l'expert.

### 6.2 Pourquoi GAIL bat IRL classique

- Pas besoin de resoudre un RL complet par iteration externe.
- Le discriminateur joue le role de "reward implicite" appris en continu.
- Sample-efficient sur les demos (beaucoup moins que BC pour matcher l'expert sur des envs MuJoCo) `[Ho & Ermon, 2016]`.

### 6.3 Limitations

- Instabilite adversariale (heritage GAN) : tuning delicat.
- Ne resout pas le probleme du distribution shift comme DAgger ; en pratique on combine souvent BC pour pre-entrainer puis GAIL pour raffiner `[Zare et al., 2024]`.
- Inferieur en robustesse aux methodes plus recentes type AIRL (recover reward) ou diffusion-based IL pour la robotique (`[CS224R L2]`).

### 6.4 Famille adjacente

- **AIRL** `[Fu et al., 2018]` : variante GAIL qui recupere une reward function reutilisable.
- **f-IRL**, **VICE**, **Score-based IL** : versions modernes utilisant differentes f-divergences ou des modeles de score.

---

## 7. Quand IL > RL ?

| Critere | RL pur | IL pur (BC/DAgger/GAIL) |
|---|---|---|
| **Reward defini ?** | Oui requis | Non requis (juste demos) |
| **Demos disponibles ?** | Optionnel | Requis |
| **Exploration risquee ?** | Probleme (crash) | Pas d'exploration, juste imite |
| **Sample efficiency** | Faible (millions de steps) | Eleve (centaines de demos) |
| **Generalisation hors demos** | Bonne (si reward dense) | Mauvaise (BC) → IRL aide |
| **Cas typique** | Atari, MuJoCo benchmarks | Conduite autonome, manip robotique, tache humaine |

> **Regle de pouce** `[CS285 L2]` : si tu as un expert et pas de reward exploitable → IL. Si tu as un reward et pas d'expert → RL. Si tu as les deux → **combiner** (pretrain BC, finetune RL ou GAIL).

C'est exactement le pattern dominant en robotique 2024-2026 : **BC sur demos teleoperees + RL en simu pour robustesse + DAgger pour corriger le distribution shift residual**. `[Zare et al., 2024]`

---

## 8. Carte mentale : famille IL

```
                   ┌───────────────────────────┐
                   │  DEMONSTRATIONS expert    │
                   │  D = {(s_i, a_i)}         │
                   └─────────────┬─────────────┘
                                 │
           ┌─────────────────────┼─────────────────────┐
           │                     │                     │
   ┌───────▼─────┐       ┌───────▼────────┐    ┌───────▼─────────┐
   │ BC          │       │ DAgger         │    │ IRL / GAIL      │
   │ supervise   │       │ on-policy      │    │ inferer reward  │
   │ (s,a) → π   │       │ + expert query │    │ ou matcher dist │
   ├─────────────┤       ├────────────────┤    ├─────────────────┤
   │ + simple    │       │ + regret O(Tε) │    │ + generalise    │
   │ + rapide    │       │ + robuste shift│    │ + explainable   │
   │ - O(T²ε)    │       │ - cout expert  │    │ - couteux RL    │
   │ - dist shift│       │ - simu friendly│    │ - instabilite   │
   │             │       │                │    │  (GAIL)         │
   └─────────────┘       └────────────────┘    └─────────────────┘
```

---

## 9. Ce qu'il faut retenir

- **BC** = supervise sur `(s, a)`. Simple. Distribution shift → erreur quadratique en T.
- **DAgger** = boucle on-policy ou l'expert relabel les etats visites par la politique. Regret lineaire en T. Simu-friendly.
- **IRL** = inferer la reward expert. Generalise mieux mais couteux (RL imbrique).
- **GAIL** = adversarial : le discriminateur fait office de reward implicite. Plus efficace que IRL classique.
- **Pattern moderne** : BC pretrain + DAgger ou GAIL ou RL fine-tune. C'est la recette VLA / Diffusion Policy de la litterature 2023-2026 `[Zare et al., 2024]` `[CS224R L2]`.

---

## 10. Flash-cards (spaced repetition)

> **Q1.** Pourquoi BC echoue sur des episodes longs en pratique ?
>
> **A1.** Distribution shift : les etats visites par la politique apprise divergent de la distribution expert, et l'erreur se compose. Borne theorique de regret O(T² ε) au lieu de O(T ε) en supervise i.i.d. `[Ross & Bagnell, 2010]`.

> **Q2.** Quel est le cout principal de DAgger par rapport a BC ?
>
> **A2.** Il faut un expert interrogeable a chaque iteration : pour chaque etat visite par la politique apprise, on demande a l'expert quelle aurait ete son action. Trivial en simu (oracle/MPC), couteux en reel (annotation humaine).

> **Q3.** Quelle est la difference fondamentale entre BC et IRL ?
>
> **A3.** BC apprend directement la politique `π(a|s)` (le **comportement**). IRL apprend la reward `R(s,a)` (l'**intention**) puis derive la politique par RL. IRL generalise mieux a un environnement modifie ; BC est plus rapide.

> **Q4.** Pourquoi GAIL a-t-il remplace IRL classique en pratique ?
>
> **A4.** GAIL evite la boucle externe RL→reward→RL en utilisant un discriminateur adversarial qui joue le role de reward implicite, mis a jour en continu. Bien plus sample-efficient sur les demos.

> **Q5.** Dans quels cas IL est-il prefere a RL ?
>
> **A5.** Quand il existe un expert (humain, controleur, autre policy) mais pas de fonction de reward exploitable, ou que l'exploration RL serait trop risquee/couteuse (conduite, manipulation reelle).

---

## Pour aller plus loin

- `[Zare et al., 2024]` *A Survey of Imitation Learning: Algorithms, Recent Developments, and Challenges*. Survey exhaustif jusqu'a 2024 — lecture canonique. <https://arxiv.org/abs/2309.02473>
- `[CS285 L2]` Levine, *Imitation Learning* — derivation rigoureuse du regret quadratique de BC, motivation DAgger.
- `[CS224R L2 — Finn]` *Imitation Learning* — vue robotique-first, lien avec diffusion policies.
- `[Ross et al., 2011]` *DAgger* — le papier original.
- `[Ho & Ermon, 2016]` *Generative Adversarial Imitation Learning* — GAIL.
