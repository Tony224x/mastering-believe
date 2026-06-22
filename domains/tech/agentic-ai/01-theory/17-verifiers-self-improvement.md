# J17 — Verifiers & self-improvement : PRM, best-of-N et boucles persistantes

> **Temps estime** : 3h | **Prerequis** : J1-J16
> **Objectif** : comprendre comment des modeles de recompense supervisent le *processus* de raisonnement (et pas seulement le resultat final) ; implementer des strategies de recherche guidees par verifier (best-of-N, beam, weighted majority) ; concevoir des boucles d'auto-amelioration dont les lecons *persistent entre les runs* plutot que d'etre perdues a chaque session.

---

## 1. Pourquoi "verifier" plutot que juste "relancer"

*Note rapide sur J4* : J4 a couvert Reflexion, CoT/ToT, self-critique intra-run, et le test-time compute etendu.
Ce module se concentre sur le **nouveau** : un *verifier* est un modele ou une heuristique qui attribue un *score* a une solution candidate ou a chaque etape d'un raisonnement, et guide la recherche vers les meilleures branches.

**Difference avec J11 (LLM-as-judge en evaluation)** : un juge d'eval mesure la qualite d'un output *apres coup* et produit un rapport ; un verifier est embarque *dans la boucle de generation* pour orienter le choix entre candidates en temps reel.

> **Analogie** : imagine un probleme de maths. Un juge de concours note ta copie a la fin. Un verifier, c'est le correcteur qui lit chaque etape de ton brouillon et te signale *pendant que tu ecris* si la ligne 3 est fausse — ce qui t'evite de perdre 10 lignes a batir sur une erreur.

---

## 2. Outcome Reward vs Process Reward

### 2.1 Outcome Reward Model (ORM)

L'ORM attribue un score **uniquement a la reponse finale**.

```
Probleme → Raisonnement → Reponse → score ORM ∈ [0, 1]
```

Probleme : si le chemin de raisonnement est faux mais que la reponse finale est correcte par hasard (ou inversement), l'ORM envoie un mauvais signal.

### 2.2 Process Reward Model (PRM)

Inspire de *"Let's Verify Step by Step"* (Lightman et al., OpenAI 2023) : le PRM attribue un score a **chaque etape** du raisonnement.

```
Probleme → Etape 1 → score_1
                   → Etape 2 → score_2
                              → ...
                                       → Etape N → score_N
```

Score agregee d'une trajectoire = `min(scores)` ou `prod(scores)` ou `mean(scores)` selon la tache.

**Avantage** : detecte les erreurs silencieuses au milieu d'un raisonnement, avant qu'elles ne propagent.

### 2.3 Comparaison empirique

| Critere | ORM | PRM |
|---|---|---|
| Signal de supervision | resultat final | chaque etape |
| Robustesse aux lucky guesses | faible | elevee |
| Cout de labelling | faible (1 label/solution) | eleve (1 label/etape) |
| Utilite pour la recherche guidee | moderate | tres elevee |

---

## 3. Strategies de recherche guidees par verifier

### 3.1 Best-of-N (BoN)

Genere N solutions candidates, score chacune avec le verifier, retourne la meilleure.

```python
def best_of_n(generator, verifier, problem, n=8):
    candidates = [generator(problem) for _ in range(n)]
    scores = [verifier(problem, c) for c in candidates]
    return candidates[scores.index(max(scores))]
```

Simple, parallele, efficace quand N est grand. Limite : pas de feedback pendant la generation.

### 3.2 Weighted Majority Voting

Au lieu de prendre uniquement le score max, on pondere les votes de chaque candidate par son score PRM.

```python
from collections import defaultdict

def weighted_majority(candidates, scores):
    tallies = defaultdict(float)
    for c, s in zip(candidates, scores):
    tallies[normalize(c)] += s
    return max(tallies, key=tallies.get)
```

Robuste aux outliers : une candidate correcte mais mal scoree ne perd pas si plusieurs autres la confirment.

### 3.3 Beam search guide par PRM

A chaque etape, on garde les K meilleures branches selon le score PRM partiel — comme un beam search NLP mais sur des pas de raisonnement.

```
Etape 0 :  [A, B, C, D, E]          # N candidats
Etape 1 :  [A1, A2, B1, B2, C1]    # expand top-K, prune le reste
Etape 2 :  [A1a, A2a, B1a, ...]    # etc.
```

### 3.4 Lookahead

Pour chaque branche candidate, on simule K etapes de plus et on revient scorer le noeud courant. Plus couteux mais evite les culs-de-sac.

*Pour aller plus loin* : Snell et al. 2024 (voir references) montrent que le bon choix entre BoN, beam et lookahead depend du budget de compute disponible et de la difficulte de la tache.

---

## 4. Self-Refine : boucle generator / critic / refiner

*Self-Refine* (Madaan, Tandon et al. 2023) est une brique importante avant d'aborder la persistance inter-runs.

```
Generator  →  output₀
Critic     →  critique₀  ("L'etape 2 suppose que x>0 sans le verifier")
Refiner    →  output₁  (reecrit en tenant compte de critique₀)
Critic     →  critique₁
...
```

**Trois roles, un seul modele** : le meme LLM joue les trois roles avec des prompts differents. Chaque iteration ameliore la qualite.

**Limite fondamentale** : les lecons apprises dans cette boucle *disparaissent* a la fin du run. Si la meme erreur se reproduit le lendemain, l'agent repart de zero.

> **Analogie** : un chirurgien qui debriefe apres chaque operation mais ne note jamais ses observations. Il progresse dans la session, mais n'a aucun carnet de bord.

---

## 5. Self-improvement persiste entre les runs

### 5.1 Le probleme de l'amnesia inter-runs

Un agent qui boucle en intra-run (self-refine, reflexion) ne retient rien entre deux sessions. Chaque nouveau run recommence avec les memes biais et les memes erreurs.

### 5.2 Architecture du lessons store

Un lessons store est une memoire externe (fichier JSON, base vectorielle, DB) dans laquelle l'agent ecrit des observations apres chaque run reussi ou echoue :

```json
{
  "lessons": [
    {
      "context": "arithmetic_chain",
      "observation": "Le modele oublie de tester x=0. Toujours commencer par les cas limites.",
      "score_before": 0.3,
      "score_after": 0.9,
      "timestamp": "2026-06-15T10:00:00"
    }
  ]
}
```

### 5.3 Injection des lecons dans le prompt

Au debut du run suivant, les lecons pertinentes sont injectees dans le contexte :

```
SYSTEM : Tu es un agent resolveur de problemes.
LECONS PASSEES (3 plus recentes) :
- Cas limites d'abord (x=0, liste vide, negatif)
- Reformule le probleme avant de coder
- Valide chaque etape intermediaire
```

### 5.4 Boucle complete

```
Run N :
  1. Charger lecons precedentes depuis le store
  2. Injecter dans le prompt du generator
  3. Generer + verifier (best-of-N ou beam)
  4. Mesurer le score final
  5. Ecrire les nouvelles lecons dans le store
  6. Sauvegarder le store

Run N+1 :
  Les lecons de Run N sont disponibles → comportement ameliore
```

**Critere d'amelioration mesurable** : compare le score median au run N vs run N-1 sur le meme jeu de problemes.

---

## 6. Exposition theorique : RL et fine-tuning pour agents

*Cette section est intentionnellement conceptuelle* — pas de GPU requis pour la comprendre.

### 6.1 Collecte de trajectoires

Un agent genere des trajectoires `(etat, action, recompense, etat_suivant)`. On accumule un dataset D de trajectoires reussies et echouees.

### 6.2 Supervised Fine-Tuning (SFT) sur les bonnes trajectoires

On filtre D pour ne garder que les trajectoires avec une recompense > seuil, puis on fait du SFT :

```
L_SFT = - Σ log P(action_t | etat_t)   sur les bonnes trajectoires
```

L'agent apprend a imiter ses propres meilleurs comportements.

### 6.3 Rejection Sampling Fine-Tuning (RFT)

Variante : pour chaque probleme, on genere K solutions, on filtre celles ou le verifier dit "correct", on fait du SFT sur ces solutions acceptees. Iterer. Chaque iteration produit un modele un peu meilleur qui genere de meilleures candidates pour la suivante.

### 6.4 Reinforcement Fine-Tuning (RLFT / GRPO / PPO)

On entraine le modele a maximiser une recompense differentiable :

```
L_RL = -E[R(trajectoire)] + β * KL(π_θ || π_ref)
```

Le terme KL evite que le modele derive trop loin du modele de reference.

**En pratique** (GPT-o1, DeepSeek-R1, Gemini Thinking) : la recompense est souvent une combinaison PRM + verifier externe.

### 6.5 Distillation gros → petit agent

Un gros modele (teacher) genere des trajectoires de haute qualite. Un petit modele (student) apprend par SFT sur ces trajectoires. Resultat : un agent compact qui imite le raisonnement du teacher sans ses couts de latence/cost.

```
Teacher (70B) → trajectoires annotees → SFT → Student (7B)
```

---

## 7. Points cles a retenir

| Concept | Essence |
|---|---|
| PRM vs ORM | PRM score chaque etape (robuste) ; ORM score la fin (cheap) |
| Best-of-N | Parallele simple, efficace avec N grand |
| Beam guide par PRM | Exploration sequentielle, economise les appels |
| Self-Refine | Generator / Critic / Refiner en boucle intra-run |
| Lessons store | Persistance inter-runs : les erreurs d'hier forment le comportement de demain |
| SFT / RFT | Fine-tuning sur les bonnes trajectoires que l'agent a lui-meme generees |
| Distillation | Teacher genere, student apprend — compression du raisonnement |

---

## Flash-cards

**Q1 :** Quelle est la difference entre un PRM et un ORM ?
> **R :** Le PRM attribue un score a chaque *etape* du raisonnement ; l'ORM score uniquement la reponse finale. Le PRM detecte les erreurs intermediaires silencieuses que l'ORM manque.

**Q2 :** Pourquoi le weighted majority voting est-il plus robuste que le simple best-of-N ?
> **R :** Le BoN prend la candidate de score maximal — un outlier peut gagner par chance. Le weighted majority vote cumule les scores de toutes les candidates qui convergent vers la meme reponse, rendant le choix statistiquement plus stable.

**Q3 :** Qu'est-ce qui distingue un verifier intra-run (self-refine) d'un lessons store inter-runs ?
> **R :** Self-refine corrige pendant le meme run mais oublie tout a la fin. Un lessons store persiste sur disque et injecte les observations du run N dans le contexte du run N+1.

**Q4 :** Expliquez le rejection sampling fine-tuning (RFT) en une phrase.
> **R :** On genere K solutions, on filtre celles validees par le verifier, et on fait du SFT sur ces bonnes solutions — en iterant, le modele apprend progressivement a produire des solutions qui passent le verifier.

**Q5 :** Quel probleme le terme KL dans la loss RL resout-il ?
> **R :** Il empeche le modele de trop s'eloigner du modele de reference (policy collapse ou mode collapse) : sans ce terme, l'optimisation exploite des failles dans la recompense plutot qu'apprendre un raisonnement generaliste.

---

## Pour aller plus loin

- **Lightman et al. (OpenAI), "Let's Verify Step by Step" (2023)** — papier fondateur des PRM : https://arxiv.org/abs/2305.20050
- **Madaan, Tandon et al., "Self-Refine: Iterative Refinement with Self-Feedback" (2023)** — boucle generator/critic/refiner : https://arxiv.org/abs/2303.17651
- **Snell, Lee, Xu, Kumar, "Scaling LLM Test-Time Compute Optimally..." (2024)** — quand utiliser BoN vs beam vs lookahead selon le budget : https://arxiv.org/abs/2408.03314
- **Shinn et al., "Reflexion" (2023)** — precurseur des boucles avec memoire externe : https://arxiv.org/abs/2303.11366
