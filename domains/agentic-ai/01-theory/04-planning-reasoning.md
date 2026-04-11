# J4 — Planning & Reasoning : CoT, ToT, ReAct, Plan-and-Execute, Reflexion

> **Temps estime** : 3h | **Prerequis** : J1 (Anatomie d'un agent), J2 (Tool Use), J3 (Memory & State)
> **Objectif** : comprendre comment forcer un LLM a raisonner explicitement, maitriser les patterns de planification (CoT, ToT, ReAct, plan-and-execute, Reflexion), et savoir quand chacun aide vs quand chacun fait perdre du temps.

---

## 1. Pourquoi le raisonnement explicite change tout

Un LLM qui repond directement sans raisonner, c'est un eleve qui ecrit la reponse sans poser le calcul. Pour les questions simples, ca passe. Pour les questions complexes, ca explose.

```
Question : "Si 3 chemises coutent 45€ et que j'ai un budget de 200€, combien puis-je en acheter ?"

Sans raisonnement (direct) :
  LLM: "13"          <-- faux (13 * 15 = 195 OK, mais le LLM a halluciné)

Avec raisonnement (CoT) :
  LLM: "1 chemise = 45/3 = 15€
        200 / 15 = 13.33
        Reponse : 13 chemises"   <-- correct, verifiable
```

> **Analogie** : demander a un LLM de faire des maths complexes sans Chain-of-Thought, c'est comme demander a un humain de multiplier 347 * 892 de tete en 1 seconde. Il faut le **temps de raisonner** — et pour un LLM, le raisonnement est du texte qu'il doit produire.

### Le principe fondamental : raisonnement = tokens

Un LLM ne "pense" pas avant de parler. Chaque token genere utilise un peu de calcul (une forward pass). Donc :

- **Plus le LLM ecrit de tokens intermediaires, plus il peut "reflechir"**
- **Moins il ecrit, moins il peut raisonner**

Forcer le raisonnement, c'est forcer le LLM a ecrire des tokens intermediaires (le raisonnement) avant la reponse finale. C'est compute = raisonnement.

### Gains observes dans la litterature

| Tache | Sans CoT | Avec CoT | Gain |
|-------|----------|----------|------|
| GSM8K (maths primaire) | 18% | 57% | +39 points |
| SVAMP (problemes de mots) | 45% | 73% | +28 points |
| Date understanding | 48% | 68% | +20 points |
| Logique multi-etapes | 30% | 65% | +35 points |

Sur les taches a plusieurs etapes, le raisonnement explicite double presque la performance.

---

## 2. Chain-of-Thought (CoT) — la technique de base

### 2.1 Zero-shot CoT

La technique la plus simple : ajouter **"Let's think step by step"** a la fin du prompt.

```python
# Sans CoT
prompt = "Un train part a 10h a 80 km/h. Un autre part a 11h a 120 km/h. Quand se rejoignent-ils ?"

# Avec CoT zero-shot
prompt = ("Un train part a 10h a 80 km/h. Un autre part a 11h a 120 km/h. "
          "Quand se rejoignent-ils ? Let's think step by step.")
```

Ce simple ajout change le comportement du LLM : il va decomposer le probleme, poser les equations, et arriver a la reponse.

**Pourquoi ca marche** : les LLMs sont entraines sur du texte ou les raisonnements corrects suivent cette structure. En activant le pattern "step by step", on declenche un mode de generation plus methodique.

### 2.2 Few-shot CoT

On fournit des exemples de raisonnements complets dans le prompt :

```python
prompt = """Q: Roger a 5 balles. Il en achete 2 boites de 3. Combien a-t-il de balles ?
A: Roger commence avec 5 balles. 2 boites de 3 balles = 6 balles. 5 + 6 = 11. La reponse est 11.

Q: Il y avait 23 pommes dans la cuisine. On en a utilise 20. On en a rachete 6. Combien reste-t-il ?
A: On commence avec 23. On en utilise 20, il reste 23 - 20 = 3. On en rachete 6, ca fait 3 + 6 = 9. La reponse est 9.

Q: {votre_question}
A:"""
```

**Avantages du few-shot** :
- Le format de raisonnement est explicite et contraint
- On peut guider vers un style particulier (concis, detaille, avec verifications)
- Plus fiable que le zero-shot sur les taches complexes

**Trade-off** : les exemples coutent des tokens (input) a chaque appel.

### 2.3 Quand le CoT AIDE et quand il NE SERT A RIEN

| Type de tache | CoT aide ? | Pourquoi |
|---------------|------------|----------|
| Arithmetique multi-etapes | Oui | Decompose le calcul |
| Logique / deduction | Oui | Force l'enchainement des implications |
| Rewriting / resume | Non | La reponse est en un seul saut |
| Classification simple | Non | Pas de decomposition utile |
| Extraction d'entites | Non | Juste du matching |
| Code generation complexe | Oui | Planifier l'architecture avant d'ecrire |
| Question factuelle | Non | Soit il sait, soit il hallucine |

> **Regle de pouce** : si un humain intelligent aurait besoin de plus de 10 secondes et de gribouiller sur un papier, le CoT aide. Sinon, c'est du token brule pour rien.

---

## 3. Self-Consistency — le vote majoritaire

### 3.1 Principe

Au lieu d'appeler le LLM une fois avec CoT, on l'appelle **N fois** avec `temperature > 0`, on obtient N raisonnements differents, et on prend la reponse **la plus frequente**.

```
Question : "Combien de r dans 'strawberry' ?"

Appel 1 : "s-t-r-a-w-b-e-r-r-y. Je compte les r : 1. Reponse : 1"   [FAUX]
Appel 2 : "s-t-r-a-w-b-e-r-r-y. r, r, r. Reponse : 3"                [CORRECT]
Appel 3 : "strawberry contient 2 r. Reponse : 2"                      [FAUX]
Appel 4 : "s-t-r-a-w-b-e-r-r-y. Je vois r au debut, puis r-r. 3"      [CORRECT]
Appel 5 : "Reponse : 3"                                               [CORRECT]

Vote majoritaire : 3 (apparait 3 fois)
```

### 3.2 Pourquoi ca marche

Les erreurs des LLMs sont **aleatoires et distribuees** : si le LLM se trompe, il se trompe de plusieurs facons differentes. Mais la bonne reponse est **attractive** — plusieurs raisonnements valides convergent vers elle.

**Analogie** : demander a 5 etudiants de resoudre un probleme. Les mauvais se trompent differemment. Les bons convergent.

### 3.3 Cout et gain

- **Cout** : N appels LLM au lieu de 1 (N = 5 typiquement)
- **Gain** : +5 a +15 points de precision sur les taches de raisonnement
- **Quand l'utiliser** : taches critiques ou la precision compte plus que le cout (medical, legal, evaluation)
- **Quand NE PAS l'utiliser** : taches conversationnelles, latence critique, budget serre

---

## 4. Tree-of-Thought (ToT) — explorer plusieurs branches

### 4.1 Le probleme du CoT lineaire

Le CoT trace **un seul chemin** de raisonnement. Si le LLM prend une mauvaise direction a l'etape 2, il est bloque — il ne sait pas revenir en arriere.

```
CoT lineaire :
  Etape 1 → Etape 2 (mauvaise) → Etape 3 (force) → Reponse (fausse)
```

### 4.2 Principe du ToT

A chaque etape, on genere **plusieurs continuations possibles**, on les evalue, et on ne garde que les meilleures. C'est un arbre de raisonnements.

```
ToT :
  Racine
    ├── Branche A1 (score: 0.8)
    │     ├── A1.1 (score: 0.9)    ← meilleure
    │     └── A1.2 (score: 0.3)    ← elaguee
    ├── Branche A2 (score: 0.4)    ← elaguee
    └── Branche A3 (score: 0.7)
          └── A3.1 (score: 0.6)
```

### 4.3 Les 4 etapes du ToT

1. **Thought generation** : demander au LLM de generer K continuations possibles ("donne-moi 3 facons de commencer a resoudre ca")
2. **State evaluation** : faire evaluer chaque continuation par le LLM ("note de 1 a 10 la viabilite de cette approche")
3. **Search strategy** : BFS (explorer en largeur) ou DFS (explorer en profondeur) ou beam search (garder les top-K a chaque niveau)
4. **Pruning** : couper les branches les moins prometteuses pour economiser le compute

### 4.4 Quand ToT est pertinent

| Tache | ToT pertinent ? | Pourquoi |
|-------|-----------------|----------|
| Jeux (echecs, Game of 24) | Tres | L'arbre de decision est naturel |
| Proofs / demonstrations | Tres | Plusieurs approches possibles |
| Creative writing | Oui | Explorer differents styles |
| Code debug | Moyen | Trop de branches possibles |
| QA factuelle | Non | Pas d'exploration necessaire |

**Cout** : 10-50x plus cher qu'un CoT simple. A n'utiliser que quand la qualite prime.

> **Opinion** : ToT est impressionnant sur papier mais rare en production. Le cout est enorme et la plupart des taches reelles sont mieux servies par un plan-and-execute avec self-consistency. Utile pour la recherche, rare dans les SaaS.

---

## 5. ReAct — Reasoning + Acting entrelaces

### 5.1 Le pattern ReAct

ReAct (Reasoning + Acting) entrelace **pensees** et **actions** :

```
Thought: J'ai besoin de savoir la population de Paris pour calculer la densite.
Action: search("population Paris 2024")
Observation: "2,161,000 habitants"
Thought: Maintenant la superficie : 105 km2. Densite = 2161000/105 = 20581/km2
Action: finish("La densite de Paris est ~20,581 hab/km2")
```

C'est le pattern par defaut des agents LLM avec tool use. On l'a vu a J2.

### 5.2 Limites de ReAct

- **Reactif** : pas de plan global, l'agent decide coup par coup
- **Myope** : peut partir dans une mauvaise direction sans voir qu'elle mene nulle part
- **Repetitif** : peut boucler sur les memes actions
- **Inefficace** : re-decide quoi faire a chaque etape, meme si le plan etait evident des le depart

### 5.3 Quand ReAct suffit vs quand il faut plus

| Situation | ReAct suffit | Il faut plus |
|-----------|--------------|--------------|
| Tache courte (< 5 etapes) | Oui | - |
| Tache exploratoire (on ne sait pas a l'avance) | Oui | - |
| Tache avec plan clair (recherche → analyse → synthese) | Non | Plan-and-execute |
| Tache avec sous-objectifs paralleles | Non | Plan-and-execute + parallelisation |
| Tache creative / ouverte | Oui | - |

---

## 6. Plan-and-Execute — le pattern production

### 6.1 L'idee

Separer **planification** et **execution** :

1. **Planner** : un LLM genere un plan complet en une seule passe (liste d'etapes)
2. **Executor** : un autre LLM (ou le meme) execute les etapes une par une, en appelant les outils
3. **Synthesizer** : un LLM final prend les resultats et produit la reponse

```
┌──────────┐     ┌───────────┐     ┌──────────────┐
│ Question │ --> │  Planner  │ --> │ Liste etapes │
└──────────┘     └───────────┘     └──────┬───────┘
                                          │
                                          v
                                   ┌─────────────┐
                                   │  Executor   │ <--> tools
                                   └──────┬──────┘
                                          │
                                          v
                                   ┌─────────────┐
                                   │ Synthesizer │
                                   └─────────────┘
```

### 6.2 Pourquoi c'est plus efficace

- **Moins d'appels LLM** : un seul appel de planning, vs un appel par etape dans ReAct
- **Meilleur plan global** : le planner voit l'ensemble de la tache
- **Parallelisable** : les etapes independantes peuvent s'executer en parallele
- **Debuggable** : le plan est visible et inspectable avant execution

### 6.3 Quand replaner

Si l'execution revele une info nouvelle qui invalide le plan, il faut replaner. Le pattern devient :

```
plan = planner(question)
while not done:
    step = next(plan)
    result = execute(step)
    if result.invalidates_plan:
        plan = replanner(question, history)
    else:
        update_state(result)
```

### 6.4 Comparaison ReAct vs Plan-and-Execute

| Critere | ReAct | Plan-and-Execute |
|---------|-------|-----------------|
| Nb appels LLM | 1 par etape | 1 (plan) + 1 par etape + 1 (synth) |
| Cout (typique 10 etapes) | 10 appels | 12 appels (marginal) |
| Cout en tokens | Eleve (context grandit a chaque etape) | Plus bas (executor a moins de context) |
| Latence | Sequentielle, chaque etape attend | Planning + execution parallelisable |
| Flexibilite | Haute (decide coup par coup) | Moyenne (plan fixe, replan si besoin) |
| Debugging | Difficile (pas de plan visible) | Facile (plan explicite) |
| Best for | Taches exploratoires | Taches structurees |

> **Opinion** : pour 80% des agents en production, **plan-and-execute** est le bon pattern. ReAct reste utile pour les chatbots conversationnels ou pour les taches ou on ne peut pas planifier a l'avance.

---

## 7. Reflexion — l'auto-critique

### 7.1 Le principe

Apres avoir produit une reponse (ou un plan), on demande au LLM de se **critiquer lui-meme** et de **retenter** si la critique est negative.

```
Tentative 1 : "La reponse est 42"
Critique : "Est-ce que ma reponse repond bien a la question ? J'ai oublie de verifier l'unite. Je dois recompter."
Tentative 2 : "La reponse est 42 pommes, pas 42 kg"
Critique : "OK, maintenant c'est correct."
```

### 7.2 Le loop Reflexion

```python
def reflexion_loop(question, max_retries=3):
    attempt = initial_response(question)
    for i in range(max_retries):
        critique = self_critique(question, attempt)
        if critique.is_satisfactory:
            return attempt
        attempt = retry_with_critique(question, attempt, critique)
    return attempt
```

### 7.3 Variantes de Reflexion

| Variante | Critere de critique | Quand l'utiliser |
|----------|--------------------|------------------|
| **Self-critique libre** | "Est-ce que c'est bon ?" | Taches creatives |
| **Checklist** | "Verifie ces 5 points : ..." | Taches structurees |
| **External validator** | Un autre LLM / un linter / un test juge | Code, maths |
| **Execution-based** | Lancer le code, verifier la sortie | Code |

### 7.4 Attention : Reflexion peut empirer les choses

**Risques** :
- Le LLM peut etre trop critique et rejeter une bonne reponse
- Le LLM peut etre trop clement et valider une mauvaise reponse
- Boucle infinie si le critere n'est jamais satisfait
- Cout multiplie par N retries

**Regle** : ne reflexionner que si tu as un critere de critique **fiable**. Un LLM qui critique ses propres reponses sans ancrage est juste un autre LLM qui peut halluciner.

---

## 8. Task decomposition — couper le probleme

### 8.1 Hierarchical decomposition

Une tache complexe se decompose en sous-taches, qui se decomposent en sous-sous-taches, etc.

```
Tache : "Ecris un rapport sur le marche de l'IA en 2026"
├── 1. Recherche des chiffres cles
│   ├── 1.1 Taille du marche global
│   ├── 1.2 Croissance YoY
│   └── 1.3 Segmentation (B2B vs B2C)
├── 2. Analyse des tendances
│   ├── 2.1 Technologies emergentes
│   └── 2.2 Acteurs dominants
└── 3. Redaction
    ├── 3.1 Introduction
    ├── 3.2 Corps
    └── 3.3 Conclusion
```

**Avantage** : chaque sous-tache est plus simple que la tache totale. Un LLM est meilleur sur des taches simples.

### 8.2 Les 3 regles de la decomposition

1. **Chaque sous-tache doit etre independamment verifiable** — sinon on ne peut pas debugger
2. **Chaque sous-tache doit produire un output concret** — pas juste "reflechir a X"
3. **La composition des outputs doit repondre a la tache totale** — sinon on perd l'objectif

### 8.3 Top-down vs bottom-up

| Approche | Principe | Quand |
|----------|----------|-------|
| **Top-down** | Partir de l'objectif, le couper en morceaux | Tache claire, objectif fixe |
| **Bottom-up** | Commencer par les taches atomiques connues, les composer | Exploration, prototypage |

---

## 9. Test-Time Compute & Extended Thinking (2024-2026)

Entre 2022 et 2024, la communaute a optimise l'**entrainement** (scaling de parametres, scaling de donnees). A partir de fin 2024, une nouvelle loi d'echelle emerge : le **scaling au moment de l'inference** — on depense plus de tokens "en reflechissant" avant de repondre, et la qualite monte.

### Chronologie rapide

- **Septembre 2024** : OpenAI sort `o1-preview` / `o1` — le modele depense des "reasoning tokens" internes avant de produire sa reponse. Breakthrough sur AIME, Codeforces, GPQA.
- **Decembre 2024** : OpenAI annonce `o3`, gains encore plus nets sur le reasoning.
- **Janvier 2025** : DeepSeek publie **R1**, un modele open-weight avec extended thinking. Premier open qui rivalise avec o1.
- **2026** : Claude Opus 4.6 propose un **extended thinking** natif, accessible via l'API. Le modele peut "reflechir" avant de parler, avec un budget de thinking configurable.

### Le concept en une phrase

**Test-time compute** = le modele genere N tokens de raisonnement **avant** de produire la reponse. Plus N est grand, meilleure la reponse sur les taches math/code/science. C'est du CoT... mais fine-tune et integre au modele.

### Patterns d'API

**Anthropic — `thinking` block** :

```python
response = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=2048,
    thinking={"type": "enabled", "budget_tokens": 8000},  # 8K tokens de reflexion
    messages=[{"role": "user", "content": "Prouve l'irrationalite de sqrt(2)"}],
)

# La reponse contient deux blocs :
# [{"type": "thinking", "thinking": "... raisonnement internal ..."},
#  {"type": "text",     "text":     "... reponse finale ..."}]
```

**OpenAI — `reasoning_effort`** :

```python
response = client.chat.completions.create(
    model="o3",
    messages=[{"role": "user", "content": "..."}],
    reasoning_effort="high",   # "low" | "medium" | "high"
)
# Les reasoning tokens sont factures mais pas visibles dans la reponse
```

### Cost / benefit

| Metric | Sans extended thinking | Avec extended thinking |
|--------|-----------------------|------------------------|
| Cout par query | 1x | 3-5x (tokens de thinking factures) |
| Latence | 2-5s | 10-60s selon budget |
| Precision math (AIME, MATH) | ~40% | ~80% |
| Precision code (SWE-bench) | ~35% | ~55% |
| Precision classification simple | 92% | 92% (pas de gain) |

Le gain est **+15 a +25 points** sur les taches de raisonnement dur, **nul** sur les taches conversationnelles ou d'extraction.

### Quand utiliser extended thinking

| Utiliser | Ne pas utiliser |
|----------|-----------------|
| Taches NP-hard (SAT, scheduling, optimisation) | Conversationnel, chitchat |
| Planning long-horizon (decomposition multi-etapes) | Classification, extraction d'entites |
| Preuves mathematiques, code complexe | Summarization |
| Decisions critiques a haut enjeu | High-throughput (batch processing) |
| Debugging de bugs non-triviaux | Latence user-facing stricte |

### Impact sur les agents

Les agents modernes **mixent** les modeles selon le role :

```
Agent supervisor :
  - Planning initial : claude-opus-4-6 avec extended thinking (decomposition critique)
  - Execution des steps : claude-haiku-4-5 (rapide, cheap, pas de thinking)
  - Synthese finale : claude-sonnet-4-6 (qualite correcte, cout raisonnable)
```

On paie cher **seulement** la ou le raisonnement est critique (le plan et les decisions hautes), et on passe a un modele cheap pour l'execution mecanique des steps. **Retour d'experience Kalira** : cette strategie divise le cout par 4 sans degrader la qualite du livrable final.

> **Regle 2026** : extended thinking = tool pour les decisions difficiles. Pas un defaut. Active-le dans les phases planning / review critiques, pas dans les phases execution / formattage.

---

## 10. Quand NE PAS utiliser de planning

Le planning n'est pas toujours beneficiaire. Il peut :

- **Augmenter le cout** (plus d'appels LLM)
- **Augmenter la latence** (sequentialite du plan)
- **Introduire des hallucinations** (le LLM invente des etapes inutiles)
- **Masquer l'echec** (le plan semble bon mais l'executor ne peut pas l'appliquer)

**Ne pas utiliser de planning quand** :
- La tache est en 1-2 etapes
- La tache est conversationnelle (pas de but clair)
- Le cout/latence est critique
- L'utilisateur veut un feedback immediat

**Utiliser du planning quand** :
- 5+ etapes sont necessaires
- Les etapes ont des dependances claires
- Le cout d'un echec est eleve (on prefere bien planifier)
- On veut parallelisation

---

## 11. Flash Cards — Test de comprehension

**Q1 : Pourquoi le Chain-of-Thought (CoT) ameliore-t-il les performances d'un LLM ?**
> R : Parce qu'il force le LLM a generer des **tokens intermediaires** qui constituent son raisonnement. Chaque token utilise une forward pass, donc plus il ecrit de raisonnement, plus il utilise de compute pour arriver a la reponse. Sans CoT, le LLM saute directement a la conclusion — sur les taches complexes, ce saut est trop long et il se trompe.

**Q2 : Quelle est la difference entre Self-Consistency et Tree-of-Thought (ToT) ?**
> R : **Self-Consistency** appelle le LLM N fois avec temperature > 0 et fait un vote majoritaire sur la reponse finale. **ToT** construit un arbre de raisonnements : a chaque etape, on genere plusieurs branches, on les evalue, et on elague. ToT explore plus largement mais coute 10-50x plus cher que Self-Consistency.

**Q3 : Quand preferer Plan-and-Execute a ReAct ?**
> R : Quand la tache a **5+ etapes structurees** avec des dependances claires, qu'on veut un **plan global inspectable avant execution**, et qu'on veut potentiellement **paralleliser** les etapes independantes. ReAct reste meilleur pour les taches courtes, exploratoires, ou conversationnelles.

**Q4 : Quels sont les 3 risques de Reflexion (auto-critique) ?**
> R : (1) **Sur-critique** : le LLM rejette une bonne reponse et la degrade en retentant. (2) **Sous-critique** : le LLM valide une mauvaise reponse par complaisance. (3) **Boucle infinie** : si le critere n'est jamais satisfait, on tourne en rond. Reflexion n'est utile que si on a un critere de critique fiable (idealement externe : test, linter, validator).

**Q5 : Donne 3 cas ou le planning NE doit PAS etre utilise.**
> R : (1) Tache courte (1-2 etapes), (2) Tache conversationnelle sans but clair, (3) Quand la latence/cout sont critiques. Le planning ajoute des appels LLM et de la complexite — ca n'est un gain que si la tache est assez complexe pour justifier l'overhead.

---

## Points cles a retenir

- Raisonnement = tokens : pour qu'un LLM "reflechisse", il doit ECRIRE son raisonnement
- CoT zero-shot ("Let's think step by step") : le gain le plus facile pour les taches de raisonnement
- Few-shot CoT : plus fiable mais plus cher en tokens
- Self-Consistency : vote majoritaire sur N CoT avec temperature > 0 — le gain le plus simple pour +5-15 points
- Tree-of-Thought : exploration en arbre, 10-50x plus cher, rare en production
- ReAct : reasoning+acting entrelaces, bon pour les taches courtes et exploratoires
- Plan-and-Execute : separer planning et execution, le pattern dominant en production
- Reflexion : auto-critique en loop, utile SI le critere est fiable (idealement externe)
- Task decomposition : chaque sous-tache doit etre verifiable, concrete, et composable
- Pas de planning pour les taches courtes — overhead inutile
