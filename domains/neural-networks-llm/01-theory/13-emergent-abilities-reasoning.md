# Jour 13 — Emergent abilities & reasoning : pourquoi les LLMs "raisonnent"

> **Temps estime** : 5h | **Prerequis** : Jours 1-12

---

## 1. Emergent abilities — la surprise de GPT-3

### Le phenomene observe

En 2022, les chercheurs d'OpenAI, Anthropic et Google ont note quelque chose d'etrange : certaines capacites des LLMs apparaissent **brusquement** a une echelle donnee.

```
Modele 1B : 0% reussite sur arithmetique a 3 chiffres
Modele 10B : 0% reussite
Modele 100B : 50% reussite  ← "emergence"!
Modele 500B : 80% reussite
```

Cette **transition brusque** (aussi appelee "phase transition" ou "grokking") a ete interpretee comme un phenomene emergent : une capacite nouvelle qui n'existe pas en dessous d'une certaine echelle.

### Exemples d'emergent abilities

- **Arithmetique multi-chiffres** : a partir de ~100B params
- **In-context learning** : a partir de ~10B params (appris spontanement)
- **Chain-of-thought reasoning** : surtout utile a partir de ~60B
- **Instruction following** : meilleur a grande echelle (mais pas discret)
- **Translation en langues peu vues** : tres non lineaire
- **Complex word problems** : ameliore fortement a l'echelle

### Le paper fondateur

Wei et al. (2022) "Emergent Abilities of Large Language Models" (Google DeepMind) documente 137 taches BIG-Bench ou l'emergence est observee. Le paper definit :

> "Une capacite est emergente si elle n'est pas presente dans les plus petits modeles mais l'est dans les plus grands. L'emergence ne peut pas etre predite simplement en extrapolant la performance des modeles plus petits."

### Le debat : emergence reelle ou artefact de mesure ?

En 2023, Schaeffer et al. ont conteste : beaucoup de "transitions brusques" sont en fait dues a la **metrique de mesure**. Si on utilise une metrique `all-or-nothing` (exact match), la transition semble brusque. Si on utilise une metrique `continue` (token-level accuracy), l'amelioration est lisse.

```
Exact match   : 0, 0, 0, 0.2, 0.7, 0.9  ← brusque
Token accuracy : 0.3, 0.4, 0.5, 0.6, 0.7, 0.8  ← lisse
```

**Conclusion actuelle** : certaines capacites sont emergentes au sens de "transition non-lineaire dans la metrique d'utilisation", mais lisses en token-level. Le debat n'est pas completement resolu.

### Pourquoi c'est important

Si l'emergence est reelle, alors :
- On ne peut pas predire les capacites d'un modele 10T avant de l'entrainer
- La scaling frontier est inconnue : GPT-5, GPT-6 peuvent debloquer des capacites imprevues
- C'est un argument majeur pour continuer a scaler (safety et capability both)

Si l'emergence est un artefact, alors :
- Le scaling est predictible, rassurant pour les entreprises
- Pas de "jump" magique a attendre — juste de l'amelioration continue
- Les agents AGI sont "juste" une question de volume compute

---

## 2. In-context learning (ICL) — apprendre sans gradient

### L'observation magique

Un LLM pre-entraine peut "apprendre" une nouvelle tache **juste en voyant des exemples dans le prompt**, sans aucun gradient update.

```
Prompt:
"Translate English to French.
 Example 1: cat -> chat
 Example 2: dog -> chien
 Example 3: book -> livre
 Example 4: house ->"

Modele: maison
```

Le modele n'a jamais ete entraine specifiquement sur ce format, mais il infere la tache a partir du contexte.

### Zero-shot, one-shot, few-shot

```
Zero-shot:  "Translate to French: hello"
One-shot:   "English -> French. Example: cat -> chat. Translate: hello ->"
Few-shot:   "... Examples (3-5 total). Translate: hello ->"
```

**Observation** : plus d'exemples donnent de meilleures performances, mais avec des diminishing returns. Quelques exemples (3-5) capturent la plupart du gain.

### Comment ca marche — hypotheses

Plusieurs hypotheses cohabitent :

**1. Implicit Bayesian inference** : le LLM apprend pendant le pretraining de nombreuses "taches" (traduction, resume, Q&A, etc.). Au moment de l'inference, il identifie laquelle correspond aux exemples du prompt et l'applique.

**2. Meta-learning** : le LLM a appris a apprendre pendant le pretraining. Les exemples font office de "gradient implicit" qui configure son comportement.

**3. Attention as induction heads** : Olsson et al. (2022) ont trouve des "induction heads" — des tetes d'attention qui font du pattern matching `(A, B), ..., A -> B`. Ce serait le mecanisme mecanistique derriere l'ICL.

### Quand ICL marche bien vs mal

**Marche bien** :
- Taches avec un format clair et une reponse courte
- Classification a etiquettes fixes
- Traduction entre langues vues
- Pattern completion (par ex. conversions d'unites)

**Marche mal** :
- Taches necessitant du raisonnement multi-etapes
- Generation de texte long et coherent
- Taches avec peu d'exemples similaires dans le pretraining
- Faits rares ou specifiques

---

## 3. Chain-of-thought (CoT) — raisonner etape par etape

### Le probleme sans CoT

Sur un probleme mathematique multi-etapes :
```
Q: Un fermier a 15 moutons. Tous meurent sauf 8. Combien en reste-t-il ?
A: 7  ← faux
```

Le modele donne une reponse immediate, souvent fausse, car il essaie de "deviner" la reponse en une seule etape.

### L'astuce du Chain-of-Thought (Wei et al., 2022)

On incite le modele a **expliciter son raisonnement** avant de donner la reponse :

```
Q: Un fermier a 15 moutons. Tous meurent sauf 8. Combien en reste-t-il ?
Let's think step by step.

A: Le fermier avait 15 moutons au depart. Tous sont morts SAUF 8, ce qui 
   signifie que 8 sont restes en vie. Donc il reste 8 moutons.
Reponse: 8
```

### Pourquoi CoT ameliore les performances

**Hypothese compute per token** : le transformer fait un nombre FIXE de calculs par token. Pour resoudre un probleme complexe, on a besoin de **plus de calculs**. Generer plus de tokens (le raisonnement) donne au modele plus de "budget de calcul".

```
Sans CoT: 1 token de reponse = ~1e12 FLOPs
Avec CoT: 200 tokens de raisonnement = ~2e14 FLOPs

Le modele a 200x plus de compute pour arriver a la reponse.
```

**Hypothese structure** : les problemes complexes se decomposent en sous-problemes. En les ecrivant, le modele peut se concentrer sur une etape a la fois, au lieu d'essayer de tout resoudre dans une seule passe.

### Zero-shot CoT (Kojima et al., 2022)

On peut declencher le CoT sans exemples, juste avec la phrase magique : "Let's think step by step."

```
"Q: {probleme}
A: Let's think step by step."
```

Cela seul ameliore enormement les performances sur les benchmarks mathematiques.

### Emergent chez les gros modeles seulement

CoT ne marche que pour les modeles > ~60B parametres. Pour les petits modeles, ecrire des etapes de raisonnement les fait en fait ECHOUER (ils se perdent dans leur propre "raisonnement" qui est du bruit).

```
LLaMA 7B  sans CoT : 20% sur GSM8K
LLaMA 7B  avec CoT : 18% (legerement pire)
LLaMA 70B sans CoT : 30%
LLaMA 70B avec CoT : 60% ← gros gain
```

C'est un exemple classique d'emergent ability.

---

## 4. Self-consistency — gagner par la majorite

### L'idee (Wang et al., 2022)

Au lieu de faire UN seul appel au modele en CoT, on fait **N appels differents** (avec temperature > 0) et on vote pour la reponse qui revient le plus souvent.

```
Q: Probleme de math difficile
CoT 1: raisonnement 1 -> reponse: 42
CoT 2: raisonnement 2 -> reponse: 42
CoT 3: raisonnement 3 -> reponse: 38
CoT 4: raisonnement 4 -> reponse: 42
CoT 5: raisonnement 5 -> reponse: 42

Vote majoritaire: 42 ← selectionnee
```

### Pourquoi ca marche

Le raisonnement peut se tromper de plein de manieres, mais la bonne reponse est souvent unique. Les erreurs se divisent en beaucoup de reponses rares ; la bonne reponse concentre les "votes".

Gains typiques : +10-15% sur les benchmarks mathematiques par rapport au CoT seul.

### Cout

Il faut N fois plus de tokens a generer. C'est un trade-off latence/qualite. En production, on utilise 5-10 samples typiquement.

---

## 5. Tree-of-thought — explorer l'espace des raisonnements

### Limite de CoT lineaire

CoT genere UN chemin de raisonnement. Si le modele se trompe a l'etape 3, il est bloque.

### Tree-of-Thought (Yao et al., 2023)

Au lieu d'un chemin lineaire, le modele explore un **arbre** de raisonnements :

```
Etat initial
├─ Branche A : "peut-etre la reponse est ..."
│  ├─ Sous-branche A1 : "si c'est X, alors ..."
│  │  └─ evalue: score faible
│  └─ Sous-branche A2 : "si c'est Y, alors ..."
│     └─ evalue: score eleve
└─ Branche B : "sinon, peut-etre ..."
   └─ evalue: score moyen
```

### Les 3 composants

1. **Generator** : le LLM genere plusieurs "thoughts" (etapes de raisonnement)
2. **Evaluator** : le LLM evalue chaque thought (probabilite de conduire a la bonne reponse)
3. **Search** : BFS, DFS, ou beam search a travers l'arbre

### Gains

Sur des taches complexes (Game of 24, Creative Writing), ToT est 10-70% meilleur que CoT simple. Mais beaucoup plus cher (N branches × profondeur).

### ToT en production

Pas tres utilise tel quel — trop cher. Mais les idees sont integrees dans :
- **o1 / o3** : test-time compute scaling via arbres de raisonnement interne
- **Agents** : ReAct, planning avec rollback

---

## 6. Test-time compute scaling — o1, R1 et au-dela

### La nouvelle scaling law (2024-2025)

2024 a revele une nouvelle dimension de scaling : non plus la taille du modele, mais la **compute consommee au moment de l'inference**. Cette decouverte a autant change le paysage que l'introduction du transformer en 2017.

```
Ancien scaling (2020-2023) :
  plus de params + plus de donnees = meilleur modele
  Chinchilla scaling law : 20 tokens par param

Nouveau scaling (2024-2025) :
  meme modele + plus de tokens de "thinking" = meilleure reponse
  o1 scaling law : performance croit lineairement avec log(test-time compute)
```

Les deux dimensions sont **multiplicatives**. On peut obtenir un gain de 10x sur le training, puis encore un 10x sur le test-time compute, pour un gain total de 100x.

### 6.1 o1 architecture (OpenAI, septembre 2024)

OpenAI o1 est le **premier modele commercial** a etre fine-tune specifiquement pour utiliser du test-time compute massif. L'architecture reste un transformer standard, mais l'entrainement et l'inference different radicalement.

**Entrainement (hypothese publique)** :
1. Partir d'un modele de base (GPT-4 class) deja pre-entraine
2. SFT sur des chaines de raisonnement tres longues, avec backtracks ("wait, that's not right, let me try...")
3. **RL avec PPO** (ou variante) sur des taches **verifiables** : math, code, science. Le modele est recompense s'il arrive a la bonne reponse ; comment il y arrive n'importe pas
4. Le modele apprend spontanement a generer des "thinking tokens" — un chain-of-thought internalise et tres long

**Inference** :
- Le modele recoit un prompt, genere N tokens de "thinking" internes (caches a l'utilisateur dans l'API publique), puis emet la reponse finale
- N peut aller de 1000 a 100 000+ tokens pour les problemes difficiles
- Plus N est grand, meilleure la reponse. OpenAI expose ce trade-off via les parametres `reasoning_effort` (low / medium / high)

**Caracteristique cle** : les thinking tokens contiennent des **backtracks explicites** ("let me reconsider", "actually, I think I made an error"), des **verifications** ("let me check this by another method"), et des **decompositions** ("this problem has three parts : ..."). Ces patterns emergeaient rarement dans le CoT classique.

### 6.2 DeepSeek R1 (janvier 2025) — open-source reasoning

DeepSeek R1 est la reponse open-source a o1, avec une recette **completement publique** (paper + weights). Ses contributions sont majeures :

**R1-Zero** : un premier modele entraine directement par RL (GRPO + RLVR) depuis le modele de base, **sans SFT intermediaire**. Le paper DeepSeek-R1 (janvier 2025) montre que le reasoning **emerge naturellement** du RL :
- Le modele apprend de lui-meme a generer des chain-of-thought longs
- Il developpe des "moments eureka" — des passages ou il s'arrete, reconsidere, change d'approche
- Pas besoin de curriculum humain pour lui apprendre a reasoning

**R1 (le modele final)** : ajoute une etape de SFT sur quelques milliers de traces de raisonnement humaines pour ameliorer la lisibilite et reduire les comportements bizarres. Puis RL GRPO+RLVR a grande echelle. Performances equivalentes a o1 sur math, code, science.

**Recette R1 simplifiee** :
```
1. Base model (DeepSeek V3, 236B MoE)
2. SFT court : 800k traces de raisonnement (OK readability)
3. RL avec GRPO :
   - Pour chaque prompt math/code, sampler 16 reponses
   - Reward = 1 si la reponse finale est correcte (verifier deterministe), 0 sinon
   - Normaliser par le groupe, update la policy
4. Repeter jusqu'a saturation
```

Impact : pour la premiere fois, la communaute comprend comment fabriquer du reasoning SOTA. Les forks R1 explosent : Qwen-R1, Llama-R1 community, DeepSeek R1-distill (modeles plus petits distillation).

### 6.3 o3 (OpenAI, decembre 2024) — scaling au maximum

o3 est l'iteration suivante d'o1. La nouveaute n'est pas l'architecture mais l'**echelle du test-time compute**. OpenAI a repousse les limites :
- Plus de thinking tokens par reponse (jusqu'a millions)
- Plus de samples en parallele avec agregation
- Fine-tune plus agressif sur les benchmarks cibles

**Resultats spectaculaires** :
- **ARC-AGI** (benchmark de reasoning pure conçu pour battre les LLMs) : 88% score (human-level ~85%). GPT-4 plafonnait a 5%
- **FrontierMath** (math niveau PhD) : 25% (les autres modeles : 2%)
- **GPQA** (sciences niveau PhD) : 87% (expert humain : 65%)

Mais **couts d'inference** : une question o3 peut couter **plusieurs milliers de dollars** en compute GPU. Le mode "o3 high compute" est reserve a des benchmarks controles.

### 6.4 Process Reward Models vs Outcome Reward Models

Quand on entraine un modele de reasoning avec du RL, quelle reward utiliser ?

**Outcome Reward Model (ORM)** : reward = 1 si la reponse **finale** est correcte, 0 sinon. Simple, deterministe (verifier), mais signal tres sparse : sur un probleme a 20 etapes, une erreur a l'etape 3 donne un reward 0, sans indication de ce qui a mal tourne.

**Process Reward Model (PRM)** : un modele appris qui score **chaque etape intermediaire** du raisonnement. Par exemple, apres l'etape 3 : "cette etape est-elle logiquement correcte ?". Signal beaucoup plus dense.

```
Probleme : "Marie a 15 pommes, donne 3 a Jean, achete 8, en mange 2. Combien?"

Raisonnement :
  Etape 1 : 15 - 3 = 12    PRM score : 0.98 (correct)
  Etape 2 : 12 + 8 = 18    PRM score : 0.99 (correct, mais erreur 20)
  Etape 3 : 20 - 2 = 18    PRM score : 0.05 (incoherent avec etape 2)

Final : 18    ORM score : 0 (la vraie reponse est 18... ou 20?)
```

**Comment entrainer un PRM** : dataset de traces de raisonnement annotees etape par etape par des humains (ou par un verifier formel). **PRM800K** (OpenAI, 2023) est le dataset de reference, 800k etapes de math annotees.

**Avantage PRM** : meilleur signal pour l'apprentissage, gradient plus dense, convergence plus rapide. Utilise dans o1 (hypothese), dans Let's Verify Step by Step (OpenAI, 2023).

**Desavantage PRM** : couteux a collecter (annotation humaine), risque de biais. RLVR (verifiable rewards) est plus scalable quand on a un verifier deterministe.

**Etat 2025** : combiner ORM (pour le signal final) + PRM (pour la credit assignment intermediaire) est la recette la plus efficace.

### 6.5 MCTS pour le reasoning

Monte Carlo Tree Search (MCTS) est un algorithme de recherche utilise dans AlphaGo, AlphaZero. En 2024-2025, il est devenu un outil standard pour le reasoning LLM.

**Principe** : au lieu de generer un seul chain-of-thought, on construit un arbre de raisonnements possibles et on explore intelligemment les branches prometteuses.

```
Root : probleme initial
├─ Thought A (PRM score 0.8) <- explorer plus
│  ├─ Thought A.1 (0.9) <- tres prometteur, expand
│  │  └─ ...
│  └─ Thought A.2 (0.6)
├─ Thought B (0.5)
└─ Thought C (0.3) <- abandonner
```

**4 phases MCTS** :
1. **Selection** : descendre dans l'arbre en choisissant les branches a score eleve (UCB formula)
2. **Expansion** : a une feuille, generer k nouveaux thoughts via le LLM
3. **Simulation** : faire un rollout (raisonnement jusqu'a la fin) et evaluer la reponse avec un verifier ou un PRM
4. **Backpropagation** : remonter le score dans l'arbre

**Utilise par** :
- **o1** (hypothese, non confirme publiquement) : le test-time compute inclut probablement des rollouts paralleles
- **DeepSeek R1** : le paper evoque des experiments MCTS mais le modele final utilise du RL pur GRPO
- **rStar-Math** (Microsoft, 2024) : un petit modele 7B qui atteint GPT-4 level sur math grace a MCTS + PRM
- **AlphaCode 2** (DeepMind, 2023) : MCTS + generation pour competitive programming

**Trade-off** : MCTS est beaucoup plus cher que CoT lineaire (k rollouts par step), mais peut debloquer des problemes impossibles en une seule passe.

### 6.6 Courbes de scaling du test-time compute

L'insight fondamental de 2024 : **la performance scale lineairement avec le log du test-time compute**. OpenAI et DeepMind ont publie des courbes similaires :

```
Score sur MATH vs. tokens de thinking (echelle log) :

o1 :
  10 tokens    -> 20%
  100 tokens   -> 45%
  1000 tokens  -> 70%
  10000 tokens -> 85%
  100000 tokens -> 92%

Pente : environ 15% par decade de compute
```

**Implication 1 — cost** : doubler la qualite coute 10x plus de compute. Une reponse o1 coute ~100x plus qu'une reponse GPT-4o classique. Une reponse o3 high-compute coute 1000x+.

**Implication 2 — strategy** : on ne veut PAS utiliser le reasoning pour toutes les taches. Les taches simples (traduction, resume, chat) n'ont pas besoin de thinking. Le pattern 2025 est :
- **Router** : classifier la difficulte du prompt
- **Easy** -> modele rapide (GPT-4o, Claude 3.5 Haiku)
- **Hard** -> modele reasoning (o1, R1)

**Implication 3 — democratisation** : contrairement au training compute (limite par le capex, quelques entreprises seulement), le test-time compute est accessible a **tous**. Avec DeepSeek R1 open-source, n'importe qui peut acheter du reasoning en API ou l'heberger.

### 6.7 Les limites du test-time compute

- **Cost** : ~100x GPT-4 pour o1, ~1000x pour o3 high compute. Economiquement viable uniquement pour des taches a haute valeur
- **Latence** : attendre 30 secondes a plusieurs minutes pour une reponse. Mauvaise UX pour le chat temps reel
- **Pas toujours mieux** : sur les taches creatives (writing), o1 est souvent **pire** que GPT-4o (le reasoning long pollue le style)
- **Difficile a controler** : quand le modele doit-il s'arreter de penser ? Heuristiques ad hoc
- **Hallucinations dans le reasoning** : le modele peut convaincre lui-meme d'une mauvaise voie pendant 10000 tokens de raisonnement. Le verifier final sauve parfois, parfois non

### 6.8 Qu'est-ce que ca change pour 2026 ?

Le paysage LLM 2026 est structure par cette bifurcation :

**Modeles "rapides"** (Claude 4 Sonnet, GPT-5-mini, Qwen 3) : pour le chat, l'ecriture, la traduction, le code simple. Latence basse, cout bas.

**Modeles "reasoning"** (o3, Claude 4 Opus extended, DeepSeek R1, Gemini 2 Thinking) : pour math, code complexe, analyse strategique, recherche scientifique. Latence et cout eleves mais performances exceptionnelles.

**Meta-modeles** (GPT-5 rumored) : un seul modele qui decide lui-meme combien de thinking allouer par prompt. "Thinking effort adaptatif". C'est le graal.

---

## 7. Flash Cards — Active Recall

### Q1 : Qu'est-ce qu'une emergent ability et pourquoi c'est controverse ?

<details>
<summary>Reponse</summary>

**Definition (Wei et al., 2022)** : une capacite est emergente si elle n'est pas presente dans les petits modeles mais l'est dans les grands, avec une transition **brusque** plutot que progressive.

**Exemples** : arithmetique multi-chiffres, in-context learning, chain-of-thought reasoning, traduction en langues peu vues.

**Controverse (Schaeffer et al., 2023)** : beaucoup de "transitions brusques" sont des artefacts de la metrique. Si on utilise une metrique `all-or-nothing` (exact match), ca a l'air emergent. Avec une metrique continue (token-level), c'est lisse.

**Etat actuel** : les capacites au sens utilisation (reponse correcte complete) sont parfois emergentes, mais le progres sous-jacent est souvent lisse. Le debat n'est pas resolu.

**Pourquoi c'est important** : si c'est reel, on ne peut pas predire les capacites des prochains modeles. Si c'est un artefact, le scaling est predictible.

</details>

### Q2 : Qu'est-ce que l'in-context learning et comment ca marche (hypotheses) ?

<details>
<summary>Reponse</summary>

**In-context learning (ICL)** : un LLM pre-entraine peut "apprendre" une nouvelle tache **juste en voyant des exemples dans le prompt**, sans aucun gradient update.

```
Prompt: "cat -> chat, dog -> chien, book -> livre, house ->"
Modele: "maison"
```

**Hypotheses** :
1. **Implicit Bayesian inference** : pendant le pretraining, le LLM a vu plein de "taches". Les exemples du prompt lui indiquent laquelle appliquer.
2. **Meta-learning** : le LLM a appris a apprendre. Les exemples configurent implicitement son comportement (sorte de "gradient implicit").
3. **Induction heads (Olsson et al., 2022)** : il existe dans le modele des tetes d'attention qui font du pattern matching `(A, B), ..., A -> B`.

**Quand ca marche bien** : taches avec format clair, classification, traduction, pattern completion.
**Quand ca marche mal** : raisonnement multi-etapes, taches rares, faits specifiques.

</details>

### Q3 : Pourquoi le Chain-of-Thought ameliore-t-il les performances sur les taches de raisonnement ?

<details>
<summary>Reponse</summary>

**Deux hypotheses principales** :

1. **Compute per token** : un transformer fait un nombre FIXE de calculs par token. Pour resoudre un probleme complexe, le modele a besoin de plus de compute. Generer 200 tokens de raisonnement = 200x plus de compute que generer la reponse directement.

2. **Structuration du probleme** : les problemes complexes se decomposent en sous-problemes. En les ecrivant explicitement, le modele se concentre sur une etape a la fois au lieu d'essayer de tout resoudre en une passe.

**Exemple** :
```
Q: Un fermier a 15 moutons, tous meurent sauf 8. Combien en reste-t-il ?
Sans CoT: 7 (faux)
Avec CoT: "Tous sauf 8 = 8 survivent. Reponse: 8" (correct)
```

**Emergent** : CoT ne marche que pour les gros modeles (~60B+). Pour les petits, le "raisonnement" est du bruit qui fait baisser la performance.

**Variante zero-shot** : "Let's think step by step" declenche le CoT sans avoir besoin d'exemples.

</details>

### Q4 : Explique self-consistency et tree-of-thought.

<details>
<summary>Reponse</summary>

**Self-consistency (Wang et al., 2022)** :
- Au lieu d'un seul CoT, generer N raisonnements differents (temperature > 0)
- Voter pour la reponse qui apparait le plus souvent
- Gain : +10-15% sur les benchmarks mathematiques
- Intuition : les erreurs se divisent, la bonne reponse concentre les votes

**Tree-of-Thought (Yao et al., 2023)** :
- Au lieu d'un chemin lineaire, explorer un arbre de raisonnements
- **Generator** : propose plusieurs thoughts a chaque etape
- **Evaluator** : score chaque thought (LLM qui juge sa qualite)
- **Search** : BFS/DFS/beam a travers l'arbre
- Gain : 10-70% sur Game of 24, creative writing
- Cout : beaucoup plus cher (N branches × profondeur)

**Comparaison** :
- Self-consistency : parallele, simple, moins cher
- Tree-of-thought : peut backtracker, plus puissant, plus cher
- En production, self-consistency est plus utilise car ratio cout/benefice meilleur

</details>

### Q5 : Qu'est-ce que le test-time compute scaling (o1, R1, o3) ?

<details>
<summary>Reponse</summary>

**Nouvelle scaling law (2024-2025)** : au lieu de scaler la taille du modele ou le training compute, on scale la **compute consommee au moment de l'inference**. La performance scale lineairement avec le **log** du test-time compute.

**Comment o1 fonctionne** :
- Genere un long "chain of thought interne" cache a l'utilisateur (thinking tokens)
- Peut faire des backtracks, verifications, "let me try differently"
- Fine-tune avec RL + RLVR (verifier deterministe sur math/code) pour bien utiliser ce temps de pensee

**DeepSeek R1 (jan 2025)** : equivalent open-source, utilise GRPO + RLVR. Contribution majeure : **R1-Zero** montre que le reasoning **emerge du RL pur sans SFT intermediaire** — le modele apprend seul a generer des CoT longs avec "moments eureka".

**o3 (dec 2024)** : scaling maximum du test-time compute. Scores remarquables :
- ARC-AGI : 88% (human-level, GPT-4 plafonnait a 5%)
- FrontierMath : 25% (les autres modeles : 2%)
- GPQA : 87% (expert humain : 65%)

Cout : une reponse o3 high-compute peut couter **plusieurs milliers de dollars** en GPU.

**Implications 2026** :
- Les modeles se scindent en 2 : rapides (Claude Sonnet, GPT-5-mini) vs reasoning (o3, R1, Gemini Thinking)
- Strategy : router le prompt selon la difficulte (easy -> rapide, hard -> reasoning)
- PRM (Process Reward Models) scorant chaque etape intermediaire + ORM (Outcome) pour le signal final
- MCTS parfois combine avec RL pour explorer l'espace des raisonnements

**Relation avec CoT** : c'est du CoT systematique et massif, internalise, combine avec verification et parfois search. Recette R1 : base model + SFT court + GRPO + RLVR a grande echelle.

</details>


---

## Pour aller plus loin

Lectures couvrant ce sujet (playlists dans [`shared/external-courses.md`](../../../shared/external-courses.md)) :

- **Berkeley CS294-280 — Lec. 11 (Learning to Reason — Weston), Lec. 12 (Inference-Time Techniques — Chen)** — vision research la plus actuelle sur reasoning.
- **Berkeley CS294-196 (Fa24) — Lec. 12 (LLM Reasoning — Denny Zhou)** — CoT et reasoning par le lead Google DeepMind.
- **CME295 — Lec. 6 (LLM Reasoning)** — synthese pedagogique des techniques de raisonnement.
