# Jour 20 — Distillation, donnees synthetiques & petits modeles specialises

> **Temps estime** : 4h | **Prerequis** : J10 (fine-tuning, SFT, RLHF), J15 (reasoning)

---

## 1. La renaissance des SLMs (Small Language Models)

En 2024-2026, la strategie industrielle gagnante pour 80% des cas produits n'est **pas** "envoyer tout a GPT-5 / Claude 4.5 Opus". C'est :

> Utiliser un gros modele (frontier) pour **generer des donnees** ou **etiqueter des traces**, puis **distiller** dans un petit modele (1B-14B) specialise pour la tache. Deployer le petit modele en prod.

Exemples concrets en production en 2026 :
- **Gemma 3 (1B-27B)** — Google, axe edge/serveur leger
- **Phi-4 / Phi-4-mini** — Microsoft, champion ratio qualite/taille
- **Llama 3.3 8B / Llama 4 Scout** — deployable sur un GPU unique
- **Qwen 3 4B / Qwen 3 14B-thinking** — excellent open source, reasoning distille
- **Claude Haiku 4.5** — distillation maison d'Anthropic pour leur API
- **DeepSeek R1 Distill (7B/14B/32B)** — distillation des traces R1 dans des bases Qwen/Llama

Le ratio qualite/taille a **triple** entre 2023 (Mistral 7B) et 2025-2026. Un Qwen3-4B 2026 ≈ GPT-3.5 2023.

---

## 2. Pourquoi un petit specialise bat un gros generaliste

### Les 4 arguments economiques

1. **Cout d'inference** : 10-50x moins cher par token
2. **Latence** : 5-20x plus rapide (TTFT + tokens/s)
3. **Souverainete** : deployable on-prem, en UE, sur hardware standard
4. **Customisation** : fine-tunable en continu sur des donnees proprietaires

### Les 2 arguments techniques

5. **Specialisation** : un modele 7B fine-tune sur ton domaine bat un 70B generaliste sur CE domaine. Un modele 400B est force d'allouer des parametres aux dinosaures et a la poesie chinoise ; un modele dedie n'en a pas besoin.
6. **Alignement local** : fine-tune pour ton format de sortie, ton ton, tes guardrails metier.

### Les limites

- **Reasoning complexe** : un 7B distille atteint ~60-70% d'un reasoning model complet sur AIME/MATH. Si ton produit depend de math/code avances, un SLM ne sera pas suffisant.
- **OOD (out-of-domain)** : un modele specialise peut "casser" sur des queries hors scope. Un routeur a l'entree limite les degats.
- **World knowledge** : un 3B ne peut pas memoriser tout Wikipedia. Pour les questions factuelles, ajouter du RAG (J17).

---

## 3. Les trois types de distillation

### Type 1 — Distillation de logits (classique, Hinton 2015)

```
teacher(x) → logits_teacher (distribution sur le vocab)
student(x) → logits_student

loss = KL_divergence(softmax(logits_teacher / T), softmax(logits_student / T))
```

Le student apprend la **distribution** de sortie du teacher, pas juste la classe gagnante. La "dark knowledge" est dans les probabilites relatives des mauvaises reponses.

Necessite : acces aux logits du teacher. Donc open-source seulement (Llama 4, Qwen, DeepSeek). Impossible avec GPT/Claude API sauf via log_probs partiels.

### Type 2 — Distillation de sequences / SFT sur outputs

```
Pour chaque prompt :
  teacher(prompt) → reponse complete
  dataset += (prompt, reponse_teacher)

Puis : student fine-tune par SFT sur ce dataset
```

Pas besoin d'acces aux logits. Marche avec les API fermees. **C'est LA methode utilisee en prod** en 2026. On appelle ca **"distillation par donnees synthetiques"**.

DeepSeek R1 Distill fait exactement ca : generer des traces reasoning avec R1 (ouvert), puis SFT un Qwen/Llama 7B-70B dessus.

### Type 3 — Distillation RL / preferences (DPO-style)

Le teacher evalue les sorties du student et donne un signal de preference. Plus proche de RLAIF (RL with AI Feedback).

```
student genere K completions
teacher (ou reward model) range les completions
student update via DPO sur ces paires
```

Utile pour aligner le student sur un juge (teacher) qui n'expose pas ses logits.

---

## 4. Le pipeline de donnees synthetiques (le vrai cœur du sujet)

En 2026, le bottleneck n'est pas l'architecture des modeles, c'est **les donnees**. Le playbook :

### Etape A — Seed diversity

Partir de quelques centaines de seeds humains (prompts varies, domaines divers). Plus les seeds sont diverses, plus le dataset synthetique l'est.

### Etape B — Generation avec un frontier model

Pour chaque seed :
- Generer N variantes (paraphrase du prompt)
- Demander au frontier model de repondre avec reasoning complet
- Varier les roles, les styles, les niveaux de difficulte

Stack 2026 typique : 1 seed → 50-200 synthetic examples.

### Etape C — Filtrage (le plus important)

80% de la qualite se joue ici. Filtres :

1. **Syntactic** : parsable (JSON valide), longueur, pas de repetitions
2. **Rule-based** : reponse presente, format respecte
3. **Verifiable** : pour math/code, tester la reponse. Si faux, jeter.
4. **LLM judge** : un autre LLM evalue "cette reponse est-elle bonne/aligned/utile ?"
5. **Near-duplicate detection** : MinHash / embedding similarity pour diversifier
6. **Toxicity / safety** : classifieur dedie

Le rendement typique : 30-60% des generations passent les filtres.

### Etape D — Task mix

Balancer les taches (classification vs generation vs reasoning) pour eviter l'overfit sur un type. Ratio typique : 40% task-specific, 40% general chat, 20% safety/refusals.

### Etape E — Fine-tune par SFT puis optional DPO

SFT classique puis, si tu as des prefs pairees (A meilleur que B), DPO pour aligner.

### Etape F — Eval et iteration

Toujours tenir un eval set **hors** du pipeline synthetic. Si tes evals montent mais les evals externes (MMLU, MT-Bench, ton domaine) stagnent, c'est un signal d'overfit sur le synthetic.

---

## 5. Techniques speciales de distillation

### Distillation "teacher-forcing" vs "on-policy"

- **Teacher-forcing** : on donne la reponse du teacher token par token au student pour calculer sa loss.
- **On-policy** : le student genere, le teacher re-scorer chaque token.

On-policy + RL est ce qu'utilise Apple en 2025-2026 ("MiniLM-R1"). Plus cher, mais resout l'exposure bias (le student ne voit jamais ses propres erreurs sinon).

### Intermediate representations matching

Au-dela des logits finaux, faire matcher les activations des couches intermediaires (hidden states, attention maps). MobileBERT (2020) a popularise ca. Utile si le student a une architecture proche du teacher.

### Distillation pour reasoning models

Le truc trouve en 2025 : distiller les **traces completes** (`<think>...</think>` + reponse finale). Le student herite de la capacite reasoning via l'imitation des tokens de reflexion. Ca marche etonnamment bien meme a 7B.

DeepSeek R1 Distill 14B bat DeepSeek V3 (236B non-reasoning) sur AIME 2024. C'est la puissance du test-time compute + distillation.

### Distillation multi-teacher

Melanger les outputs de plusieurs teachers (Claude + GPT + Gemini). Gain : couverture plus large, robustesse. Cout : compliqué a orchestrer, biais croise.

---

## 6. Tailles et cibles de deploiement 2026

| Taille student | Cible | Example 2026 | Frontier possible si |
|---|---|---|---|
| 0.5B-1B | Embedded, mobile, IoT | Gemma 3 1B, Qwen 3 0.6B | Tache tres specifique, latence <50ms |
| 1-3B | Browser (WebGPU), edge server | Phi-4-mini, Llama 3.2 3B | Classification, extraction, chat simple |
| 7-14B | Un seul GPU commodity (A100 24-40GB) | Llama 3.3 8B, Qwen 3 8B, DS R1 14B | Agents leger, RAG, reasoning intermediaire |
| 27-32B | Un seul A100 80GB / H100 | Gemma 3 27B, Qwen 3 32B thinking | Agents, domain expert, reasoning |
| 70-120B | 2-4 H100 | Llama 4 Maverick, Qwen 3 72B | Quasi-frontier, self-host gros produit |

**Regle d'or** : commencer par tester la cible avec un modele generaliste existant a cette taille. Si tu atteins 80% de la qualite voulue, fine-tune/distille. Si tu n'atteins que 40%, il faut un modele plus gros OU la tache est mal definie.

---

## 7. Economie concrete : distiller ou pas ?

### Le calcul a faire avant tout projet de distillation

```
Cout API frontier (generaliste) :
  = volume × tokens_moyen × price_per_token

Cout distillation :
  = cout_generation_dataset (une fois, ~$10k-100k)
  + cout_training (une fois, ~$5k-50k)
  + cout_inference_selfhost (cloud/infra, mensuel)
  + cout_maintenance (equipe ML, mensuel)
```

**Break-even typique** : > 10-50M tokens/mois sur la tache.

En dessous, rester sur API frontier + prompt caching + Haiku/Gemma 3 27B.
Au-dessus, distillation gagne.

### Anti-pattern : distiller sans avoir prouve la qualite

Distiller un modele pour un cas d'usage mal defini = 3 mois perdus. Toujours :
1. Mesurer la qualite du frontier model en baseline sur un eval set
2. Mesurer la qualite d'un SLM generaliste (Gemma, Phi, Qwen) sans fine-tune
3. Mesurer le gap
4. Seulement alors : decider si le gap est comble par distillation

---

## 8. Pieges frequents en distillation 2026

1. **Data contamination** : tes synthetic data contiennent l'eval set (car le frontier model a lu ton eval set public pendant son pre-training). Toujours verifier avec des eval sets internes jamais publies.
2. **Mode collapse** : le student apprend UNE façon de repondre et perd la diversite. Mitigation : varier la temperature lors de la generation, diversifier les prompts.
3. **Distillation de biais** : si le teacher hallucine ou a des biais, le student les herite amplifies. Les filtres LLM judge doivent etre rigoureux.
4. **Overfit synthetic** : le student performe en eval synthetic mais chute sur vrais utilisateurs. Solution : inclure des traces de prod reelles (anonymisees) dans le train mix.
5. **Forgetting** : la distillation ecrase les capacites generales du modele de base. Melanger avec des donnees generales (Open-Orca, etc.) durant le fine-tune.
6. **Licence** : les outputs des API proprietaires ne sont pas forcement utilisables pour entrainer un modele concurrent. Lire les CGU d'OpenAI, Anthropic, Google avant de batir dessus.

---

## 9. Route pratique pour un AI engineer en 2026

```
1. Tache bien definie, volume eleve prevu ?
   │
   ├── Non  → Reste sur API frontier + prompt caching + routeur modele
   │
   └── Oui  → continuer
         │
2. Prototype avec Claude/GPT pour valider le concept et la qualite cible
         │
3. Evaluer un SLM generaliste existant (Gemma/Phi/Qwen) sur ton eval set
         │
         ├── Qualite OK (>= 90% de la cible) → deployer, pas de distillation
         │
         └── Gap >= 10 points → pipeline synthetic data + SFT
               │
4. Gap persistant apres SFT ?
         │
         ├── Oui → DPO ou RL on-policy
         │
         └── Non → stop, deploy
```

La majorite des equipes **sautent l'etape 3** et vont directement a distiller. Resultat : 3 mois de travail pour rattraper ce qu'un Qwen 3 8B non-distille faisait deja.

---

## Key takeaways (flashcards)

**Q1** — Quelle est la methode de distillation dominante en 2026 pour les API fermees ?
> Distillation par donnees synthetiques : generer un dataset avec le frontier model (prompt, reponse complete), puis SFT un petit modele dessus. Pas besoin d'acces aux logits du teacher.

**Q2** — Pourquoi un SLM specialise peut-il battre un frontier generaliste sur une tache ?
> Il alloue ses parametres a la tache au lieu de tout le monde. Cout 10-50x plus bas, latence 5-20x. Payer la difference en maintenance/infra si volume eleve.

**Q3** — Quelle est l'etape la plus importante du pipeline synthetic ?
> Le filtrage (rule-based, verifiable, LLM judge, dedup, safety). 30-60% des generations passent. La qualite finale du student en depend directement.

**Q4** — Quand la distillation devient-elle economiquement justifiee ?
> Break-even typique : 10-50M tokens/mois sur la tache. En dessous, rester sur API frontier + Haiku/Gemma en routeur.

**Q5** — Pourquoi le DeepSeek R1 Distill 14B peut-il battre DeepSeek V3 non-reasoning ?
> Le student herite des traces reasoning completes. Test-time compute + heritage reasoning fait que 14B + thinking bat 236B sans thinking.

**Q6** — Quels sont les 3 principaux pieges de la distillation ?
> (1) Data contamination (ton eval est dans le teacher), (2) mode collapse (perte de diversite), (3) forgetting (ecrase les capacites generales si pas de data mix).
