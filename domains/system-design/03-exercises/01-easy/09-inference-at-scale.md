# Exercices Easy — Inference at Scale

---

## Exercice 1 : Choisir son framework de serving

### Objectif
Savoir matcher un use case a un framework de serving inference.

### Consigne
Pour chaque situation, recommande un framework (**vLLM**, **TGI**, **TensorRT-LLM**, **Triton**, **TorchServe**, **llama.cpp / Ollama**) et justifie en 2-3 phrases :

1. Une startup veut servir Llama-3-8B pour un chatbot interne. Equipe de 2 devs, budget limite, 1 seul GPU A100.
2. Une banque veut un modele de classification document base sur BERT, deploye sur 4 GPUs T4, avec du multi-model hosting (10+ modeles differents).
3. Un gros editeur AI sert 100K+ users sur Mistral-large, doit maximiser la marge (cost par token), equipe ML infra dediee.
4. Un dev veut faire tourner Llama-3-70B quantise sur son MacBook M3 Pro pour un usage personnel.
5. Une equipe HuggingFace centric veut deployer un modele fine-tune avec minimum d'effort, streaming supporte.

### Livrables
Un tableau : use case / framework / 2-3 lignes de justification.

### Criteres de reussite
- [ ] vLLM choisi pour au moins 1 use case (Llama bas cost ou haute perf)
- [ ] TensorRT-LLM pour le cas "maximiser la marge" (plus optimise mais plus complexe)
- [ ] llama.cpp / Ollama pour le cas MacBook (CPU / Apple Silicon)
- [ ] Triton pour le cas multi-model hosting (c'est sa force)
- [ ] TGI pour le cas HuggingFace-centric
- [ ] Chaque justification cite au moins un argument technique precis

---

## Exercice 2 : Dimensionnement d'un serveur LLM

### Objectif
Calculer concretement combien de GPUs il faut pour une charge donnee.

### Consigne
Tu deploies Llama-3-70B quantise en int8 pour un produit SaaS. Donnees :

- Tu cibles **1000 requetes concurrentes** en peak
- Le modele fp16 pese 140 Go, int8 pese 70 Go
- Tu as des H100 80Go disponibles
- Une H100 en int8 + continuous batching sert environ **~30 requetes concurrentes** avec p99 < 3 s sur des outputs de 500 tokens
- Une H100 coute $3/heure en cloud
- Ton SLA est p99 < 5 s, uptime 99.9%

**Questions :**
1. Combien de H100 faut-il au minimum pour la charge peak ?
2. Quelle marge de securite recommandes-tu et pourquoi (headroom) ?
3. Calcule le cout mensuel approximatif (peak soutenu 30% du temps, off-peak 10%, reste 3%)
4. Si tu passes a du int4, tu peux charger 2 modeles par H100. Comment ca change ta reponse ?
5. Pour atteindre le 99.9% uptime, combien de replicas supplementaires prevois-tu ?

### Livrables
Calculs detailles + recommandation finale de taille de pool.

### Criteres de reussite
- [ ] Le calcul de base donne 34+ H100 pour le peak (1000 / 30)
- [ ] La marge de securite est mentionnee (> 20%, donc ~40 H100)
- [ ] Le cout mensuel est approximatif mais dans le bon ordre de grandeur (milliers $/mois)
- [ ] Le int4 permet une reduction (facteur ~2 sur le nombre de GPUs)
- [ ] L'uptime 99.9% implique au moins du N+1 voire N+2 et une repartition sur 2 zones

---

## Exercice 3 : Tuner le dynamic batching

### Objectif
Comprendre le tradeoff `max_batch_size` vs `max_wait` et savoir le tuner.

### Consigne
Tu exploites un serveur d'inference avec dynamic batching configure ainsi :
`max_batch_size=32`, `max_wait=100ms`.

Les metriques actuelles :
- p50 latency : 180 ms
- p99 latency : 950 ms
- Throughput : 180 req/s
- GPU utilization : 65%

Tu as 3 objectifs business differents (exclusifs). Pour chacun, propose :
- Comment tu changerais `max_batch_size` et `max_wait`
- Ce que tu ferais d'autre
- Quelles metriques tu surveilles apres le changement

**Objectifs :**
1. **Reduire le cout par requete de 30%** (meme SLA accepte)
2. **Reduire le p99 a < 500 ms** (on peut accepter moins de throughput)
3. **Tenir un nouveau SLA tres strict : p99 < 200 ms a isothroughput**

### Livrables
3 mini-analyses structurees (1 par objectif).

### Criteres de reussite
- [ ] Objectif 1 : augmenter max_batch_size et/ou max_wait (on accepte plus de latence pour plus de throughput)
- [ ] Objectif 2 : baisser max_wait (ou baisser max_batch_size) pour reduire la tail latency
- [ ] Objectif 3 : on ne peut pas tenir avec le meme hardware -> il faut scaler horizontalement (ajouter des replicas) ou changer de modele (distillation, quantization plus agressive)
- [ ] La GPU utilization est mentionnee dans au moins 2 reponses
- [ ] La notion de queue depth comme metrique de monitoring est citee
