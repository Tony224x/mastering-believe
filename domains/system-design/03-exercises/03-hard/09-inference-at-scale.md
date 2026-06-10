# Exercices Hard — Inference at Scale

---

## Exercice 1 : Plateforme d'inference mutualisee sous contrainte de cout

### Objectif
Concevoir une plateforme de serving LLM multi-workloads en arbitrant explicitement cout, latence et utilisation GPU.

### Consigne
Tu concois la plateforme d'inference d'une entreprise qui sert 3 workloads sur des modeles open-source :

**Workloads :**
- **Chat interactif** : 600 req/s en pic (9h-22h), TTFT < 500 ms p95, TPOT < 50 ms, prompts ~2K tokens, generation ~400 tokens
- **Extraction structuree** (API interne) : 250 req/s constant 24/7, latence totale < 5 s p99, prompts ~3K, generation ~150 tokens
- **Batch nocturne** (enrichissement de catalogue) : 80M de documents/nuit, fenetre 6 h, pas de contrainte de latence unitaire

**Contraintes chiffrees :**
- GPU disponibles : H100 a 4.50 $/h (on-demand) ou 2.70 $/h (reserved 1 an) ; spot a 1.60 $/h (preemption ~2 fois/jour)
- Un H100 avec le modele 8B quantise sert : ~15 req/s de chat au SLO, ~25 req/s d'extraction, ~40 req/s en batch sature
- Budget cible : < 90 000 $/mois
- Le trafic chat varie de 60 req/s (nuit) a 600 req/s (pic)

**Livre :**
1. **Dimensionnement par workload** : nombre de GPUs pour le chat (pic et nuit), l'extraction, et le batch (80M docs en 6 h a 40 docs/s/GPU : calcule le debit requis et le nombre de GPUs).
2. **Mutualisation** : faut-il 3 pools dedies, ou mutualiser ? Concois la politique : le batch nocturne peut-il tourner sur les GPUs du chat la nuit ? L'extraction peut-elle absorber les bursts du chat ? Quels mecanismes (priorites, preemption des requetes batch, admission control) ?
3. **Mix d'achat** : repartis reserved / on-demand / spot par workload. Calcule le cout mensuel total de ta proposition et compare au budget. Quel workload peut tolerer le spot et avec quel mecanisme (checkpoint/retry) ?
4. **SLO engineering** : le chat exige TTFT < 500 ms p95. Quelle utilisation GPU cible max pour tenir ce p95 (explique pourquoi viser 85%+ d'utilisation casse les latences — queueing theory qualitative) ? Quel est le cout de ce headroom ?
5. **Degradation** : un pic imprevu a 900 req/s de chat (1.5x le pic prevu). Sequence de defense : autoscaling (delai ?), routing vers un modele plus petit, admission control, file prioritaire. Qui est degrade en premier ?
6. **3 tradeoffs chiffres** (ex : reserved vs on-demand sur le delta de cout, qualite 8B vs 70B, mutualisation vs isolation).

### Criteres de reussite
- [ ] Dimensionnement pose : chat pic = 600/15 = 40 GPUs, nuit = 4 ; extraction = 250/25 = 10 GPUs constants ; batch = 80M/21 600 s = ~3 700 docs/s -> /40 = ~93 GPUs pendant 6 h
- [ ] La mutualisation exploite la complementarite temporelle : les ~36 GPUs du chat liberes la nuit servent le batch ; mecanismes : priorites strictes + preemption des jobs batch + quotas
- [ ] Le mix d'achat est justifie : reserved pour la base 24/7 (extraction + chat minimum), on-demand pour le pic chat previsible, spot pour le batch (tolerant aux preemptions avec checkpointing) ; cout mensuel calcule et confronte aux 90 K$
- [ ] Le lien utilisation/latence est explique (file d'attente non lineaire pres de la saturation) avec une cible ~60-70% pour le pool latence-sensible et le surcout assume
- [ ] La sequence de degradation est ordonnee et protege le chat interactif (1. preempter le batch, 2. puiser dans l'extraction (marge 5s), 3. router vers modele plus petit, 4. admission control)
- [ ] 3 tradeoffs avec chiffres a l'appui

---

## Exercice 2 : Servir un modele 70B a l'echelle mondiale — parallelisme et topologie

### Objectif
Concevoir le serving d'un grand modele multi-region en raisonnant parallelisme, memoire et reseau.

### Consigne
Ton entreprise doit servir son modele fine-tune de **70B parametres** (le 8B ne suffit pas en qualite) a des clients en Amerique, Europe et Asie.

**Contraintes chiffrees :**
- 70B params en FP16 = 140 Go de poids ; GPUs disponibles : H100 80 Go (NVLink intra-noeud 900 Go/s, reseau inter-noeud 400 Gb/s)
- KV cache par token (70B, FP16) : ~2.5 Mo ; profil : prompt 4K tokens, generation 800 tokens
- Trafic global : 220 req/s en pic, reparti 45% US / 35% EU / 20% APAC, avec des pics regionaux decales
- SLO : TTFT < 1 s p95, TPOT < 70 ms p95 partout
- Un noeud = 8x H100 ; cout noeud : 25 $/h
- Conformite : les prompts des clients europeens ne doivent pas quitter l'Europe

**Livre :**
1. **Plan de parallelisme** : le modele ne tient pas sur 1 GPU. Combien de GPUs minimum pour les poids + KV cache utile ? Tensor parallelism intra-noeud (TP=4 ? TP=8 ?) vs pipeline parallelism inter-noeuds : lequel et pourquoi (pense a la latence des all-reduce sur NVLink vs reseau) ?
2. **Capacite par noeud** : avec TP=8 sur un noeud (640 Go - 140 Go de poids - overhead ~60 Go = ~440 Go de KV), combien de sequences de 4.8K tokens concurrentes ? Quel throughput en req/s si une generation dure ~55 s (800 tokens x 70 ms) ?
3. **Topologie mondiale** : combien de noeuds par region (pics regionaux : US 100 req/s, EU 80, APAC 45) ? La contrainte de residence EU empeche le spillover du trafic EU : comment dimensionner differemment EU vs les autres ?
4. **Quantization** : passer en FP8/int8 (poids 70 Go, KV 1.25 Mo/token) change quoi : GPUs minimum, concurrence, cout/region ? Quelle validation avant de basculer (benchmark de qualite, A/B) ?
5. **Cout total** : calcule le cout mensuel de ta topologie (FP16 vs FP8) et le cout par 1M de tokens generes. Compare a un appel API frontier (~10 $/M tokens) : le self-hosting se justifie-t-il ?
6. **Failure modes** : perte d'un GPU dans un noeud TP=8 (que devient le noeud ?), perte d'une region. Plans de mitigation.

### Criteres de reussite
- [ ] Le raisonnement memoire est pose : 140 Go de poids -> minimum 2 GPUs theorique mais ~4+ avec KV utile ; TP intra-noeud retenu (all-reduce a chaque couche = NVLink obligatoire, le pipeline inter-noeud ajoute de la latence par token)
- [ ] Capacite calculee : 1 seq = 4.8K x 2.5 Mo = 12 Go -> ~36 sequences concurrentes/noeud ; throughput ~36/55 = ~0.65 req/s/noeud — ce chiffre DOIT declencher la discussion (il faut beaucoup de noeuds, ou du FP8, ou revoir le profil)
- [ ] La topologie respecte la residence EU (capacite EU dimensionnee pour son pic sans spillover, N+1 local) et exploite le spillover US<->APAC si acceptable
- [ ] La quantization est evaluee quantitativement (2x concurrence, GPUs/2 possibles) avec validation qualite obligatoire avant rollout
- [ ] Le cout mensuel est calcule avec les hypotheses posees et le cout/M tokens compare honnetement a l'API frontier (la conclusion peut etre que le self-hosting ne gagne qu'a fort volume)
- [ ] Failure modes : TP=8 -> 1 GPU perdu = le noeud entier hors service (blast radius assume), perte de region = capacite N+1 ou degradation annoncee
