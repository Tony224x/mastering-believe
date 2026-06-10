# Exercices Medium — Inference at Scale

---

## Exercice 1 : Dimensionner le KV cache et la capacite concurrente

### Objectif
Calculer la memoire GPU reellement disponible pour le KV cache et en deduire la concurrence maximale.

### Consigne
Tu sers un modele **13B parametres en FP16** sur un GPU de **80 Go** (A100/H100).

**Donnees :**
- Poids du modele : 13B params x 2 bytes = 26 Go
- Overhead runtime (activations, buffers, CUDA) : ~6 Go
- KV cache par token : ~800 Ko (modele 13B, FP16)
- Profil des requetes : prompt moyen 1 500 tokens, generation moyenne 500 tokens (sequence totale 2 000 tokens)

1. Calcule la memoire disponible pour le KV cache.
2. Calcule la memoire KV d'UNE requete complete (2 000 tokens). Combien de requetes concurrentes max ?
3. Avec un batching statique naif qui pre-alloue la sequence MAX (4 096 tokens) par requete, combien de requetes concurrentes ? Quel pourcentage de memoire est gaspille en moyenne ?
4. vLLM avec PagedAttention alloue par blocs a la demande (fragmentation < 4%). Recalcule la concurrence effective et le gain vs le statique.
5. On passe le KV cache en **int8** (2x plus compact). Nouvelle concurrence ? Quel est le risque a verifier avant de deployer ?

### Criteres de reussite
- [ ] Memoire KV disponible = 80 - 26 - 6 = 48 Go
- [ ] 1 requete = 2000 x 0.8 Mo = 1.6 Go -> ~30 requetes concurrentes
- [ ] Statique 4096 tokens = ~3.3 Go/req -> ~14 requetes, avec ~50% de memoire allouee mais inutilisee
- [ ] PagedAttention ~ 29-30 requetes (quasi le theorique), soit ~2x le batching statique
- [ ] int8 KV -> ~60 requetes ; risque : degradation de qualite a evaluer sur un benchmark avant rollout

---

## Exercice 2 : SLO TTFT/TPOT et tuning du continuous batching

### Objectif
Relier les metriques TTFT / TPOT / throughput aux parametres de serving et arbitrer latence vs cout.

### Consigne
Ton endpoint de chat a ces SLOs : **TTFT < 800 ms au p95** et **TPOT < 60 ms** (vitesse de generation percue > ~16 tokens/s). Mesures actuelles sur ton deploiement vLLM : TTFT p95 = 2.1 s, TPOT = 45 ms, GPU utilization = 92%, queue depth moyenne = 12 requetes.

1. Diagnostique : lequel des deux SLOs est viole, et que t'indique la combinaison (queue depth eleve + TPOT OK) sur la cause racine ?
2. Explique pourquoi TTFT et TPOT ne reagissent pas pareil a la charge (phase prefill vs phase decode, compute-bound vs memory-bound).
3. On te propose 4 actions. Predis l'effet de chacune sur TTFT, TPOT et cout :
   a) Ajouter un replica GPU (scale out)
   b) Augmenter `max_num_seqs` (batch plus gros)
   c) Activer le chunked prefill
   d) Prioriser les prompts courts dans la queue
4. Les prompts longs (8K tokens, 10% du trafic) font exploser le TTFT des petits prompts. Propose une architecture de separation (pools, routing par longueur) et ses tradeoffs.
5. Definis les 3 alertes de monitoring que tu poses, avec seuils, pour proteger ces SLOs.

### Criteres de reussite
- [ ] TTFT viole ; queue depth eleve + GPU sature = sous-capacite en prefill/admission, pas un probleme de decode
- [ ] Prefill = compute-bound (tout le prompt d'un coup), decode = memory-bound (1 token/step) — la file d'attente retarde surtout le premier token
- [ ] Effets corrects : (a) baisse TTFT, cout +1 GPU ; (b) ameliore le throughput mais peut degrader TPOT ; (c) lisse le TTFT en evitant qu'un long prefill bloque les decodes ; (d) baisse le TTFT p95 global mais risque de famine pour les longs prompts
- [ ] La separation propose 2 pools (court/long) ou un routing par longueur estimee, tradeoff : utilisation GPU moins bonne, capacite a dimensionner par pool
- [ ] Alertes plausibles : TTFT p95 > seuil, queue depth > N pendant M minutes, KV cache utilization > 90%

---

## Exercice 3 : Autoscaling d'une fleet d'inference

### Objectif
Concevoir une politique d'autoscaling adaptee aux specificites GPU (cold start, cout, trafic bursty).

### Consigne
Ta fleet sert un modele 7B : trafic de **5 req/s la nuit a 80 req/s en pic** (pattern journalier previsible + bursts imprevisibles x2 en 5 minutes). Un replica GPU (L4) tient **4 req/s** au SLO, coute **0.80 $/h**, et met **4 minutes** a demarrer (pull image + load weights).

1. Calcule la taille de fleet necessaire : la nuit, au pic prevu, au pic avec burst x2.
2. Pourquoi le HPA classique base sur le CPU ne fonctionne pas ici ? Quelles metriques utiliser a la place ?
3. Le cold start de 4 min est plus long que la montee d'un burst (5 min pour x2). Montre par le calcul pourquoi un scaling purement reactif arrive trop tard, et ce que ca provoque pendant l'intervalle.
4. Concois la politique complete : scaling predictif (pattern journalier) + reactif (bursts) + marge de capacite. Quel pourcentage de headroom gardes-tu et combien ca coute par jour ?
5. Optimisations du cold start : propose 3 techniques pour passer de 4 min a < 1 min.

### Criteres de reussite
- [ ] Tailles : nuit = 2 replicas (5/4 arrondi + min HA), pic = 20, burst = 40
- [ ] CPU inutile (le GPU travaille, le CPU est idle) ; metriques : queue depth, concurrence, latence p95, GPU/KV utilization
- [ ] Calcul du retard : burst x2 en 5 min, scale-up detecte puis 4 min de boot -> plusieurs minutes en sous-capacite -> queue qui explose, TTFT degrade, timeouts
- [ ] La politique combine schedule predictif + reactif sur queue depth + headroom chiffre (ex : 25% -> cout calcule sur la journee)
- [ ] Cold start : image pre-pulled / replicas warm en pause, poids sur volume local/NVMe ou streaming des weights, modele deja en RAM (snapshot), pool de spare instances
