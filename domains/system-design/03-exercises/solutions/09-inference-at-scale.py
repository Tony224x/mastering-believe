"""
Solutions -- Jour 9 : Inference at Scale
"""


def solution_exercice_1() -> None:
    """
    Exercice 1 -- Choix du framework.

    1) Startup, Llama-3-8B chatbot, 1 A100 :
       -> **vLLM**. Raisons :
         - Continuous batching out-of-the-box, tres simple a lancer
           (`python -m vllm.entrypoints.openai.api_server --model ...`).
         - API compatible OpenAI, branchement instant au client.
         - Meilleure perf/cout du marche en 2025 pour les modeles open source.
         - Zero infra complexe, parfait pour une petite equipe.

    2) Banque, BERT classification, 4 T4, multi-model hosting :
       -> **Triton Inference Server**. Raisons :
         - Multi-model serving natif (un seul serveur sert 10+ modeles).
         - Model ensembling, dynamic batching, versioning.
         - Supporte PyTorch, TensorRT, ONNX, Python backend -> flexible pour
           des equipes data science heterogenes.
         - Excellent monitoring et metriques Prometheus.

    3) Editeur AI, 100K users, maximiser la marge :
       -> **TensorRT-LLM**. Raisons :
         - Optimisation la plus poussee par GPU NVIDIA (graph compilation,
           fused kernels, int8/int4 calibre pour chaque architecture).
         - Gain de 20-40% sur vLLM a l'echelle, ce qui represente des millions
           de dollars quand on tourne a 100K users.
         - Contrepartie : courbe d'apprentissage raide, equipe experte
           necessaire (compilation du modele, quantization, etc.).

    4) MacBook M3 Pro, Llama-3-70B perso :
       -> **llama.cpp / Ollama**. Raisons :
         - Optimise pour Apple Silicon (Metal) et CPU.
         - Quantization GGUF (int4 -> 35 Go, tient en RAM unified d'un M3 Pro).
         - Ollama est le plus simple : `ollama run llama3:70b-q4`.

    5) Equipe HF-centric, fine-tune deploy rapide :
       -> **TGI (Text Generation Inference)**. Raisons :
         - Integration native avec HuggingFace Hub, un modele privee deploye
           en une commande docker.
         - Streaming SSE standard.
         - Support continuous batching et quantization.
         - Bon compromis perf / facilite quand vLLM parait trop "raw".
    """


def solution_exercice_2() -> None:
    """
    Exercice 2 -- Dimensionnement Llama-3-70B int8.

    Hypotheses :
      - 1 H100 en int8 sert ~30 req concurrentes avec p99 < 3 s
      - Peak : 1000 concurrent
      - SLA : p99 < 5 s, 99.9% uptime
      - H100 : $3/h cloud

    Q1 -- Nombre de H100 minimum pour le peak :
      1000 / 30 = 33.33 -> arrondi sup = **34 H100**.
      Cette taille te mettra au bord du SLA. En pratique, il faut du headroom.

    Q2 -- Marge de securite :
      Recommandation : **+25% de headroom** -> ~42 H100 pour le peak.
      Raisons :
        - Variance naturelle du trafic (bursts)
        - Certaines requetes ont des outputs plus longs (queue se remplit)
        - Pour tenir le p99 < 5 s (pas seulement le p50), on doit rester
          loin de la saturation. A 90% util, les queues explosent.

    Q3 -- Cout mensuel approximatif :
      Cas moyen : on ne tourne pas 42 H100 tout le temps.
      - 30% du temps au peak (42 GPUs) = 0.30 * 720h * 42 * $3 = $27,216
      - 10% du temps a mid (20 GPUs)   = 0.10 * 720h * 20 * $3 = $4,320
      - 60% du temps a low (10 GPUs)   = 0.60 * 720h * 10 * $3 = $12,960
      Total : ~**$44K / mois**.
      (Enonce dit peak 30%, off-peak 10% -- interprete comme mid 10%, low 60%
      implicite. Dans tous les cas, on est dans les $30-50K/mois pour une
      charge de 1000 concurrent 24/7.)

    Q4 -- Int4 + 2 modeles par H100 :
      En int4 (~35 Go par modele), on peut charger 2 replicas du meme modele
      par H100. Attention : on ne double pas forcement le throughput, car ils
      partagent les unites de calcul. En pratique : gain 1.5-1.8x.
      Avec 1.7x effectif : 34 / 1.7 = 20 H100 -> avec headroom 25 H100.
      Economie : ~$20K / mois. MAIS : perte de qualite int4 a valider sur
      benchmark metier avant de valider.

    Q5 -- 99.9% uptime :
      99.9% = 8.7h de downtime max par an. Une maintenance de routine d'un
      GPU, une instance pre-emptee, etc. consomment du budget d'erreur vite.
      Recommandation :
        - N+2 au minimum (2 replicas en plus du minimum requis)
        - Deployer sur 2 availability zones distinctes
        - Health checks + eviction auto
        - Rolling deploys avec canary, jamais de full swap
      -> +2-4 H100 supplementaires par rapport a N, mais repartis sur 2 AZ.
    """


def solution_exercice_3() -> None:
    """
    Exercice 3 -- Tuner le dynamic batching.

    Etat actuel :
      max_batch_size=32, max_wait=100ms
      p50=180ms, p99=950ms, throughput=180 req/s, GPU util=65%

    Observation : GPU util 65% = marge. p99/p50 = 5.3x = la tail domine.
    Causes probables : max_wait=100ms pousse certaines requetes a attendre
    presque 100ms avant d'etre traitees.

    === Objectif 1 : -30% de cout par requete ===
      Levier : augmenter le throughput pour amortir le cout fixe du GPU.
      Action :
        - max_batch_size : 32 -> 48 ou 64 (si la memoire KV le permet)
        - max_wait       : 100 -> 150 ms (on accepte un peu de latence)
      Resultat attendu : GPU util 85%+, throughput +30-40%, p50 +50ms, p99 +200ms.
      Si le SLA tient toujours : mission accomplie.
      Metriques a surveiller : throughput (doit monter), GPU util (doit
        monter), p99 (ne doit pas exploser), cost per 1K tokens (doit baisser).

    === Objectif 2 : p99 < 500ms ===
      Levier : reduire la variance. La tail vient principalement de
      requetes qui attendent un batch plein ou le timeout.
      Action :
        - max_wait : 100 -> 20 ms (flush plus tot)
        - max_batch_size : 32 -> 16 (limiter la variance par batch)
      Resultat attendu : p99 chute a ~350-450ms, throughput baisse ~20%.
      Contrepartie : plus cher par requete, GPU util baisse a ~50%.
      Metriques : p99 (cible), queue depth (doit rester basse),
        GPU util (va baisser, c'est normal).

    === Objectif 3 : p99 < 200ms a isothroughput ===
      Verdict : **on ne peut pas tenir a iso hardware**.
      Raisons : meme en batch=1, gpu_process_time est > 40ms. p99=200ms
      avec batch = 32 impossible sur un seul replica.
      Options :
        1. Scale horizontalement : ajouter des replicas (doubler la fleet)
           pour repartir la charge. Chaque replica a moins de queue -> p99 bas.
        2. Modele plus petit : distillation / quantization plus agressive
           -> gpu_process_time divise par 2-3.
        3. Architecture speculative : un petit modele predit, un grand
           modele verifie (speculative decoding). Gain x2 sur les latences
           en moyenne.
        4. Changer de GPU (H100 au lieu de A100 : ~2x plus rapide).
      Monitoring : p99, queue depth par replica, coherence du LB.

    Note transverse : dans TOUS les cas, surveiller la **queue depth**
    devant le batcher. C'est le signal leading qui precede une degradation
    de p99. Si queue > 0 soutenu -> trigger scale-up.
    """


if __name__ == "__main__":
    for fn in (solution_exercice_1, solution_exercice_2, solution_exercice_3):
        print(f"\n--- {fn.__name__} ---")
        print(fn.__doc__)
