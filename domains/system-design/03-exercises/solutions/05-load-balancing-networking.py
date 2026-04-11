"""
Solutions -- Exercices Jour 5 : Load Balancing & Networking

Ce fichier contient les solutions detaillees des exercices Easy.

Usage:
    python 05-load-balancing-networking.py
"""

SEPARATOR = "=" * 60


# =============================================================================
# EASY -- Exercice 1 : Choisir l'algorithme de LB
# =============================================================================


def easy_1_choose_lb_algorithm():
    """Solution : mapping scenario -> algo."""
    print(f"\n{SEPARATOR}")
    print("  EASY 1 : Choisir l'algorithme de LB")
    print(SEPARATOR)

    choices = [
        (
            "10 web servers stateless homogenes, req ~50 ms",
            "Round Robin",
            "Requetes homogenes + serveurs identiques = RR simple et equitable. "
            "Aucun besoin de tracker l'etat des backends. C'est le defaut."
        ),
        (
            "Canary deploy 5% V2 / 95% V1",
            "Weighted Round Robin",
            "Poids 19:1 (V1:V2) = 95%/5% du trafic. WRR est LA methode "
            "standard pour piloter finement un deploy progressif."
        ),
        (
            "Cluster Memcached 20 noeuds, eviter reshuffle sur panne",
            "Consistent Hashing",
            "C'est exactement le cas d'usage central de consistent hashing. "
            "Quand un noeud tombe, seuls ~5% des cles (1/20) sont "
            "redistribuees. Avec hash % N, on perdrait 95% du cache."
        ),
        (
            "Analytics : rapports de 100 ms a 30s",
            "Least Connections",
            "Les durees sont tres variables. RR enverrait aveuglement, "
            "et un serveur pourrait se retrouver avec 5 rapports de 30s "
            "pendant qu'un autre traite 100 rapides. Least connections "
            "detecte la charge reelle via le count et rebalance."
        ),
        (
            "WebSocket chat, sticky session necessaire",
            "IP Hash (ou cookie-based sticky au L7)",
            "Le serveur a l'etat de la connexion en memoire. Chaque user "
            "DOIT retourner au meme serveur. IP Hash est simple mais "
            "fragile (un user qui change de reseau reshuffle). Preferer "
            "une sticky session par cookie (ALB, nginx ip_hash ou cookie)."
        ),
        (
            "Cluster heterogene : 4x16CPU + 2x64CPU",
            "Weighted Round Robin",
            "Les 64-core ont 4x la capacite des 16-core. Poids 4:4:4:4:16:16 "
            "= la charge est proportionnelle a la capacite. Sans poids, les "
            "16-core saturent pendant que les 64-core idlent."
        ),
    ]

    for i, (scenario, algo, reason) in enumerate(choices, 1):
        print(f"\n  {i}. {scenario}")
        print(f"     Algo : {algo}")
        print(f"     Raison : {reason}")


# =============================================================================
# EASY -- Exercice 2 : Rate limiter pour API publique
# =============================================================================


def easy_2_rate_limiter_design():
    """Solution : design d'un rate limiter multi-tier."""
    print(f"\n{SEPARATOR}")
    print("  EASY 2 : Rate limiter pour API publique")
    print(SEPARATOR)

    print("""
  1. Algorithme per-minute : TOKEN BUCKET (ou sliding window counter)
     - Token bucket autorise les bursts legitimes (un client qui envoie
       80 req d'un coup puis rien pendant 30s reste dans son quota).
     - Sliding window counter est l'autre bon choix : precision
       ~99% et memoire constante. Cloudflare l'utilise.
     - Eviter le fixed window : l'effet "double au bord de fenetre"
       permet de depasser la limite a cheval sur deux fenetres.

  2. Algorithme per-day : FIXED WINDOW ou SLIDING WINDOW
     - Sur 24h, la precision au bord compte moins (100 req de plus ou
       de moins sur 50K est negligeable).
     - Fixed window est le plus simple a implementer en Redis (INCR +
       EXPIRE). Se reset chaque nuit a minuit UTC.
     - Implementation naive : un counter Redis par cle, avec TTL 24h.

  3. Cle de rate-limiting :
     - PRIMAIRE : API key (le client s'authentifie avec un token).
       -> Permet de distinguer free/pro/enterprise et de facturer.
     - SECONDAIRE (fallback) : IP address, pour les endpoints anonymes
       (ex : GET /public, login page, etc.).
     - JAMAIS le user_id seul : le client peut se deconnecter et
       refaire un compte pour contourner.

  4. Ou stocker les compteurs :
     - EN REDIS, en central.
     - Pourquoi pas en memoire par pod : si on a 20 pods, le client
       pourrait faire 20x la limite en hittant chaque pod. Impossible
       a coordonner sans un store central.
     - Pourquoi pas dans le LB : L7 LBs ne partagent pas d'etat
       entre instances (nginx est stateless).
     - Pattern : Redis + Lua script atomique (INCRBY + GET + EXPIRE
       en une seule operation, pas de race condition).

  5. Headers HTTP a renvoyer (RFC 9239 draft, ou headers legacy) :
     - RateLimit-Limit: 100         (ou X-RateLimit-Limit)
     - RateLimit-Remaining: 42       (tokens restants)
     - RateLimit-Reset: 45           (secondes avant refill)
     - Retry-After: 30               (si la limite est atteinte)
     Les clients bien codes lisent ces headers et adaptent leur debit.

  6. Status HTTP : 429 Too Many Requests
     - Avec un body JSON :
       {
         "error": "rate_limited",
         "message": "100 req/min exceeded",
         "retry_after_seconds": 45
       }
     - Jamais 503 Service Unavailable (ca sous-entend que le service
       est down, alors que c'est juste le client qui abuse).

  7. Gestion du burst :
     - Le token bucket autorise NATIVEMENT les bursts jusqu'a 'capacity'.
     - Pour un Pro 1000 req/min : capacity = 1000, refill = 1000/60 = 16.7/s.
       -> Le client peut envoyer 1000 req d'un coup, puis regenere 16.7/s.
     - Si on veut limiter le burst a 200 : capacity=200, refill=16.7/s.
       Le bucket se vide en 12s si on atteint la limite, puis refill lent.
     - C'est la difference cle avec sliding window qui interdit les bursts.

  Bonus : tiers differents via des buckets differents en Redis :
     key = f"rl:{api_key}:minute"  avec TTL=60
     key = f"rl:{api_key}:day"     avec TTL=86400
     On charge le tier depuis la DB user la premiere fois (cache-aside).
    """)


# =============================================================================
# EASY -- Exercice 3 : Debugger une cascade de pannes
# =============================================================================


def easy_3_cascading_failure():
    """Solution : root cause et remediation d'une cascade classique."""
    print(f"\n{SEPARATOR}")
    print("  EASY 3 : Cascade de pannes recommendations-api -> home-page")
    print(SEPARATOR)

    print("""
  1. Pourquoi home-page est tombe :
     - Quand recommendations-api ralentit (50 ms -> 2 s), chaque
       requete home-page bloque un thread pendant 2 secondes au lieu de 50 ms.
     - Avec un debit constant, le nombre de threads/connexions OCCUPES
       explose par 40x. Thread pool de 200 ? Sature en quelques secondes.
     - Toutes les requetes suivantes attendent un thread disponible ->
       elles aussi prennent 2s+. Effet boule de neige.
     - Les retries empirent : chaque echec declenche 3 retries, donc
       x4 la charge sur recommendations-api -> il tombe plus fort.
     - Resultat : home-page est up techniquement, mais tous ses
       workers sont bloques dans des appels lents. Pour les users,
       c'est down.

  2. Pourquoi home-page reste bloque apres la reprise :
     - Les threads existants sont toujours dans des appels en cours
       avec timeout 30s. Ils ne se liberent qu'a l'expiration.
     - La queue de requetes accumulees est enorme ; meme apres que
       reco-api reponde, il y a un gros backlog a digerer.
     - Si les connection pools sont leaked (not closed on error),
       ils restent fermes meme apres la reprise.
     - Il faut soit attendre, soit redemarrer home-page (ce que les
       ops auraient du faire).

  3. 4+ ameliorations concretes :
     a) CIRCUIT BREAKER sur l'appel reco-api :
        - failureThreshold = 10 erreurs en 30s
        - OPEN : echec immediat + fallback (1 ms au lieu de 30s)
        - HALF_OPEN apres 30s pour tester la reprise
        -> Arrete d'envoyer quand ca va mal, reduit la pression sur reco-api
           et evite la saturation de home-page.

     b) TIMEOUT COURT : 200-500 ms au lieu de 30 s
        - Si reco-api repond en 50 ms normalement, un timeout de 500 ms
          est largement suffisant (10x la p99).
        - Mieux vaut echouer vite et montrer un fallback que de bloquer
          un thread 30 secondes.

     c) LIMITER LES RETRIES : max 1 retry avec jitter, budget global
        - 3 retries = x4 la charge = retry-storm.
        - 0 ou 1 retry + budget (pas plus de 10% de retries dans le
          trafic total).

     d) BULKHEAD (isolation des ressources) :
        - Thread pool dedie a reco-api (ex : 50 threads), pas le pool
          general. Quand il sature, les autres appels (DB, cache, user
          service) fonctionnent encore.
        - Pattern Hystrix.

     e) FALLBACK GRACIEUX :
        - En cas d'echec, retourner une liste de produits generique
          (top 10 des ventes, cache stale, ou section masquee).
        - Le user voit une page fonctionnelle sans reco personnalises
          plutot qu'une page 500.

     f) APPEL ASYNCHRONE / NON BLOQUANT :
        - En ideal, la reco est chargee en AJAX apres le render de
          la page principale. Si reco-api tombe, la page est la,
          seul le bloc reco est manquant.
        - Decouplage maximal : le core path ne depend plus de reco-api.

     g) MONITORING ET ALERTES :
        - Alerte sur p99 reco-api > 500 ms pendant 1 min.
        - Alerte sur taux de circuit breaker ouvert.
        - Dashboard : latence, throughput, erreurs par dependance.

  4. Valeur de timeout recommandee :
     - 200 ms a 500 ms, basee sur la p99 normale x ~10.
     - Si reco-api est a p99 50 ms, alors 500 ms = 10x headroom.
     - Un timeout de 30 s signifie 'j'attends que ca reponde meme si
       c'est catastrophique'. C'est la philosophie inverse du fail fast.
     - En SLA : 'Mieux vaut une reponse degradee en 100 ms qu'une
       reponse parfaite en 30 s'.

  5. Fallback pour reco :
     - Cache Redis : 'top-products-fallback' refreshe toutes les heures,
       contient le top 20 des ventes globales.
     - Si le cache existe : on l'affiche, banniere discrete 'Populaires'.
     - Si le cache est vide aussi : on MASQUE carrement la section
       (pas d'erreur visible cote user).
     - Jamais afficher une erreur rouge au user pour un echec reco.

  Resume : le probleme n'etait PAS reco-api. Le probleme etait le
  couplage fort + timeouts longs + retries aggressifs cote home-page.
  Une dependance secondaire ne doit JAMAIS pouvoir tuer un service
  principal. C'est la loi fondamentale de l'architecture distribuee.
    """)


def main():
    easy_1_choose_lb_algorithm()
    easy_2_rate_limiter_design()
    easy_3_cascading_failure()
    print(f"\n{SEPARATOR}")
    print("  Fin des solutions Jour 5.")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
