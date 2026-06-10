"""
Solutions -- Exercices Jour 5 : Load Balancing & Networking

Ce fichier contient les solutions detaillees des exercices Easy,
Medium et Hard.

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


# =============================================================================
# MEDIUM -- Exercice 1 : Consistent hashing
# =============================================================================


def medium_1_consistent_hashing():
    """Solution calculee : modulo vs consistent hashing, vnodes, panne."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 1 : Consistent hashing -- dimensionner et rebalancer")
    print(SEPARATOR)

    nodes = 8
    keys = 80_000_000

    # 1. Modulo : fraction de cles qui restent en place lors de 8 -> 9
    # Une cle reste si hash % 8 == hash % 9. Sur un grand espace de hash,
    # cela n'arrive que pour ~1/9 des cles (les deux modulos coincident
    # sur lcm(8,9)=72 valeurs -> 8 cas favorables sur 72 = 1/9).
    staying_fraction = 1 / 9
    moved_modulo = keys * (1 - staying_fraction)
    print(f"\n  1. Sharding modulo, ajout du 9e noeud")
    print(f"     P(meme noeud avant/apres) = 1/9 (~{staying_fraction:.1%})")
    print(f"     Cles deplacees ~ 8/9 = {moved_modulo:,.0f} ({(1-staying_fraction):.0%})")
    print(f"     Quasi TOUT le cluster est rebalance : catastrophe pour un cache.")

    # 2. Consistent hashing
    moved_ch = keys / (nodes + 1)
    print(f"\n  2. Consistent hashing, ajout du 9e noeud")
    print(f"     Seules les cles du segment repris par le nouveau noeud bougent :")
    print(f"     ~1/9 des cles = {moved_ch:,.0f} (~11%). 8x moins que le modulo.")

    print("""
  3. Sans virtual nodes (8 points sur l'anneau)
     Les 8 positions aleatoires decoupent l'anneau en segments tres
     inegaux : il est courant qu'un noeud porte 2-3x la charge moyenne
     (et un autre 3x moins). Avec si peu de points, la variance est
     enorme : ecart type ~ moyenne.

  4. Avec 150 vnodes par noeud (1 200 points sur l'anneau)
     - Variance de charge : chaque noeud agrege 150 petits segments ->
       la loi des grands nombres lisse la distribution (ecart < 5-10%).
     - Cout de lookup : bisect sur 1 200 points au lieu de 8 ->
       O(log n), toujours negligeable (~microsecondes).
     - Rebalancing : l'ajout d'un noeud prend 150 petits segments
       voles a TOUS les noeuds, pas un gros segment a un seul.

  5. Panne d'un noeud
     Modulo : N passe de 8 a 7 -> ~7/8 des cles changent de noeud ->
       hit rate du cache s'effondre quasiment a zero -> la DB derriere
       prend toute la charge (risque de cascade).
     Consistent hashing + vnodes : seules les cles du noeud mort
       (~12.5%) sont relocalisees, reparties sur TOUS les survivants
       (grace aux vnodes). Hit rate global : -12.5 points environ,
       le temps que le cache se rechauffe. Pas de cascade.""")


# =============================================================================
# MEDIUM -- Exercice 2 : Rate limiter distribue
# =============================================================================


def medium_2_distributed_rate_limiter():
    """Solution : du token bucket local au sliding window Redis + hybride."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 2 : Rate limiter distribue")
    print(SEPARATOR)

    print("""
  1. QUOTA EFFECTIF AVEC BUCKETS LOCAUX
     Chaque gateway compte dans son coin : un client dont les requetes
     sont reparties en round robin peut consommer 100 req/min sur CHACUNE
     des 10 instances -> jusqu'a 1 000 req/min, 10x le contrat.

  2. POURQUOI 100/10 = 10 PAR GATEWAY NE MARCHE PAS
     a) Le round robin n'est pas parfaitement uniforme par client (sticky
        connections, keep-alive, scaling) : un client legitime peut etre
        bloque a 40 req/min parce que ses requetes se concentrent sur
        3 instances.
     b) Le quota depend de la taille de la flotte : a chaque scale-up/down
        il faut recalculer -- et pendant un deploy (12 instances actives)
        le quota global derive.

  3. SOLUTION CENTRALISEE REDIS (sliding window)
     Option compteur : INCR rate:{api_key}:{window} + EXPIRE -- simple
       mais fenetre fixe (burst en bord de fenetre).
     Option sorted set : ZADD timestamp, ZREMRANGEBYSCORE (purge),
       ZCARD (comptage) -- vraie fenetre glissante.
     ATOMICITE : les 3-4 commandes doivent etre un script Lua unique.
     Sinon deux gateways peuvent lire '99' simultanement et accepter
     toutes les deux la 100e ET la 101e requete (race condition).

  4. ARCHITECTURE HYBRIDE (latence + resilience)
     - Chaque gateway garde un budget local pre-alloue (ex : tranches de
       10 tokens demandees a Redis par lot) -> 0 appel Redis sur le chemin
       chaud dans 90% des cas.
     - Sync periodique (100-500 ms) avec le compteur central.
     - Si Redis tombe : fail-open avec le budget local seul (on prefere
       laisser passer un exces borne plutot que bloquer toute l'API).
     Tradeoff de precision : un client peut depasser de ~5-10% pendant
     une fenetre de sync. Contractuellement acceptable.

  5. REPONSE CLIENT
     HTTP 429 Too Many Requests
     X-RateLimit-Limit: 100
     X-RateLimit-Remaining: 0
     X-RateLimit-Reset: 1718020800   (epoch de reouverture)
     Retry-After: 23                 (secondes)
     Les headers sont envoyes aussi sur les reponses 2xx (le client peut
     s'auto-reguler avant de se faire limiter).""")


# =============================================================================
# MEDIUM -- Exercice 3 : Multi-region
# =============================================================================


def medium_3_multi_region():
    """Solution chiffree : GeoDNS, TTL, failover EU -> US, replication."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 3 : Plan de deploiement multi-region")
    print(SEPARATOR)

    eu, us, ap = 0.50, 0.35, 0.15

    print("""
  1. CHAINE DE ROUTAGE
     Client -> GeoDNS (latency-based routing + healthchecks)
            -> LB regional L7 (nginx/ALB/Envoy : TLS termination,
               routing par path, retries locaux)
            -> services de la region
     GeoDNS plutot qu'anycast pur : plus simple a operer, suffisant pour
     un SLO de 150 ms (anycast = utile pour DDoS/edge, complexite BGP).
     Healthchecks DNS : une region unhealthy sort de la rotation.

  2. TTL DNS
     Choix : 60 secondes.
     - TTL court -> failover rapide (les resolvers re-demandent vite)
       MAIS plus de requetes DNS (cout marginal) et certains resolvers
       ignorent les TTL < 30s.
     - TTL long (1h) -> failover catastrophique : des clients tapent une
       region morte pendant 1h.
     60s = bon compromis standard.""")

    surge = us + eu
    print(f"  3. PANNE DE eu-west")
    print(f"     Detection : healthchecks GeoDNS (3 echecs / 10s) + alerting.")
    print(f"     Bascule : eu-west sort du DNS ; propagation ~TTL (60s) +")
    print(f"     trainee des resolvers non conformes (quelques minutes).")
    print(f"     Le trafic EU ({eu:.0%}) part vers us-east (latence +80-100 ms,")
    print(f"     SLO degrade mais service UP).")
    print(f"     us-east passe de {us:.0%} a {surge:.0%} du trafic global")
    print(f"     soit x{surge/us:.1f} sa charge nominale.")

    print(f"""
  4. DIMENSIONNEMENT N+1
     us-east doit pouvoir absorber {surge:.0%} du trafic global, soit
     x{surge/us:.1f} sa charge nominale. Deux strategies :
     - Reserve permanente : payer ~2.4x en continu -> cher.
     - Autoscaling rapide + surdimensionnement partiel (headroom 40-50%)
       + degradation controlee (rate limiting des endpoints non critiques)
       pendant les 5-10 min de scale-up. C'est le choix pragmatique.

  5. REPLICATION POSTGRES
     Pattern recommande ici : PARTITIONNEMENT des users par region
     (home region) : les writes d'un user europeen vont au primary EU.
     + read replicas cross-region pour les lectures globales.
     Probleme des writes cross-region (si un seul primary global) :
     80-150 ms de RTT ajoutes a CHAQUE write des regions distantes,
     et conflits si multi-primary (resolution complexe, a eviter).
     En cas de panne de region : promotion d'une replica (RPO de
     quelques secondes en async -- perte potentielle assumee et
     documentee, ou sync intra-continent pour la reduire).""")


# =============================================================================
# HARD -- Exercice 1 : Edge layer plateforme video live
# =============================================================================


def hard_1_live_edge_layer():
    """Solution : chaine d'entree mondiale, WebSockets, burst x300, DDoS."""
    print(f"\n{SEPARATOR}")
    print("  HARD 1 : Edge layer video live (API + chat)")
    print(SEPARATOR)

    total_conns = 20_000_000
    conns_per_gw = 200_000
    gateways = total_conns // conns_per_gw

    print(f"""
  1. CHAINE DE ROUTAGE
     Resolver -> GeoDNS (TTL 30-60s, sante par region)
              -> PoPs ANYCAST (absorption DDoS volumetrique au plus
                 pres de la source, TLS termination edge)
              -> LB L4 regional (les WebSockets sont des connexions
                 longues : L4 est efficace pour les tenir par millions)
              -> Gateway L7 WebSocket (auth, routing par stream/channel)
              -> services chat/API
     API HTTP classique : L7 direct (ALB/Envoy) derriere les memes PoPs.
     Justification anycast : 5M req/s d'attaque ne doivent JAMAIS
     atteindre une region unique.""")

    print(f"  2. PROBLEME WEBSOCKET")
    print(f"     {total_conns:,} conn / {conns_per_gw:,} par instance = {gateways} gateways min")
    print(f"     + marge 20-30% (rebalancement, deploys) = ~{int(gateways*1.25)} instances")
    print("""     Deploiement d'une nouvelle version :
     - drain progressif : on arrete d'ASSIGNER de nouvelles connexions,
       puis on ferme les existantes par vagues (1-5% a la fois) avec un
       message de reconnexion.
     - cote client : reconnexion avec JITTER aleatoire (0-30s) + backoff.
       Sans jitter, 200K clients reviennent dans la meme seconde =
       reconnection storm qui tue le gateway suivant (cascade).""")

    print("""
  3. BURST x300 (10K -> 3M viewers en 10 min)
     ~3M connexions nouvelles / 600 s = 5 000 conn/s rien que pour ce
     stream, plus le fan-out chat qui explose.
     Sature en premier : le pub/sub du channel chat (un seul stream =
     une seule cle logique -> hot partition) puis les gateways WS.
     Absorption :
     - shuffle sharding : le channel d'un gros stream est decoupe en
       N sous-channels repartis sur des brokers differents (les viewers
       sont assignes a un sous-channel ; les messages sont fan-out entre
       sous-channels par un relai dedie).
     - pre-scaling sur signal : un stream qui gagne +50K viewers/min
       declenche le scale-up AVANT la saturation ; events planifies
       (finales) = capacite pre-provisionnee.
     - admission control : au-dela de la capacite, les nouveaux viewers
       recoivent le video SANS le chat interactif (lecture seule).
     Rythme de scale-up : si 1 gateway absorbe 200K conn, il faut
     ~15 instances en 10 min = 1.5 instance/min + le warm-up -> d'ou
     l'interet d'un pool warm permanent (5-10% de capacite).

  4. DDOS (5M req/s malveillants)
     Couche 1 -- anycast/scrubbing edge : filtrage volumetrique (SYN
       floods, amplification) reparti sur tous les PoPs.
     Couche 2 -- L7 edge : challenge JS/cookie pour les clients sans
       token, rate limit par IP ET par token, reputation IP.
     Couche 3 -- application : rate limit par user authentifie, par
       channel, scoring comportemental (un 'viewer' qui envoie 50 msg/s).
     L'IP seule ne suffit pas : les botnets ont des millions d'IP
     residentielles, et un NAT d'universite = des milliers d'users
     legitimes derriere UNE IP (les bloquer = faux positifs massifs).

  5. LOAD SHEDDING (3 niveaux, on protege la connexion + l'envoi)
     Niveau 1 (105% capacite) : couper l'historique du chat au join,
       desactiver les features cosmetiques (emotes animees, badges).
     Niveau 2 (115%) : sampling de l'affichage sur les gros streams
       (chaque viewer voit un sous-ensemble des messages -- deja le cas
       au-dela d'un certain debit, personne ne lit 10K msg/s).
     Niveau 3 (130%) : refuser les NOUVELLES connexions chat des
       non-abonnes sur les streams satures (le video continue) ;
       l'envoi de messages des users connectes reste prioritaire.

  6. TROIS TRADEOFFS
     1) Anycast + PoPs plutot que GeoDNS seul : protection DDoS et RTT
        reduits, MAIS complexite BGP/infra edge et cout fixe eleve.
     2) L4 pour les WS plutot que L7 partout : 20M de connexions tenues
        a moindre cout CPU, MAIS moins de visibilite/routing fin a ce
        niveau (compense par le gateway L7 derriere).
     3) Pool warm 5-10% en permanence : du cash brule chaque jour, MAIS
        c'est l'assurance anti-burst x300 ; moins cher qu'un seul
        incident de finale e-sport rate.""")


# =============================================================================
# HARD -- Exercice 2 : Budget de retry et anti-cascade
# =============================================================================


def hard_2_retry_budget():
    """Solution chiffree : amplification, timeouts, hedging, chaos test."""
    print(f"\n{SEPARATOR}")
    print("  HARD 2 : Resilience inter-services")
    print(SEPARATOR)

    retries = 3
    qps = 5_000

    # 1. Amplification
    amp = (1 + retries) ** 2  # BFF retrie Pricing, Pricing retrie Promo
    print(f"\n  1. AMPLIFICATION DES RETRIES")
    print(f"     Si Promo degrade : Pricing fait 1+{retries} appels, et le BFF")
    print(f"     retente Pricing 1+{retries} fois -> ({retries+1}) x ({retries+1}) = {amp} appels a Promo")
    print(f"     pour UNE requete client.")
    print(f"     Formule : (R+1)^N pour N etages a R retries.")
    print(f"     A {qps:,} req/s -> jusqu'a {qps*amp:,} req/s sur Promo : le service")
    print(f"     deja degrade recoit 16x sa charge = mort assuree (retry storm).")

    print("""
  2. BUDGET DE LATENCE (SLO 800 ms p99) -- timeouts DECROISSANTS
     Edge -> BFF                 : 750 ms
     BFF -> Cart (p99 80)        : 200 ms
     BFF -> Shipping (p99 150)   : 400 ms  (inclut le sous-appel Carrier)
     BFF -> User (p99 40)        : 100 ms
     Cart -> Pricing (p99 120)   : 150 ms
     Pricing -> Promo (p99 60)   : 100 ms
     Shipping -> Carrier         : 250 ms  <- INCOMPATIBLE avec p99 2000 ms
     Regle : chaque timeout > p99 du service appele, et la somme du
     chemin critique (avec 1 retry eventuel) < budget parent.
     Le Carrier API viole structurellement le budget -> il faut le
     sortir du chemin synchrone (question 4).

  3. POLITIQUE DE RETRY
     - UN SEUL etage retrie : le caller DIRECT de la feuille (ex : le BFF
       pour ses appels directs). Les etages intermediaires ne retryent
       JAMAIS -> amplification max = (1+1) au lieu de 16.
     - 1 retry max, backoff 50-100 ms + jitter, UNIQUEMENT sur erreurs
       transitoires (timeout, 503, connexion) et requetes idempotentes.
     - RETRY BUDGET global : max 10% de requetes additionnelles par
       fenetre de 10 s. Budget epuise -> les echecs remontent sans retry.
       (C'est le mecanisme qui rend le retry storm IMPOSSIBLE.)

  4. CARRIER API (p99 2 s, 2% erreurs)
     - Circuit breaker : open si error rate > 10% OU p99 > 800 ms sur
       une fenetre de 30 s ; half-open apres 15 s (1 requete sonde).
     - Cache des tarifs : cle (origine, destination, poids arrondi),
       TTL 4-24 h (les grilles tarifaires bougent rarement) -> ~90%+
       de hit rate attendu, le Carrier sort du chemin critique.
     - Fallback : tarif ESTIME depuis la derniere grille connue +
       marge de securite ; flag 'estimated' dans la reponse.
     - Impact business assume : ecart possible de quelques % sur les
       frais de port, absorbe ou regularise a la confirmation.""")

    # 5. Hedging
    p95_shipping = 150
    print(f"  5. HEDGED REQUESTS (Shipping)")
    print(f"     Envoyer une 2e requete si pas de reponse a ~{p95_shipping} ms (p95),")
    print(f"     garder la premiere reponse arrivee.")
    print(f"     Gain : le p99 (queue aleatoire, GC, noeud lent) tombe vers")
    print(f"     ~p95 + p50 du second essai, typiquement 150-200 ms au lieu de 400+.")
    print(f"     Surcout : ~5% de QPS en plus (seules les requetes > p95 hedgent).")
    print(f"     Conditions : idempotent + le service a de la marge de capacite.")

    print("""
  6. CHAOS TEST DE VALIDATION
     Faute injectee : +1 900 ms de latence sur Promo (100% des appels)
     pendant 10 min en pre-prod sous 5 000 req/s simules.
     Metriques observees :
       - p99 endpoint : doit rester < 800 ms (timeouts + fallback)
       - QPS recu par Promo : doit rester < 1.1x le QPS nominal
         (preuve que l'amplification est morte)
       - error rate client : les requetes degradees renvoient une reponse
         partielle (sans promo), pas des 500.
     Critere de succes : les 3 conditions tenues pendant toute la faute
     + retour a la normale < 60 s apres la fin de l'injection.""")


def main():
    easy_1_choose_lb_algorithm()
    easy_2_rate_limiter_design()
    easy_3_cascading_failure()
    medium_1_consistent_hashing()
    medium_2_distributed_rate_limiter()
    medium_3_multi_region()
    hard_1_live_edge_layer()
    hard_2_retry_budget()
    print(f"\n{SEPARATOR}")
    print("  Fin des solutions Jour 5.")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
