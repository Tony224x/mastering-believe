# Exercices Hard — Caching & CDN

---

## Exercice 1 : Design du caching layer pour un e-commerce Black Friday

### Objectif
Concevoir une architecture de cache resiliente capable de gerer un pic de trafic 20x avec des contraintes de consistance sur les prix et le stock.

### Consigne
Tu concois le caching layer pour un e-commerce (type Amazon simplifie) qui prepare le Black Friday :

**Chiffres normaux :**
- 50K requetes/sec en lecture (catalogue, prix, stock)
- 2K ecritures/sec (commandes, mises a jour de stock)
- Catalogue : 5M de produits, taille moyenne 2 Ko par produit
- Hit rate actuel : 88%

**Chiffres Black Friday (pic prevu) :**
- 1M requetes/sec en lecture (20x)
- 40K ecritures/sec (20x)
- 100 "flash deals" annonces a heure precise (pic instantane sur ces produits)
- Duree des flash deals : 30 minutes chacun
- Stock limite : certains deals ont 500 unites

**Contraintes :**
- Le prix affiche doit etre le prix reel (pas de stale price > 10s)
- Le stock doit etre coherent (pas de survente)
- La latence de lecture doit rester < 50ms au p99, meme pendant le pic
- Budget : $50K/mois max pour l'infrastructure cache

**Livre :**

1. **Architecture multi-tier** :
   - Combien de niveaux de cache ? Lesquels ?
   - Quel role pour chaque niveau ? Quelle donnee a quel niveau ?
   - Comment gerer le fait que les prix ne doivent pas etre stale > 10s ?

2. **Dimensionnement Redis** :
   - Combien de memoire pour le catalogue complet ? (5M * 2 Ko)
   - Combien de noeuds Redis Cluster pour 1M reads/sec ?
   - Comment partitionner les donnees (shard key) ?

3. **Flash deals** :
   - Comment gerer le pic instantane quand un flash deal est annonce ? (cache stampede sur 100 produits en meme temps)
   - Comment eviter la survente quand 100K users ajoutent le meme produit au panier ?
   - Propose un mecanisme de "reservation de stock" avec Redis

4. **Cache warming** :
   - Quelles donnees pre-charger avant le Black Friday ?
   - Comment eviter de surcharger la DB pendant le warm ?
   - Timing : combien de temps avant le pic ?

5. **Resilience** :
   - Que se passe-t-il si Redis tombe pendant le Black Friday ?
   - Propose un fallback qui ne tue pas la DB
   - Comment detecter et reagir a un hit rate qui chute (< 50%) ?

6. **Monitoring et alerting** :
   - Les 8 metriques critiques a surveiller
   - Les seuils d'alerte pour chaque metrique
   - Le runbook d'urgence en 5 etapes si le cache degrade

### Criteres de reussite
- [ ] Architecture 3-tier : L1 in-process (prix/config) + L2 Redis Cluster (catalogue) + CDN (assets)
- [ ] Dimensionnement Redis coherent (~25 Go pour le catalogue, 10-20 noeuds pour le throughput)
- [ ] Cache warming des 100 flash deals + top 10K produits + config globale
- [ ] Redis DECR atomique propose pour le stock (evite la survente)
- [ ] Le fallback en cas de Redis down utilise un circuit breaker + rate limiting DB
- [ ] Le prix a un TTL < 10s ou utilise une invalidation event-driven
- [ ] Le budget $50K est respecte (estimation : ~20-30 noeuds Redis = ~$15-25K + CDN ~$5-10K)

---

## Exercice 2 : Post-mortem — Le cache qui a casse le systeme de paiement

### Objectif
Analyser un incident complexe lie au cache, identifier les causes racines, et concevoir les protections.

### Consigne
Voici le rapport d'incident (resume) :

**Contexte** : Un e-commerce a implemente un cache-aside sur Redis pour les donnees de prix des produits. Le systeme fonctionne bien depuis 6 mois. Le cache a un TTL de 5 minutes.

**Timeline de l'incident** :

| Heure | Evenement |
|---|---|
| 10:00 | L'equipe marketing lance une promo : -50% sur 500 produits. Les prix sont mis a jour en DB via un script batch. |
| 10:01 | Le script met a jour les prix en DB mais **ne touche pas le cache Redis**. (Le script contourne le service applicatif et ecrit directement en DB.) |
| 10:01 - 10:06 | Les utilisateurs voient les anciens prix (cache stale). Mais les commandes sont creees avec l'ancien prix car le service lit le prix depuis le cache. |
| 10:06 | Les premiers TTL expirent. Certains users voient le nouveau prix (-50%). |
| 10:06 - 10:10 | **Etat incoherent** : certains users voient l'ancien prix, d'autres le nouveau. Les plaintes affluent sur les reseaux sociaux ("pourquoi mon ami a -50% et pas moi ?"). |
| 10:10 | L'equipe realise le probleme et decide de forcer un cache flush global (`FLUSHALL`). |
| 10:10:01 | **FLUSHALL execute.** 15M de cles supprimees d'un coup. |
| 10:10:02 | **Cache stampede massif.** 50K requetes/sec font un cache miss simultane et requetent la DB. |
| 10:10:05 | PostgreSQL atteint 100% CPU. Les connexions sont saturees (max_connections = 200). |
| 10:10:10 | Le pool de connexions DB est epuise. Les requetes echouent en cascade. Le site retourne des erreurs 500. |
| 10:10:30 | L'equipe tente de redemarrer les app servers pour reduire les connexions. |
| 10:11:00 | Le redemarrage aggrave le probleme (cache encore vide, plus de stampede). |
| 10:15:00 | L'equipe met le site en maintenance (page statique). |
| 10:30:00 | Le cache se reconstruit progressivement avec un rate limiter d'urgence. |
| 10:45:00 | Le site est remis en ligne. 35 minutes de downtime. |
| 11:00 | Post-mortem : ~$200K de commandes perdues, 15K tickets support. |

**Questions :**

1. **Root cause analysis** :
   - Identifie la chaine de causes (pas un seul point, la cascade complete)
   - Pour chaque maillon de la chaine, identifie quel guardrail manquait
   - Classe les causes par categorie : processus, architecture, monitoring

2. **Le FLUSHALL etait-il la bonne decision ?** :
   - Analyse les alternatives qui auraient evite le stampede
   - Propose la procedure correcte pour invalider 500 cles specifiques
   - Si tu DEVAIS faire un flush massif, comment l'aurais-tu fait sans casser la DB ?

3. **Architecture corrigee** :
   - Comment empecher les ecritures directes en DB qui contournent le cache ?
   - Propose 3 mecanismes independants qui auraient evite l'inconsistance de prix
   - Concois un "safe cache invalidation" pattern qui protege contre le stampede

4. **Le probleme du script marketing** :
   - Comment garantir que TOUT changement de prix invalide le cache, quel que soit l'outil qui ecrit ?
   - Propose une architecture event-driven (CDC, outbox pattern, ou autre) qui decouple l'ecriture DB de l'invalidation cache

5. **Resilience pattern** :
   - Concois un circuit breaker entre l'app et la DB qui se declenche quand le pool est > 80%
   - Comment implementer un "graceful degradation" : servir les prix stale plutot que des erreurs 500 ?
   - Propose un runbook de 10 etapes pour un incident cache en production

### Criteres de reussite
- [ ] La chaine causale complete est identifiee : script direct DB -> pas d'invalidation -> stale cache -> FLUSHALL -> stampede -> DB saturee -> downtime
- [ ] Le FLUSHALL est identifie comme l'erreur qui a transforme un probleme mineur en incident majeur
- [ ] L'alternative au FLUSHALL est proposee : invalider les 500 cles specifiques avec un script `DEL key1 key2 ...` espace dans le temps
- [ ] CDC (Change Data Capture) via Debezium est propose pour capturer les ecritures directes en DB
- [ ] Le circuit breaker est decrit avec des seuils concrets (ex: > 160 connexions = fallback stale cache)
- [ ] Le graceful degradation sert les donnees stale avec un header d'avertissement plutot que des 500
- [ ] Le runbook est actionable et commence par "NE PAS FLUSHALL"
