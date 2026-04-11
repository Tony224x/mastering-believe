# Exercices Medium — Caching & CDN

---

## Exercice 1 : Concois la couche de cache pour un feed social

### Objectif
Appliquer les strategies de caching a un cas reel avec plusieurs niveaux et des tradeoffs de consistance.

### Consigne
Tu concois la couche de cache pour le feed d'un reseau social (type Twitter/X) avec les caracteristiques suivantes :

**Chiffres :**
- 200M d'utilisateurs actifs
- Chaque utilisateur suit en moyenne 200 comptes
- 500K nouveaux posts/minute en pic
- Access pattern principal : "afficher les 50 derniers posts du feed d'un utilisateur"
- Le feed est personnalise (chaque user voit un feed different selon ses abonnements)
- SLA : latence < 200ms au p99

**Questions :**
1. Quel type de cache utiliser (in-process, Redis, CDN) ? Justifie pourquoi le CDN ne convient pas ici.
2. Propose une structure de donnees Redis pour stocker le feed d'un utilisateur. Quelle commande Redis pour "les 50 derniers posts" ?
3. Comment gerer la taille du feed en memoire ? (un user suit 200 comptes * 50 posts chacun = 10K posts potentiels)
4. Quand un nouveau post est publie, comment mettre a jour les feeds des followers ? (fanout-on-write vs fanout-on-read). Analyse les deux approches.
5. Comment gerer les "celebrities" (comptes avec 50M+ followers) sans surcharger Redis ?
6. Estime la memoire Redis totale necessaire (200M users * taille du feed cache)
7. Quelle strategie d'invalidation pour le feed ? TTL ? Event-driven ? Les deux ?

### Criteres de reussite
- [ ] Redis Sorted Set est propose pour le feed (ZADD avec timestamp comme score, ZREVRANGE pour les top 50)
- [ ] Le CDN est ecarte car le feed est personnalise (contenu different par user)
- [ ] Le fanout-on-write est identifie comme lent pour les celebrities (50M writes par post)
- [ ] Une solution hybride est proposee (fanout-on-write pour les users normaux, fanout-on-read pour les celebrities)
- [ ] L'estimation memoire est coherente (attendu : 1-10 To de Redis)
- [ ] L'invalidation combine event-driven (nouveau post) + TTL (filet de securite)

---

## Exercice 2 : Diagnostic et resolution d'un cache miss rate eleve

### Objectif
Analyser un probleme de performance cache et proposer des solutions concretes.

### Consigne
Ton equipe observe les metriques suivantes sur le cache Redis de production :

**Metriques actuelles :**
- Hit rate global : 45% (attendu : > 85%)
- Eviction rate : 12K evictions/sec
- Memoire Redis : 8 Go / 8 Go (100% utilise)
- TTL moyen des cles : 1 heure
- Nombre de cles : 15M
- Taille moyenne par cle : ~500 bytes
- Top 3 prefixes : `session:` (40%), `product:` (35%), `search:` (25%)
- Les cles `search:` ont un TTL de 1h mais un taux de re-acces < 5%

**Questions :**
1. Identifie la cause racine du hit rate bas (pas les symptomes)
2. Calcule la memoire ideale pour stocker 15M cles * 500 bytes avec l'overhead Redis
3. Propose 3 actions concretes pour ameliorer le hit rate, ordonnees par impact
4. Pour chaque action, estime l'amelioration attendue du hit rate
5. Faut-il augmenter la RAM Redis ou optimiser l'usage ? Justifie avec des chiffres
6. Propose un monitoring dashboard avec les 5 metriques les plus importantes pour le cache

### Criteres de reussite
- [ ] La cause racine est identifiee : cache sous-dimensionne + cles search: a faible re-acces qui gaspillent de la memoire
- [ ] Le calcul montre que 15M * 500B * 2.5 = ~18 Go necessaires vs 8 Go disponibles
- [ ] Les actions incluent : reduire le TTL des search: (ou les supprimer), augmenter la RAM, et/ou ajouter un L1 cache
- [ ] L'optimisation des search: est la premiere action (impact le plus grand pour le cout le plus faible)
- [ ] Les 5 metriques du dashboard incluent : hit rate, eviction rate, memory usage, latency p99, key count

---

## Exercice 3 : CDN strategy pour une application multi-region

### Objectif
Concevoir une strategie CDN complete pour une application deployee en multi-region.

### Consigne
Tu deployes une application SaaS de gestion de documents avec les contraintes suivantes :

**Architecture :**
- Frontend : React SPA (build = 5 Mo de JS/CSS/images)
- API : FastAPI deployee en 3 regions (US-East, EU-West, APAC)
- Stockage : S3 pour les documents (PDF, images), PostgreSQL pour les metadata
- Users : 60% EU, 30% US, 10% APAC

**Requirements :**
- Les assets statiques doivent etre charges en < 1 seconde dans toutes les regions
- Les documents PDF doivent etre accessibles en < 3 secondes
- L'API doit repondre en < 200ms (p95) dans toutes les regions
- Les documents sont confidentiels (acces authentifie uniquement)
- Un document modifie doit etre visible dans sa nouvelle version en < 5 minutes

**Questions :**
1. Propose une architecture CDN complete (quel provider, combien de niveaux, quels cache headers pour chaque type de contenu)
2. Comment securiser l'acces aux documents sur le CDN ? (les documents sont confidentiels)
3. Comment gerer l'invalidation quand un document est modifie ? (< 5 min)
4. Propose les cache headers pour chacun de ces 4 types de contenu :
   - Assets statiques React (`app.hash.js`)
   - API responses (`/api/documents/list`)
   - Documents PDF (`/documents/{id}/download`)
   - HTML index (`index.html`)
5. Comment mesurer le hit rate du CDN et savoir si la strategie est efficace ?
6. Estime le cout mensuel CDN (CloudFront pricing : $0.085/Go pour les premiers 10 To)

### Criteres de reussite
- [ ] Le CDN est configure differemment pour les assets (cache long), l'API (cache court ou pas de cache), et les documents (cache signe)
- [ ] Les signed URLs ou signed cookies sont proposes pour securiser les documents
- [ ] L'invalidation < 5 min est geree par purge API + TTL court (s-maxage=300)
- [ ] index.html utilise `no-cache` (toujours revalider pour avoir le dernier build)
- [ ] Le monitoring inclut hit rate par type de contenu, latence edge, bandwidth savings
- [ ] L'estimation de cout est realiste (quelques centaines de dollars/mois pour un SaaS moyen)
