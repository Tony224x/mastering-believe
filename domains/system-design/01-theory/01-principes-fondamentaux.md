# Jour 1 — Principes fondamentaux du System Design

## Pourquoi le system design est un skill distinct du coding

**Exemple d'abord** : Tu sais ecrire un endpoint FastAPI qui renvoie une liste d'utilisateurs. Maintenant, cet endpoint doit servir 50 000 requetes/seconde, rester disponible si un datacenter tombe, et repondre en < 100ms au p99. Le code ne change presque pas — c'est l'architecture autour qui change tout.

Le system design, c'est l'art de **composer des composants** (DB, cache, queues, services) pour satisfaire des **contraintes non-fonctionnelles** : performance, disponibilite, consistance, cout.

**Key takeaway** : Le coding resout "comment implementer X". Le system design resout "comment X fonctionne a l'echelle, de maniere fiable, pour des millions d'utilisateurs".

---

## Scalabilite horizontale vs verticale

### Verticale (Scale Up)

Ajouter des ressources a une seule machine : plus de RAM, CPU plus rapide, SSD plus gros.

**Exemple concret** : Ta base PostgreSQL rame avec 10M de lignes. Tu passes de 16 Go a 64 Go de RAM — les index tiennent en memoire, les requetes sont 10x plus rapides.

| Avantage | Inconvenient |
|---|---|
| Simple (pas de code a changer) | Plafond physique (la plus grosse machine AWS = ~24 To RAM) |
| Pas de complexite de distribution | Single point of failure |
| Transactions ACID faciles | Cout exponentiel (2x RAM =/= 2x prix) |

### Horizontale (Scale Out)

Ajouter plus de machines qui se partagent la charge.

**Exemple concret** : Ton API FastAPI sature a 5000 req/s sur une machine. Tu deploies 10 instances derriere un load balancer — chacune gere 500 req/s.

| Avantage | Inconvenient |
|---|---|
| Pas de plafond theorique | Complexite de distribution |
| Resilience (1 noeud tombe, les autres tiennent) | Consistance des donnees plus dure |
| Cout lineaire | Latence reseau entre noeuds |

### Quand choisir quoi ?

| Situation | Choix | Pourquoi |
|---|---|---|
| DB relationnelle < 1 To, < 10K QPS | **Vertical** | Moins de complexite, ACID natif |
| API stateless, charge variable | **Horizontal** | Auto-scaling, pas d'etat partage |
| Cache (Redis) | **Vertical d'abord, puis horizontal** (Redis Cluster) | Vertical simple jusqu'a ~100 Go |
| ML inference | **Horizontal** | Chaque GPU sert des requetes independamment |

**Key takeaway** : Commence vertical (plus simple), passe horizontal quand tu atteins le plafond ou quand tu as besoin de resilience.

---

## CAP Theorem

### L'intuition

Imagine 3 serveurs de base de donnees qui repliquent les memes donnees. Un cable reseau entre 2 d'entre eux est coupe (partition reseau).

Tu as **un choix** :
- **Consistency** : les 3 serveurs refusent de repondre tant qu'ils ne sont pas d'accord (donnees identiques partout)
- **Availability** : chaque serveur repond avec ce qu'il a, meme si les donnees sont potentiellement desynchronisees

Tu ne peux pas avoir les deux pendant une partition. C'est le **CAP theorem**.

### Les 3 proprietes

| Propriete | Definition simple |
|---|---|
| **C**onsistency | Tous les noeuds voient les memes donnees au meme moment |
| **A**vailability | Chaque requete recoit une reponse (pas d'erreur) |
| **P**artition tolerance | Le systeme continue de fonctionner malgre des pertes de messages entre noeuds |

### Les 3 combinaisons

> **Important** : En systeme distribue, les partitions reseau ARRIVENT. P n'est pas optionnel. Le vrai choix est entre **CP** et **AP**.

#### CP — Consistency + Partition tolerance

Le systeme refuse de repondre s'il ne peut pas garantir la consistance.

**Exemples reels** :
- **ZooKeeper** : coordination de cluster, les locks doivent etre exacts
- **etcd** : stockage de config Kubernetes, une mauvaise valeur = catastrophe
- **MongoDB** (config par defaut avec write concern majority)
- **Systeme bancaire** : un solde incorrect = perte financiere

**Analogie** : Un notaire qui refuse de signer un acte tant qu'il n'a pas TOUS les documents — plus lent, mais garanti correct.

#### AP — Availability + Partition tolerance

Le systeme repond toujours, quitte a renvoyer des donnees potentiellement obsoletes.

**Exemples reels** :
- **Cassandra** : reseau social, un like en retard de 2 secondes n'est pas grave
- **DynamoDB** (eventual consistency mode)
- **DNS** : un enregistrement perime pendant quelques minutes est tolerable
- **Cache CDN** : une page legrement outdated vaut mieux que pas de page

**Analogie** : Un journal qui publie avec les infos disponibles — parfois une correction le lendemain, mais les gens ont l'info.

#### CA — Consistency + Availability (sans Partition tolerance)

Possible uniquement sur **un seul noeud** (pas de reseau = pas de partition).

**Exemples reels** :
- **PostgreSQL single node** : ACID complet, toujours disponible, mais si le serveur tombe, c'est fini
- **SQLite** : meme logique

**Key takeaway** : En distribue, le vrai choix est CP vs AP. Pose-toi la question : "Vaut-il mieux une reponse fausse ou pas de reponse ?"

---

## Latence vs Throughput

### Definitions

| Concept | Definition | Unite |
|---|---|---|
| **Latence** | Temps pour traiter UNE requete | ms, us |
| **Throughput** | Nombre de requetes traitees par unite de temps | req/s, MB/s |

### Relation

Ce ne sont **pas** des inverses. Un systeme peut avoir une latence haute ET un throughput haut (pipeline, batching).

**Exemple concret** : Un avion met 12h pour traverser l'Atlantique (latence haute) mais transporte 400 personnes (throughput enorme). Un jet prive met 8h (latence basse) mais transporte 10 personnes (throughput faible).

### Comment mesurer

| Metrique | Signification |
|---|---|
| **p50 (median)** | 50% des requetes sont plus rapides |
| **p95** | 95% des requetes sont plus rapides — la vraie experience utilisateur |
| **p99** | 99% des requetes sont plus rapides — les cas limites |
| **p99.9** | Les cas extremes (souvent lies au GC, cold start, etc.) |

> **Regle** : Optimise pour le p99, pas la moyenne. La moyenne masque les outliers.

### Ordres de grandeur a connaitre par coeur

| Operation | Latence approximative |
|---|---|
| L1 cache reference | **1 ns** |
| L2 cache reference | **4 ns** |
| Branch mispredict | **3 ns** |
| RAM (main memory) reference | **100 ns** |
| Compress 1 Ko (Snappy) | **3 us** |
| Read 1 MB sequentially from RAM | **3 us** |
| SSD random read | **16 us** |
| Read 1 MB sequentially from SSD | **49 us** |
| Round trip dans le meme datacenter | **500 us** |
| Read 1 MB sequentially from disk (HDD) | **825 us** |
| Disk seek (HDD) | **2 ms** |
| Read 1 MB sequentially from network (1 Gbps) | **10 ms** |
| Round trip US coast to coast | **40 ms** |
| Round trip Europe - US | **80 ms** |
| Round trip Europe - Asie | **150 ms** |

> **Regle du pouce** : RAM est ~100x plus rapide que SSD, SSD est ~100x plus rapide que le reseau long-distance.

**Key takeaway** : Un cache en memoire evite un round-trip reseau. La difference est de 100 ns vs 500 us — un facteur 5000.

---

## Consistency Models

### Strong Consistency

Apres une ecriture, TOUTE lecture renvoie la valeur ecrite.

**Exemple** : Virement bancaire. Apres le debit de 100 EUR, le solde affiche doit etre a jour immediatement, sur tous les canaux (app, web, ATM).

**Cout** : Latence plus elevee (il faut synchroniser tous les noeuds avant de confirmer).

### Eventual Consistency

Apres une ecriture, les lectures FINIRONT par renvoyer la nouvelle valeur, mais pas immediatement.

**Exemple** : Tu postes un tweet. Pendant 2-3 secondes, certains followers ne le voient pas encore. C'est acceptable.

**Avantage** : Tres faible latence, haute disponibilite.

### Causal Consistency

Les operations causalement liees sont vues dans le bon ordre. Les operations independantes peuvent etre vues dans n'importe quel ordre.

**Exemple** : Sur un forum, si Alice poste un message puis Bob repond, TOUT LE MONDE voit le message d'Alice avant la reponse de Bob. Mais deux messages independants peuvent apparaitre dans un ordre different selon les lecteurs.

**Compromis** : Entre strong et eventual — plus de garanties que eventual, moins de latence que strong.

| Modele | Garantie | Latence | Use case type |
|---|---|---|---|
| Strong | Toujours a jour | Haute | Banque, inventaire, reservations |
| Causal | Ordre causal respecte | Moyenne | Social feeds, messaging |
| Eventual | Finira par converger | Faible | Likes, vues, analytics |

**Key takeaway** : Le bon modele depend du domaine metier. Un compte bancaire exige strong. Un compteur de likes tolere eventual.

---

## SLA / SLO / SLI

### Definitions

| Terme | Definition | Exemple |
|---|---|---|
| **SLI** (Service Level Indicator) | La **metrique** mesuree | Latence p99, taux d'erreur, uptime |
| **SLO** (Service Level Objective) | L'**objectif interne** pour cette metrique | p99 < 200ms, erreurs < 0.1% |
| **SLA** (Service Level Agreement) | Le **contrat** avec le client (avec penalites) | 99.9% uptime, sinon credit |

**Analogie** : SLI = le thermometre, SLO = "je veux garder la temperature sous 38", SLA = "si ca depasse 38, j'appelle le medecin".

### Les "nines" — Uptime et downtime autorise

| Uptime | Downtime/an | Downtime/mois | Downtime/jour |
|---|---|---|---|
| 99% (deux nines) | 3.65 jours | 7.3 heures | 14.4 min |
| 99.9% (trois nines) | 8.76 heures | 43.8 min | 1.44 min |
| 99.95% | 4.38 heures | 21.9 min | 43.2 sec |
| 99.99% (quatre nines) | 52.6 min | 4.38 min | 8.64 sec |
| 99.999% (cinq nines) | 5.26 min | 26.3 sec | 0.864 sec |

> **Realite** : Passer de 99.9% a 99.99% coute souvent **10x plus cher** en infrastructure. Chaque "nine" supplementaire est exponentiellement plus difficile.

### Comment definir un SLO

1. Mesure tes SLIs actuels pendant 1 mois
2. Fixe un objectif legerement en dessous de la performance observee
3. Definis un "error budget" : si tu es a 99.95% et ton SLO est 99.9%, tu as un budget de 0.05% a "depenser" pour deployer des features

**Key takeaway** : Ne vise pas 99.99% si tu n'en as pas besoin. Chaque nine supplementaire est un multiplicateur de cout et de complexite.

---

## Back-of-the-envelope estimation

### Pourquoi c'est crucial

En entretien, on te demande : "Combien de serveurs pour supporter 1M d'utilisateurs actifs ?" Si tu ne sais pas estimer, tu ne peux pas designer.

### Puissances de 2 a connaitre

| Puissance | Valeur exacte | Approximation |
|---|---|---|
| 2^10 | 1 024 | ~1 000 (1 Ko) |
| 2^20 | 1 048 576 | ~1 million (1 Mo) |
| 2^30 | 1 073 741 824 | ~1 milliard (1 Go) |
| 2^40 | | ~1 trillion (1 To) |

### Chiffres de reference

| Donnee | Valeur |
|---|---|
| Taille d'un caractere (UTF-8 ASCII) | 1 octet |
| Taille d'un int | 4 octets |
| Taille d'un long/timestamp | 8 octets |
| Taille d'un UUID | 16 octets |
| Taille d'un tweet (~280 chars + metadata) | ~1 Ko |
| Taille d'une image compressée (JPEG) | ~300 Ko |
| Taille d'une minute de video HD | ~100 Mo |
| Taille d'une heure de video HD | ~6 Go |
| Secondes dans un jour | 86 400 (~100 000) |
| Secondes dans un mois | ~2.5 millions |
| Secondes dans un an | ~31.5 millions (~30M) |

### Framework d'estimation

1. **Estimer les utilisateurs** : DAU, pic vs moyenne (pic = 2-5x moyenne)
2. **Estimer le QPS** : DAU x actions/jour / 86400 secondes
3. **Estimer le stockage** : taille d'un objet x nombre d'objets x retention
4. **Estimer la bande passante** : QPS x taille moyenne de reponse

**Exemple — Estimer le stockage pour un service type Twitter** :
- 500M tweets/jour
- ~1 Ko/tweet (texte + metadata)
- Stockage/jour = 500M x 1 Ko = 500 Go/jour
- Stockage/an = 500 Go x 365 = ~180 To/an
- Avec 20% d'images (300 Ko chacune) : 100M x 300 Ko = 30 To/jour = ~11 Po/an

**Key takeaway** : Arrondis agressivement. Le but n'est pas la precision, c'est l'ordre de grandeur.

---

## Flash Cards — Q&A

### Q1
**Q** : Tu dois choisir entre CP et AP pour un systeme de reservation de billets d'avion. Lequel et pourquoi ?

**R** : **CP**. Une double reservation du meme siege est inacceptable (perte financiere, probleme legal). Mieux vaut refuser temporairement une requete que vendre deux fois le meme siege.

---

### Q2
**Q** : Quelle est la difference entre latence p50 et p99 ? Laquelle optimiser en priorite ?

**R** : p50 = experience de la moitie des utilisateurs. p99 = experience du 1% le plus lent. **Optimiser le p99** car ces utilisateurs sont souvent les plus actifs (requetes complexes, gros comptes) et le p99 revele les problemes systemiques (GC pauses, cold starts, contention).

---

### Q3
**Q** : Quelle est la difference entre SLO et SLA ?

**R** : Le SLO est un **objectif interne** (on veut p99 < 200ms). Le SLA est un **contrat externe** avec le client (si uptime < 99.9%, on rembourse 10%). Le SLO est generalement plus strict que le SLA pour avoir une marge de securite.

---

### Q4
**Q** : Estime le QPS pour un service avec 10M DAU ou chaque utilisateur fait en moyenne 20 requetes/jour.

**R** : QPS moyen = 10M x 20 / 86400 = 200M / 86400 ≈ **2300 req/s**. QPS pic (x3) ≈ **7000 req/s**.

---

### Q5
**Q** : Pourquoi eventual consistency est-elle plus performante que strong consistency ?

**R** : En eventual consistency, une ecriture est confirmee des qu'UN seul noeud l'a enregistree (pas besoin d'attendre la replication). Les lectures ne bloquent pas pour verifier la version. Resultat : latence d'ecriture et de lecture beaucoup plus faibles, mais les donnees peuvent etre temporairement incoherentes entre noeuds.

---

## Pour aller plus loin

Ressources canoniques sur le sujet :

- **MIT 6.824 — Distributed Systems** (Robert Morris, MIT) — cours de reference sur les systemes distribues. Lecture 1 (Introduction) pose CAP, fault tolerance, consistency. Playlist Spring 2020 sur YouTube. https://www.youtube.com/playlist?list=PLrw6a1wE39_tb2fErI4-WkMbsvGQk9_UB
- **Designing Data-Intensive Applications** (Martin Kleppmann, O'Reilly 2017) — Ch 1 (Reliable, Scalable, Maintainable) et Ch 2 (Data Models) sont la fondation incontournable. https://dataintensive.net/
- **Google SRE Book** (Beyer et al., O'Reilly, libre en ligne) — Ch 3 "Embracing Risk" et Ch 4 "Service Level Objectives" pour quantifier disponibilite et latence. https://sre.google/sre-book/table-of-contents/
- **System Design Interview Vol 1** (Alex Xu, ByteByteGo 2020) — Ch 1 "Scale from zero to millions of users" parcourt scaling vertical/horizontal/replication etape par etape. https://www.amazon.com/System-Design-Interview-insiders-Second/dp/B08CMF2CQF
