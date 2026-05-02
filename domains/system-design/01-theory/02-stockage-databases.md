# Jour 2 — Stockage & Databases

## Le stockage comme fondation de tout system design

**Exemple d'abord** : Tu concois un systeme de messagerie. Le premier reflexe est de penser aux websockets, aux notifications push, au scaling. Mais la vraie question fondamentale : **ou et comment stocker les messages ?** Est-ce que les conversations sont interrogeables ? Les messages sont-ils immutables ? Faut-il une retention de 7 jours ou 7 ans ? Chaque reponse change radicalement l'architecture.

Chaque composant d'un systeme (cache, queue, search engine) est au fond **un systeme de stockage specialise**. Redis est un store cle-valeur en memoire. Kafka est un log append-only distribue. Elasticsearch est un index inverse persistant. Comprendre les fondamentaux du stockage, c'est comprendre TOUS ces outils.

**Key takeaway** : Le choix de la base de donnees n'est pas une decision technique isolee. C'est la decision qui contraint tout le reste : schema, queries, scaling, consistency, et meme la structure de ton equipe (cf. Conway's Law).

---

## SQL vs NoSQL — Les vrais tradeoffs

### Ce qu'on te dit (et qui est reducteur)

"SQL = donnees structurees, NoSQL = donnees non structurees." C'est une simplification trompeuse. MongoDB a des schemas (meme optionnels). PostgreSQL gere le JSON natif (JSONB). La vraie question est ailleurs.

### Les axes de decision reels

#### 1. Modele de consistance : ACID vs BASE

| Propriete | ACID (SQL) | BASE (NoSQL) |
|---|---|---|
| **A**tomicity / **B**asically Available | Transaction tout-ou-rien | Le systeme repond toujours, meme degrade |
| **C**onsistency / **S**oft state | Etat toujours valide apres une transaction | L'etat peut etre temporairement incoherent |
| **I**solation / **E**ventual consistency | Les transactions concurrentes ne se voient pas | Les donnees convergent a terme |
| **D**urability | Donnee ecrite = donnee persistee | Idem (la plupart des NoSQL aussi) |

**Quand ACID est non-negociable** : transferts financiers, gestion de stock, reservations.
**Quand BASE suffit** : compteurs, feeds sociaux, analytics, sessions.

#### 2. Schema flexibility

| | SQL | NoSQL (Document) |
|---|---|---|
| Schema | Defini a l'avance (DDL) | Schema-on-read (flexible) |
| Migration | `ALTER TABLE` (potentiellement long et bloquant) | Ajouter un champ = juste l'ecrire |
| Avantage | Le schema protege contre les donnees invalides | Iteration rapide, attributs variables |
| Inconvenient | Rigide, migrations complexes a l'echelle | Schema chaos si pas de discipline |

**Exemple concret** : Un catalogue produit ou chaque categorie a des attributs differents (vetements : taille/couleur, electronique : voltage/connectique). MongoDB gere ca nativement. En SQL, tu fais du EAV (Entity-Attribute-Value) ou du JSONB — les deux ont des couts.

#### 3. Join performance

| | SQL | NoSQL |
|---|---|---|
| Jointures | Natives, optimisees par le query planner | Generalement inexistantes ou couteuses |
| Denormalisation | Evitee (3NF) | Encouragee (embed plutot que reference) |
| Consequence | 1 source de verite, queries flexibles | Donnees dupliquees, queries pre-definies |

**Regle** : Si ton modele de donnees est fortement relationnel (commandes -> lignes -> produits -> categories), SQL est naturel. Si tes donnees sont des agregats independants (profils utilisateur, documents), NoSQL est naturel.

#### 4. Scaling model

| | SQL | NoSQL |
|---|---|---|
| Scaling vertical | Excellent (une grosse machine) | Aussi |
| Scaling horizontal | Complexe (sharding manuel, Citus, Vitess) | Natif (sharding automatique) |
| Replication | Leader-follower (read replicas) | Multi-leader, leaderless (Cassandra, DynamoDB) |
| Write scaling | Difficile sans sharding | Distribue nativement |

**Key takeaway** : SQL scale verticalement tres bien jusqu'a ~10 To et ~50K QPS. Au-dela, le scaling horizontal NoSQL natif est un avantage reel, pas juste theorique.

### Tableau comparatif final

| Critere | SQL (PostgreSQL, MySQL) | NoSQL (MongoDB, Cassandra, DynamoDB) |
|---|---|---|
| Consistance | Strong (ACID) | Configurable (eventual par defaut) |
| Schema | Rigide, protege | Flexible, risque de chaos |
| Jointures | Natives, performantes | Couteuses ou impossibles |
| Scaling writes | Complexe | Natif |
| Transactions complexes | Excellent | Limite (single-document ou 25 items max) |
| Queries ad hoc | Flexible (SQL complet) | Limite aux access patterns prevus |
| Latence | ms (avec index) | Sub-ms (key-value), ms (document) |
| Courbe d'apprentissage | Mature, bien documentee | Variable selon le type |

---

## Types de NoSQL

### Key-Value Store

**Principe** : Table de hachage geante. `GET key` / `SET key value`. Rien de plus simple.

**Exemples** : Redis, DynamoDB (mode simple), Memcached

**Quand l'utiliser** :
- **Cache** : session utilisateur, resultats de requetes couteuses, rate limiting
- **Compteurs** : vues, likes, quotas API (Redis INCR atomique)
- **Feature flags** : activation/desactivation de fonctionnalites en temps reel

**Quand NE PAS l'utiliser** : Quand tu as besoin de requeter par autre chose que la cle. Pas de `WHERE age > 25`.

**Chiffres** :
- Redis : ~100K-1M ops/sec par noeud, latence < 1ms
- DynamoDB : single-digit ms, auto-scaling illimite

**Exemple concret** : Un rate limiter API. Cle = `user_id:minute_bucket`, valeur = compteur. `INCR` + `EXPIRE` 60s. Simple, rapide, efficace.

### Document Store

**Principe** : Stocke des documents JSON/BSON. Chaque document est un agregat autonome avec sa propre structure.

**Exemples** : MongoDB, CouchDB, Firestore

**Quand l'utiliser** :
- **Catalogues** avec attributs variables par categorie
- **CMS** : articles, pages, contenus semi-structures
- **Profils utilisateur** enrichis (preferences, historique, metadata)
- **Prototypage** rapide quand le schema evolue vite

**Quand NE PAS l'utiliser** : Relations complexes entre entites (commandes/lignes/produits), transactions multi-documents frequentes.

**Chiffres** :
- MongoDB : ~20K-80K ops/sec par noeud (depend des queries)
- Latence : 1-10ms pour les lookups indexes

**Exemple concret** : Un catalogue e-commerce. Un document produit contient tout : nom, prix, description, images, attributs specifiques a la categorie. Pas de jointure necessaire pour afficher une fiche produit.

### Column-Family Store

**Principe** : Les donnees sont stockees par colonnes, pas par lignes. Optimise pour ecrire et lire des sous-ensembles de colonnes sur des volumes massifs.

**Exemples** : Cassandra, HBase, ScyllaDB

**Quand l'utiliser** :
- **Time-series** : metriques IoT, logs, evenements (millions d'ecritures/sec)
- **Messaging** : boites de reception, historiques de conversation
- **Ecritures massives** : tout systeme ou le write throughput est critique

**Quand NE PAS l'utiliser** : Queries ad hoc, aggregations complexes, donnees fortement relationnelles. Cassandra ne supporte pas les jointures et les queries doivent correspondre exactement a la partition key.

**Chiffres** :
- Cassandra : ~10K-100K writes/sec par noeud, scaling lineaire
- ScyllaDB : ~1M ops/sec par noeud (rewrite C++ de Cassandra)

**Exemple concret** : L'historique de messages d'une app de chat. Partition key = `conversation_id`, clustering key = `timestamp`. Cassandra ecrit a une vitesse folle et tu recuperes les messages tries par date sans effort.

### Graph Database

**Principe** : Stocke des noeuds (entites) et des aretes (relations). Optimise pour traverser des relations.

**Exemples** : Neo4j, Amazon Neptune, JanusGraph

**Quand l'utiliser** :
- **Reseaux sociaux** : "amis d'amis", suggestions de connexion
- **Moteur de recommandation** : "les utilisateurs qui ont aime X ont aussi aime Y"
- **Detection de fraude** : patterns de transactions suspectes entre comptes
- **Knowledge graphs** : ontologies, relations semantiques

**Quand NE PAS l'utiliser** : Donnees tabulaires simples, analytics de masse, donnees sans relations complexes.

**Chiffres** :
- Neo4j : traversals multi-niveaux en ms (la ou SQL ferait des jointures en secondes)
- Plus le graph est profond, plus l'avantage est grand

**Exemple concret** : Detection de fraude bancaire. Tu cherches "Existe-t-il un chemin de 3+ transferts entre le compte A et le compte B en moins de 24h ?" En SQL, c'est un cauchemar de jointures recursives. En Neo4j, c'est une query Cypher de 3 lignes.

### Resume : quel NoSQL pour quel use case

| Type | Exemples | Force | Use case type |
|---|---|---|---|
| Key-Value | Redis, DynamoDB | Latence sub-ms, simplicite | Cache, sessions, compteurs |
| Document | MongoDB, Firestore | Schema flexible, agregats autonomes | Catalogues, CMS, profils |
| Column-Family | Cassandra, ScyllaDB | Write throughput massif, time-series | IoT, logs, messaging |
| Graph | Neo4j, Neptune | Traversal de relations profondes | Social, fraude, recommendations |

---

## Indexing — Comment un index accelere les reads

### L'analogie de l'annuaire

Imagine un annuaire telephonique de 500 pages. Sans index (pas d'ordre alphabetique), trouver "Dupont" oblige a lire chaque page -> **full table scan**. Avec l'index alphabetique, tu ouvres directement a la lettre D -> **index lookup**. Passer de O(n) a O(log n).

### B-Tree Index (le standard SQL)

**Structure** : Arbre equilibre ou chaque noeud a plusieurs enfants. Les feuilles contiennent les pointeurs vers les lignes.

**Comment ca marche** :
1. L'index est trie. Pour trouver `id=42`, on parcourt l'arbre depuis la racine.
2. A chaque niveau, on choisit la branche correcte (comme une recherche dichotomique).
3. Profondeur typique = 3-4 niveaux pour des millions de lignes.

**Performance** :
- Lecture : O(log n) — ~3-4 acces disque pour 10M de lignes
- Ecriture : O(log n) — l'arbre doit etre maintenu equilibre
- Range queries : excellentes (les feuilles sont chainees)

**Utilisation** : PostgreSQL, MySQL, Oracle — index par defaut.

### LSM-Tree (Log-Structured Merge-Tree)

**Structure** : Les ecritures vont dans un memtable (en memoire), puis sont flushees en SSTables (fichiers tries sur disque). Des compactions periodiques fusionnent les SSTables.

**Comment ca marche** :
1. Ecriture : append dans le memtable (O(1) en memoire)
2. Quand le memtable est plein, il est flushe sur disque comme une SSTable triee
3. Lecture : chercher dans le memtable, puis dans les SSTables recentes, puis les anciennes
4. Compaction : fusionner les SSTables pour reduire les reads

**Performance** :
- Ecriture : O(1) amorti — beaucoup plus rapide que B-Tree pour les writes
- Lecture : potentiellement O(k * log n) ou k = nombre de niveaux de SSTables
- Range queries : bonnes (les SSTables sont triees)

**Utilisation** : Cassandra, RocksDB, LevelDB, HBase.

**Tradeoff cle** : LSM-Tree favorise les ecritures au detriment des lectures. B-Tree equilibre les deux.

### Hash Index

**Structure** : Table de hachage en memoire. Chaque cle est hashee vers un offset dans un fichier sur disque.

**Performance** :
- Lookup exact : O(1)
- Range queries : impossibles (pas d'ordre)

**Utilisation** : Redis (en memoire), Bitcask (Riak).

**Quand l'utiliser** : Uniquement pour des lookups par cle exacte ou le dataset tient en memoire.

### Comparaison des structures d'index

| Structure | Read | Write | Range Query | Use case |
|---|---|---|---|---|
| B-Tree | O(log n) | O(log n) | Excellent | DB relationnelles (defaut) |
| LSM-Tree | O(k log n) | O(1) amorti | Bon | Write-heavy (Cassandra, RocksDB) |
| Hash Index | O(1) | O(1) | Impossible | Key-value en memoire |

### Composite Indexes (index multi-colonnes)

Un index sur `(country, city)` permet de repondre a :
- `WHERE country = 'FR'` -> utilise l'index
- `WHERE country = 'FR' AND city = 'Paris'` -> utilise l'index completement
- `WHERE city = 'Paris'` -> **N'utilise PAS l'index** (le prefix n'est pas present)

**Regle du leftmost prefix** : un composite index n'est utile que si la query commence par la/les premiere(s) colonne(s) de l'index.

### Covering Indexes

Un covering index inclut TOUTES les colonnes de la query dans l'index. Le moteur SQL peut repondre uniquement a partir de l'index, sans aller lire la table (index-only scan).

```sql
-- Index: (user_id, created_at) INCLUDE (status)
SELECT status FROM orders WHERE user_id = 42 ORDER BY created_at DESC;
-- -> Index-only scan : pas besoin d'acceder a la table
```

### Quand NE PAS indexer

1. **Tables petites** (< 10K lignes) : le full scan est plus rapide que l'overhead de l'index
2. **Colonnes a faible cardinalite** : indexer `gender` (M/F) est inutile — l'index ne filtre que 50%
3. **Tables write-heavy** : chaque INSERT/UPDATE doit aussi mettre a jour l'index. Si tu as 50K writes/sec et peu de reads, les index ralentissent le systeme
4. **Colonnes rarement filtrees** : un index non utilise consomme du stockage et ralentit les ecritures pour rien

**Key takeaway** : Un index n'est pas gratuit. Il accelere les reads mais ralentit les writes. Chaque index occupe de l'espace disque et doit etre maintenu. Indexer tout est une anti-pattern.

---

## Sharding — Horizontal Partitioning

### Le probleme

Ta base PostgreSQL fait 5 To. Les queries sont lentes. Tu as deja scale verticalement au maximum (512 Go RAM, 96 vCPUs). Il faut distribuer les donnees sur plusieurs machines. C'est le **sharding** (ou horizontal partitioning).

### Strategies de sharding

#### 1. Range-based Sharding

Les donnees sont reparties par plages de la shard key.

| Shard | Plage |
|---|---|
| Shard 1 | user_id 1 — 1M |
| Shard 2 | user_id 1M — 2M |
| Shard 3 | user_id 2M — 3M |

**Avantage** : Range queries efficaces (`WHERE user_id BETWEEN 1M AND 1.5M` -> une seule shard).

**Inconvenient** : **Hot spots**. Si les users recents sont les plus actifs, Shard 3 recoit tout le trafic. Distribution tres inegale.

**Quand l'utiliser** : Donnees avec un acces temporel (time-series : chaque mois = une shard).

#### 2. Hash-based Sharding

On hash la shard key et on prend le modulo : `shard = hash(key) % num_shards`.

**Avantage** : Distribution uniforme (le hash disperse les cles).

**Inconvenient** : Les range queries sont impossibles (les cles consecutives sont sur des shards differentes). Et si tu ajoutes une shard, **TOUTES les donnees doivent etre redistribuees** (rehashing catastrophique).

**Quand l'utiliser** : Access pattern uniquement par cle exacte (user lookups).

#### 3. Consistent Hashing

Resout le probleme du rehashing. Les shards et les cles sont placees sur un anneau (ring) de 0 a 2^32. Chaque cle est assignee a la prochaine shard dans le sens horaire.

**Avantage** : Ajouter/supprimer une shard ne deplace que ~1/N des cles (au lieu de toutes).

**Virtual nodes** : Chaque shard physique a plusieurs positions sur l'anneau pour eviter les desequilibres.

**Utilisation** : DynamoDB, Cassandra, systemes de cache distribue.

#### 4. Directory-based Sharding

Un service central (directory) maintient un mapping explicite `key -> shard`.

**Avantage** : Flexibilite totale. Tu peux deplacer un gros client sur sa propre shard sans toucher au reste.

**Inconvenient** : Le directory est un single point of failure et un bottleneck.

**Quand l'utiliser** : Multi-tenant SaaS ou certains tenants sont beaucoup plus gros que d'autres.

### Hot Spots

Meme avec du hash sharding, des hot spots peuvent apparaitre si un petit nombre de cles recoit un trafic disproportionne.

**Exemple** : Un tweet viral de Beyonce. La cle `user_id=beyonce` recoit 1M de lectures/sec. La shard qui la contient est surchargee.

**Solutions** :
1. **Key splitting** : ajouter un suffixe aleatoire (`beyonce_1`, `beyonce_2`, ... `beyonce_10`) et distribuer les reads sur 10 shards
2. **Read replicas** par shard : multiplier les lecteurs
3. **Cache** devant la shard chaude

### Resharding — La douleur

Ajouter des shards a un systeme en production est l'une des operations les plus complexes :
1. Les donnees doivent etre rebalancees pendant que le systeme tourne
2. Les ecritures continuent pendant la migration -> il faut un mecanisme de dual-write ou de rattrapage
3. Les clients doivent etre re-routes vers les nouvelles shards

**Key takeaway** : Concois ton sharding pour eviter le resharding. Choisis une shard key stable et prevois largement (start with 64 shards meme si 4 suffisent aujourd'hui).

---

## Replication

### Pourquoi repliquer

1. **Haute disponibilite** : si un noeud tombe, les replicas prennent le relais
2. **Scalabilite en lecture** : distribuer les reads sur plusieurs replicas
3. **Localite geographique** : un replica en Europe, un en Asie = latence reduite

### Single-leader Replication

**Principe** : Un seul noeud (leader) accepte les ecritures. Les followers recoivent les ecritures en repliquant le Write-Ahead Log (WAL) du leader.

```
Client (write) -> Leader -> WAL -> Follower 1
                                -> Follower 2
Client (read) -> Follower 1 (ou 2, ou leader)
```

**Sync vs Async** :
- **Synchrone** : le leader attend que le follower confirme avant d'ACK le client. Garantie de durabilite mais latence plus elevee.
- **Asynchrone** : le leader ACK immediatement. Le follower rattrape plus tard. Risque de perte si le leader crash avant la replication.
- **Semi-synchrone** : 1 follower sync + les autres async. Compromis courant (PostgreSQL `synchronous_commit`).

**Probleme majeur** : Replication lag. Le follower peut avoir quelques ms a quelques secondes de retard. Si tu lis depuis un follower juste apres une ecriture sur le leader, tu peux lire des donnees obsoletes.

**Solution** : Read-your-writes consistency. Apres une ecriture, forcer les reads de cet utilisateur vers le leader pendant quelques secondes.

**Utilisation** : PostgreSQL, MySQL, MongoDB (par defaut).

### Multi-leader Replication

**Principe** : Plusieurs noeuds acceptent les ecritures. Chaque leader replique vers les autres.

**Use case principal** : Multi-datacenter. Un leader par datacenter evite les latences cross-continent pour les ecritures.

**Le cauchemar** : Les conflits. Deux leaders ecrivent des valeurs differentes pour la meme cle en meme temps.

**Strategies de resolution de conflit** :
1. **Last-write-wins (LWW)** : le timestamp le plus recent gagne. Simple mais perte de donnees.
2. **Merge** : combiner les ecritures (pour les CRDTs — Conflict-free Replicated Data Types).
3. **Application-level** : presenter le conflit a l'utilisateur (comme Git merge conflicts).

**Utilisation** : CouchDB, Dynamo-style systems. Rarement utilise sauf multi-DC.

### Leaderless Replication

**Principe** : Pas de leader. Chaque noeud accepte les reads ET les writes. Le client ecrit sur W noeuds et lit depuis R noeuds.

**Quorum** : Pour garantir la consistance, il faut `W + R > N` (N = nombre total de replicas).

| Config | W | R | N | Propriete |
|---|---|---|---|---|
| Strong consistency | 2 | 2 | 3 | W + R = 4 > 3 |
| Write-optimized | 1 | 3 | 3 | Writes rapides, reads lents |
| Read-optimized | 3 | 1 | 3 | Reads rapides, writes lents |

**Quand W + R <= N** : eventual consistency. Un read peut retourner une valeur obsolete.

**Utilisation** : Cassandra, DynamoDB, Riak.

**Key takeaway** : Le choix du modele de replication depend du ratio read/write et de la tolerance aux donnees obsoletes. Single-leader = simple et correct. Leaderless = performant mais complexe.

---

## Partitioning — Horizontal vs Vertical

### Horizontal Partitioning (Sharding)

Diviser les **lignes** sur plusieurs machines. Chaque partition a le meme schema mais un sous-ensemble des donnees.

| Partition | Donnees |
|---|---|
| Partition 1 | Users 1-1M |
| Partition 2 | Users 1M-2M |

**C'est le sharding** vu plus haut. Toute la section "Sharding" s'applique ici.

### Vertical Partitioning

Diviser les **colonnes** sur plusieurs tables/services.

| Table principale | Table details |
|---|---|
| user_id, name, email | user_id, bio, avatar_url, preferences_json |

**Quand l'utiliser** :
- Certaines colonnes sont lues 100x plus souvent que d'autres (separer hot columns des cold columns)
- Des colonnes BLOB (images, documents) alourdissent la table principale
- Microservices : chaque service possede ses colonnes

**Exemple concret** : Table `users` avec 50 colonnes. Les colonnes `id`, `name`, `email` sont lues a chaque requete. La colonne `avatar_blob` (5 Mo) est lue 1 fois sur 100. Separer ces donnees en deux tables evite de charger 5 Mo quand tu n'as besoin que du nom.

### Quand partitionner

| Signal | Action |
|---|---|
| Table > 100 Go | Envisager le horizontal partitioning |
| Queries lentes malgre les index | Analyser si le working set depasse la RAM |
| Write contention sur une seule table | Sharding pour distribuer les ecritures |
| Colonnes hot vs cold | Vertical partitioning |
| Pas encore de probleme | **NE PAS partitionner**. Premature partitioning = complexite gratuite |

### Partition Key Design

Le choix de la partition key est **la decision la plus importante** du sharding. Une mauvaise partition key -> hot spots, cross-partition queries, resharding.

**Bonne partition key** :
- Haute cardinalite (beaucoup de valeurs distinctes)
- Distribution uniforme du trafic
- Correspond a l'access pattern principal

**Exemples** :

| Systeme | Bonne partition key | Mauvaise partition key |
|---|---|---|
| E-commerce | `order_id` | `country` (desequilibre FR vs autres) |
| Chat | `conversation_id` | `user_id` (un user a 10 conversations) |
| Analytics | `(tenant_id, date)` | `date` seul (un jour = une partition surcharee) |

---

## Comment choisir sa base de donnees — Decision Framework

### Par use case

| Use case | DB recommandee | Pourquoi |
|---|---|---|
| **OLTP** (transactions) | PostgreSQL, MySQL | ACID, jointures, SQL mature |
| **OLAP** (analytics) | ClickHouse, BigQuery, Redshift | Stockage colonnaire, aggregations rapides |
| **Time-series** | TimescaleDB, InfluxDB, Cassandra | Optimise pour l'ecriture sequentielle temporelle |
| **Search** (full-text) | Elasticsearch, OpenSearch | Index inverse, scoring, fuzzy matching |
| **Cache** | Redis, Memcached | Sub-ms en memoire, TTL, structures de donnees |
| **Graph** | Neo4j, Neptune | Traversals de relations multi-niveaux |
| **Key-Value (scale)** | DynamoDB, Cassandra | Auto-scaling, latence previsible |
| **Document** | MongoDB, Firestore | Schema flexible, agregats autonomes |
| **Queue/Stream** | Kafka, Redis Streams | Ordering, replay, consumer groups |

### Decision tree simplifie

```
1. Ai-je besoin de transactions ACID multi-tables ?
   OUI -> PostgreSQL/MySQL
   NON -> continuer

2. Mon access pattern est-il uniquement par cle ?
   OUI -> Redis (si ca tient en RAM) ou DynamoDB
   NON -> continuer

3. Mon workload est-il > 80% en ecriture ?
   OUI -> Cassandra/ScyllaDB (append-only optimized)
   NON -> continuer

4. Ai-je besoin de traverser des relations profondes ?
   OUI -> Neo4j
   NON -> continuer

5. Ai-je besoin de recherche full-text ?
   OUI -> Elasticsearch (+ une DB primaire)
   NON -> continuer

6. Mes donnees ont-elles un schema variable ?
   OUI -> MongoDB
   NON -> PostgreSQL (le choix par defaut safe)
```

**Regle d'or** : En cas de doute, PostgreSQL. C'est la DB la plus polyvalente. Tu peux toujours ajouter Redis (cache), Elasticsearch (search), ou un data warehouse (analytics) plus tard.

---

## Chiffres a connaitre

### Throughput par type de DB

| DB | Type | Throughput typique par noeud | Latence typique |
|---|---|---|---|
| PostgreSQL | OLTP SQL | 5K-50K QPS | 1-10 ms |
| MySQL | OLTP SQL | 5K-50K QPS | 1-10 ms |
| MongoDB | Document | 20K-80K ops/s | 1-10 ms |
| Redis | Key-Value (RAM) | 100K-1M ops/s | < 1 ms |
| Cassandra | Column-family | 10K-100K writes/s | 1-5 ms |
| ScyllaDB | Column-family | 500K-1M ops/s | < 1 ms |
| DynamoDB | Key-Value (managed) | "illimite" (auto-scaling) | 1-10 ms |
| ClickHouse | OLAP | 100M-1B rows/s (scan) | 100 ms-10s (aggregations) |
| Elasticsearch | Search | 5K-20K searches/s | 10-100 ms |
| Neo4j | Graph | 100K+ traversals/s | 1-10 ms |

### Latences typiques

| Operation | Latence |
|---|---|
| Redis GET (local) | 0.1-0.5 ms |
| PostgreSQL SELECT par PK (indexe) | 1-5 ms |
| MongoDB find par _id | 1-5 ms |
| Elasticsearch query simple | 10-50 ms |
| Cross-datacenter query | 50-150 ms |
| ClickHouse aggregation (1B rows) | 500 ms - 5s |

**Key takeaway** : Redis est ~1000x plus rapide que PostgreSQL pour un lookup simple. Mais PostgreSQL peut faire des jointures, des transactions, et du full-text search. Le bon outil pour le bon job.

---

## Flash Cards — Q&A

### Q1
**Q** : Tu concois un systeme IoT qui recoit 500K events/sec de capteurs. Les events doivent etre stockes pendant 2 ans et interrogeables par capteur et par plage de temps. Quelle DB et pourquoi ?

**R** : **Cassandra** (ou TimescaleDB si tu veux du SQL). Partition key = `sensor_id`, clustering key = `timestamp`. Cassandra excelle en write throughput massif et les range queries par timestamp sont natives avec les clustering keys. 500K writes/sec = 5-50 noeuds Cassandra. TimescaleDB est une alternative si tu veux garder SQL et que tu acceptes moins de write throughput natif.

---

### Q2
**Q** : Explique pourquoi un index B-Tree accelere un SELECT mais ralentit un INSERT.

**R** : Le B-Tree est un arbre trie maintenu sur disque. Pour un SELECT, au lieu de scanner N lignes (O(n)), on parcourt l'arbre en O(log n) — 3-4 acces disque pour 10M de lignes. Mais pour un INSERT, il faut **aussi** inserer la nouvelle entree dans l'index, ce qui peut impliquer le split de noeuds pour maintenir l'arbre equilibre. Plus il y a d'index, plus chaque INSERT est couteux. Un INSERT dans une table avec 5 index fait 1 write dans la table + 5 writes dans les index.

---

### Q3
**Q** : Ton systeme a 3 replicas avec W=1, R=1, N=3. Est-ce consistent ? Que se passe-t-il si un noeud tombe ?

**R** : **Non, ce n'est pas consistent.** W + R = 2, qui n'est pas > N (3). Un write peut aller au noeud A, et un read peut aller au noeud B qui n'a pas encore la donnee -> stale read. Si un noeud tombe, le systeme continue de fonctionner (W=1 et R=1 sont satisfaits avec 2 noeuds restants), mais la consistance est encore pire car un des 2 noeuds restants peut ne pas avoir la derniere ecriture.

---

### Q4
**Q** : Tu shardes une table `orders` par `customer_id`. Un client corporate passe 10M de commandes. Quel est le probleme et comment le resoudre ?

**R** : **Hot spot.** Ce client surcharge une seule shard. Solutions : (1) **Composite shard key** `(customer_id, order_id % 16)` pour distribuer les commandes de ce client sur 16 shards. (2) **Directory-based sharding** pour deplacer ce client sur une shard dediee plus puissante. (3) **Identifier les "whales"** a l'avance et les traiter differemment. Le cout : les queries "toutes les commandes du client X" deviennent des scatter-gather sur plusieurs shards.

---

### Q5
**Q** : Quelle est la difference entre replication et sharding ? Peut-on les combiner ?

**R** : La **replication** duplique les memes donnees sur plusieurs noeuds (pour la disponibilite et la lecture). Le **sharding** distribue des donnees differentes sur differents noeuds (pour le scaling). On les combine presque toujours en production : chaque shard a ses propres replicas. Exemple : 4 shards * 3 replicas = 12 noeuds au total. Chaque shard contient 1/4 des donnees, et chaque donnee est presente sur 3 noeuds pour la disponibilite.

---

## Pour aller plus loin

Ressources canoniques sur le sujet :

- **Designing Data-Intensive Applications** (Martin Kleppmann, O'Reilly 2017) — Ch 2-3 (Data Models, Storage Engines), Ch 5 (Replication), Ch 6 (Partitioning), Ch 7 (Transactions) couvrent l'integralite du chapitre. https://dataintensive.net/
- **CMU 15-445 — Database Systems** (Andy Pavlo, CMU) — cours universitaire reference sur les SGBD : storage, indexes, concurrency control, recovery. Playlist YouTube officielle. https://www.youtube.com/playlist?list=PLSE8ODhjZXjYDBpQnSymaectKjxCy6BYq
- **Database Internals** (Alex Petrov, O'Reilly 2019) — deep-dive sur B-Trees, LSM-Trees, WAL, paxos/raft. Le complement technique de DDIA. https://www.oreilly.com/library/view/database-internals/9781492040330/
- **CMU 15-721 — Advanced Database Systems** (Andy Pavlo, CMU) — cours avance sur les systemes en memoire, columnar, distributed. Playlist YouTube Spring 2019. https://www.youtube.com/playlist?list=PLSE8ODhjZXja7K1hjZ01UTVDnGQdx5v5U
