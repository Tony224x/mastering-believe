# Exercices Medium — Stockage & Databases

---

## Exercice 1 : Concois le schema de sharding pour une messagerie

### Objectif
Appliquer les principes de sharding et de partition key design a un cas reel.

### Consigne
Tu concois le stockage pour un service de messagerie (type Slack/WhatsApp) avec les caracteristiques suivantes :

**Chiffres :**
- 100M d'utilisateurs actifs
- 50 messages/user/jour en moyenne
- Taille moyenne d'un message : 500 octets (texte + metadata)
- Retention : 3 ans
- Access pattern principal : "afficher les 50 derniers messages d'une conversation"
- Access pattern secondaire : "rechercher un message par mot-cle dans une conversation"

**Questions :**
1. Calcule le stockage total pour 3 ans
2. Propose une partition key. Justifie pourquoi les alternatives sont inferieures.
3. Propose un clustering key pour optimiser l'access pattern principal
4. Comment gerer une conversation de groupe avec 500 personnes (tres active) sans creer un hot spot ?
5. Comment implementer la recherche par mot-cle sans scanner toutes les partitions ?
6. Dessine le schema de la table (colonnes, types, primary key)

### Criteres de reussite
- [ ] Estimation de stockage coherente (~2.5 Po pour 3 ans)
- [ ] Partition key = `conversation_id` avec justification vs `user_id` et `message_id`
- [ ] Clustering key = `timestamp DESC` ou `message_id DESC`
- [ ] Solution pour le hot spot du groupe actif (sub-partitioning ou bucketing)
- [ ] Elasticsearch mentionne pour la recherche full-text
- [ ] Le schema est complet et coherent

---

## Exercice 2 : Migration SQL vers NoSQL — Analyse de tradeoffs

### Objectif
Evaluer les consequences d'une migration de base de donnees sur un systeme en production.

### Consigne
Ton equipe gere un service e-commerce dont la base PostgreSQL (2 To, 100M de lignes dans `orders`) atteint ses limites :
- Les writes sont a 95% de la capacite du serveur
- Le replication lag vers les read replicas depasse regulierement 10 secondes
- Les jointures entre `orders`, `order_items`, et `products` prennent > 5 secondes

Un collegue propose de migrer vers DynamoDB. Analyse la proposition :

1. **Avantages concrets** de DynamoDB pour CE cas (pas des avantages generiques)
2. **Couts caches** de la migration (au moins 5)
3. **Ce que tu perds** en quittant PostgreSQL (au moins 4 points)
4. **Alternative** : propose une solution qui resout les 3 problemes SANS quitter PostgreSQL
5. **Decision framework** : dans quelles conditions recommanderais-tu la migration ?

### Criteres de reussite
- [ ] Les avantages sont specifiques au cas (pas juste "DynamoDB scale mieux")
- [ ] Les couts caches incluent : migration des donnees, rewrite des queries, perte de jointures, formation equipe, double-run pendant la transition
- [ ] Les pertes incluent : ACID multi-tables, SQL ad hoc, ecosysteme d'outils
- [ ] L'alternative PostgreSQL est realiste (Citus, partitioning, CQRS, materialized views)
- [ ] Le decision framework a des criteres mesurables, pas subjectifs

---

## Exercice 3 : Design un index strategy pour un dashboard analytics

### Objectif
Concevoir une strategie d'indexation optimale pour un workload mixte.

### Consigne
Tu geres une table `events` dans PostgreSQL qui alimente un dashboard analytics :

```sql
CREATE TABLE events (
    id            BIGSERIAL PRIMARY KEY,
    tenant_id     UUID NOT NULL,
    event_type    VARCHAR(50) NOT NULL,   -- 'click', 'purchase', 'signup', ...
    user_id       UUID,
    amount        DECIMAL(12,2),
    metadata      JSONB,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

**Workload :**
- 50K inserts/sec (append-only, jamais d'update)
- Query 1 (90% du trafic) : `SELECT count(*), sum(amount) FROM events WHERE tenant_id = ? AND event_type = ? AND created_at BETWEEN ? AND ?`
- Query 2 (8% du trafic) : `SELECT * FROM events WHERE tenant_id = ? AND user_id = ? ORDER BY created_at DESC LIMIT 20`
- Query 3 (2% du trafic) : `SELECT DISTINCT event_type FROM events WHERE tenant_id = ? AND created_at > NOW() - INTERVAL '24h'`

**Questions :**
1. Propose les index necessaires pour chaque query. Explique l'ordre des colonnes.
2. Y a-t-il un covering index possible ? Pour quelle query ?
3. Estime l'overhead en espace disque des index (chaque index = ~30% de la taille de la table pour les colonnes indexees)
4. Les 50K inserts/sec seront-ils impactes ? Quantifie.
5. Propose un partitioning strategy pour cette table (type, cle, granularite)
6. A quel moment faut-il passer a ClickHouse au lieu de rester sur PostgreSQL ?

### Criteres de reussite
- [ ] Index composite dans le bon ordre (tenant_id en premier, puis les colonnes de filtre)
- [ ] Le covering index est identifie pour Query 1
- [ ] L'overhead des index est estime en Go
- [ ] L'impact sur les writes est quantifie (chaque index = 1 write supplementaire par insert)
- [ ] Le partitioning par `created_at` (monthly/weekly) est propose
- [ ] Le seuil de migration vers ClickHouse est argumente (~1B+ lignes ou latence > SLO)
