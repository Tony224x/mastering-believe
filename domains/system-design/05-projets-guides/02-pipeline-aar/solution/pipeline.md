# Reference design — Pipeline AAR

## Vue d'ensemble

```
  [Sim worker]                                                        [AAR UI]
       |                                                                  ^
       | 1. events                                                        |
       v                                                                  |
  [Local agent]  --- 2. batch WAL append ----> [Hot store : NVMe + DuckDB]
       |                                               |
       |                                               | 3. promotion nightly
       |                                               v
       +-----> (if down, buffer disk)          [Cold store : Parquet + tape]
                                                       ^
                                                       |
                                               [Query service : gRPC]
                                                       ^
                                                       |
                                                  [Replay service]
                                                       |
                                                       v
                                                    [AAR UI]
```

## Format d'event — Protobuf

```proto
message SimEvent {
  uint32 exercise_id = 1;
  uint64 seq = 2;           // monotone per exercise, pour detect drop
  double t_sim = 3;         // temps simule en secondes
  uint32 tick = 4;
  uint32 unit_id = 5;
  EventKind kind = 6;       // FIRE, MOVE, DAMAGE, ORDER, ...
  bytes payload = 7;        // subfield specific a kind
}
```

Pourquoi protobuf et pas JSON :
- ~4-5x plus compact sur ce schema
- Typing strict, pas de drift de format
- Backward-compat via champs numerotes

Pourquoi pas flatbuffer : complexite > gain quand on doit quand meme deserialiser cote reader.

## Ingestion — agent local + WAL

Le sim worker **ne parle pas directement** au store : il pousse les events dans un `unix socket` local vers un **agent** qui tourne sur la meme machine. Raisons :
- Isole les pics de debit : si le store est lent, c'est l'agent qui prend la contrainte
- Si le store est injoignable, l'agent buffer sur disque local (WAL), retry au retour
- Le sim worker ne doit **jamais** bloquer sur I/O reseau

Ack protocol : le sim worker ne considere un tick comme "committed" qu'apres que l'agent a fsync'e le WAL local. C'est la garantie **at-least-once**. L'idempotence est assuree par `(exercise_id, seq)` unique.

## Hot store — DuckDB sur NVMe

Pour les 48 premieres heures : DuckDB pointing sur des fichiers Parquet append-only, un fichier par `(exercise_id, hour)`.

Pourquoi DuckDB :
- Embedded, pas de service a operer
- Requetes SQL analytiques tres rapides (columnar)
- Read-only sur fichiers partages, pas de lock
- Parfait pour "filtre par unit_id sur une fenetre de 30 min"

Pourquoi pas Postgres :
- Le row store est mauvais pour les scans analytiques sur 10k+ events
- Operations DBA (VACUUM, WAL bloat) lourdes sur un deploiement air-gap

Pourquoi pas Kafka + Flink :
- Kafka/Flink air-gap c'est possible mais l'equipe MASA n'en a pas besoin : 20 exercices paralleles c'est 200k events/sec total, un seul NVMe gere ca
- Complexite operationnelle injustifiee

## Cold store — Parquet + tape archive

Apres 48 h sans acces, promotion automatique vers :
- Parquet partitionne par `(year, month, exercise_id)` sur disque HDD ou SAN
- Apres 2 ans : migration tape (LTO) pour retention 10 ans (norme defense)

## Indexation

Les Parquet files sont deja "indexes" par partition. Pour les queries point (par unit_id) :
- Hot : DuckDB fait du pushdown sur les row groups Parquet, suffisant
- Cold : index z-order ou bloom filter sur `(unit_id)` au moment de l'ecriture

Pour "tous les events d'une unite sur 30 min" : 1 partition Parquet, filtre sur `unit_id` via bloom, scan les row groups concernes = < 200 ms en pratique.

## Replay avec scrubbing

Le rejeu doit pouvoir sauter a `t=3h27m`. Strategie :
- **Snapshots cles** toutes les 5 minutes = snapshot complet de l'etat simu (positions, sante, possession). ~10 Mo chaque.
- **Deltas** entre snapshots (les events eux-memes).
- Pour scrub a `t=3h27m` : charger snapshot a `t=3h25m`, rejouer 2 minutes d'events.
- Couts de scrub = constant < 1s.

Les snapshots sont stockes a part (pas dans la meme table events) pour eviter de les confondre a la requete.

## Resilience et garanties

| Probleme | Garantie | Mecanisme |
|---|---|---|
| Crash sim worker | Pas de perte | WAL agent local, replay au restart |
| Crash agent | Pas de perte | Sim worker bloque sur ack, buffer local |
| Crash hot store | Pas de perte | Agent bufferise sur disque, retry |
| Corruption disque | Recovery | Replication 2x sur NAS |
| Suppression accidentelle | Recovery | Soft-delete + tombstone, purge effective apres 30 jours |
| Classification | Confidentialite | LUKS disk + audit trail gravee en WORM |

## Questions de revue — reponses

**Pourquoi pas Kafka/Kinesis ?**
Air-gap : pas de Kinesis, et Kafka impose Zookeeper/KRaft a operer. Debit requis atteignable avec un agent + NVMe. Simplicite.

**Pourquoi pas Postgres ?**
Row store inefficace sur scans analytiques lourds. DuckDB/Parquet donne x10-x50 sur ces patterns.

**Go/sec a pleine charge ?**
200k events/sec * ~200 bytes/event = 40 Mo/sec. Un NVMe PCIe 3 tient 3 Go/s sequentiel. OK.

**Suppression accidentelle ?**
Soft-delete 30 jours, restauration via CLI admin. L'UI utilisateur ne propose pas de delete (trop risque).

**Stat fratricide sur 50 exercices ?**
Query DuckDB sur la partition `hot` + `cold` : filtre `kind=FIRE AND friendly_fire=true`, aggregate par exercice. Estimation : 2-10 secondes. Pour plus rapide, materialised view calculee au commit d'exercice.

## Trade-offs non choisis

- **TimescaleDB** : bon, mais la granularite "time-series" n'est pas si forte ici, DuckDB/Parquet est plus simple.
- **S3 + Iceberg** : genial en cloud, impossible air-gap.
- **ClickHouse** : viable alternative. Choisi DuckDB pour le "embedded, no service to operate".
