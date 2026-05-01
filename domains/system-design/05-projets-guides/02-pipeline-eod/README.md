# Projet 02 — Pipeline EOD (End-of-Day Review)

## Contexte metier

Apres chaque shift, l'OCC fait un **EOD Review** : on replay le shift, on pointe les moments cles, on explique les bonnes et mauvaises decisions aux operateurs et aux planners. C'est souvent la phase la plus importante de l'optimisation continue (c'est la que les SOPs s'ameliorent).

Pour que ca marche, le pipeline doit :
- Enregistrer **tous** les events du shift en temps reel, sans perdre un seul tick
- Permettre le **rejeu** a n'importe quel point du temps (scrubbing)
- Permettre des **requetes analytiques** : "quels AGV ont ete en zone B-12 ?", "combien de pickups par la flotte Alpha ?"
- Survivre a un crash du serveur de simu (un shift = 8 heures d'activite operationnelle, pas negociable de perdre les donnees)

Un shift typique : 2000 unites, 8 heures, 10 ticks/sec = 576 millions de ticks, 10-100 events par tick = **6-60 milliards d'events** par shift. En realite on stocke plutot les deltas (ce qui change), ce qui fait revenir a ~100 Go brut.

## Objectif technique

Designer le pipeline EOD : ingestion, stockage, indexation, requete, rejeu.

## Contraintes

| Contrainte | Valeur |
|---|---|
| Ingestion debit | 10k events/sec par shift, 20 shifts paralleles |
| Pertes tolerees | 0 event |
| Latence d'une requete EOD | < 500 ms pour "tous les events d'une unite sur 30 min" |
| Stockage | 100 Go / shift, 10 ans de retention (archivage SOC 2 / contrat client) |
| Rejeu | pouvoir scrub a n'importe quel `t` en < 1 s |
| Deploiement | on-premise, pas de SaaS |

## Etapes guidees

1. **Format d'event** — binaire compact (protobuf, capnp, flatbuffer) ou JSON ? Justifie.
2. **Ingestion** — le serveur de simu pousse ou un agent sidecar pull ? Avec quel buffer pour absorber les pics (Black Friday, surge) ?
3. **Stockage chaud vs froid** — les 48 dernieres heures en hot storage (SSD), le reste en cold (HDD / tape). Quel critere de promotion/demotion ?
4. **Indexation** — pour repondre a "tous les events de l'unite U entre t1 et t2" en < 500 ms, quelle structure d'index ? B-tree sur `(shift_id, unit_id, t)` ? Parquet partitionne ?
5. **Replay** — comment "sauter" a `t=3h27m15s` sans rejouer depuis zero ? Snapshots periodiques (toutes les 5 minutes) + deltas ?
6. **Resilience** — WAL ? Replication ? Acknowledgement avant de confirmer a la simu ?
7. **Confidentialite** — les donnees client sont sensibles (flux, volumes, partenaires). Chiffrement au repos ? Audit trail ? Effacement sur demande contractuelle ?

## Questions de revue

- Pourquoi pas Kafka/Kinesis pour l'ingestion ? (Indice : air-gap, sites isoles)
- Pourquoi pas Postgres direct ?
- Combien de Go/sec genere un shift a pleine charge ? Ton design tient ?
- Un operateur supprime accidentellement un EOD — qu'est-ce qu'il se passe ?
- Un auditeur te demande "montre moi toutes les quasi-collisions inter-flotte sur les 50 derniers shifts". Combien de temps pour repondre ?

## Solution

Voir `solution/pipeline.md` pour le design detaille avec schemas.

## Pour aller plus loin

- **Projection materialisee** — stats precalculees pour les dashboards OCC (par unite, par type d'event, par zone)
- **Differential sync** — pour un client multi-site, synchroniser les EOD vers le hub central sans pousser 100 Go a travers le LAN
- **LLM sur les traces** — embedder les traces et permettre de chercher "trouve des moments similaires a celui-ci" (vector DB)
