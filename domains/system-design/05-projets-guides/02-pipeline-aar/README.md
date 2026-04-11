# Projet 02 — Pipeline AAR (After-Action Review)

## Contexte metier

Apres chaque exercice, le formateur fait un **AAR** : il replay l'exercice, pointe les moments cles, explique les bonnes et mauvaises decisions aux stagiaires. C'est souvent la phase la plus importante du training (c'est la que les gens apprennent).

Pour que ca marche, le pipeline doit :
- Enregistrer **tous** les events de l'exercice en temps reel, sans perdre un seul tick
- Permettre le **rejeu** a n'importe quel point du temps (scrubbing)
- Permettre des **requetes analytiques** : "quelles unites ont ete en contact ?", "combien de munitions tirees par le peloton Alpha ?"
- Survivre a un crash du serveur de simu (un exercice = 4 heures de travail des stagiaires, pas negociable de perdre les donnees)

Un exercice typique : 2000 unites, 4 heures, 10 ticks/sec = 288 millions de ticks, 10-100 events par tick = **3-30 milliards d'events** par exercice. En realite on stocke plutot les deltas (ce qui change), ce qui fait revenir a ~100 Go brut.

## Objectif technique

Designer le pipeline AAR : ingestion, stockage, indexation, requete, rejeu.

## Contraintes

| Contrainte | Valeur |
|---|---|
| Ingestion debit | 10k events/sec par exercice, 20 exercices paralleles |
| Pertes tolerees | 0 event |
| Latence d'une requete AAR | < 500 ms pour "tous les events d'une unite sur 30 min" |
| Stockage | 100 Go / exercice, 10 ans de retention (archivage) |
| Rejeu | pouvoir scrub a n'importe quel `t` en < 1 s |
| Deploiement | on-premise, pas de SaaS |

## Etapes guidees

1. **Format d'event** — binaire compact (protobuf, capnp, flatbuffer) ou JSON ? Justifie.
2. **Ingestion** — le serveur de simu pousse ou un agent sidecar pull ? Avec quel buffer pour absorber les pics ?
3. **Stockage chaud vs froid** — les 48 dernieres heures en hot storage (SSD), le reste en cold (HDD / tape). Quel critere de promotion/demotion ?
4. **Indexation** — pour repondre a "tous les events de l'unite U entre t1 et t2" en < 500 ms, quelle structure d'index ? B-tree sur `(exercise_id, unit_id, t)` ? Parquet partitionne ?
5. **Replay** — comment "sauter" a `t=3h27m15s` sans rejouer depuis zero ? Snapshots periodiques (toutes les 5 minutes) + deltas ?
6. **Resilience** — WAL ? Replication ? Acknowledgement avant de confirmer a la simu ?
7. **Classification** — les donnees peuvent etre classifiees. Chiffrement au repos ? Audit trail ? Effacement sur demande ?

## Questions de revue

- Pourquoi pas Kafka/Kinesis pour l'ingestion ? (Indice : air-gap)
- Pourquoi pas Postgres direct ?
- Combien de Go/sec genere un exercice a pleine charge ? Ton design tient ?
- Un stagiaire accidentellement supprime un AAR — qu'est-ce qu'il se passe ?
- Un auditeur te demande "montre moi tous les tirs amis (fratricide) dans les 50 derniers exercices". Combien de temps pour repondre ?

## Solution

Voir `solution/pipeline.md` pour le design detaille avec schemas.

## Pour aller plus loin

- **Projection materialisee** — stats precalculees pour les dashboards (par unite, par type d'event)
- **Differential sync** — si on a un exercice LVC avec 3 sites, synchroniser les AAR sans pousser 100 Go a travers le LAN
- **LLM sur les traces** — embedder les traces et permettre de chercher "trouve des moments similaires a celui-ci" (vector DB)
