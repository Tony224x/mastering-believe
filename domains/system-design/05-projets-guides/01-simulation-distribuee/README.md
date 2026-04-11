# Projet 01 — Simulation distribuee tick-based

## Contexte metier

SWORD simule jusqu'a ~2000 unites autonomes en temps reel. Sur un seul serveur ca passe jusqu'a ~500 unites. Au-dela, il faut **distribuer** : partitionner le monde sur N serveurs, synchroniser leurs horloges simulees, gerer les unites qui traversent les frontieres. Le tout en restant **deterministe** (rejeu pour AAR) et **resilient** (un crash serveur ne doit pas corrompre tout l'exercice en cours).

Tu es l'AI Engineer sur l'equipe R&D. Le tech lead te demande une proposition d'architecture pour un prototype "SWORD distribue" capable de faire tourner 5000 unites sur 5 serveurs en parallele.

## Objectif technique

Produire un document d'architecture qui couvre :
1. Le modele de partitioning (comment on decoupe le monde)
2. La boucle de tick synchronisee (comment tous les serveurs avancent au meme pas)
3. La gestion des unites cross-boundary (hand-off)
4. Le determinisme (PRNG, ordre d'application des events)
5. La tolerance aux pannes (crash d'un worker pendant l'exercice)
6. Les metriques et SLOs

## Contraintes non fonctionnelles

| Contrainte | Valeur cible |
|---|---|
| Unites simulees | 5000 sur 5 workers (1000 / worker) |
| Tick rate | 10 Hz (100 ms par tick) |
| Latence max d'un tick | 80 ms a p99 |
| Determinisme | rejeu bit-exact sur meme seed |
| Air-gap | aucun service cloud, tout on-premise |
| Resilience | crash d'un worker = 1 exercice perdu max, pas de corruption des autres |

## Niveau 1 — version simplifiee (a faire d'abord)

Avant d'attaquer l'architecture complete, commence par une **version simplifiee** pour acquerir les intuitions :

- **2 workers seulement** (plus facile a raisonner qu'N)
- **Zones fixes**, pas de hand-off dynamique : une unite qui sort de sa zone est juste ignoree (on accepte la perte, v0)
- **Tick master centralise** avec un simple barrier : envoie "TICK n", attend 2 ACK, envoie "COMMIT n"
- **Pas de crash recovery** — si un worker tombe, l'exercice est perdu
- **Pas de replication** — chaque worker est seul sur sa zone

Ce niveau 1 te permet de comprendre le tick sync, la serialisation des events, et le cout d'un round-trip master-worker, avant d'ajouter la complexite du hand-off, du determinisme et de la resilience. **Fais-le tourner en local avec 2 processus Python (multiprocessing) avant de passer au niveau 2 decrit ci-dessous.**

Quand le niveau 1 tourne, passe au niveau 2 en ajoutant une contrainte a la fois : d'abord le hand-off, puis le determinisme, puis la crash recovery.

## Etapes guidees (niveau 2 — architecture complete)

1. **Decoupage spatial** — grille reguliere, quadtree, ou zones metier ? Expose le pro/contra.
2. **Tick sync** — master-slave avec barrier, peer-to-peer avec lockstep, ou virtual time (Jefferson) ? Pour 10 Hz sur LAN, quelle est l'option la plus simple qui marche ?
3. **Hand-off** — quand une unite a `(r, c)` sort de la zone du worker W1 et entre dans W2, qui est authoritative ? Comment on evite les doublons ou les disparitions ?
4. **Ordering** — deux events a meme `t` sur deux workers differents : quel ordre d'application ? (Indice : hash stable sur l'id de l'event.)
5. **Crash worker** — rejouer depuis un snapshot ? Redistribuer sa zone ? Accepter la perte ?
6. **Diagramme** — fais au moins 2 vues : vue deploiement (qui parle a qui) + vue sequence d'un tick complet.

## Questions de revue

- Pourquoi pas un simple "big lock" sur un serveur unique avec des cores ?
- Quelle est la taille max d'une zone avant que le hand-off devienne dominant en cout ?
- Comment on prouve le determinisme en test ? Quel est le test CI ?
- Qu'est-ce qui empeche le deterministe de se casser des qu'on a des timers OS ?
- Comment un crash de la zone centrale (celle avec le plus d'action) impacte l'exercice ?

## Solution

Voir `solution/architecture.md` pour le reference design, les diagrammes, et les arbitrages.

## Pour aller plus loin

- **Predictive hand-off** — commencer a repliquer une unite dans la zone voisine 5 ticks avant qu'elle franchisse la frontiere, pour masquer la latence
- **Interest management** — un worker ne voit que les unites a moins de X metres de sa zone (economise bande passante)
- **Lockstep via HLA/DIS** — les normes IEEE qui gerent deja tout ca dans le monde defense, pourquoi ne pas les utiliser directement
