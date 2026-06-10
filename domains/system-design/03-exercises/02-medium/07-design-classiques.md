# Exercices Medium — Design classiques (entretiens)

---

## Exercice 1 : Design Web Crawler

### Objectif
Derouler le framework en 6 etapes sur un design classique oriente throughput et politeness.

### Consigne
Concois un web crawler qui doit indexer **1 milliard de pages par mois**.

**Contraintes :**
- Taille moyenne d'une page HTML : 100 Ko (on ne stocke que le HTML)
- Respecter robots.txt et ne pas marteler un meme domaine (max 1 req/sec/domaine)
- Detecter les contenus dupliques
- Re-crawler les pages selon leur frequence de changement

**Deroule les 6 etapes :**
1. **Estimation** : pages/sec moyen, bande passante d'ingestion, stockage pour 1 an (avec re-crawls : considere 1 version conservee).
2. **High-level design** : composants (frontier/URL queue, fetchers, parser, dedup, storage, scheduler) et flux entre eux.
3. **Deep dive — politeness** : comment garantir 1 req/sec/domaine avec 1000 fetchers paralleles ? (indice : une queue par domaine ou un mapping domaine -> queue)
4. **Deep dive — deduplication** : URL dedup vs content dedup. Quelle structure pour tester "URL deja vue" sur 10 milliards d'URLs avec une RAM raisonnable ?
5. **Bottlenecks** : identifie les 2 principaux et leur mitigation.
6. **Extension** : comment prioriser le re-crawl (une page de news vs une page statique) ?

### Criteres de reussite
- [ ] Estimation : ~385 pages/sec en moyenne (1B / 2.6M sec), ~38 Mo/s d'ingestion, ~100 To+ de stockage HTML pour 1B de pages
- [ ] L'architecture separe frontier (queues par domaine), fetchers stateless, parsing async, et storage
- [ ] La politeness utilise un sharding par domaine (une sous-queue par domaine + delai min entre fetches)
- [ ] Le Bloom filter (ou equivalent) est propose pour l'URL dedup, avec mention des faux positifs ; content dedup par hash/simhash
- [ ] Le re-crawl est pilote par un score de fraicheur (frequence de changement observee)

---

## Exercice 2 : Design Rate-Limited Ticketing (vente de billets flash)

### Objectif
Combiner les patterns queue, cache, lock et consistency sur un cas a fort pic de contention.

### Consigne
Concois le systeme de vente de billets d'un concert : **50 000 places**, mises en vente a 10h00 precises, **2 millions d'utilisateurs** attendus dans les 5 premieres minutes.

1. **Estimation** : QPS au pic (suppose 80% du trafic dans les 60 premieres secondes). Le site marchand classique (200 req/s par serveur) peut-il encaisser frontalement ?
2. **Anti-pattern** : pourquoi "SELECT stock FROM tickets WHERE id=X puis UPDATE" provoque de l'overselling sous concurrence ? Montre l'interleaving fautif.
3. **Design** : propose une architecture avec virtual waiting room (file d'attente) + reservation a duree limitee (hold de 10 min) + paiement asynchrone. Diagramme des composants et des etats d'un billet (AVAILABLE -> HELD -> SOLD / RELEASED).
4. **Decompte du stock** : compare 3 implementations (lock SQL SELECT FOR UPDATE, decrement atomique Redis DECR, partitionnement du stock en N sous-pools). Laquelle tient 25K+ req/s ?
5. **Equite** : comment empecher les bots et garantir un ordre de passage a peu pres juste ?

### Criteres de reussite
- [ ] QPS pic calcule : ~27K req/s (1.6M requetes / 60 s), incompatible avec un acces frontal direct a la DB
- [ ] L'interleaving read-check-write est montre avec 2 transactions concurrentes vendant le meme billet
- [ ] L'architecture contient : waiting room (token + queue), hold avec TTL, confirmation asynchrone du paiement
- [ ] Redis DECR atomique (ou stock partitionne) retenu pour le pic ; SELECT FOR UPDATE identifie comme goulot (lock contention)
- [ ] Les mesures d'equite incluent au moins : token par session, CAPTCHA/proof-of-work, randomisation de l'ordre d'admission

---

## Exercice 3 : Design Google Docs (edition collaborative)

### Objectif
Raisonner sur la consistency temps reel et la resolution de conflits.

### Consigne
Concois le backend d'un editeur de documents collaboratif : plusieurs utilisateurs editent le meme document simultanement et voient les modifications des autres en < 500 ms.

1. **Clarification** : quelles questions poses-tu avant de designer ? (taille max d'un doc, nb max d'editeurs simultanes, offline ?)
2. **Transport** : WebSocket, SSE ou polling pour propager les modifications ? Justifie.
3. **Conflits** : deux users inserent du texte au meme endroit au meme moment. Explique pourquoi "last write wins" detruit des donnees, puis decris a haut niveau les deux familles de solutions (Operational Transformation vs CRDT) et leur tradeoff principal.
4. **Architecture** : dessine le flux d'une frappe clavier (client -> ? -> autres clients), en placant : le serveur de session du document, la persistence (snapshots + log d'operations), et la gestion des reconnexions.
5. **Scaling** : 10M de documents actifs, ~5 editeurs max par doc. Comment shardes-tu ? Pourquoi le sharding par document est naturel ici ?

### Criteres de reussite
- [ ] Au moins 3 questions de clarification pertinentes
- [ ] WebSocket retenu (bidirectionnel, latence basse) avec fallback mentionne
- [ ] OT vs CRDT : OT = serveur central qui transforme les operations, CRDT = merge sans coordination mais surcout memoire/complexite des structures
- [ ] Le flux contient un log d'operations ordonne par le serveur de session + snapshots periodiques pour le chargement rapide
- [ ] Sharding par document_id : toutes les operations d'un doc passent par le meme serveur de session (ordre garanti), avec un service de routage doc -> serveur
