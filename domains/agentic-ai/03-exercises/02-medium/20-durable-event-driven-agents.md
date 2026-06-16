# Exercices Medium — Durable & event-driven agents (J20)

---

## Exercice 1 : Etat event-sourced reconstruit par replay (avec snapshot)

### Objectif
Implementer le coeur de l'event sourcing (cours section 2.1) : un etat qui n'est jamais stocke directement mais **derive** d'un log append-only en rejouant tous les evenements. Tu dois prouver deux invariants que beaucoup ratent : (a) le replay complet reconstruit exactement le meme etat, et (b) un **snapshot + replay de la queue** (tail) donne le meme etat que le replay complet — la base de toute optimisation de demarrage durable.

### Consigne
En t'inspirant de la `DurableEventLog` du module 20 (tu peux la reembarquer dans ta solution ou la garder en memoire dans une simple `list`) :

1. Modelise un agent de panier d'achat dont l'**etat** est uniquement `{"items": {sku: qty}, "version": int}`. Cet etat n'est **jamais** muté directement par l'exterieur.
2. Definis 3 types d'evenements : `ITEM_ADDED {sku, qty}`, `ITEM_REMOVED {sku, qty}`, `CART_CLEARED {}`. Ecris une fonction pure `apply(state, event) -> state` qui applique un seul evenement (un quantite a 0 supprime le sku ; `version` s'incremente a chaque evenement).
3. Ecris `replay(events) -> state` qui part de l'etat vide et applique tous les evenements dans l'ordre.
4. Genere une sequence d'au moins 6 evenements melangeant les 3 types, puis construis l'etat `full = replay(tous)`.
5. Implemente un **snapshot** : `snapshot = {"state": replay(events[:k]), "at": k}` pour un `k` au milieu de la sequence. Puis reconstruis l'etat via `tail = replay_from(snapshot, events[k:])` qui repart de l'etat du snapshot et applique uniquement la queue.
6. **Prouve par assertions** que `full == tail` (snapshot+tail equivaut au replay complet) et que rejouer deux fois le meme log donne deux etats identiques (determinisme).

### Criteres de reussite
- [ ] L'etat n'est jamais mute hors de `apply()` (fonction pure, retourne un nouvel etat)
- [ ] `replay()` reconstruit l'etat depuis l'etat vide en appliquant les evenements dans l'ordre
- [ ] Un sku dont la quantite tombe a 0 disparait du dict `items`
- [ ] `version` est egal au nombre d'evenements appliques
- [ ] `replay(events[:k])` puis tail-replay de `events[k:]` == `replay(events)` (verifie par assertion)
- [ ] Rejouer le meme log deux fois donne le meme etat (determinisme verifie)
- [ ] Execution offline, deterministe, sans dependance

---

## Exercice 2 : Harnais crash-and-resume avec checkpoint (exactly-once)

### Objectif
Construire un harnais qui prouve la garantie centrale de la durable execution (cours section 2.1, table section 1) : un workflow qui **crashe au milieu** reprend depuis son checkpoint **sans refaire** les etapes deja committees. Tu dois mesurer, compteur a l'appui, que chaque etape a effet de bord s'execute **exactement une fois** malgre le crash et la reprise.

### Consigne
1. Modelise un workflow lineaire de 4 etapes a effet de bord : `step_a`, `step_b`, `step_c`, `step_d`. Chacune incremente un compteur global `SIDE_EFFECTS[name]` (preuve d'execution reelle) et retourne un resultat deterministe.
2. Implemente un `CheckpointStore` (en memoire, un dict suffit) qui persiste : les resultats des etapes deja completees (`done: {name: result}`) et l'index de la prochaine etape a executer.
3. Ecris `run_workflow(steps, store, crash_after=None)` :
   - pour chaque etape, **si son resultat est deja dans le checkpoint**, le retourne sans re-executer (memoisation depuis le checkpoint),
   - sinon execute l'etape, **persiste immediatement** son resultat dans le store (commit), puis avance l'index,
   - si `crash_after == name`, leve `RuntimeError("CRASH")` **apres** le commit de cette etape (le crash arrive entre deux etapes, pas au milieu d'une).
4. Scenario : lance le workflow avec `crash_after="step_b"` (il crashe), puis **relance** `run_workflow` sur le **meme store** sans crash → il doit reprendre a `step_c` et finir.
5. **Prouve par assertions** :
   - apres le run complet, `SIDE_EFFECTS[name] == 1` pour les 4 etapes (aucune re-execution des etapes committees),
   - le run 2 n'a ré-execute que `step_c` et `step_d` (et `SIDE_EFFECTS` de `step_a`/`step_b` valent toujours 1),
   - le resultat final est identique a un run sans crash (run de reference sur un store neuf).

### Criteres de reussite
- [ ] `CheckpointStore` persiste les resultats des etapes completees et l'index courant
- [ ] Une etape deja dans le checkpoint est memoisee, jamais re-executee
- [ ] Le crash survient apres le commit de l'etape (pas de perte ni de double-commit)
- [ ] Apres reprise, chaque etape a effet de bord a un compteur a exactement 1 (exactly-once)
- [ ] Le run 2 ne re-execute que les etapes restantes (`step_c`, `step_d`)
- [ ] Le resultat final == resultat d'un run de reference sans crash (verifie par assertion)
- [ ] Execution offline, deterministe, sans dependance

---

## Exercice 3 : Couche d'idempotence pour livraison at-least-once

### Objectif
Implementer la deduplication d'evenements (cours section 2.3 : cle d'idempotence) au-dessus d'un canal **at-least-once** qui livre parfois le meme evenement plusieurs fois. Tu dois prouver que, malgre les doublons, chaque effet de bord ne s'applique **qu'une seule fois**, et distinguer concretement at-least-once (doublons possibles) de exactly-once (vu par le consommateur grace a la dedup).

### Consigne
1. Definis un evenement `{"event_id": str, "type": str, "payload": dict}` ou `event_id` est la cle d'idempotence (UUID-like, mais deterministe dans le test).
2. Implemente une `IdempotentConsumer` :
   - garde un set `processed_ids` (le "store d'idempotence", en memoire),
   - methode `handle(event) -> dict` : si `event_id` est deja dans `processed_ids`, retourne `{"status": "duplicate", ...}` **sans** appeler le handler metier ; sinon appelle le handler, enregistre l'id, et retourne `{"status": "applied", ...}`.
   - le handler metier mute un etat reel (ex : credite un solde, ou accumule une liste) — c'est ce qui ne doit pas etre applique deux fois.
3. Simule un canal **at-least-once** : une fonction `deliver(consumer, events)` qui livre une liste d'evenements ou **certains apparaissent en double** (rejoue volontairement 2-3 ids), dans un ordre eventuellement entrelace.
4. **Prouve par assertions** :
   - le solde / l'etat final correspond a l'application de **chaque event_id une seule fois** (les doublons n'ont rien change),
   - le nombre d'appels au handler metier == nombre d'event_id **distincts** (pas le nombre de livraisons),
   - chaque livraison en double retourne bien `{"status": "duplicate"}`.
5. Bonus de robustesse : verifie que l'ordre de livraison entrelace ne casse pas la dedup (livrer `[id1, id2, id1, id3, id2]` donne le meme etat que `[id1, id2, id3]`).

### Criteres de reussite
- [ ] L'evenement porte une `event_id` servant de cle d'idempotence
- [ ] Un `event_id` deja vu retourne `duplicate` sans rejouer le handler metier
- [ ] L'etat metier final == application de chaque id distinct exactement une fois
- [ ] Le compteur d'appels handler == nombre d'ids distincts (pas de livraisons)
- [ ] Un ordre de livraison entrelace avec doublons donne le meme etat final
- [ ] Execution offline, deterministe, sans dependance
