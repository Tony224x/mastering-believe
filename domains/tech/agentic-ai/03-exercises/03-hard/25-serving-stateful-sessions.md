# Exercices Hard — Serving stateful & sessions (J25)

> Tout est **simule en memoire**, deterministe, offline. La concurrence est rejouee via une **sequence d'operations explicite** (pas de vrais threads non-deterministes) : on controle l'entrelacement pour rendre les bugs reproductibles. Pas de Redis/Postgres reel, pas de reseau.

---

## Exercice 1 : Serving stateful tolerant aux pannes — failover + migration de session

### Objectif

Cabler un mini-cluster stateful conforme au cours (sections 1, 4.1, 4.2) : un load balancer a affinite de session route les tours vers des workers stateless adosses a un store de checkpoints partage. Quand un noeud **tombe**, le routeur **fail-over** la session vers un autre noeud qui la **restaure** depuis le dernier checkpoint durable — **zero tour perdu**, prouve par assertions de continuite. C'est la demonstration que la resilience l'emporte sur les sticky sessions (theorie section 4.2 / flash card Q5).

### Consigne

En reutilisant `Checkpoint` / `BaseCheckpointer` de `02-code/25-serving-stateful-sessions.py` (embarque, pas importe) :

1. Construis un `Cluster` compose de :
   - un **store de checkpoints partage** (in-memory durable, survit a la mort d'un worker).
   - N `Worker` **stateless** (`alive: bool`), chacun avec `handle(thread_id, message) -> str` qui load → traite (reponse mock taggee `worker_id`) → save.
   - un `Router` a affinite de session : `owner(thread_id) -> worker_id` (consistent hashing ou hash stable modulo l'ensemble des workers **vivants**), et un re-routage automatique vers un worker vivant quand l'owner courant est mort.
2. Implemente `Cluster.dispatch(self, thread_id, message) -> dict` :
   - resout l'owner vivant ; si l'owner historique est mort, choisit un nouvel owner vivant (failover) et le **logge** (`migrations` : liste de tuples `(thread_id, from_worker, to_worker)`).
   - le nouveau worker **restaure** l'etat depuis le store partage avant de traiter (donc aucun tour anterieur n'est perdu).
   - retourne `{"worker": worker_id, "reply": ..., "migrated": bool}`.
3. Implemente `Cluster.kill(worker_id)` (passe `alive=False`).
4. Scenario de continuite (entrelacement explicite, deterministe) :
   - 3 workers, 1 session `S`. Envoie 3 tours → ils tombent tous sur le meme owner (affinite).
   - `kill` l'owner de `S`. Envoie 2 tours supplementaires → ils sont traites par un **autre** worker (failover) qui a restaure l'historique.
   - **Assertions** : le `step` final du store == 5 (3 + 2, aucun tour perdu) ; l'historique contient les 5 tours dans l'ordre ; au moins une `migration` a ete enregistree pour `S` ; le worker des 2 derniers tours est vivant et **different** du worker mort.
5. Cas limite : tuer **tous** les workers sauf un et verifier que la session converge vers l'unique survivant sans crash ; tuer le dernier doit lever une erreur claire `NoLiveWorker` (le store survit, mais plus personne pour servir).

### Criteres de reussite

- [ ] Les workers sont stateless : le store partage est la seule source de verite (un worker mort ne fait perdre aucun checkpoint)
- [ ] Le routeur a affinite envoie un meme `thread_id` au meme owner tant qu'il est vivant
- [ ] La mort de l'owner declenche un **failover** vers un worker vivant, journalise dans `migrations`
- [ ] Continuite prouvee : `step` final == nombre total de tours, historique ordonne, **zero tour perdu ni duplique**
- [ ] Plus aucun worker vivant → `NoLiveWorker` levee proprement (pas de boucle infinie, pas d'`IndexError` brut)

---

## Exercice 2 : Manager de sessions concurrent — optimistic locking, conflit & retry idempotent

### Objectif

Resoudre le probleme de concurrence sur un meme `thread_id` du cours (section 4.3) sans vrais threads non-deterministes : detecter un **write conflict** entre deux requetes simultanees via un **controle de version optimiste** (compare-and-swap), faire **echouer** le perdant, puis le **rejouer** ; et garantir l'**idempotence** des retries via une cle d'idempotence (un meme tour rejoue ne s'applique qu'une fois).

### Consigne

1. Cree un `VersionedStore` ou chaque `thread_id` mappe vers `(version: int, state: dict)`. Methodes :
   - `read(thread_id) -> (version, state_copy)` (renvoie une **copie** de l'etat, jamais la reference partagee).
   - `compare_and_swap(thread_id, expected_version, new_state) -> bool` : applique `new_state` et incremente la version **seulement si** `expected_version == version_courante` ; sinon retourne `False` (conflit).
2. Cree un `ConcurrentSessionManager(store)` avec `apply_turn(thread_id, user_message, idempotency_key, max_retries=3) -> dict` :
   - lit `(version, state)`, verifie d'abord la cle d'idempotence : si `idempotency_key` figure deja dans `state["applied_keys"]`, **n'applique pas** le tour (retourne `{"status": "duplicate", "version": ...}`).
   - sinon construit le nouvel etat (append message + enregistre la cle dans `applied_keys`), tente le `compare_and_swap`.
   - en cas de conflit (`False`) : **retry** (relit la derniere version et recommence) jusqu'a `max_retries` ; retourne `{"status": "applied", "retries": k, "version": ...}` ou `{"status": "conflict_exhausted"}` si tous les essais echouent.
3. Simule un **entrelacement explicite** de deux requetes R1 et R2 sur le **meme** thread (pas de vrais threads — on ordonne les operations a la main) :
   - R1 lit la version v0. R2 lit aussi v0. R1 CAS(v0) → succes (version devient v1). R2 CAS(v0) → **conflit** (version != v0). R2 retry : relit v1, CAS(v1) → succes (version v2).
   - **Assertions** : R1 applique au 1er essai (`retries == 0`), R2 applique apres exactement 1 retry (`retries == 1`), version finale == 2, les **deux** messages sont presents dans l'etat (aucune ecriture perdue — *lost update* evite).
4. Prouve l'**idempotence** : rejoue R1 avec la **meme** `idempotency_key` → `status == "duplicate"`, la version **n'avance pas**, et le message n'est **pas** duplique dans l'historique.

### Criteres de reussite

- [ ] `compare_and_swap` echoue (False) si la version attendue ne correspond plus (detection de conflit)
- [ ] Deux requetes concurrentes sur le meme thread : le perdant detecte le conflit puis **retry** et reussit (aucun *lost update*, les 2 messages presents)
- [ ] Le nombre de retries est correct (`R1.retries == 0`, `R2.retries == 1`) et la version finale est exacte
- [ ] Un retry avec la meme `idempotency_key` est **idempotent** : statut `duplicate`, version inchangee, pas de message duplique
- [ ] `read` renvoie une copie defensive (modifier la copie ne corrompt pas l'etat du store)
