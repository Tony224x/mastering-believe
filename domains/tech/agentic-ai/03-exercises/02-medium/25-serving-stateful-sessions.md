# Exercices Medium — Serving stateful & sessions (J25)

> **Prerequis** : avoir lu `01-theory/25-serving-stateful-sessions.md` et execute `02-code/25-serving-stateful-sessions.py`.
> Tout est **simule en memoire**, deterministe, offline : pas de Redis/Postgres reel, pas de reseau, pas de cle API. Les backends sont modelises par des stores Python.

---

## Exercice 1 : Store de sessions LRU + TTL avec eviction prouvee

### Objectif

Aller plus loin que le `MemoryCheckpointer` du cours (section 2) : construire un store de sessions borne en taille (eviction **LRU**) ET dans le temps (**TTL**), comme un cache L1 Redis (theorie sections 3.4 et 5.2). On prouve l'ordre exact d'eviction et l'expiration TTL sans jamais dependre de l'horloge reelle.

### Consigne

En partant des classes `Checkpoint` / `BaseCheckpointer` de `02-code/25-serving-stateful-sessions.py` :

1. Cree une classe `LRUTTLSessionStore(capacity: int, ttl: float, clock=...)` :
   - `clock` est un **callable** retournant un timestamp (par defaut `time.time`). En test on injecte une horloge fausse pour etre deterministe — n'utilise jamais `time.sleep`.
   - Stockage interne via un `collections.OrderedDict` keye par `thread_id`.
2. Implemente `save(self, checkpoint: Checkpoint) -> list[str]` :
   - met a jour (ou insere) l'entree, l'estampille avec `clock()` et la marque **most-recently-used** (deplacee en fin d'`OrderedDict`).
   - si la taille depasse `capacity`, evince les entrees **least-recently-used** (debut de l'`OrderedDict`) jusqu'a revenir a `capacity`.
   - retourne la **liste ordonnee** des `thread_id` evinces par cet appel (vide si aucun).
3. Implemente `load(self, thread_id: str) -> Checkpoint | None` :
   - retourne `None` si la cle est absente.
   - retourne `None` (et **supprime** l'entree, expiration paresseuse) si `clock() - stamp >= ttl`.
   - sinon, marque l'entree comme most-recently-used et la retourne.
4. Implemente `purge_expired(self) -> list[str]` qui supprime toutes les entrees expirees et retourne leurs `thread_id`.
5. Demontre :
   - capacite 3, on sauve A, B, C, on `load(A)` (A redevient recent), puis on sauve D → l'evince est **B** (le vrai LRU), pas A.
   - avec une horloge fausse avancee au-dela du `ttl`, `load` d'une vieille session retourne `None` et `purge_expired` la liste.

### Criteres de reussite

- [ ] `LRUTTLSessionStore` accepte une horloge injectable (deterministe, sans `sleep`)
- [ ] `save` evince le **least-recently-used** quand `capacity` est depassee et retourne la liste ordonnee des evinces
- [ ] Un `load` recent protege une session de l'eviction (ordre LRU correct : B evince avant A)
- [ ] `load` d'une session expiree (TTL) retourne `None` et supprime l'entree
- [ ] `purge_expired` retourne exactement les `thread_id` expires

---

## Exercice 2 : Load balancer a session affinity par consistent hashing

### Objectif

Implementer le routage stateful du cours (section 4.1) : router chaque `thread_id` vers un worker via un **anneau de hachage coherent**, garantir l'**affinite de session** (un meme thread retombe toujours sur le meme noeud) et limiter le remappage quand on ajoute un noeud — contrairement a un `hash % N` naif.

### Consigne

1. Cree une classe `ConsistentHashRouter(nodes: list[str], vnodes: int = 100)` :
   - construit un anneau : pour chaque noeud, `vnodes` points virtuels positionnes via `hashlib.md5(f"{node}#{i}".encode()).hexdigest()` converti en `int` (hash stable, deterministe, **pas** `hash()` builtin qui varie entre runs).
   - garde les positions triees.
2. Implemente `route(self, thread_id: str) -> str` : hashe le `thread_id`, trouve le **premier** point virtuel `>=` sur l'anneau (avec `bisect`), wrap-around vers le debut sinon, retourne le noeud proprietaire.
3. Implemente `add_node(self, node: str) -> None` et `remove_node(self, node: str) -> None` qui reconstruisent l'anneau.
4. Demontre l'affinite et la stabilite :
   - route 1000 `thread_id` distincts sur 3 noeuds → memo l'assignation initiale.
   - verifie l'**affinite** : re-router les memes ids donne exactement la meme assignation (deterministe).
   - ajoute un 4e noeud, re-route les 1000 ids, et verifie que la **fraction remappee est < 50 %** (en pratique ~1/4), bien meilleure que le `hash % N` naif. Compare en calculant aussi le remappage qu'aurait produit `hash % N` (qui remappe la grande majorite des cles) et verifie que le consistent hashing remappe **strictement moins**.

### Criteres de reussite

- [ ] L'anneau utilise un hash **stable** (`hashlib`, pas le builtin `hash`) → resultats reproductibles entre runs
- [ ] `route` est deterministe : meme `thread_id` → meme noeud a chaque appel (affinite de session)
- [ ] Tout `thread_id` est route vers un noeud existant (jamais de `None`)
- [ ] Apres `add_node`, la fraction de cles remappees est < 0.5 et **strictement inferieure** au remappage d'un `hash % N` naif
- [ ] `remove_node` reroute les sessions de ce noeud sans toucher (ou peu) les autres

---

## Exercice 3 : Checkpoint / restore — reprise apres crash worker

### Objectif

Materialiser le pattern "workers stateless + store externe" du cours (sections 1 et 4) : un worker traite des tours, **crashe** au milieu d'un tour, et un **autre** worker reprend la session depuis le dernier checkpoint durable **sans perdre de tour valide**.

### Consigne

En reutilisant `Checkpoint`, `SQLiteCheckpointer` (ou un store en memoire equivalent) et l'idee de `SessionManager.process_turn` de `02-code/25-serving-stateful-sessions.py` :

1. Cree une classe `ResilientWorker(worker_id: str, store: BaseCheckpointer)` avec `process_turn(self, thread_id, user_message, crash_after_compute=False) -> str` qui :
   - charge le dernier checkpoint (ou en cree un vide a `step=0`).
   - ajoute le message user, calcule une reponse assistant mock taggee `worker_id`.
   - **si `crash_after_compute=True`** : leve `WorkerCrash` **avant** d'appeler `store.save(...)` (le tour n'est donc PAS persiste — atomicite : on ne sauve qu'a la fin).
   - sinon : sauvegarde le nouveau checkpoint (`step + 1`) et retourne la reponse.
2. Scenario :
   - worker `W1` traite 2 tours valides sur `thread-42` (step passe a 2).
   - `W1` tente un 3e tour avec `crash_after_compute=True` → `WorkerCrash` levee, rien de persiste.
   - on instancie `W2` (nouveau worker, **meme store**), qui reprend `thread-42` et rejoue le 3e tour normalement.
3. Verifie la **continuite sans perte** :
   - apres le crash, `store.load("thread-42").step == 2` (le tour crashe n'a pas avance le step).
   - apres reprise par `W2`, `step == 3` et l'historique contient les 2 tours de W1 **plus** le tour repris par W2 (le tag worker du dernier message assistant doit etre `W2`).
   - aucun message duplique (pas de double-comptage du tour crashe).

### Criteres de reussite

- [ ] `process_turn` ne persiste **qu'apres** le calcul complet (un crash avant `save` ne laisse aucun etat partiel)
- [ ] Apres le crash, le `step` du dernier checkpoint est inchange (pas de tour fantome)
- [ ] Un autre worker (meme store) reprend la session et fait avancer le `step`
- [ ] L'historique final est coherent : tours de W1 conserves + tour repris par W2, sans doublon
- [ ] Le tag `worker_id` permet de prouver quel worker a traite le dernier tour
