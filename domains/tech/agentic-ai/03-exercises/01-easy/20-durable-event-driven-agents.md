# Exercices — Durable & event-driven agents (J20)

---

## Exercice 1 : Ajouter un event type `ACTIVITY_FAILED` au moteur durable

### Objectif
Comprendre comment un moteur de durable execution gere les echecs d'activities et les distingue des completions, afin de ne pas cacher un echec derriere une absence d'entree dans le log.

### Consigne
En partant de `02-code/20-durable-event-driven-agents.py` :

1. Modifie `DurableEngine.run_activity()` pour capturer les exceptions levees par l'activity :
   - Si l'activity leve une exception, journalise un evenement `ACTIVITY_FAILED` avec les champs `{"name": ..., "error": str(exc), "error_type": type(exc).__name__}`
   - Re-leve l'exception apres journalisation (le workflow peut decider de la gerer)
2. Modifie `replay()` pour charger les echecs connus : cree un dict `self._failed_activities: dict[str, str]` qui mappe nom d'activity → message d'erreur
3. Dans `run_activity()`, si l'activity est dans `_failed_activities`, affiche un avertissement `[REPLAY] Activity '...' previously failed: ...` et **re-execute l'activity** (on suppose que les echecs peuvent etre transitoires)
4. Cree une activity `activity_flaky(attempt_count: list)` qui echoue a la premiere invocation (`attempt_count[0] == 0`) et reussit a la deuxieme
5. Ecris un test qui :
   - Lance le workflow, l'activity echoue (journalise `ACTIVITY_FAILED`)
   - Recharge le moteur, replay, re-execute (l'activity reussit)
   - Verifie que le log contient les deux evenements (`ACTIVITY_FAILED` puis `ACTIVITY_COMPLETED`)

### Criteres de reussite
- [ ] `ACTIVITY_FAILED` est journalise avec `name`, `error`, `error_type`
- [ ] `replay()` peuple `_failed_activities` depuis le log
- [ ] Une activity precedemment echouee est RE-executee (pas skippee)
- [ ] `activity_flaky` echoue a l'appel 1 et reussit a l'appel 2
- [ ] Le log final contient dans l'ordre : `ACTIVITY_STARTED`, `ACTIVITY_FAILED`, `ACTIVITY_STARTED`, `ACTIVITY_COMPLETED`

---

## Exercice 2 : Filtrage par topic avec wildcards dans l'EventBus

### Objectif
Rendre l'EventBus plus expressif en supportant des patterns de topic avec wildcards, comme dans les systemes de messaging professionnels (AMQP topic exchange).

### Consigne
1. Etends `EventBus` pour supporter deux types de wildcards :
   - `*` remplace exactement un segment (ex. `order.*` matche `order.created` et `order.updated` mais pas `order.item.added`)
   - `#` remplace zero ou plusieurs segments (ex. `order.#` matche `order.created`, `order.item.added`, `order`)
   - Les segments sont separes par `.`
2. Implemente une methode privee `_matches(pattern: str, topic: str) -> bool` qui evalue le matching
3. Modifie `publish()` pour iterer sur tous les patterns enregistres et appeler les handlers dont le pattern matche le topic publie
4. Cree 4 souscriptions :
   - `order.*` -> handler_A
   - `order.#` -> handler_B
   - `payment.*` -> handler_C
   - `#` -> handler_all (matche tout)
5. Publie 4 evenements sur les topics : `order.created`, `order.item.added`, `payment.failed`, `system.health`
6. Verifie que chaque handler a recu exactement les bons evenements

### Criteres de reussite
- [ ] `_matches("order.*", "order.created")` retourne True
- [ ] `_matches("order.*", "order.item.added")` retourne False
- [ ] `_matches("order.#", "order.item.added")` retourne True
- [ ] `_matches("#", "anything.goes.here")` retourne True
- [ ] handler_A recoit 1 evenement (order.created), handler_B en recoit 2 (order.*)
- [ ] handler_all recoit les 4 evenements

---

## Exercice 3 : Saga pattern avec compensations

### Objectif
Implementer le pattern Saga (transactions distribuees avec compensation) par-dessus le moteur durable, pour garantir la coherence en cas d'echec partiel.

### Consigne
1. Definis une classe `SagaStep` avec les champs :
   - `name: str`
   - `activity: Callable` (l'action principale)
   - `compensation: Callable` (l'annulation, appelee si une etape suivante echoue)
2. Cree une classe `SagaEngine` qui prend un `DurableEngine` et une liste de `SagaStep`
3. Implemente `SagaEngine.run(args: dict) -> dict` :
   - Execute les steps dans l'ordre via `engine.run_activity()`
   - Si une step leve une exception, execute les compensations des steps deja completees dans l'ordre inverse
   - Journalise un evenement `SAGA_COMPENSATING` avant les compensations et `SAGA_FAILED` a la fin
   - Retourne `{"status": "compensated", "failed_at": step_name}` en cas d'echec
4. Definis 3 steps avec leurs compensations :
   - `reserve_stock` / `release_stock`
   - `charge_payment` / `refund_payment`
   - `ship_order` (cette step echoue intentionnellement) / `cancel_shipment`
5. Lance la saga, verifie que les compensations s'executent dans l'ordre inverse (`charge_payment` puis `reserve_stock`), et que le log contient les evenements de compensation

### Criteres de reussite
- [ ] `SagaStep` a les champs `name`, `activity`, `compensation`
- [ ] Les steps s'executent dans l'ordre
- [ ] Si `ship_order` echoue, `refund_payment` puis `release_stock` sont appeles (ordre inverse)
- [ ] `SAGA_COMPENSATING` est journalise avant les compensations
- [ ] Le resultat retourne `{"status": "compensated", "failed_at": "ship_order"}`
- [ ] La saga complete sans lever d'exception (les compensations absorbent l'echec)
