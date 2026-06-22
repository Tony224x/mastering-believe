# Exercices Hard — Durable & event-driven agents (J20)

---

## Exercice 1 : Moteur de workflow durable avec journal de steps et replay deterministe

### Objectif
Construire un mini-moteur de durable execution complet (cours sections 2.1-2.2) : chaque step est **journalise** dans un log append-only, le code de workflow est **deterministe** (sources de non-determinisme — `now()`, `uuid()`, `random()` — passees par des side-effects journalises facon Temporal), et apres un **crash simule au milieu**, le replay re-execute le workflow depuis le debut mais **ne rejoue PAS les steps deja committees** (memoisees depuis le journal). Tu dois prouver l'**exactly-once** des effets de bord a travers le crash, et que les valeurs non-deterministes capturees au run 1 sont **rejouees a l'identique** au run 2.

### Consigne
Construis un `WorkflowEngine` (stdlib, log append-only en `list` ou JSON-lines comme le module 20) :

1. **Journal append-only** : chaque entree `{seq, kind, key, value}`. Types (`kind`) : `STEP_COMPLETED` (resultat d'une step), `SIDE_EFFECT` (valeur non-deterministe capturee), `WORKFLOW_DONE`.
2. **API determinisme** exposee au workflow :
   - `engine.step(name, fn, *args)` : si `name` a un `STEP_COMPLETED` dans le journal → retourne la valeur journalisee **sans appeler `fn`** ; sinon execute `fn`, journalise le resultat, le retourne. (compteur reel d'executions a verifier)
   - `engine.side_effect(key, producer)` : pareil pour une valeur non-deterministe (ex : un id, un timestamp). Au run 1 elle est **produite et journalisee** ; au replay elle est **relue** (donc identique). `producer` ne doit JAMAIS etre rappele au replay.
3. **Workflow deterministe** : ecris une fonction `booking_workflow(engine)` qui enchaine 3-4 steps a effet de bord (chacune incremente un compteur `EXEC[name]`), en utilisant `side_effect` pour generer un `booking_id` et un `created_at` (sources de non-determinisme).
4. **Crash + replay** :
   - run 1 : execute le workflow jusqu'a un crash simule apres la 2e step (leve `RuntimeError` apres le commit) ;
   - run 2 : recree un engine **sur le meme journal**, re-execute `booking_workflow` du debut → les 2 premieres steps sont memoisees, les suivantes s'executent.
5. **Prouve par assertions** :
   - chaque step a effet de bord a `EXEC[name] == 1` apres le run complet (exactly-once malgre le replay),
   - le `booking_id` et le `created_at` du run 2 sont **identiques** a ceux du run 1 (determinisme rejoue ; le `producer` n'a pas ete rappele — verifie par un compteur),
   - le journal final contient les steps dans l'ordre, et le `WORKFLOW_DONE` n'apparait qu'une fois,
   - un run de reference sans crash produit le **meme** etat final que le run crash-puis-resume.

### Criteres de reussite
- [ ] Le journal est append-only ; `step()` et `side_effect()` journalisent leur resultat
- [ ] Une step deja journalisee est memoisee, `fn` n'est jamais rappele (compteur a 1)
- [ ] `side_effect()` rejoue la valeur journalisee au replay ; le `producer` n'est pas rappele
- [ ] Apres crash + resume, chaque effet de bord s'execute exactement une fois (exactly-once)
- [ ] `booking_id` et `created_at` sont identiques entre run 1 et run 2 (determinisme)
- [ ] L'etat final crash+resume == etat final d'un run de reference sans crash
- [ ] Execution offline, deterministe, sans dependance ni cle API

---

## Exercice 2 : Orchestrateur de saga durable avec compensation, replay et dead-letter

### Objectif
Aller au-dela de la saga "en memoire" de l'exercice easy : construire un orchestrateur de saga **durable** dont chaque action ET chaque compensation sont **journalisees**, qui survit a un crash **pendant la compensation** (la reprise ne rejoue pas les compensations deja faites), et qui route vers une **dead-letter queue** les echecs **non-compensables** (compensation elle-meme en echec). Tu dois prouver l'etat final coherent (tout est rollback) et l'idempotence des compensations a travers un crash.

### Consigne
Construis un `DurableSagaOrchestrator` au-dessus d'un journal append-only :

1. **Modele** : un `SagaStep(name, action, compensation, compensable=True)`. L'`action` mute un etat-monde (ex : un dict `world`) et l'orchestrateur enregistre sa compensation pour rollback.
2. **Journal** : evenements `STEP_DONE`, `STEP_FAILED`, `COMPENSATING` (debut, liste a compenser), `COMPENSATED {name}` (une compensation reussie), `SAGA_ROLLED_BACK`, `DEAD_LETTER {step, error}`.
3. **Execution** : exécute les steps dans l'ordre. A la **premiere** action qui echoue, demarre la compensation des steps deja committees **dans l'ordre inverse**.
4. **Compensation idempotente + reprise** : chaque compensation reussie journalise `COMPENSATED {name}`. Si l'orchestrateur est relance apres un crash **au milieu** de la phase de compensation, il **relit le journal**, identifie quelles compensations sont deja faites (`COMPENSATED`), et **ne rejoue que les restantes** (preuve par compteur que chaque compensation ne tourne qu'une fois).
5. **Dead-letter** : si une compensation **echoue** (step `compensable=False` ou compensation qui leve), journalise `DEAD_LETTER {step, error}`, place l'item dans une `dead_letter_queue` consultable, et **continue** les autres compensations (on ne bloque pas tout le rollback sur un seul item non-compensable).
6. **Scenario a tester** :
   - une saga `reserve_stock → charge_payment → ship_order(echoue)` : prouve que `refund_payment` puis `release_stock` sont compensees dans l'ordre inverse et que `world` revient a l'etat initial ;
   - un **crash simule** apres la 1ere compensation (`refund_payment`) : relance l'orchestrateur, prouve qu'il **ne rejoue pas** `refund_payment` (deja `COMPENSATED`) et termine `release_stock` ;
   - une saga ou une compensation est **non-compensable** : prouve que l'item part en `dead_letter_queue`, que `DEAD_LETTER` est journalise, et que les **autres** compensations s'executent quand meme.
7. **Prouve par assertions** : ordre inverse des compensations ; `world` coherent (rollback complet) sur le cas nominal ; chaque compensation idempotente (compteur a 1) malgre le crash ; la dead-letter queue contient exactement le(s) item(s) non-compensable(s).

### Criteres de reussite
- [ ] Actions et compensations sont journalisees (`STEP_DONE`, `COMPENSATED`, `DEAD_LETTER`, ...)
- [ ] A l'echec d'une action, les steps committees sont compensees dans l'ordre **inverse**
- [ ] Apres crash pendant la compensation, la reprise rejoue **uniquement** les compensations restantes (compteur prouvant l'exactly-once par compensation)
- [ ] Sur le cas nominal, `world` revient exactement a l'etat initial (rollback complet verifie)
- [ ] Une compensation non-compensable part en `dead_letter_queue` et journalise `DEAD_LETTER`, sans bloquer les autres compensations
- [ ] La saga absorbe l'echec sans laisser l'etat dans un etat incoherent
- [ ] Execution offline, deterministe, sans dependance ni cle API
