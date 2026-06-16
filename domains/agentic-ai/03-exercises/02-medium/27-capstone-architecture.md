# Exercices Medium — Capstone architecture (J27)

> Ces exercices ETENDENT les briques durables du capstone (`02-code/27-capstone-architecture.py` :
> `VirtualFS`, `SQLiteCheckpointer`, `DurableEngine`, `ModelRouter`, `SubAgent`).
> Ne reimplemente pas le systeme entier : durcis/ajoute une brique a la fois.
> Les solutions fournies embarquent une mini-version fidele des briques pour tourner offline.

---

## Exercice 1 : DurableEngine avec retry-with-backoff et journalisation au succes

### Objectif

Durcir le `DurableEngine` (J20) face aux echecs **transitoires** : une etape peut echouer plusieurs fois avant de reussir (reseau, rate-limit), et l'engine doit reessayer avec un backoff borne **sans jamais journaliser un echec** ni rejouer une etape deja journalisee.

### Consigne

En partant de `02-code/27-capstone-architecture.py` :

1. Cree `RetryingDurableEngine(DurableEngine)` qui ajoute un parametre `max_attempts` (defaut 3) et un `backoff_base` (defaut 0, pour rester deterministe et instantane en test).
2. Surcharge `run` : pour chaque etape non journalisee, appelle `step.fn(ctx)` dans une boucle de retry. Si `fn` leve une exception, reessaie (jusqu'a `max_attempts`), en accumulant le delai logique `backoff_base * 2**(attempt-1)` dans un compteur `self.total_backoff` (sans `sleep` reel).
3. **Journalise uniquement au succes** : on n'ecrit `step::<name>` et `__ctx__` dans le checkpointer **qu'apres** un appel reussi. Si toutes les tentatives echouent, propage la derniere exception (l'etape reste non journalisee → reprise possible).
4. Trace le nombre d'essais par etape dans `self.attempts: dict[str, int]`.
5. Teste un scenario ou une etape echoue **deux fois puis reussit** : verifie qu'elle est journalisee **une seule fois**, qu'une reprise (nouvel engine, meme `run_id` + meme fichier db) la **skip** (0 nouvelle execution de son `fn`), et que `attempts` reflete bien les 3 essais au premier run.

### Criteres de reussite

- [ ] `RetryingDurableEngine` reessaie une etape transitoire jusqu'a `max_attempts`
- [ ] Une etape n'est journalisee qu'apres un appel reussi (jamais sur echec)
- [ ] `self.attempts` compte les essais (ex : 3 pour l'etape flaky)
- [ ] `self.total_backoff` accumule le backoff logique sans `sleep` reel
- [ ] A la reprise, l'etape journalisee est skip et son `fn` n'est pas rappele (compteur de side-effect inchange)
- [ ] Une etape qui echoue toujours propage l'exception et reste non journalisee

---

## Exercice 2 : ModelRouter avec plafond de budget et downgrade sous pression

### Objectif

Etendre le `ModelRouter` (J24) avec un **plafond de cout dur** et une politique de **downgrade sous pression** : router une tache complexe vers `strong` tant que le budget restant le permet, sinon basculer sur `weak` (degradation gracieuse) plutot que de depasser le plafond.

### Consigne

1. Cree `CappedRouter` (sous-classe ou dataclass inspiree de `ModelRouter`) avec un champ `budget: float` (plafond dur) en plus de `threshold`, `cost_weak`, `cost_strong`.
2. Surcharge `route(task)` :
   - calcule le tier "souhaite" comme le `ModelRouter` (`strong` si `len(task.split()) >= threshold`, sinon `weak`) ;
   - si le tier souhaite est `strong` mais que `total_cost + cost_strong > budget`, **downgrade** vers `weak` (note l'evenement dans `self.downgrades += 1`) ;
   - si meme `weak` ferait depasser le budget, **n'execute pas** : retourne `None` (ou leve `BudgetExceeded`, au choix documente) et n'incremente ni cout ni compteur.
3. Garantis l'invariant : `total_cost <= budget` **a tout moment**.
4. Teste : (a) un budget large route une tache complexe en `strong` ; (b) un budget serre force au moins un `downgrade` strong→weak observable ; (c) verifie `total_cost <= budget` apres une rafale de taches mixtes ; (d) un budget epuise refuse la tache suivante proprement.

### Criteres de reussite

- [ ] Tache complexe routee `strong` quand le budget le permet
- [ ] Downgrade strong→weak quand `strong` depasserait le budget (`self.downgrades >= 1`)
- [ ] Invariant `total_cost <= budget` verifie a chaque etape
- [ ] Refus propre (None/exception) quand meme `weak` ne rentre pas, sans corrompre `total_cost`
- [ ] Les complexes restent routees `strong` tant que c'est finançable

---

## Exercice 3 : Audit d'integrite VirtualFS <-> checkpointer

### Objectif

Detecter une **desynchronisation** entre le journal du `DurableEngine` (etapes finies dans SQLite) et les artefacts ecrits sur le `VirtualFS` (scratchpad). En production durable, une etape journalisee dont l'artefact disque manque (ou inversement) est une corruption a signaler.

### Consigne

1. Definis une convention etape → artefact attendu, ex : `{"plan": "todo.md", "research": "research.md", "code": "report.md"}`.
2. Fais tourner un `DurableEngine` ou chaque `Step.fn` ecrit son artefact sur un `VirtualFS` partage (en plus de retourner son resultat).
3. Cree `audit_integrity(checkpointer, fs, run_id, expected) -> dict` qui retourne :
   - `journaled`: les etapes presentes dans le journal (cles `step::*`) ;
   - `missing_artifacts`: etapes journalisees dont l'artefact attendu est **absent** du `VirtualFS` ;
   - `orphan_artifacts`: artefacts presents sur le FS mais dont l'etape n'est **pas** journalisee ;
   - `consistent: bool` (True ssi les deux listes sont vides).
4. Teste 3 cas : (a) run complet et coherent → `consistent True` ; (b) on **supprime** `report.md` apres coup alors que `code` est journalise → `missing_artifacts == ["code"]`, `consistent False` ; (c) on ecrit un artefact orphelin sans etape journalisee → `orphan_artifacts` non vide.

### Criteres de reussite

- [ ] `audit_integrity` lit le journal via `checkpointer.keys(run_id)` et l'etat du `VirtualFS`
- [ ] Un artefact manquant pour une etape journalisee est detecte (`missing_artifacts`)
- [ ] Un artefact orphelin (sans etape journalisee) est detecte (`orphan_artifacts`)
- [ ] `consistent` est True seulement quand journal et FS concordent
- [ ] Les 3 cas (coherent / artefact manquant / orphelin) donnent les bons verdicts
