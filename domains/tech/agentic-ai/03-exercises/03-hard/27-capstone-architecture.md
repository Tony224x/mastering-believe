# Exercices Hard — Capstone architecture (J27)

> Ces exercices DURCISSENT les briques durables du capstone au niveau production
> (`02-code/27-capstone-architecture.py` : `DurableEngine`, `SQLiteCheckpointer`, ...).
> Les solutions embarquent une mini-version fidele des briques pour tourner offline.

---

## Exercice 1 : Matrice de robustesse crash-resume + detection de checkpoint corrompu

### Objectif

Prouver que la reprise apres crash du `DurableEngine` (J20) est **correcte a chaque frontiere d'etape**, pas seulement sur le scenario unique de la demo. Puis ajouter une detection de **checkpoint a moitie ecrit** (corruption) qui force la re-execution idempotente de l'etape concernee.

### Consigne

Construis une suite qui pousse l'engine bien au-dela du cas unique du `02-code` :

1. **Run de reference (no-crash)** : execute les N etapes sans crash et capture le contexte final `ref_ctx`.
2. **Ablation par frontiere** : pour CHAQUE etape `i` (de 0 a N), simule un crash `crash_before=steps[i].name` sur un fichier db neuf, puis **reprends** (nouvel engine, meme `run_id` + meme db) sans crash. Pour chaque point de crash, verifie :
   - (a) les etapes du prefixe deja journalise sont **skipped** ;
   - (b) seules les etapes du suffixe restant sont **executed** ;
   - (c) le contexte final est **identique** a `ref_ctx` (la reprise ne change pas le resultat).
   - Le cas `i == N` (crash apres la derniere etape) ne doit rien re-executer.
3. **Detection de corruption** : simule un checkpoint a moitie ecrit pour une etape — par exemple `step::<x>` journalise mais `__ctx__` incoherent (ne contient pas `x`), OU une valeur JSON invalide. Ecris `audit_corruption(checkpointer, run_id, steps)` qui detecte l'incoherence et liste les etapes a **rejouer**.
4. **Idempotence** : prouve qu'apres avoir invalide le checkpoint corrompu (suppression de `step::<x>`), une reprise re-execute exactement cette etape et reconverge vers `ref_ctx` — un compteur de side-effect montre qu'aucune AUTRE etape n'est rejouee.

### Criteres de reussite

- [ ] Un run de reference no-crash fournit `ref_ctx`
- [ ] Pour chaque frontiere de crash : prefixe skipped, suffixe executed, contexte final == `ref_ctx`
- [ ] Le crash apres la derniere etape ne re-execute rien a la reprise
- [ ] `audit_corruption` detecte un checkpoint incoherent (ctx ne reflete pas une etape journalisee, ou JSON invalide)
- [ ] Apres invalidation du checkpoint corrompu, la reprise re-execute SEULEMENT l'etape concernee (idempotence verifiee par compteur de side-effect)
- [ ] La matrice complete passe (toutes frontieres + cas corruption)

---

## Exercice 2 : Idempotence concurrente — journal-claim/lock anti double side-effect

### Objectif

Durcir le `DurableEngine` pour le cas **deux workers concurrents** partageant le meme `run_id` (re-livraison d'evenement, autoscaling). Sans verrou, les deux pourraient executer la meme etape → double effet de bord. Avec un **journal-claim** (lock atomique en SQLite), chaque etape s'execute **au plus une fois** ; le perdant reprend depuis le journal.

### Consigne

Implemente un `ClaimingDurableEngine` qui simule la concurrence de maniere deterministe (entrelacement explicite, pas de vrais threads requis) :

1. **Side-effect counter** : chaque `Step.fn` incremente un compteur global partage `EXEC_COUNTER[name]` a son execution. C'est l'instrument qui prouve l'unicite.
2. **Journal-claim atomique** : avant d'executer une etape, un worker tente de **reclamer** le droit via une ecriture conditionnelle SQLite (`INSERT` sur une table `claims(run_id, step)` avec contrainte d'unicite ; un `INSERT OR IGNORE` + verification du `rowcount`/relecture suffit). Le gagnant execute et journalise ; le perdant **n'execute pas** et attend/recharge le resultat journalise.
3. **Entrelacement adverse** : ecris un scenario ou worker A reclame et execute `code` pendant que worker B tente `code` juste apres — B doit perdre le claim et recharger le resultat de A, pas re-executer `fn`.
4. **Garantie at-most-once** : a la fin, `EXEC_COUNTER[name] == 1` pour CHAQUE etape, meme si chaque etape a ete "tentee" par les deux workers. Les deux workers convergent vers le **meme contexte final**.
5. **Crash du gagnant avant journal** : (bonus exige) si le gagnant crash APRES avoir reclame mais AVANT de journaliser, le claim doit pouvoir etre **repris** (TTL/relache ou re-claim sur absence de resultat journalise) pour eviter un blocage permanent — montre qu'apres ce crash, l'etape est finalement executee exactement une fois au total.

### Criteres de reussite

- [ ] Le claim SQLite est atomique : un seul worker gagne une etape donnee
- [ ] `EXEC_COUNTER[name] == 1` pour chaque etape malgre les tentatives des deux workers (at-most-once)
- [ ] Le worker perdant recharge le resultat journalise au lieu de re-executer `fn`
- [ ] Les deux workers convergent vers un contexte final identique
- [ ] Le cas "crash du gagnant avant journal" est repris sans blocage et l'etape finit executee exactement une fois au total
- [ ] Aucun double effet de bord observable sur tout le run
