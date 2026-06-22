# J27 — Capstone (architecture & setup) : concevoir un deep ops agent durable

> **Temps estime** : 4h | **Prerequis** : J1-J26
> **Objectif** : concevoir l'architecture d'un agent autonome avance qui assemble les patterns frontier du parcours (deep agent, durabilite, routing, isolation de contexte, eval), et poser les briques de base runnable avant le build complet du J28.

---

## 1. Le projet : "Deep ops agent durable, observable et auto-evalue"

On construit un agent qui recoit une tache d'ingenierie (ex : reparer un bug dans un petit repo) et la mene a bout **seul**, en combinant ce qu'on a appris :

- un **deep agent** : un planner qui decompose la tache en todos, ecrit dans un scratchpad / virtual filesystem (J15) plutot que de tout garder en contexte ;
- des **sous-agents a contexte isole** (J15/J9) : research, code, verify — chacun avec sa propre fenetre de contexte privee ;
- un **outil de coding** edit/search/run (J21) execute dans un sous-process restreint (J23) ;
- la **durabilite** (J20) : l'etat est checkpointe dans SQLite ; si le process crash, l'agent **reprend exactement** ou il en etait ;
- un **routeur de modele** mocke (J24) pour la conscience du cout ;
- un **harness d'evaluation** (J11/J26) : pass^k + rapport de regression sur l'agent lui-meme.

> **Contrainte de conception clef** : tout doit tourner **en local, sans cle API, sans GPU, sans serveur**. Le LLM est mocke par des politiques deterministes. On n'enseigne pas Temporal ni le computer-use *dans le livrable* (trop d'infra / trop fragile) — ils restent des encarts theoriques. Cette contrainte force une architecture propre et reproductible.

> **Analogie** : un chef de chantier (planner) distribue des taches a des ouvriers specialises (sous-agents) qui travaillent chacun dans leur atelier (contexte isole) ; un carnet de chantier (SQLite) note ce qui est fini, de sorte qu'apres une coupure de courant (crash) on reprend sans tout refaire.

---

## 2. Decoupage en contrats (interfaces)

La regle d'or d'un systeme durable : **separer ce qui est deterministe de ce qui ne l'est pas**, et journaliser les resultats des etapes non-deterministes. On definit 5 briques (implementees dans `02-code/27-capstone-architecture.py`).

### 2.1 VirtualFS — offloading de contexte

Un store `nom -> texte` sur disque. Les sous-agents y ecrivent notes et artefacts (todo.md, research.md, report.md) au lieu de gonfler le prompt. C'est la materialisation du context offloading vu en J15.

### 2.2 SQLiteCheckpointer — etat durable

Un store cle/valeur durable (`sqlite3`, fichier ou `:memory:`). Quand on utilise un vrai fichier, l'etat **survit au redemarrage du process** — c'est exactement la propriete sur laquelle repose la reprise apres crash.

```python
cp.put(run_id, "step::research", result)   # journalise une etape finie
cp.get(run_id, "step::research")           # relue au redemarrage
```

### 2.3 DurableEngine — sequence reprenable

Execute une liste de `Step` (unites de travail idempotentes). Apres chaque etape reussie, il **journalise** le resultat. Au redemarrage avec le meme `run_id` :

1. pour chaque etape, si un resultat est deja journalise → on le **recharge** (skip, pas de re-execution) ;
2. sinon on **execute**, on journalise, on continue.

C'est la difference de fond avec le checkpointing de graphe de J6 : ici, c'est le **workflow** qui survit a la mort du process, pas juste un snapshot d'etat en memoire.

### 2.4 ModelRouter — conscience du cout

Route une tache vers un modele "weak" (cheap) ou "strong" (cher) selon une heuristique de complexite, et cumule le cout (J24). Mocke : aucun appel reseau.

### 2.5 SubAgent — contexte isole

Classe de base d'un sous-agent qui garde un **buffer de contexte prive** (`_context`). L'orchestrateur ne recoit que le **resultat compact**, jamais l'historique verbeux : c'est l'isolation de contexte de J15 appliquee a une topologie multi-agent (J9).

---

## 3. Le flux durable, etape par etape

```
plan      -> ecrit todo.md (planner)
research  -> ResearchSubAgent  -> research.md
code      -> CoderSubAgent     -> edit/search/run sur repo jouet
verify    -> VerifierSubAgent  -> verdict
```

Chaque etape est un `Step` du `DurableEngine`. Si le process meurt **avant** `verify`, un second lancement avec le meme `run_id` recharge `plan/research/code` depuis SQLite et n'execute que `verify`. On *prouve* la reprise dans la demo : `skipped == ["plan","research","code"]`, `executed == ["verify"]`.

> **Pourquoi l'idempotence est obligatoire** : si une etape a un effet de bord (ecrire un fichier, appeler une API), rejouer son resultat journalise ne doit pas le re-declencher. Ici, on journalise le **resultat** et on saute l'execution — l'effet de bord ne se reproduit pas. En production durable (Temporal), c'est le meme principe : les activities sont rejouees depuis l'historique, pas re-executees.

---

## 4. Criteres d'acceptation du capstone

Le J28 devra satisfaire, **sans cle API** :

- [ ] `python 28-capstone-build-eval.py` tourne et termine une tache (bug fixe + verifie) ;
- [ ] une demo de **reprise apres crash** : relance qui ne refait pas le travail journalise ;
- [ ] un **rapport d'eval chiffre** : pass^k par cas + verdict de regression baseline vs candidate ;
- [ ] l'outil de coding edit/search/run tourne en **sous-process** sur un mini-repo jouet ;
- [ ] aucune dependance externe (stdlib + sqlite3), aucun reseau.

---

## 5. Decisions de design (et ce qu'on a ecarte)

| Decision | Choix | Pourquoi |
|----------|-------|----------|
| Durabilite | SQLite app-level | Survit au crash, zero serveur ; Temporal = encart theorique |
| Computer-use | Hors livrable | Trop fragile/non reproductible (J22 le couvre) |
| LLM | Mock deterministe | Reproductible, testable, pass^k mesurable |
| Coding tool | Sous-process sur repo jouet | Isolation simple (J23), pas de sandbox infra lourde |
| Routing | Heuristique mockee | Montre la conscience du cout sans multi-API |

---

## Flash-cards

**Q1 :** Quelle est la difference entre le DurableEngine (J27/J20) et le checkpointing de graphe vu en J6 ?
> **R :** Le checkpointing J6 sauvegarde un snapshot d'etat (souvent en memoire/DB) pour le human-in-the-loop ou le time-travel. Le DurableEngine fait survivre le **workflow** a la mort du process : au redemarrage, il recharge les etapes deja finies depuis SQLite et ne re-execute que les etapes manquantes.

**Q2 :** Pourquoi les etapes du DurableEngine doivent-elles etre idempotentes ?
> **R :** Parce qu'apres un crash, on rejoue le resultat journalise d'une etape sans la re-executer. Si une etape avait un effet de bord non protege, le rejouer pourrait le re-declencher. On journalise donc le resultat et on saute l'execution.

**Q3 :** Qu'est-ce que l'isolation de contexte des sous-agents apporte ici ?
> **R :** Chaque sous-agent garde un buffer de contexte prive et ne renvoie que son resultat compact a l'orchestrateur. Cela evite que l'historique verbeux d'un agent pollue la fenetre de contexte des autres (J15), tout en gardant une topologie multi-agent (J9).

**Q4 :** Pourquoi mocker le LLM dans le capstone plutot que d'appeler une vraie API ?
> **R :** Pour la reproductibilite et la testabilite : le harness pass^k a besoin de runs deterministes et l'ensemble doit tourner partout sans cle API ni cout. Une vraie cle reste branchable, mais n'est jamais requise.

**Q5 :** Pourquoi avoir ecarte le computer-use et Temporal du livrable ?
> **R :** Le computer-use (J22) est trop fragile/non reproductible pour un binaire qui doit tourner partout ; Temporal demande un serveur (infra lourde). Les deux sont enseignes mais restent hors du livrable executable, qui privilegie la durabilite app-level via SQLite.

---

## Points cles a retenir

- Le capstone **assemble** les patterns du parcours : deep agent (J15), memoire/sous-agents isoles (J15/J16/J9), verifiers (J17), durabilite (J20), coding tool (J21/J23), routing (J24), eval (J11/J26)
- **5 briques contractuelles** : VirtualFS, SQLiteCheckpointer, DurableEngine, ModelRouter, SubAgent
- La **durabilite app-level** (SQLite + reprise) donne 80% de la valeur de Temporal sans serveur
- **Separer deterministe / non-deterministe** et journaliser les resultats = coeur de la reprise sur crash
- Contraintes de design = reproductibilite : stdlib, mock LLM, zero reseau, zero GPU

---

## Pour aller plus loin

- Revoir J15 (context engineering), J20 (durable execution), J21 (coding agents), J24 (inference engineering), J26 (benchmarking) — le capstone est leur synthese
- Temporal, "Durable Execution meets AI" — https://temporal.io/blog/durable-execution-meets-ai-why-temporal-is-the-perfect-foundation-for-ai
- LangChain Deep Agents — https://docs.langchain.com/oss/python/deepagents/overview
- Sources tier-1 detaillees : voir [`REFERENCES.md`](../REFERENCES.md)
