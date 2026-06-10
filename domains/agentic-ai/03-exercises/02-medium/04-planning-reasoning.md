# Exercices Medium — Planning & Reasoning (J4)

---

## Exercice 1 : Self-consistency — vote majoritaire sur N echantillons

### Objectif
Implementer le pattern self-consistency : echantillonner plusieurs raisonnements CoT et voter sur la reponse finale pour reduire les erreurs ponctuelles.

### Consigne
En partant du `MockLLM` de `02-code/04-planning-reasoning.py` :

1. Etends le `MockLLM` pour accepter un parametre `temperature` et un `seed` :
   - A `temperature=0`, il retourne toujours le meme raisonnement
   - A `temperature>0`, il retourne une variante parmi 5 raisonnements pre-ecrits pour la meme question (4 corrects, 1 faux — le faux doit arriver de maniere deterministe selon le seed)
2. Ecris une fonction `self_consistency(llm, question, n_samples=5) -> dict` qui :
   - Appelle le LLM `n_samples` fois avec des seeds differents
   - Extrait la reponse finale de chaque raisonnement (ligne `Reponse : X`)
   - Compte les votes par reponse et retourne `{"answer": ..., "votes": {...}, "confidence": votes_majoritaires / n_samples}`
3. Teste sur une question arithmetique multi-etapes ou 1 echantillon sur 5 donne une reponse fausse
4. Compare : la reponse a `temperature=0` seule vs la reponse self-consistency
5. Affiche le detail des votes et la confidence

### Criteres de reussite
- [ ] Le MockLLM produit des variantes deterministes selon le seed (pas de `random` sans seed)
- [ ] L'extraction de la reponse finale fonctionne sur les 5 variantes
- [ ] Le vote majoritaire retourne la bonne reponse malgre 1 echantillon faux
- [ ] La confidence est calculee correctement (ex: 4/5 = 0.8)
- [ ] Le script est 100% deterministe d'une execution a l'autre

---

## Exercice 2 : Replanning dynamique apres echec d'un step

### Objectif
Rendre le plan-and-execute robuste : quand un step echoue, le planner doit produire un plan corrige au lieu de continuer aveuglement.

### Consigne
Etends le pattern plan-and-execute :

1. Modifie `mock_search_tool` pour qu'une requete precise echoue : `search('population Bordeaux 2024')` retourne `"no_result"`
2. L'executor doit detecter le `no_result` et marquer le step comme `FAILED` (au lieu de stocker une valeur invalide dans le scratchpad)
3. Quand un step echoue, appelle le planner en mode **replan** :
   - Le prompt de replan contient : la question, le plan initial, le step qui a echoue, et le scratchpad courant
   - Le MockLLM retourne un plan corrige (ex: remplacer le step par `search('Bordeaux nombre habitants INSEE')` qui, lui, reussit)
4. L'executor reprend le plan corrige **a partir du step repare** (ne pas re-executer les steps deja reussis)
5. Limite a 2 replans maximum — au-dela, retourne une reponse d'echec explicite
6. Teste avec la question "Quelle est la densite de Bordeaux ?" (population via la requete reparee, area via une requete qui marche du premier coup)

### Criteres de reussite
- [ ] Le step en echec est detecte et marque FAILED (jamais de "no_result" dans le scratchpad)
- [ ] Le prompt de replan contient bien le step echoue et le scratchpad
- [ ] Les steps deja reussis ne sont pas re-executes apres replan
- [ ] La limite de 2 replans est respectee (teste avec une requete qui echoue toujours)
- [ ] La trace affiche clairement : plan initial -> FAILED -> replan -> succes

---

## Exercice 3 : Decomposition en DAG de sous-taches

### Objectif
Passer d'un plan lineaire a un graphe de dependances : certaines sous-taches peuvent s'executer en parallele, d'autres dependent de resultats precedents.

### Consigne
1. Definis une dataclass `SubTask` avec : `id: str`, `description: str`, `depends_on: list[str]`, `status: str` (pending/done/failed), `result: str | None`
2. Ecris une fonction `decompose(question) -> list[SubTask]` (mock deterministe) qui, pour la question "Compare la densite de Paris et de Lyon", retourne 5 sous-taches :
   - `t1` : population Paris (aucune dependance)
   - `t2` : surface Paris (aucune dependance)
   - `t3` : population Lyon (aucune dependance)... plus `t4` surface Lyon
   - `t5` : calculer et comparer les 2 densites (`depends_on: [t1, t2, t3, t4]`)
3. Ecris un scheduler `run_dag(tasks, executor) -> dict` qui :
   - A chaque "vague", identifie toutes les taches `pending` dont les dependances sont `done` (executables en parallele)
   - Les execute (sequentiellement dans le code, mais affiche `wave 1: [t1, t2, t3, t4]`, `wave 2: [t5]`)
   - Detecte les cycles (si aucune tache executable et des taches pending restent -> `RuntimeError("Cycle or unsatisfiable dependency")`)
4. Verifie avec un assert que `t5` recoit bien les 4 resultats et produit la comparaison correcte
5. Teste aussi un DAG avec un cycle volontaire (`t1` depend de `t2`, `t2` depend de `t1`) pour voir l'erreur

### Criteres de reussite
- [ ] Le DAG est construit avec les bonnes dependances
- [ ] Le scheduler regroupe les taches en vagues correctes (vague 1 = 4 taches, vague 2 = 1 tache)
- [ ] Une tache ne s'execute jamais avant ses dependances
- [ ] Le cycle est detecte avec une erreur claire
- [ ] L'assert sur la comparaison finale passe (Paris plus dense que Lyon)
