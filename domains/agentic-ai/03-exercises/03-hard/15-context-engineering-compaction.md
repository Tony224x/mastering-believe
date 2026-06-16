# Exercices Hard — Context engineering & token budgeting (J15)

---

## Exercice 1 : Allocateur de budget hierarchique + circuit-breaker global sur un arbre ReAct

### Objectif
Implementer le budgeting fin de la section 5 du cours : un budget de tokens **par profondeur d'arbre ReAct**, qui **decroit** a chaque niveau (facteur de decay), couple a un **budget global** servant de circuit-breaker. Un sous-arbre qui ferait exploser le budget global doit etre **vetote** (refuse) ou **escalade** vers le parent, et l'on doit prouver qu'aucun depassement global ne se produit, meme avec un arbre profond et large.

### Consigne
Construis un `HierarchicalBudgetAllocator` qui gouverne un arbre de noeuds ReAct (superviseur -> sous-agents -> sous-sous-agents) :

1. **Budget par profondeur** : `budget_at_depth(depth) = base_budget * (decay ** depth)` (ex : `base_budget=10_000`, `decay=0.4`). Un noeud a la profondeur `d` ne peut PAS consommer plus que `budget_at_depth(d)`.
2. **Budget global (circuit-breaker)** : un compteur `global_spent` borne par `global_budget`. Toute consommation est d'abord testee contre le budget local du noeud **ET** contre le budget global restant.
3. **Methode `request(node_id, depth, tokens) -> str`** qui retourne un verdict parmi :
   - `"granted"` : la consommation tient dans le budget local du noeud ET dans le budget global → on l'enregistre (incremente `global_spent` et le `local_spent` du noeud).
   - `"veto_local"` : depasse le budget local du noeud (a profondeur d) → refus, rien n'est consomme.
   - `"escalate_global"` : tient localement mais ferait depasser le budget global → refus, rien n'est consomme, et l'evenement est trace pour escalade au parent.
4. **Methode `spawn(parent_id, depth) -> bool`** : limite la profondeur a `max_depth` (ex : 3). Au-dela, le spawn est refuse (retourne `False`).
5. **Invariant a garantir** : a tout instant, `global_spent <= global_budget` et, pour chaque noeud, `local_spent[node] <= budget_at_depth(depth[node])`.
6. Teste un arbre realiste : 1 superviseur (depth 0), K sous-agents (depth 1), K^2 sous-sous-agents (depth 2), avec des demandes variees dont **certaines** depassent le budget local et **certaines** declenchent le circuit-breaker global. Construis le scenario de sorte qu'au moins un `veto_local`, un `escalate_global` et plusieurs `granted` soient produits.

### Criteres de reussite
- [ ] `budget_at_depth` decroit avec la profondeur selon le facteur de decay
- [ ] `request` retourne les 3 verdicts (`granted`, `veto_local`, `escalate_global`), chacun observe au moins une fois (assertion)
- [ ] Une demande vetotee/escaladee ne consomme **rien** (ni local ni global), verifie par assertion
- [ ] L'invariant `global_spent <= global_budget` tient a la fin ET a chaque etape (assertion)
- [ ] L'invariant `local_spent[node] <= budget_at_depth(depth)` tient pour tous les noeuds (assertion)
- [ ] `spawn` refuse de depasser `max_depth` (assertion)
- [ ] Execution offline, deterministe, sans dependance

---

## Exercice 2 : Boucle deep-agent complete (auto-compaction + offloading VFS + isolation par sous-agent)

### Objectif
Assembler les 3 leviers du module en une boucle deep-agent de bout en bout et **prouver** deux proprietes que les implementations naives ratent : (a) un run long se **termine sans overflow** de contexte grace a l'auto-compaction + offloading, et (b) l'**isolation par sous-agent** garde le contexte du parent **petit** pendant que les sous-agents font le gros du travail (le parent ne recoit que des resultats compactes, pas les logs internes — section 4.3).

### Consigne
Construis un mini framework deep-agent self-contained :

1. **`ContextWindow`** : encapsule une liste de messages + un `token_limit`. Methode `add(role, content)` qui auto-compacte si on depasse un seuil (`threshold = ratio * token_limit`), en gardant les `keep_tail` derniers messages + un resume fixe des autres. Expose `tokens()` et `compactions`.
2. **`VFS`** : `write/read/list_files`, comme au module.
3. **`SubAgent`** : possede sa **propre** `ContextWindow` (contexte isole). Methode `run(task: str, big_inputs: list[str]) -> str` qui :
   - recoit un prompt de tache **compact** (~ quelques dizaines de tokens), pas l'historique du parent,
   - traite des entrees volumineuses (`big_inputs`) qu'il **offload** dans le VFS (ne les garde pas inline),
   - effectue plusieurs tours internes (qui font grossir SON contexte et declenchent SES compactions),
   - retourne un resultat **compacte** (~ <= 50 tokens) au parent.
4. **`ParentAgent`** : possede sa propre `ContextWindow`. Methode `orchestrate(subtasks: list[tuple[str, list[str]]]) -> dict` qui, pour chaque sous-tache :
   - cree un `SubAgent` frais (contexte vierge),
   - lui delegue la tache avec un prompt court,
   - n'ajoute a SON contexte que le **resultat compacte** rendu par le sous-agent (jamais les tours internes ni les `big_inputs`).
5. **Scenario de stress** : au moins 5 sous-taches, chacune avec plusieurs entrees volumineuses (milliers de chars), et chaque sous-agent faisant plusieurs tours internes.
6. **Prouve par assertions** :
   - le run **complet** se termine et, a la fin, le contexte du parent **ET** celui de chaque sous-agent restent **sous leur `token_limit`** (aucun overflow),
   - le contexte final du parent reste **petit** (ex : nettement inferieur a la somme des tokens reellement traites par les sous-agents) — quantifie le ratio "tokens traites par les sous-agents / tokens dans le parent" et verifie qu'il est grand (ex : > 10x),
   - les gros inputs ne sont **jamais** dans le contexte du parent (verifie en cherchant un marqueur unique de gros input dans le contexte parent : absent),
   - les gros inputs restent **recuperables** depuis le VFS (round-trip),
   - au moins une compaction a eu lieu cote sous-agent (le scenario stresse vraiment le contexte).

### Criteres de reussite
- [ ] Chaque `SubAgent` a un contexte **isole** (sa propre `ContextWindow`)
- [ ] Le run complet se termine sans qu'aucun contexte ne depasse son `token_limit` (assertion)
- [ ] Le contexte du parent reste petit : ratio (tokens traites par sous-agents / tokens parent) > 10x (assertion)
- [ ] Les gros inputs sont absents du contexte parent mais recuperables depuis le VFS (assertions)
- [ ] Au moins une auto-compaction se declenche cote sous-agent (assertion)
- [ ] Le parent n'integre que des resultats compactes (<= plafond de tokens par resultat)
- [ ] Execution offline, deterministe, sans dependance reseau
