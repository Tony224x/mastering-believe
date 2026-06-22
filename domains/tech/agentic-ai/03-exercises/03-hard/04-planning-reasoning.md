# Exercices Hard — Planning & Reasoning (J4)

---

## Exercice 1 : Tree-of-Thought avec beam search et pruning

### Objectif
Implementer un vrai Tree-of-Thought (ToT) : generation de branches, evaluation, beam search avec elagage. Le cas d'ecole du module (Game of 24) rend le succes verifiable sans LLM reel.

### Consigne
Resous le **Game of 24** avec ToT : etant donne 4 nombres, trouver une combinaison d'operations (+, -, *, /) qui donne 24.

1. **State** : un noeud de l'arbre = `(remaining_numbers: tuple, operations: list[str])`.
2. **Thought generation** : depuis un state, generer tous les enfants possibles = choisir 2 nombres, une operation, et les remplacer par le resultat. (C'est le "LLM qui propose des continuations", ici deterministe et exhaustif par paire.)
3. **State evaluation** : l'evaluateur d'etat est le "mock LLM judge". Implemente-en **deux** pour montrer que la qualite du juge compte autant que la largeur du beam :
   - `evaluate_cheap(state)` : juge faible, sans lookahead (juste la proximite des nombres restants a 24 / facteurs utiles). Il elague facilement le bon chemin.
   - `evaluate_strong(state)` : juge fort, avec une sonde de solvabilite bornee (recursion exacte sur <= 4 nombres). Il garde les etats reellement solvables.
   - Documente que ces evaluateurs seraient des appels LLM en prod (low-effort vs high-effort).
4. **Beam search** : a chaque profondeur, ne garder que les `beam_width` meilleurs noeuds (pruning). `beam_width` et `evaluator` sont des parametres.
5. **Search** : explorer jusqu'a ce qu'un state atteigne un seul nombre == 24 (a epsilon pres pour les flottants), ou epuisement.
6. Compare **juge faible + beam_width=1** (greedy, echoue : il elague le bon chemin) vs **juge fort + beam_width=5** (trouve la solution) sur au moins 2 jeux : `(4, 6, 8, 2)` et `(1, 3, 4, 6)` (solution classique : 6/(1-3/4)=24).
7. Reconstruis et affiche le chemin d'operations de la solution trouvee, et **verifie-le** en rejouant les operations sur le multiset de nombres de depart.

### Criteres de reussite
- [ ] La generation d'enfants est correcte (toutes les paires × 4 operations + 2 divisions, division par zero geree)
- [ ] Deux evaluateurs (faible/fort) sont implementes et branchables sur le search
- [ ] Le pruning ne garde que `beam_width` noeuds par niveau
- [ ] juge fort + beam_width=5 trouve une solution pour `(1, 3, 4, 6)` la ou juge faible + greedy echoue
- [ ] La solution affichee est verifiable : rejouer les operations sur le multiset redonne bien 24
- [ ] Le code compte les noeuds explores et montre que le pruning borne l'exploration
- [ ] Tout est deterministe et tourne offline (aucune dependance)

---

## Exercice 2 : Pipeline de raisonnement multi-strategie auto-evalue

### Objectif
Construire un orchestrateur qui combine CoT, self-consistency ET reflexion en un pipeline, avec un juge externe verifiable, et qui mesure le gain de chaque etage. C'est le "Agent supervisor" decrit en fin de module (mix de strategies par role).

### Consigne
Cree un `ReasoningPipeline` sur un mini-benchmark de 5 problemes de logique a **reponse verifiable localement** (la vraie reponse est connue, donc on peut mesurer la justesse sans LLM).

1. **Benchmark** : 5 problemes `(question, ground_truth)` ou la verite est calculable (arithmetique, deduction). Un `verify(answer, ground_truth) -> bool` deterministe.
2. **Etage 1 — baseline direct** : reponse en un coup. Mesure l'accuracy.
3. **Etage 2 — CoT + self-consistency** : N echantillons avec un MockLLM dont la "temperature" simule des erreurs aleatoires (parfois bon, parfois faux), puis vote majoritaire. Mesure l'accuracy.
4. **Etage 3 — Reflexion ancree** : sur les reponses encore fausses apres l'etage 2, applique une boucle de reflexion ou la critique est le **juge verifiable** (execution-based : on sait si c'est faux), et on autorise 1 retry guide. Mesure l'accuracy finale.
5. **Rapport** : un tableau qui montre accuracy et nb d'appels LLM par etage, et qui prouve que chaque etage ameliore (ou non) le resultat — avec un `assert` que l'accuracy finale >= accuracy baseline.
6. **Cost accounting** : compte total des appels LLM, et calcule le ratio "points d'accuracy gagnes par appel LLM" (efficience marginale).

### Criteres de reussite
- [ ] Le benchmark a 5 problemes a verite verifiable localement
- [ ] Le MockLLM simule un taux d'erreur realiste controlable (seed deterministe)
- [ ] L'etage self-consistency ameliore l'accuracy vs baseline (mesure chiffree)
- [ ] La reflexion n'est appliquee qu'aux cas encore faux (pas de gaspillage)
- [ ] La critique de reflexion est ancree (juge verifiable, pas une auto-evaluation libre)
- [ ] Le rapport montre accuracy + nb d'appels LLM par etage
- [ ] L'efficience marginale (accuracy gagnee / appel) est calculee et commentee
- [ ] `assert accuracy_finale >= accuracy_baseline` passe, et le tout est deterministe
