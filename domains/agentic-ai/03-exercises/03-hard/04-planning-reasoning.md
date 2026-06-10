# Exercices Hard — Planning & Reasoning (J4)

---

## Exercice 1 : Tree-of-Thought avec beam search et backtracking

### Objectif
Implementer un vrai solveur Tree-of-Thought : generer plusieurs "pensees" candidates a chaque niveau, les scorer, garder les meilleures (beam), et revenir en arriere quand une branche est sans issue.

### Consigne
Construis un solveur ToT pour un puzzle deterministe, par exemple le "Game of 24" simplifie : a partir de 4 nombres (ex: `[4, 6, 8, 2]`), trouver une expression qui donne 24 en utilisant chaque nombre une fois (+, -, *, /).

1. **Generateur de pensees** : a chaque niveau, une "pensee" est une operation entre 2 nombres restants, ex : `"4 * 6 = 24, restants: [24, 8, 2]"`. Genere toutes les pensees possibles du niveau (ou un sous-ensemble si trop nombreuses)
2. **Evaluateur** : score chaque pensee de 0 a 1 (heuristique deterministe : 1.0 si un des nombres restants est 24 et qu'il ne reste que lui, sinon proximite de 24 atteignable, 0.0 si impasse evidente)
3. **Beam search** : garde les `beam_width=3` meilleures pensees par niveau, explore en profondeur
4. **Backtracking** : si une branche n'a plus de pensees possibles sans solution, remonte au niveau precedent et essaie la pensee suivante du beam
5. Retourne la solution complete (suite d'operations) + des stats : noeuds explores, noeuds elagues, profondeur max
6. Compare avec une approche "CoT lineaire" (une seule trajectoire gloutonne qui prend toujours la meilleure pensee) : montre un cas ou le glouton echoue et ou ToT trouve la solution

Architecture :
```
                [4, 6, 8, 2]
              /      |       \
       4*6=24     8-2=6      6-4=2      <- pensees candidates (scorees)
        /            |          \
   [24,8,2]      [4,6,6]      [8,2,2]   <- beam garde les 3 meilleures
       ...           ...          ...
```

### Criteres de reussite
- [ ] Le generateur produit toutes les operations valides entre paires de nombres restants
- [ ] L'evaluateur est deterministe et borne entre 0 et 1
- [ ] Le beam search garde au plus `beam_width` branches par niveau
- [ ] Le backtracking explore une autre branche quand la courante est une impasse
- [ ] Une solution valide est trouvee pour au moins 2 instances (ex: [4,6,8,2] et [1,5,5,5])
- [ ] Les stats (noeuds explores/elagues) sont affichees
- [ ] Le cas glouton-echoue / ToT-reussit est demontre

---

## Exercice 2 : Meta-controller — choisir la strategie de raisonnement selon la tache

### Objectif
Construire un meta-controller qui choisit dynamiquement la strategie de raisonnement (direct, CoT, plan-and-execute, reflexion) en fonction de la complexite de la question et d'un budget, puis mesurer le tradeoff cout/qualite.

### Consigne
1. Implemente 4 strategies derriere une interface commune `Strategy.run(llm, question) -> StrategyResult` avec `StrategyResult = {answer, llm_calls, tokens_estimes, trace}` :
   - `DirectStrategy` : 1 appel
   - `CoTStrategy` : 1 appel avec "think step by step"
   - `PlanExecuteStrategy` : planner + 1 appel par step + synthese
   - `ReflexionStrategy` : tentative + critique locale (checklist) + revision, max 2 rounds
2. Implemente un `MetaController` qui :
   - Classifie la question avec des heuristiques deterministes : nombre d'entites, presence de calcul, presence de comparaisons, longueur, mots-cles ("compare", "ratio", "etapes")
   - Choisit la strategie : trivial -> direct, calcul simple -> CoT, multi-source/comparaison -> plan-and-execute, exigence de format/qualite -> reflexion
   - Respecte un `budget_llm_calls` : si le budget restant ne permet pas la strategie ideale, **degrade** vers une strategie moins chere (et le note dans la trace)
3. Construis un banc de test de 8 questions variees avec leur reponse attendue et la strategie ideale attendue
4. Lance le banc 2 fois : budget large (30 calls) et budget serre (8 calls)
5. Produis un tableau comparatif : question | strategie choisie | calls utilises | reponse OK ?
6. Affiche les stats globales : taux de bonnes reponses, calls totaux, nombre de degradations

### Criteres de reussite
- [ ] Les 4 strategies sont interchangeables derriere la meme interface
- [ ] Le classifieur choisit la strategie attendue sur au moins 7 des 8 questions
- [ ] Le mode budget serre declenche au moins une degradation visible dans la trace
- [ ] Le tableau comparatif s'affiche proprement pour les 2 budgets
- [ ] Tout est deterministe (mock LLM, pas d'aleatoire non seede)
- [ ] Le compteur de calls est exact (verifie par un assert sur le total)
