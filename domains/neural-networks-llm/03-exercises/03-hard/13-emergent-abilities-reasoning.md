# Exercices Hard — Jour 13 : Emergent abilities & reasoning

---

## Exercice 7 : Best-of-N, majority vote et reward models (PRM vs ORM)

### Objectif

Implementer et comparer trois strategies de test-time compute scaling — majority vote, best-of-N avec un Outcome Reward Model (ORM), et best-of-N avec un Process Reward Model (PRM) — et reproduire la forme des courbes de scaling du cours.

### Consigne

On simule un reasoner produisant des "chaines" de raisonnement. Chaque chaine a `k` etapes ; chaque etape est correcte avec proba `p_step`. La reponse finale est correcte ssi toutes les etapes sont correctes (modele simplifie).

1. **Generateur de chaines** : produire `N` chaines, chacune = liste de booleens (etape correcte ou non) + une reponse finale (l'entier "vrai" si toutes correctes, sinon une valeur erronee plausible).

2. **Strategie A — Majority vote (self-consistency)** : prendre la reponse la plus frequente parmi les N.

3. **Strategie B — Best-of-N + ORM** : un Outcome Reward Model NOISY note la reponse FINALE (score = proba que ce soit bon + bruit). On garde la reponse au meilleur score. Modeliser l'ORM comme : `score = 1.0 si reponse correcte else 0.0`, plus un bruit gaussien (l'ORM est imparfait).

4. **Strategie C — Best-of-N + PRM** : un Process Reward Model note CHAQUE etape. Le score d'une chaine = produit (ou somme des log) des scores d'etapes. Modeliser : chaque etape correcte a un score moyen plus eleve qu'une etape fausse, + bruit. Le PRM detecte mieux les chaines partiellement fausses.

5. **Courbes de scaling** : pour N ∈ {1, 2, 4, 8, 16, 32, 64}, tracer l'accuracy des 3 strategies. Reproduire les observations du cours :
   - Toutes s'ameliorent avec N (test-time compute scaling)
   - PRM > ORM > majority vote (en general), surtout sur les problemes a beaucoup d'etapes
   - Le rendement est decroissant (log-lineaire)

6. Analyser :
   - Pourquoi le PRM bat l'ORM sur les longues chaines ? (il peut crediter une chaine "presque juste" et detecter ou ca derape)
   - Pourquoi la majority vote echoue quand le modele a un biais systematique (toutes les chaines font la meme erreur) ?
   - Lien avec o1 / DeepSeek R1 : qu'est-ce que le test-time compute change par rapport au scaling du pre-training ?

### Criteres de reussite

- [ ] Les 3 strategies (majority, ORM, PRM) sont implementees correctement
- [ ] L'accuracy croit avec N pour les 3 (scaling de test-time compute)
- [ ] PRM >= ORM >= majority sur les chaines longues (reproduit, avec explication)
- [ ] Le rendement decroissant (log-lineaire) est visible
- [ ] L'analyse PRM vs ORM et le lien avec o1/R1 sont corrects

---

## Exercice 8 : Tree-of-Thought — recherche dans l'espace des raisonnements

### Objectif

Implementer un Tree-of-Thought (ToT) sur un probleme concret (le jeu "Game of 24" simplifie ou un probleme de recherche), avec exploration BFS/DFS guidee par une heuristique d'evaluation, et comparer a un CoT lineaire.

### Consigne

On choisit un probleme de recherche : etant donne un ensemble de nombres et une cible, trouver une suite d'operations (+, -, *) qui atteint la cible (variante simplifiee du Game of 24).

1. **Etat et expansion** : un etat = (nombres restants, expression courante). L'expansion genere les etats successeurs en combinant deux nombres par une operation.

2. **CoT lineaire (baseline)** : a chaque etape, choisir UNE seule continuation (greedy ou aleatoire). Mesurer le taux de reussite sur un jeu de problemes — il echoue souvent car il ne revient jamais en arriere.

3. **Tree-of-Thought** :
   - **Generation** : a chaque noeud, generer plusieurs "thoughts" (continuations candidates)
   - **Evaluation** : une heuristique note chaque etat (ex: distance a la cible atteignable, ou "sure/maybe/impossible")
   - **Recherche** : BFS avec beam (garder les `b` meilleurs etats par niveau) OU DFS avec backtracking
   - Implementer au moins une des deux (BFS beam ou DFS), avec elagage des etats "impossible"

4. **Comparaison** : sur un jeu de N problemes (certains solubles, certains non), comparer CoT lineaire vs ToT :
   - Taux de reussite (problemes solubles resolus)
   - Nombre d'etats explores (cout)

5. **Largeur du beam** : faire varier `b` (1, 3, 5) et montrer le trade-off largeur/cout. b=1 ≈ CoT greedy.

6. Analyser : pourquoi le ToT bat le CoT lineaire sur les problemes ou il faut explorer/revenir en arriere ? Quel est le cout (plus d'appels au modele) ? Quand le ToT n'apporte-t-il rien ?

### Criteres de reussite

- [ ] L'expansion d'etats (combinaisons de nombres) est correcte et complete
- [ ] Le CoT lineaire baseline est implemente et echoue sur les problemes "a backtrack"
- [ ] Le ToT (BFS beam ou DFS) avec evaluation + elagage est implemente correctement
- [ ] ToT resout strictement plus de problemes que CoT lineaire, au prix de plus d'etats explores
- [ ] L'effet de la largeur du beam est demontre (trade-off largeur/cout)
- [ ] L'analyse explique quand ToT aide vs quand il est inutile
