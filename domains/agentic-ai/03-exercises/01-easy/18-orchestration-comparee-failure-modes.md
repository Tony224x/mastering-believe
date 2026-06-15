# Exercices — Orchestration comparee (J18)

---

## Exercice 1 : Ajouter le checkpointing au graphe jouet

### Objectif
Comprendre le mecanisme de checkpointing de LangGraph en l'implementant dans le mini-orchestrateur `ToyLangGraph` fourni dans `02-code/18-orchestration-comparee-failure-modes.py`.

### Consigne
En repartant de `ToyLangGraph` et `GraphState` :

1. Modifie `ToyLangGraph.run()` pour sauvegarder l'etat apres chaque noeud dans une liste `self.history : list[GraphState]` (copie superficielle du dict suffit)
2. Ajoute une methode `replay(from_step: int) -> GraphState` qui reprend l'execution depuis un checkpoint : elle recharge l'etat sauvegarde a l'index `from_step` et continue le graphe depuis le noeud suivant
3. Simule un crash apres le noeud `writer` (leve une `RuntimeError` conditionnelle si `state.step == 2`) puis reprends depuis le checkpoint `step=1` via `replay`
4. Affiche les etats sauvegardes a chaque pas et montre que la reprise produit le meme resultat final que sans crash

### Criteres de reussite
- [ ] `ToyLangGraph` stocke un checkpoint apres chaque noeud
- [ ] `replay(from_step)` recharge l'etat correct et reprend l'execution
- [ ] La simulation de crash leve bien une `RuntimeError`
- [ ] La reprise depuis le checkpoint produit le resultat final attendu
- [ ] Le test affiche clairement : etat avant crash, checkpoint charge, resultat apres reprise

---

## Exercice 2 : Budget guard — coupe le pipeline si les tokens depassent un seuil

### Objectif
Implementer un garde-fou de cout qui arrete le pipeline multi-agent quand la consommation de tokens depasse un budget, illustrant la defense contre l'explosion de cout (failure mode 4.3 du cours).

### Consigne
En repartant de `mock_llm` et `_CALL_LOG` (ou en reimplementant une version proche) :

1. Cree une classe `BudgetGuard` :
   - `__init__(self, max_tokens_in: int)` : seuil en tokens d'entree
   - `check(self) -> None` : leve `BudgetExceededError` si `tokens_used()[0] > max_tokens_in`
   - `reset(self) -> None` : remet le compteur a zero
2. Cree `BudgetExceededError(Exception)` avec un message indiquant le seuil et le consomme reel
3. Integre `BudgetGuard` dans un pipeline de 5 agents (inspire de `demo_token_explosion`) : apres chaque appel LLM, appelle `guard.check()`
4. Fixe le budget a 30 tokens d'entree (valeur basse pour la demo, sans vrai LLM) et montre que le pipeline est coupe avant le 5e agent
5. Affiche l'agent qui a declenche la coupure et le nombre de tokens consommes a ce moment

### Criteres de reussite
- [ ] `BudgetExceededError` est levee avant la fin du pipeline
- [ ] Le message d'erreur mentionne le seuil et le consomme reel
- [ ] Le pipeline s'arrete proprement (pas de crash non gere)
- [ ] Le test montre quel agent a declenche la coupure
- [ ] `reset()` permet de relancer un nouveau pipeline depuis zero

---

## Exercice 3 : Tiebreaker — resoudre un desaccord entre deux agents

### Objectif
Implementer le pattern "agent arbitre" pour sortir de la boucle de desaccord demontree dans `demo_disagreement_loop`, et comparer le cout avec et sans arbitre.

### Consigne
En repartant de la simulation de desaccord (`demo_disagreement_loop`) :

1. Cree une classe `TiebreakerAgent` :
   - `name: str = "arbitre"`
   - `resolve(self, position_a: str, position_b: str, criteria: str) -> str` : appelle `mock_llm` avec les deux positions et un critere de decision, et retourne la position retenue
2. Modifie la boucle de debat pour qu'apres **3 tours** de desaccord l'arbitre soit appele et le debat s'arrete
3. Compte et affiche le cout en tokens (a) avec arbitre (3 tours + 1 appel arbitre) vs (b) sans arbitre (6 tours)
4. Le resultat final doit etre une decision claire : "Position A retenue" ou "Position B retenue"
5. Assure-toi que `resolve()` ne boucle pas — il s'appelle une seule fois

### Criteres de reussite
- [ ] `TiebreakerAgent.resolve` appelle `mock_llm` une seule fois
- [ ] Le debat s'arrete apres 3 tours de desaccord
- [ ] Le cout avec arbitre est inferieur au cout sans arbitre (6 tours)
- [ ] La decision finale est affichee clairement
- [ ] Le code demontre que l'arbitre evite l'explosion de tokens
