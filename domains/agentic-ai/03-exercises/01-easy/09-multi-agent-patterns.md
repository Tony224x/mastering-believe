# Exercices Faciles — Multi-Agent Patterns (J9)

---

## Exercice 1 : Ajouter un 4e specialiste

### Objectif
Comprendre comment etendre un systeme multi-agent avec un nouveau role sans casser l'existant.

### Consigne
En partant de `02-code/09-multi-agent-patterns.py` :

1. Ajoute un 4e role `documenter` au `MockLLM` :
   - Son output : une docstring en format reStructuredText pour le code produit par le coder
   - Il doit mentionner les parametres et le type de retour
2. Modifie le plan du supervisor pour inclure une etape `documenter` apres le `reviewer`
3. Modifie la chaine de handoff du swarm pour inclure `documenter` apres `reviewer`
4. Mets a jour la liste `agents` et le `_handoff` dict
5. Teste que les deux patterns (supervisor et swarm) utilisent bien le nouveau role et produisent une sortie coherente

### Criteres de reussite
- [ ] `_documenter` existe et retourne une docstring realiste
- [ ] Le plan du supervisor contient 4 etapes
- [ ] La chaine swarm inclut `documenter` en avant-derniere position
- [ ] Les deux patterns s'executent sans erreur et montrent la sortie du documenter

---

## Exercice 2 : Budget de hops dans le swarm

### Objectif
Comprendre comment proteger un swarm contre les boucles et les couts qui explosent.

### Consigne
1. Ajoute un parametre `max_llm_calls: int = 10` a `SwarmPattern.__init__`
2. Dans `run()`, maintient un compteur local `local_calls` qui s'incremente a chaque appel LLM
3. Si `local_calls >= max_llm_calls`, arrete la boucle proprement et ajoute un message "[Swarm] budget reached" a la fin de la trajectoire
4. Ajoute aussi une detection de cycle explicite : si le meme agent est sollicite 2 fois de suite via handoff (sans qu'un autre agent ne parle entre temps), leve une `RuntimeError("Detected tight loop")`
5. Teste le cas normal (budget suffisant) et le cas budget insuffisant (`max_llm_calls=2`)

### Criteres de reussite
- [ ] Le budget est respecte : ni appel LLM au-dela du seuil
- [ ] Le message "[Swarm] budget reached" apparait quand on depasse
- [ ] La detection de tight loop leve bien une RuntimeError
- [ ] Le test avec budget normal termine proprement

---

## Exercice 3 : Scoring et vote entre agents

### Objectif
Implementer un mini debate pattern : 3 agents donnent un avis chiffre sur une proposition, un moderateur tranche.

### Consigne
1. Cree une classe `DebatePattern` qui prend un LLM, un moderateur et une liste d'agents
2. Sa methode `decide(proposal: str) -> dict` doit :
   - Demander a chaque agent un score 0-10 et une courte justification sur la proposition
   - Collecter tous les scores
   - Demander au moderateur de trancher : "accept" si la moyenne >= 6, "reject" si < 4, "debate" sinon
3. Cree un `DebateMockLLM` qui etend le MockLLM et donne des scores deterministes :
   - `researcher` donne toujours 7 (prudent)
   - `coder` donne toujours 8 (enthousiaste si c'est code)
   - `reviewer` donne toujours 5 (critique)
4. Le moderateur est un appel LLM qui recoit les scores en JSON et retourne le verdict
5. Teste avec 3 propositions et affiche les scores, la moyenne et le verdict final

### Criteres de reussite
- [ ] `DebatePattern.decide` retourne un dict avec `scores`, `average`, `verdict`
- [ ] Chaque agent produit un score deterministe et une justification
- [ ] Le moderateur tranche correctement selon les seuils
- [ ] Le test montre clairement les 3 avis et le verdict
- [ ] Aucune boucle de debat infinie
