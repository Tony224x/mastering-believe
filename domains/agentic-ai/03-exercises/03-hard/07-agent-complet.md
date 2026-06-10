# Exercices Hard — Agent complet (J7, capstone semaine 1)

---

## Exercice 1 : Agent de session multi-questions avec resolution d'anaphores

### Objectif
Transformer l'agent one-shot en agent conversationnel : gerer une session de plusieurs questions qui se referencent entre elles, avec memoire partagee et statistiques de session.

### Consigne
1. Cree une classe `SessionAgent` qui encapsule le graph + une memoire de session :
   - `session_memory` : faits accumules (`{"africa_population": ..., "paris_density": ...}`)
   - `dialogue_history` : liste des (question, reponse)
   - `last_entities` : dernieres entites mentionnees (ex: `["africa"]`), avec leur type (lieu)
   - `last_intent` : derniere intention (ex: `density_of_X`)
2. Implemente un resolveur d'anaphores **local et deterministe** `resolve(question, context) -> str` :
   - "And for Paris?" -> reutilise `last_intent` avec la nouvelle entite -> "What is the density of Paris?"
   - "What about its area?" -> reutilise `last_entities[-1]` -> "What is the area of Paris?"
   - "Compare them" -> les 2 dernieres entites -> "Compare the density of Africa and Paris"
   - Une question complete passe sans modification
3. Chaque question resolue est traitee par l'agent ; les faits decouverts enrichissent `session_memory` et le planner court-circuite les searches deja couvertes (cf. medium 1)
4. Scenario de demo (5 tours, dans cet ordre) :
   - "What is the population density of Africa?"
   - "And for Paris?"
   - "What about its area?"
   - "Compare them" (doit reutiliser les densites SANS refaire les searches)
   - "What is the capital of Mars?" (hors domaine : reponse honnete "no data")
5. A la fin, affiche les stats de session : questions traitees, resolutions d'anaphores effectuees, tool calls total, tool calls economises par la memoire, faits en memoire
6. Asserts : le tour 4 fait 0 search ; le tour 2 contient la densite de Paris ; le tour 5 ne contient aucun chiffre invente

### Criteres de reussite
- [ ] Les 3 formes d'anaphore (entite, possessif, "them") sont resolues correctement
- [ ] La question resolue est affichee a chaque tour ("resolved: ...")
- [ ] Le tour "Compare them" n'appelle aucun outil de recherche
- [ ] Le hors-domaine ne produit pas d'hallucination
- [ ] Les stats de session sont exactes (verifiees par asserts)
- [ ] Une question complete n'est jamais alteree par le resolveur

---

## Exercice 2 : Planner cost-aware avec optimisation de plan

### Objectif
Ajouter une passe d'optimisation entre le planner et l'executor : estimer le cout de chaque step, supprimer les redondances, et choisir la variante de plan la moins chere qui repond a la question.

### Consigne
1. Definis un modele de cout par action : `COSTS = {"search": 10, "read_doc": 4, "compute": 1, "recall": 0, "format": 1}` (unites arbitraires) et un budget par question `budget=20`
2. Ecris `estimate_plan_cost(plan) -> int` et un optimiseur `optimize_plan(plan, memory, scratchpad) -> tuple[plan, report]` qui applique 3 regles dans l'ordre :
   - **Dedup** : 2 steps identiques -> en garder un seul
   - **Memory substitution** : un `search:` dont le fait est deja en memoire -> remplace par `recall:` (cout 0)
   - **Downgrade** : si le cout depasse le budget, remplace les `search:` restants par `read_doc:` (moins cher mais marque `confidence: "medium"` dans le rapport)
3. Le rapport d'optimisation liste chaque transformation : `{"rule": "dedup", "before": ..., "after": ...}`
4. Genere ensuite **2 variantes de plan** pour les questions de comparaison (ex: "Compare density of Africa and Paris") :
   - Variante A : sequentielle complete (4 searches + 2 computes + 1 compare)
   - Variante B : factorisee (reutilise un fait deja en memoire, 2 searches seulement)
   - Choisis la variante au cout estime le plus bas
5. Demo en 3 runs :
   - Run 1 : "density of Africa" (plan normal, remplit la memoire)
   - Run 2 : "Compare the density of Africa and Paris" (l'optimiseur substitue les faits Africa, choisit la variante B)
   - Run 3 : meme question avec `budget=8` (downgrade visible, confidence medium)
6. Affiche pour chaque run : plan brut, plan optimise, cout avant/apres, rapport des regles appliquees, et verifie que la reponse finale reste correcte

### Criteres de reussite
- [ ] L'estimation de cout est exacte par rapport a la table COSTS
- [ ] Les 3 regles d'optimisation s'appliquent chacune au moins une fois dans la demo
- [ ] La selection de variante choisit B quand la memoire couvre Africa
- [ ] Le downgrade ne s'active QUE quand le budget est depasse, et la confidence est degradee
- [ ] Les reponses finales des runs optimises restent correctes (asserts sur les chiffres)
- [ ] Le rapport d'optimisation est complet et lisible
