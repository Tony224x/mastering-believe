# Exercices Faciles — Anatomie d'un Agent (J1)

---

## Exercice 1 : Ajouter un outil "weather"

### Objectif
Comprendre comment definir et integrer un nouvel outil dans un agent ReAct.

### Consigne
En partant du code `02-code/01-anatomie-agent.py` :

1. Definis un nouvel outil `get_weather` qui prend un parametre `city` (string) et retourne une meteo fictive (ex: "Paris: 18°C, partly cloudy")
2. Ajoute-le dans la liste `TOOLS` avec un nom, une description, et un schema de parametres
3. Implemente la fonction d'execution (mock — pas besoin d'API reelle)
4. Ajoute-le dans `TOOL_FUNCTIONS`
5. Teste avec la question : "What's the weather like in Tokyo?"

### Criteres de reussite
- [ ] L'outil apparait dans le system prompt genere
- [ ] L'agent (en mode simule ou live) appelle correctement `get_weather` avec `{"city": "Tokyo"}`
- [ ] L'observation retournee contient une meteo fictive
- [ ] L'agent produit une reponse finale qui mentionne Tokyo et la meteo

---

## Exercice 2 : Compteur d'iterations et guard-rails

### Objectif
Comprendre l'importance des limites de securite dans un agent.

### Consigne
Modifie l'agent pour ajouter les mecanismes de protection suivants :

1. **Compteur d'appels** : affiche a chaque etape "Step X/max" (ex: "Step 3/10")
2. **Detection de boucle** : si les 2 dernieres actions sont identiques (meme outil + memes params), arrete l'agent avec un message d'erreur
3. **Budget tokens** : ajoute un parametre `max_tokens_budget` (defaut: 5000). A chaque appel LLM, estime le nombre de tokens utilises (approximation : `len(text) / 4`) et arrete si le budget est depasse

Teste en forcant une boucle (ex: modifie les reponses simulees pour repeter la meme action).

### Criteres de reussite
- [ ] Le compteur d'etapes s'affiche correctement
- [ ] L'agent detecte et arrete une boucle repetitive (2 actions identiques consecutives)
- [ ] L'agent s'arrete quand le budget tokens est depasse
- [ ] Les 3 mecanismes fonctionnent independamment (tester chacun)

---

## Exercice 3 : Trace d'execution formatee

### Objectif
Apprendre a instrumenter un agent pour le debugging et l'observabilite.

### Consigne
Cree une fonction `format_trace(steps: list[dict]) -> str` qui prend l'historique d'execution de l'agent et produit une trace lisible :

```
=== Agent Trace ===
Question: What is 25 * 47?
Steps: 2 | Duration: 1.3s | Tools used: calculator, get_current_time

  [1] Thought: I need to calculate 25 * 47
      Action: calculator({"expression": "25 * 47"})
      Observation: 1175
      Duration: 0.8s

  [2] Thought: I have the answer
      Action: finish("25 * 47 = 1175")
      Duration: 0.5s

Final Answer: 25 * 47 = 1175
=================
```

Modifie `react_agent` pour :
1. Collecter chaque etape dans une liste de dicts (thought, action, action_input, observation, duration)
2. Mesurer le temps de chaque etape avec `time.time()`
3. Appeler `format_trace` a la fin et l'afficher

### Criteres de reussite
- [ ] La trace affiche chaque etape avec thought, action, observation, et duree
- [ ] Le resume en haut indique le nombre d'etapes, la duree totale, et les outils utilises
- [ ] Le format est lisible et alignable (indentation correcte)
- [ ] Fonctionne aussi bien en mode simule qu'en mode live
