# Exercices Hard — Tool Use & Function Calling (J2)

---

## Exercice 1 : Dynamic tool discovery — agent qui decouvre ses outils

### Objectif
Implementer un agent qui ne connait pas ses outils a l'avance. Il decouvre les outils disponibles, lit leurs schemas, et apprend a les utiliser dynamiquement — comme un developpeur qui decouvre une API pour la premiere fois.

### Consigne
Cree un systeme avec deux composants :

1. **Tool Server** : un registry qui expose ses outils via une API (simulee) :
   - `GET /tools` → liste les noms et descriptions courtes de tous les outils
   - `GET /tools/{name}/schema` → retourne le schema complet d'un outil specifique
   - `POST /tools/{name}/execute` → execute l'outil avec les parametres fournis

2. **Discovery Agent** : un agent qui :
   - Ne recoit AUCUNE description d'outil dans son system prompt initial
   - A un meta-outil `discover_tools` qui appelle `GET /tools` pour lister les outils disponibles
   - A un meta-outil `get_tool_schema` qui appelle `GET /tools/{name}/schema` pour obtenir le schema d'un outil
   - A un meta-outil `call_tool` qui appelle `POST /tools/{name}/execute` pour executer un outil
   - Doit resoudre une tache en 3 phases : **discover → learn → execute**

Scenario de test :
```
User: "Find the most expensive product and calculate 15% tax on it."

Agent Step 1: discover_tools() → ["calculator", "database_query", "search_web"]
Agent Step 2: get_tool_schema("database_query") → {name, description, parameters...}
Agent Step 3: call_tool("database_query", {"query": "SELECT name, price FROM products ORDER BY price DESC LIMIT 1"})
Agent Step 4: get_tool_schema("calculator") → {name, description, parameters...}
Agent Step 5: call_tool("calculator", {"expression": "1299.99 * 0.15"})
Agent Step 6: finish("The most expensive product is Laptop Pro 16 at $1,299.99. 15% tax = $195.00")
```

**Mode simule** : hardcode les reponses du "Tool Server" et les decisions de l'agent.

### Criteres de reussite
- [ ] Le Tool Server expose 3 endpoints simules (list, schema, execute)
- [ ] L'agent demarre SANS connaitre les outils — le system prompt ne liste aucun outil specifique
- [ ] L'agent utilise `discover_tools` pour lister les outils disponibles
- [ ] L'agent utilise `get_tool_schema` pour apprendre le schema AVANT d'appeler un outil
- [ ] L'agent utilise `call_tool` pour executer les outils decouverts
- [ ] La trace montre les 3 phases : discover → learn → execute
- [ ] Le pattern est generalizable : ajouter un nouvel outil au server ne necessite aucun changement cote agent

---

## Exercice 2 : Tool composition engine — combiner des outils atomiques

### Objectif
Construire un moteur qui compose automatiquement des outils atomiques en outils complexes. L'idee : definir des "recettes" (sequences d'outils) que l'agent peut appeler comme un seul outil.

### Consigne
Cree un `CompositionEngine` qui :

1. **Definit des recettes** (composite tools) a partir d'outils atomiques :
   ```python
   engine.register_recipe(
       name="get_product_report",
       description="Generate a complete report for a product category",
       steps=[
           {"tool": "database_query", "params_template": {"query": "SELECT * FROM products WHERE category = '{category}'"}},
           {"tool": "calculator", "params_template": {"expression": "{total_price} / {count}"}},
           {"tool": "analyze_text", "params_template": {"text": "{report_text}", "analysis_type": "stats"}},
       ],
       input_schema={"category": "string"},
       output_mapping={...}  # How to pass results between steps
   )
   ```

2. **Gere le passage de donnees** entre les etapes :
   - L'output de l'etape N est accessible a l'etape N+1 via des templates (`{variable}`)
   - Un mapping definit comment extraire des valeurs des resultats (ex: JSONPath-like)

3. **Gere les erreurs dans la chaine** :
   - Si une etape echoue, la recette entiere echoue avec un message qui indique quelle etape a echoue
   - Option `continue_on_error: true` pour les etapes non-critiques

4. **Expose les recettes comme des outils** dans le registry :
   - L'agent voit `get_product_report` comme un outil normal (nom, description, schema)
   - Il ne sait pas que c'est une composition — pour lui, c'est atomique

Teste avec :
- Une recette a 3 etapes qui reussit
- Une recette ou l'etape 2 echoue (et verifie le message d'erreur)
- Une recette avec `continue_on_error` ou une etape non-critique echoue

### Criteres de reussite
- [ ] Les recettes sont definies de maniere declarative (pas de code Python inline)
- [ ] Le passage de donnees entre etapes fonctionne via un template system (`{variable}`)
- [ ] Les recettes composees apparaissent comme des outils normaux dans le registry
- [ ] Une erreur a l'etape N rapporte clairement "Step N failed: ..."
- [ ] `continue_on_error` permet de sauter les etapes non-critiques
- [ ] Au moins 2 recettes differentes testees
- [ ] Le engine fait < 200 lignes (complexite maitrisee)
- [ ] Le system est type (type hints) et documente
