# Exercices Medium — Tool Use & Function Calling (J2)

---

## Exercice 1 : Agent avec function calling et self-correction

### Objectif
Implementer un agent qui utilise le function calling natif (pas le parsing ReAct texte) et qui se corrige automatiquement quand un outil echoue.

### Consigne
Cree un agent en mode simule qui :

1. Recoit une question utilisateur
2. Genere un `tool_call` (en simulant la reponse LLM avec le format OpenAI : `tool_calls[].function.{name, arguments}`)
3. Execute l'outil via le registry
4. Si l'outil echoue, **renvoie l'erreur au LLM** dans un message `role: "tool"` et laisse le LLM re-essayer (simuler une reponse corrigee)
5. Si l'outil reussit, le LLM genere la reponse finale (`content` texte, pas de `tool_calls`)

Scenario de test (a simuler) :
```
User: "How many products are in each category, and what's the average price?"

Step 1 (LLM): tool_call -> database_query("SELECT category, COUNT(*), AVG(cost) FROM products GROUP BY category")
Step 2 (Tool): ERROR — no column 'cost', available columns: id, name, category, price, stock
Step 3 (LLM): tool_call -> database_query("SELECT category, COUNT(*) as count, AVG(price) as avg_price FROM products GROUP BY category")
Step 4 (Tool): SUCCESS — [{category: "electronics", count: 4, avg_price: 482.49}, ...]
Step 5 (LLM): "Here are the results: ..."
```

Construis le tableau `messages` complet a chaque etape (system, user, assistant+tool_calls, tool, assistant+tool_calls, tool, assistant+content).

### Criteres de reussite
- [ ] Les messages respectent le format OpenAI : `role: "assistant"` avec `tool_calls`, puis `role: "tool"` avec `tool_call_id`
- [ ] L'erreur de l'etape 1 est transmise au LLM dans le bon format
- [ ] Le LLM corrige sa requete (colonne `cost` → `price`) a l'etape 3
- [ ] L'historique complet des messages est affiche et valide
- [ ] Le nombre total de messages dans l'historique est correct (system + user + assistant + tool + assistant + tool + assistant = 7)

---

## Exercice 2 : Structured output — extraction de donnees

### Objectif
Utiliser le trick `tool_choice` + fake tool pour forcer le LLM a retourner des donnees structurees.

### Consigne
1. Definis un "outil" `extract_invoice_data` dont le schema represente une facture :
   ```json
   {
     "invoice_number": "string",
     "date": "string (YYYY-MM-DD)",
     "vendor": "string",
     "items": [{"description": "string", "quantity": "integer", "unit_price": "number"}],
     "total": "number",
     "currency": "string (enum: USD, EUR, GBP)"
   }
   ```
2. Simule l'input suivant (texte brut d'une facture) :
   ```
   Invoice #INV-2026-0042
   Date: April 10, 2026
   Vendor: TechCorp Solutions

   Items:
   - 3x Laptop Pro 16 @ $1,299.99 each
   - 10x Wireless Mouse @ $29.99 each
   - 1x Standing Desk @ $599.00

   Total: $4,798.87 USD
   ```
3. Simule la reponse du LLM qui remplit le schema (comme si `tool_choice` forcait l'appel)
4. Ecris une fonction `validate_extraction(data: dict, schema: dict) -> list[str]` qui :
   - Verifie que tous les champs `required` sont presents
   - Verifie les types (string, number, integer, boolean, array)
   - Verifie les enums
   - Retourne une liste d'erreurs (vide = valide)
5. Teste avec une extraction valide ET une extraction avec des erreurs (mauvais type, enum invalide, champ manquant)

### Criteres de reussite
- [ ] Le schema de l'outil `extract_invoice_data` est un JSON Schema valide avec `items` (array d'objets)
- [ ] La simulation produit un dict qui respecte le schema
- [ ] `validate_extraction` detecte : champs manquants, mauvais types, enums invalides
- [ ] Au moins 3 types d'erreur testes et correctement detectes
- [ ] Le code montre comment `tool_choice` serait utilise dans l'appel API

---

## Exercice 3 : Tool middleware — logging, timing, rate limiting

### Objectif
Implementer un systeme de middlewares qui s'intercale entre l'agent et l'execution des outils pour ajouter de l'observabilite et du controle.

### Consigne
Cree un systeme de middleware pour le `ToolRegistry` :

1. **Middleware `log_calls`** : log chaque appel d'outil avec timestamp, nom, parametres, resultat (ou erreur), et duree
2. **Middleware `timing`** : mesure le temps d'execution de chaque outil et l'ajoute au resultat
3. **Middleware `rate_limiter`** : limite le nombre d'appels par outil a N appels par minute. Si la limite est atteinte, retourne une erreur
4. **Middleware `retry`** : si un outil echoue, retry jusqu'a 3x avec backoff exponentiel (0.1s, 0.2s, 0.4s)

Architecture :
```python
class ToolMiddleware:
    """Base class for tool middleware."""
    def __call__(self, tool_name: str, params: dict, next_fn: Callable) -> dict:
        """Wrap the next middleware/tool execution."""
        return next_fn(tool_name, params)

class MiddlewareRegistry(ToolRegistry):
    """ToolRegistry with middleware support."""
    def add_middleware(self, middleware: ToolMiddleware): ...
    def execute(self, name, params):
        # Chain: middleware_1 → middleware_2 → ... → actual tool execution
```

Teste en executant 5 appels d'outils avec tous les middlewares actifs. Verifie que :
- Les logs contiennent toutes les infos
- Le timing est mesure
- Le rate limiter bloque apres N appels
- Le retry fonctionne (simuler un outil qui echoue 2 fois puis reussit)

### Criteres de reussite
- [ ] Les 4 middlewares sont implementes et fonctionnels
- [ ] Les middlewares sont chainables (l'ordre compte : log → rate_limit → retry → timing → execute)
- [ ] Le rate limiter bloque correctement apres N appels (tester avec N=3)
- [ ] Le retry fonctionne avec backoff (mesurer que le temps augmente entre les retries)
- [ ] Les logs contiennent : timestamp, tool_name, params, result/error, duration
- [ ] Le code est extensible : ajouter un nouveau middleware ne modifie pas le code existant
