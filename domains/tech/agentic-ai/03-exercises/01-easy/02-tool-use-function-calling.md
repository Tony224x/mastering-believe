# Exercices Faciles — Tool Use & Function Calling (J2)

---

## Exercice 1 : Ajouter un outil "translate_text"

### Objectif
Comprendre comment definir un outil avec des parametres types (string, enum) et l'integrer dans le tool registry.

### Consigne
En partant du code `02-code/02-tool-use-function-calling.py` :

1. Definis un nouvel outil `translate_text` avec les parametres suivants :
   - `text` (string, required) : le texte a traduire
   - `target_language` (string, required, enum: ["en", "fr", "es", "de", "pt"]) : la langue cible
   - `formality` (string, optional, enum: ["formal", "informal"], default: "formal") : niveau de formalite
2. Implemente la fonction (mock — retourne une traduction fictive basee sur la langue cible)
3. Enregistre l'outil dans le registry avec une description qui dit **quand** l'utiliser et quand **ne pas** l'utiliser
4. Teste l'execution : succes, parametre manquant, langue invalide

### Criteres de reussite
- [ ] L'outil a un schema JSON valide avec `enum` pour les langues et la formalite
- [ ] Le parametre `formality` a une valeur par defaut ("formal")
- [ ] L'execution reussit avec des parametres valides et retourne une traduction mock
- [ ] Un parametre manquant (`text` absent) retourne une erreur claire
- [ ] La description mentionne quand utiliser ET quand ne pas utiliser l'outil

---

## Exercice 2 : Convertir des outils entre formats OpenAI et Anthropic

### Objectif
Comprendre les differences de format entre les deux principaux providers et savoir convertir de l'un a l'autre.

### Consigne
1. Prends 3 outils du registry (calculator, search_web, database_query)
2. Exporte-les en format OpenAI (`to_openai_format()`)
3. Ecris une fonction `openai_to_anthropic(tool: dict) -> dict` qui convertit un outil du format OpenAI vers le format Anthropic :
   - `function.parameters` → `input_schema`
   - Retire le wrapper `"type": "function"` + `"function": {}`
   - Garde `name` et `description` au meme niveau
4. Ecris la fonction inverse `anthropic_to_openai(tool: dict) -> dict`
5. Verifie que la conversion aller-retour (OpenAI → Anthropic → OpenAI) preserve toutes les informations

### Criteres de reussite
- [ ] `openai_to_anthropic` produit un dict avec `name`, `description`, `input_schema` (pas de wrapper `function`)
- [ ] `anthropic_to_openai` produit un dict avec `type: "function"` et `function: {name, description, parameters}`
- [ ] La conversion aller-retour est lossless (le dict final == le dict initial)
- [ ] Les 3 outils convertis sont valides dans les deux formats

---

## Exercice 3 : Messages d'erreur actionnables

### Objectif
Apprendre a ecrire des messages d'erreur que le LLM peut utiliser pour se corriger.

### Consigne
1. Cree un wrapper `execute_with_feedback(registry, tool_name, params) -> str` qui :
   - Execute l'outil via le registry
   - En cas de succes, retourne le resultat brut
   - En cas d'erreur, retourne un message structure :
     ```
     TOOL_ERROR: {tool_name} failed.
     REASON: {error message}
     SUGGESTION: {what the LLM should do differently}
     AVAILABLE_TOOLS: {list of registered tools}
     ```
2. Implemente la logique de suggestion basee sur le type d'erreur :
   - `KeyError` / outil inconnu → "Check available tools listed above"
   - `PermissionError` → "This operation is not allowed. Try a read-only alternative"
   - `ValueError` → "Check your parameters against the tool schema"
   - `FileNotFoundError` → "Verify the file path exists"
   - Autre → "Retry with different parameters or try a different approach"
3. Teste avec 5 scenarios d'erreur differents

### Criteres de reussite
- [ ] Le message d'erreur contient toujours : TOOL_ERROR, REASON, SUGGESTION, AVAILABLE_TOOLS
- [ ] La suggestion est specifique au type d'erreur (pas un message generique pour tout)
- [ ] Le LLM recevrait assez d'info pour se corriger sans aide humaine
- [ ] Le wrapper fonctionne aussi en cas de succes (retourne le resultat directement)
- [ ] 5 scenarios d'erreur differents testes avec les bonnes suggestions
