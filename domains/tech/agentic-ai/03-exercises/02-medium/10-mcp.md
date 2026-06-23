# Exercices Medium — MCP (J10)

---

## Exercice 1 : Validation de schema generique (JSON Schema mini-validator)

### Objectif
Comprendre comment un serveur MCP valide les arguments d'un tool contre son `inputSchema` AVANT d'appeler le handler — au lieu de laisser exploser le handler avec un `TypeError`.

### Consigne
En partant de `02-code/10-mcp.py`, le serveur appelle aujourd'hui `tool.handler(**arguments)` sans verifier les arguments contre `tool.input_schema`. Implemente une vraie validation generique :

1. Ecris une fonction `validate_against_schema(schema: dict, arguments: dict) -> list[str]` qui retourne la liste des erreurs (vide = OK) en supportant un sous-ensemble de JSON Schema :
   - `required` : chaque champ requis doit etre present
   - `properties[x].type` : verifie le type (`"integer"`, `"string"`, `"number"`, `"boolean"`, `"array"`, `"object"`)
   - Champ inconnu (pas dans `properties`) → erreur "unexpected property"
   - **Piege** : en Python `True` est une instance de `int` — un `bool` ne doit PAS valider un champ `"integer"`
2. Modifie `_tools_call` pour appeler ce validateur. Si erreurs, retourne `jsonrpc_error(req_id, -32602, ...)` avec un message qui liste les erreurs (ne PAS appeler le handler)
3. Teste les cas :
   - `add(a=3, b=4)` → OK, resultat 7
   - `add(a="3", b=4)` → erreur -32602 (type)
   - `add(a=3)` → erreur -32602 (required manquant)
   - `add(a=3, b=4, c=9)` → erreur -32602 (propriete inattendue)
   - `add(a=True, b=4)` → erreur -32602 (bool n'est pas integer)

### Criteres de reussite
- [ ] `validate_against_schema` gere `required`, `type`, et les proprietes inattendues
- [ ] Le cas `bool` vs `integer` est correctement rejete
- [ ] `_tools_call` n'appelle JAMAIS le handler si la validation echoue
- [ ] Les erreurs retournent le code JSON-RPC `-32602` (invalid params)
- [ ] Les 5 cas de test affichent un verdict clair

---

## Exercice 2 : Sampling — le serveur emprunte le LLM du host

### Objectif
Implementer le pattern **sampling** de la spec MCP (section 10 du cours) : un serveur qui demande au client (host) d'invoquer son LLM pour une sous-tache, sans avoir sa propre cle API.

### Consigne
1. Ajoute au `MiniMCPClient` une capacite `sampling` : un callback `sampler: Callable[[str], str]` (mock du LLM hote) passe a la construction
2. Cote serveur, ajoute une methode `request_sampling(prompt: str) -> str` qui delegue au client via une nouvelle direction de message JSON-RPC `sampling/createMessage` (server → client)
   - Pour rester en-process, donne au serveur une reference vers un callback `on_sampling_request` que le client branche au moment du `initialize`
3. Cree un tool `summarize_notes()` qui :
   - Lit la resource `notes://acme` (contenu interne)
   - **Demande au LLM hote** de la resumer via sampling (le serveur n'a pas de LLM)
   - Retourne le resume
4. Le mock du LLM hote : une fonction qui prend un prompt et retourne `"SUMMARY: <les 5 premiers mots du contenu>..."` (deterministe)
5. Teste : appelle `summarize_notes()`, verifie que le resume vient bien du sampler du client, et que le serveur n'a aucune logique LLM en dur

### Criteres de reussite
- [ ] Le serveur peut declencher un appel `sampling/createMessage` vers le client
- [ ] Le client repond en invoquant son sampler (mock LLM)
- [ ] Le tool `summarize_notes` combine resource-read + sampling
- [ ] Le serveur ne contient aucune logique de generation (le texte vient du sampler)
- [ ] Un compteur prouve que le sampler du client a bien ete appele exactement 1 fois

---

## Exercice 3 : Routeur multi-serveurs (aggregating client)

### Objectif
Comprendre comment un host gere PLUSIEURS serveurs MCP en parallele et route un appel de tool vers le bon serveur, en gerant les collisions de noms.

### Consigne
1. Cree une classe `MCPRouter` qui agrege plusieurs `MiniMCPServer` (via leurs clients)
2. A l'init, le routeur appelle `tools/list` sur chaque serveur et construit une table `{namespaced_name: (server_id, tool_name)}`
   - **Collision de noms** : si deux serveurs exposent un tool `search`, le routeur les expose sous `serverA.search` et `serverB.search`
   - Si un nom est unique, il reste accessible sans prefixe ET avec prefixe
3. Methode `list_all_tools() -> list[dict]` : retourne la vue agregee avec un champ `server`
4. Methode `call(tool_name: str, arguments: dict) -> str` : route vers le bon serveur ; si le nom est ambigu (non prefixe et present sur 2 serveurs), leve une erreur claire
5. Teste avec 2 serveurs : un serveur "math" (tools `add`, `search`) et un serveur "kb" (tools `lookup`, `search`)
   - `call("add", {...})` → serveur math (unique)
   - `call("math.search", {...})` → serveur math
   - `call("kb.search", {...})` → serveur kb
   - `call("search", {...})` → erreur "ambiguous"

### Criteres de reussite
- [ ] Le routeur decouvre les tools de N serveurs au demarrage
- [ ] Les collisions de noms sont resolues par namespacing (`server.tool`)
- [ ] Un tool unique reste appelable sans prefixe
- [ ] Un appel ambigu (nom partage, non prefixe) leve une erreur explicite
- [ ] La vue agregee indique le serveur source de chaque tool
