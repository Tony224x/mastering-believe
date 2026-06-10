# Exercices Medium — MCP (J10)

---

## Exercice 1 : Notifications tools/list_changed et refresh du client

### Objectif
Implementer le mecanisme dynamique de MCP : un serveur qui enregistre un tool a chaud doit notifier le client, qui rafraichit alors sa liste — la base des serveurs MCP evolutifs.

### Consigne
En partant de `02-code/10-mcp.py` :

1. Etends `MiniMCPServer` :
   - Une liste d'abonnes `self._subscribers: list[Callable]` et une methode `subscribe(callback)`
   - `register_tool_runtime(name, description, schema, handler)` : enregistre le tool PUIS envoie a chaque abonne une notification JSON-RPC **sans id** : `{"jsonrpc": "2.0", "method": "notifications/tools/list_changed"}`
2. Etends `MiniMCPClient` :
   - Maintient un cache local `self._tools_cache` rempli par `list_tools()`
   - `on_notification(message)` : si la methode est `notifications/tools/list_changed`, invalide le cache et le marque `stale`
   - `call_tool` verifie le cache : s'il est stale, re-fetch via `tools/list` avant l'appel
3. Declare les capabilities a l'initialize : le serveur repond `"capabilities": {"tools": {"listChanged": True}}` — le client ne souscrit QUE si cette capability est presente
4. Demo :
   - Le client liste les tools (N tools), appelle `add`
   - Le serveur enregistre `weather` a chaud -> notification
   - Le client appelle `weather` : la trace montre le re-fetch automatique puis l'appel reussi
   - Assert : le cache du client contient N+1 tools a la fin
5. Contre-test : un serveur sans la capability `listChanged` -> le client ne souscrit pas, et l'appel a un tool ajoute a chaud echoue avec une erreur claire `-32601`

### Criteres de reussite
- [ ] La notification est bien un message JSON-RPC sans `id` (fire-and-forget)
- [ ] Le client invalide son cache sur notification et re-fetch avant le prochain appel
- [ ] La negociation de capability conditionne la souscription
- [ ] Le contre-test sans capability echoue proprement avec `-32601`
- [ ] Les asserts sur l'etat du cache passent

---

## Exercice 2 : Resource templates avec parametres d'URI

### Objectif
Implementer les resource templates MCP (`products://{category}/{id}`) : exposer une famille de resources parametrees plutot que d'enumerer chaque resource statiquement.

### Consigne
1. Etends `MiniMCPServer` avec `resource_template(uri_template, name, mime_type)` :
   - `uri_template` du type `"products://{category}/{id}"`
   - Le reader recoit les parametres extraits : `def get_product(category: str, id: str) -> str`
2. Implemente le matching d'URI :
   - `_match_template("products://{category}/{id}", "products://laptops/42")` -> `{"category": "laptops", "id": "42"}`
   - Un segment `{param}` matche tout sauf `/` ; le matching est exact sur le reste
3. Ajoute la methode protocole `resources/templates/list` qui retourne les templates (`uriTemplate`, `name`, `mimeType`)
4. `resources/read` doit maintenant : essayer les resources statiques d'abord, puis les templates ; si aucun ne matche -> erreur `-32002` ("Resource not found")
5. Demo avec un mini-catalogue en dict :
   - `products://laptops/42` -> JSON du produit
   - `products://laptops/99` -> erreur applicative propre ("unknown id 99") dans le contenu, PAS une erreur JSON-RPC
   - `products://nope` (ne matche pas le template, segment manquant) -> erreur `-32002`
   - Une resource statique existante continue de fonctionner
6. Asserts sur les 4 cas

### Criteres de reussite
- [ ] Le matching de template extrait correctement les parametres multiples
- [ ] `resources/templates/list` expose les templates separement des resources statiques
- [ ] La distinction erreur applicative (id inconnu) vs erreur protocole (-32002) est respectee
- [ ] Les resources statiques ne sont pas cassees par l'ajout des templates
- [ ] Les 4 cas de demo passent leurs asserts

---

## Exercice 3 : Adapter MCP -> registre d'outils d'agent

### Objectif
Brancher un serveur MCP sur un agent : convertir dynamiquement les tools MCP en definitions d'outils format function-calling, et router les appels de l'agent a travers le client MCP.

### Consigne
1. Ecris `mcp_to_function_defs(client) -> list[dict]` :
   - Appelle `client.list_tools()` et convertit chaque tool au format OpenAI function calling (`{"type": "function", "function": {"name", "description", "parameters"}}`)
   - Prefixe les noms : `"mcp__<server_name>__<tool_name>"` (evite les collisions avec les outils natifs)
2. Ecris une classe `MCPToolBridge` :
   - `__init__(client, server_name)`
   - `is_mcp_tool(name) -> bool` et `execute(name, arguments) -> str` qui de-prefixe et route vers `client.call_tool`
   - Les erreurs JSON-RPC sont converties en string d'observation pour l'agent : `"[MCP ERROR -32602] ..."` (l'agent doit pouvoir les lire, pas crasher)
3. Construis un mini agent loop (simule, 3 etapes hardcodees comme en J1) qui dispose :
   - D'un outil natif `local_time`
   - Des outils MCP du serveur de demo (`add`, `greet`)
   - Scenario : "Greet Alice then compute 19 + 23" -> appel MCP `greet`, appel MCP `add`, reponse finale
4. Le scenario doit aussi montrer une erreur recuperable : l'agent appelle `add` avec un argument string -> observation `[MCP ERROR -32602]` -> l'agent corrige et rappelle avec des ints
5. Affiche la table des outils exposes a l'agent (natifs + MCP prefixes) et la trace des appels

### Criteres de reussite
- [ ] La conversion produit des schemas function-calling valides pour tous les tools MCP
- [ ] Le prefixage/de-prefixage est transparent pour l'agent
- [ ] L'outil natif et les outils MCP cohabitent dans le meme registre
- [ ] L'erreur -32602 devient une observation lisible et l'agent se corrige
- [ ] Le scenario complet (greet + add + correction) se deroule avec trace claire
