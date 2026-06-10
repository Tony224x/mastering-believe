# Exercices Hard — MCP (J10)

---

## Exercice 1 : Host multi-serveurs avec namespacing, policy et audit

### Objectif
Construire la piece centrale d'un host MCP type Claude Desktop : agreger plusieurs serveurs, gerer les collisions de noms, appliquer une politique de securite par serveur, et auditer tous les appels.

### Consigne
1. Cree une classe `MCPHost` :
   - `add_server(alias, server, policy)` : connecte un `MiniMCPClient` au serveur et enregistre sa policy
   - `list_all_tools() -> list[dict]` : agrege les tools de tous les serveurs, prefixes par alias (`"files__read_file"`, `"web__fetch"`)
   - `call(tool_full_name, arguments, user_approved=False) -> dict`
2. Implemente 2 serveurs de demo distincts :
   - Serveur `files` : tools `read_file`, `delete_file` ; le tool `read_file` existe AUSSI sur le serveur `web` (collision a demontrer : les 2 versions restent accessibles via leurs prefixes)
   - Serveur `web` : tools `fetch`, `read_file` (lit une page comme un fichier)
3. **Policy par serveur** (dataclass `ServerPolicy`) :
   - `allowed_tools: list[str] | None` (None = tout), `denied_tools: list[str]`
   - `require_approval: list[str]` : ces tools exigent `user_approved=True`, sinon refus avec un dict `{"error": "approval_required", ...}`
   - `max_calls_per_session: int` : au-dela, refus `{"error": "rate_limited"}`
4. **Audit log** : chaque appel (autorise OU refuse) est journalise : timestamp, serveur, tool, arguments (avec valeurs > 50 chars tronquees), decision (`allowed`/`denied:<reason>`), resultat ok/erreur. Methode `audit_report()` qui affiche le journal et des stats par serveur
5. Demo complete :
   - Lister les tools agreges (avec la collision visible)
   - Appel normal (`web__fetch`)
   - Appel a un tool denied (`files__delete_file` avec policy qui le denie)
   - Appel necessitant approbation : refuse sans approval, puis accepte avec
   - Depassement du rate limit sur le serveur web (max 3 calls)
   - Affiche l'audit final
6. Asserts sur chaque decision et sur le contenu de l'audit (nb d'entrees, nb de denied)

### Criteres de reussite
- [ ] Les tools des 2 serveurs cohabitent, la collision est resolue par prefixage
- [ ] La policy bloque denied, exige l'approbation, et applique le rate limit
- [ ] Un refus n'appelle JAMAIS le serveur sous-jacent (verifiable par compteur serveur)
- [ ] L'audit contient toutes les decisions avec leurs raisons
- [ ] Le rapport d'audit affiche des stats correctes par serveur
- [ ] Tous les asserts passent

---

## Exercice 2 : Couche JSON-RPC robuste — batch, correlation, timeouts et messages malformes

### Objectif
Blinder l'implementation JSON-RPC du transport MCP : gerer les batchs, les reponses dans le desordre, les timeouts avec retry, et tous les codes d'erreur standards — avec une suite de tests de messages hostiles.

### Consigne
1. **Codes d'erreur standards** : le serveur doit retourner :
   - `-32700` Parse error : message non-JSON (le transport recoit des strings brutes)
   - `-32600` Invalid request : pas de `jsonrpc: "2.0"`, ou `method` absent/non-string, ou `id` de type invalide (liste/dict)
   - `-32601` Method not found : methode inconnue
   - `-32602` Invalid params : params manquants ou de mauvais type pour `tools/call`
2. **Batch** : si le transport recoit une liste JSON de requetes, retourne une liste de reponses (les notifications du batch ne produisent PAS de reponse) ; une liste vide -> `-32600`
3. **Transport asynchrone simule** : cree un `UnreliableTransport` qui enveloppe le serveur et, selon une sequence de pannes configurable (deterministe), peut :
   - Retarder une reponse (la livrer apres la suivante -> reponses dans le desordre)
   - Perdre une reponse (timeout cote client)
4. **Client robuste** :
   - Correlation par `id` : les reponses dans le desordre sont reassociees a leur requete (table `pending[id]`)
   - Timeout simule (compteur de "ticks", pas de vraie attente) + retry avec un NOUVEL id (max 2 retries), et deduplication cote serveur si la requete originale arrive quand meme
   - Verification de version de protocole a l'initialize : un serveur qui annonce une version inconnue -> le client refuse la connexion avec un message clair
5. **Suite de tests hostiles** : au moins 8 cas — JSON casse, jsonrpc manquant, method numerique, id objet, methode inconnue, params invalides, batch vide, batch mixte requetes+notifications — chaque cas verifie le code d'erreur exact retourne
6. Affiche un tableau recapitulatif des cas et de leur verdict

### Criteres de reussite
- [ ] Les 4 codes d'erreur standards sont retournes dans les bons cas
- [ ] Le batch fonctionne et les notifications n'y produisent pas de reponse
- [ ] Les reponses dans le desordre sont correctement correlees par id
- [ ] Le timeout declenche un retry avec nouvel id, sans doublon d'execution cote serveur
- [ ] Le mismatch de version de protocole est refuse a l'initialize
- [ ] Les 8+ tests hostiles passent tous avec le code exact attendu
