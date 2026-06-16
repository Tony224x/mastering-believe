# Exercices Hard — MCP (J10)

---

## Exercice 1 : Serveur MCP securise — approval gate + audit + canary

### Objectif
Construire la couche de securite que tout serveur MCP exposant des actions devrait avoir (section 8 du cours) : approbation humaine sur les tools dangereux, audit log inviolable, et detection de fuite de resource via canary token.

### Consigne
En partant du `MiniMCPServer`, cree une sous-classe `SecureMCPServer` qui ajoute :

1. **Tools dangereux** : `ToolDef` gagne un champ `dangerous: bool`. A l'enregistrement d'un tool dangereux (ex: `delete_file`), un appel `tools/call` doit d'abord passer par un **approval callback** branche sur le serveur (`set_approver(callback)`)
   - Si l'approbation est refusee → `jsonrpc_error(-32001, "human rejected")`, le handler n'est PAS appele
   - Le callback recoit `(tool_name, arguments)` et retourne un `bool`
2. **Audit log avec hash chaining** : chaque `tools/call` (approuve ou refuse, succes ou erreur) ajoute une entree `{seq, ts, tool, args_hash, outcome, prev_hash}` ou `prev_hash` = SHA-256 de l'entree precedente (la 1re a `"0"*64`)
   - Methode `verify_audit() -> bool` qui detecte toute alteration
3. **Canary dans les resources** : une resource secrete `secret://config` contient un canary token. Avant de renvoyer le resultat de N'IMPORTE quel tool, le serveur scanne la sortie : si le canary y apparait → bloque avec `jsonrpc_error(-32002, "canary leak")` et logge l'incident
4. **Least privilege** : un tool `read_public` (non dangereux) et `delete_file` (dangereux) ; un tool `leaky` qui tente de renvoyer le canary (simule une exfiltration)

Scenario de test :
- `read_public` → OK, audit grandit
- `delete_file` avec approver=yes → OK ; avec approver=no → rejete
- `leaky` → bloque par le canary scan, incident loggue
- Apres N appels, `verify_audit()` == True ; puis on falsifie une entree → `verify_audit()` == False

### Criteres de reussite
- [ ] Les tools dangereux exigent une approbation avant execution
- [ ] Un refus d'approbation empeche l'appel du handler
- [ ] L'audit log chaine les hash et `verify_audit()` detecte toute falsification
- [ ] Le canary scan bloque toute sortie de tool contenant le token secret
- [ ] Chaque tentative (OK / rejet / blocage) produit une entree d'audit
- [ ] Le code reste structure et chaque mecanisme est testable independamment

---

## Exercice 2 : Transport stdio realiste — framing, buffering, requetes concurrentes

### Objectif
Remplacer le bus "in-process" (deque + appel direct) par un vrai transport stdio simule : messages encadres ligne par ligne (newline-delimited JSON), buffering partiel, multiplexing de requetes concurrentes par `id`, et gestion des notifications (sans `id`).

### Consigne
1. Cree une classe `StdioTransport` qui simule deux pipes (`client→server`, `server→client`) avec des buffers `bytes` :
   - `write(msg: dict)` serialise en JSON + `\n` et pousse les **octets** dans le buffer
   - **Framing** : le lecteur doit gerer un message arrivant en plusieurs morceaux (ecris la moitie des octets, puis l'autre moitie, et verifie que le parser attend la ligne complete)
   - `read_messages() -> list[dict]` extrait toutes les lignes COMPLETES disponibles, garde le reste partiel en buffer
2. Cree un `AsyncishClient` qui peut avoir **plusieurs requetes en vol** : il envoie 3 `tools/call` d'affilee (ids 1, 2, 3) AVANT de lire aucune reponse. Le serveur traite les 3 et repond. Le client doit **matcher chaque reponse a sa requete par `id`** (pas par ordre d'arrivee — simule un ordre de reponse melange : 2, 1, 3)
3. **Notifications** : le message `initialized` (sans `id`) ne doit produire AUCUNE reponse ; verifie que le buffer serveur→client ne contient rien apres une notification
4. Gere aussi une ligne JSON malformee dans le flux : le transport ne doit pas crasher, il logge l'erreur et continue avec les lignes suivantes

Scenario de test :
- Envoi fragmente d'un `initialize` (2 morceaux) → parse correct quand complet
- 3 requetes concurrentes, reponses dans le desordre → chacune matchee a son id
- Une notification → 0 reponse
- Une ligne corrompue au milieu du flux → ignoree, les suivantes passent

### Criteres de reussite
- [ ] Le framing newline-delimited gere les messages fragmentes (partial reads)
- [ ] Le client matche N reponses a N requetes par `id`, meme dans le desordre
- [ ] Une notification (sans `id`) ne genere pas de reponse
- [ ] Une ligne malformee n'interrompt pas le flux
- [ ] Aucun message n'est perdu ou attribue a la mauvaise requete
- [ ] Le buffer partiel est correctement conserve entre deux `read_messages()`
