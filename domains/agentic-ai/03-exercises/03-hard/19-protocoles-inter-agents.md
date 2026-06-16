# Exercices Hard — Protocoles inter-agents (J19)

---

## Exercice 1 : Interop A2A bout-en-bout entre deux "frameworks" + negociation de version

### Objectif
Implementer un mini-protocole A2A complet et prouver l'**interoperabilite** : deux agents implementes comme s'ils venaient de **frameworks differents** (deux classes sans aucun code partage hormis le protocole) doivent cooperer **uniquement** via le contrat reseau — Agent Card, handshake versionne, cycle de vie de tache, artefacts structures. Le test ultime : un client "framework A" doit pouvoir piloter un agent "framework B" sans rien connaitre de son implementation, et une incompatibilite de version doit etre **negociee ou rejetee proprement**, jamais provoquer un plantage.

### Consigne
Tout reembarquer (stdlib pur). Le **seul** point de contact entre les deux mondes est le dict JSON-serialisable echange.

1. **Couche protocole partagee** (le "fil") :
   - `AgentCard` : `name`, `protocol_versions: list[str]`, `skills: dict[skill_id -> {input_modes, output_modes}]`.
   - Envelopes JSON-RPC-like : `make_request(method, params, req_id)`, `make_response`, `make_error` (codes : `-32601` method not found, `-32602` invalid params, `-32000` server error).
   - Cycle de vie de tache : etats `submitted → working → completed/failed`, artefacts = liste de parts `{"type": "data", "data": {...}}`.
2. **Agent "Framework B"** : une classe `BravoAgent` (genre BeeAI/ACP) qui expose `get_card()` et `handle(raw_request_dict) -> response_dict`. Elle gere `tasks/send` (cree la tache, l'execute synchroniquement, renvoie la tache completee avec artefact) et `tasks/get`. Skill : `optimize_routes` qui trie des stops et renvoie une distance mockee. Versions supportees : `["1.0", "1.1"]`.
3. **Client "Framework A"** : une classe `AlphaClient` (genre LangGraph) qui ne connait **que** le protocole. Elle :
   - recupere la card (`discover`), execute un **handshake** : choisit la version commune **la plus elevee** (gere `"1.10" > "1.9"`) ; si pas de version commune → echec propre `version_mismatch` ;
   - verifie que le skill voulu est declare ;
   - envoie la tache et lit l'artefact structure.
4. **Test d'interop** : `AlphaClient` pilote `BravoAgent` de bout en bout et recupere l'ordre optimise (assertion sur le contenu de l'artefact + etat `completed`).
5. **Test de version** :
   - un `BravoAgent` qui ne supporte que `["2.0"]` face a un client `["1.0", "1.1"]` → le handshake renvoie `version_mismatch` **sans** envoyer de tache ;
   - prouve aussi que le client choisit bien `"1.1"` (et non `"1.0"`) quand les deux sont communs, et que `"1.10"` serait prefere a `"1.9"`.
6. **Test d'erreur protocole** : envoyer une methode inconnue (`tasks/teleport`) renvoie une erreur JSON-RPC `-32601` propre, pas une exception remontee.

### Criteres de reussite
- [ ] Le client et l'agent ne partagent **aucun** etat/code hormis les dicts du protocole (interop reelle)
- [ ] Handshake : la version commune retenue est la **plus elevee**, avec un ordre de version correct (`"1.10" > "1.9"`)
- [ ] Une tache bout-en-bout aboutit a `completed` avec un artefact structure verifiable par assertion
- [ ] Un agent sans version commune declenche `version_mismatch` **avant** tout envoi de tache
- [ ] Une methode inconnue renvoie une erreur JSON-RPC `-32601`, sans exception non geree
- [ ] stdlib pur, offline, deterministe, aucune cle API

---

## Exercice 2 : Couche de confiance protocolaire — enveloppes signees (HMAC), anti-rejeu, autorisation scopee

### Objectif
Securiser le canal inter-agents au niveau du protocole (section 6 du cours). Beaucoup de demos A2A oublient que le contenu d'un agent externe est **untrusted** et que le transport peut etre altere ou rejoue. Tu vas implementer une couche de confiance avec : **enveloppes signees HMAC** (integrite + authenticite via `hmac`/`hashlib` de la stdlib), **protection anti-rejeu** (nonce + horodatage + fenetre de validite), et **autorisation scopee par capacite**. L'objectif : un message **altere**, **rejoue** ou **non autorise** est rejete, tandis qu'un message legitime passe.

### Consigne
Tout en stdlib (`hmac`, `hashlib`, `json`). Pas de vrai reseau.

1. **Cles partagees** : un registre `sender_id -> shared_secret` (les deux pairs connaissent le secret ; modele HMAC symetrique). L'autorisation est un registre `sender_id -> set(capabilities autorisees)`.
2. **Signature canonique** : ecris `sign(payload: dict, secret: str) -> str` qui :
   - serialise le payload de maniere **canonique** (`json.dumps(..., sort_keys=True, separators=(",", ":"))`) — sinon deux serialisations differentes du meme dict cassent la verif ;
   - calcule `hmac.new(secret, canonical, hashlib.sha256).hexdigest()`.
   Et `verify_signature(payload, secret, sig) -> bool` qui recompare en **temps constant** (`hmac.compare_digest`).
3. **Enveloppe signee** : `make_signed_envelope(sender, capability, body, secret, nonce, ts)` qui construit `{"sender", "capability", "body", "nonce", "ts"}` puis y ajoute `"sig"` = signature du dict **sans** la cle `sig`.
4. **Verificateur cote recepteur** `TrustGate.accept(env, now) -> tuple[bool, str]` qui rejette dans cet ordre, avec une raison distincte :
   - `unknown_sender` (pas de secret connu) ;
   - `bad_signature` (signature recalculee != `sig` → detecte toute alteration du body, du sender ou de la capability) ;
   - `expired` (`abs(now - ts) > window`, ex window = 30 s) ;
   - `replay` (nonce deja vu — garde un set des nonces consommes) ;
   - `unauthorized` (la `capability` n'est pas dans le scope autorise du sender) ;
   - sinon `(True, "accepted")` et le nonce est **consomme**.
5. **Scenario de test** (assertions) :
   - un message legitime, bien signe, frais, nonce neuf, capacite autorisee → **accepte** ;
   - **altere** : on modifie `body` apres signature → `bad_signature` ;
   - **usurpation de capacite** : on change `capability` apres signature → `bad_signature` (la capacite est couverte par la signature) ;
   - **rejeu** : on renvoie deux fois la **meme** enveloppe valide → la 2e est `replay` ;
   - **perime** : `ts` hors fenetre → `expired` ;
   - **non autorise** : un sender connu et bien signe mais demandant une capacite hors de son scope → `unauthorized` ;
   - **sender inconnu** → `unknown_sender`.

### Criteres de reussite
- [ ] La signature HMAC utilise une serialisation **canonique** (`sort_keys`), sinon la verif est instable
- [ ] `verify_signature` utilise `hmac.compare_digest` (comparaison en temps constant)
- [ ] Toute alteration du `body` **ou** de la `capability` apres signature donne `bad_signature`
- [ ] Le rejeu de la meme enveloppe (meme nonce) est detecte et rejete (`replay`)
- [ ] Un message hors fenetre temporelle est rejete (`expired`), via une horloge injectee deterministe
- [ ] Une capacite hors du scope autorise du sender est rejetee (`unauthorized`) meme si la signature est valide
- [ ] Un message legitime traverse toutes les verifications et est accepte ; stdlib pur, offline
