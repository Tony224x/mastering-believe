# Exercices Medium — Protocoles inter-agents (J19)

---

## Exercice 1 : Enveloppe de message typee + routeur par capacite

### Objectif
Aller au-dela du simple `tasks/send` du module : construire une **enveloppe de message typee** (l'equivalent du `Message` A2A enrichi d'un en-tete de routage) et un **routeur inter-agents** qui valide chaque enveloppe, verifie l'autorisation de l'expediteur, et l'achemine vers l'agent qui declare la **capacite** demandee. Un routeur reel rejette autant de messages qu'il en route : malformes, non autorises, ou sans destinataire competent. C'est ce comportement de rejet propre qu'on cible ici.

### Consigne
Tout reembarquer dans ta solution (stdlib pur, pas d'import du `02-code`) :

1. Definis un `@dataclass Envelope` qui modelise une enveloppe de message inter-agents avec au minimum :
   - `msg_id: str`, `sender: str`, `capability: str` (la capacite demandee, ex `"optimize_routes"`),
   - `payload: dict`, `protocol_version: str` (ex `"1.0"`).
2. Ecris une fonction `validate_envelope(env) -> tuple[bool, str]` qui rejette une enveloppe **malformee** :
   - champ obligatoire manquant ou vide (`msg_id`, `sender`, `capability`),
   - `payload` qui n'est pas un dict,
   - `protocol_version` non supportee (n'appartient pas a un ensemble `SUPPORTED = {"1.0", "1.1"}`).
3. Modelise au moins 2 agents recepteurs declarant chacun leurs capacites (ex : un agent `route` declare `optimize_routes`, un agent `weather` declare `get_weather`). Garde une **ACL** (access control list) `capability -> set(senders autorises)`.
4. Ecris une classe `MessageRouter` avec `route(env) -> dict` qui, dans l'ordre :
   - **valide** l'enveloppe (rejet si invalide, raison explicite) ;
   - verifie l'**autorisation** : le `sender` est-il dans l'ACL de la `capability` ? (rejet sinon) ;
   - trouve l'agent qui declare la `capability` (rejet `no_route` si aucun) ;
   - sinon, dispatche au bon agent et retourne `{"status": "routed", "agent": ..., "result": ...}`.
5. **Prouve par assertions** : une enveloppe valide+autorisee est routee au bon agent ; une enveloppe malformee est rejetee avec la bonne raison ; un sender non autorise est rejete ; une capacite sans destinataire renvoie `no_route`.

### Criteres de reussite
- [ ] `Envelope` est un dataclass typé avec en-tete de routage (sender, capability, version, payload)
- [ ] `validate_envelope` rejette champ manquant, payload non-dict et version non supportee, avec une raison distincte par cas
- [ ] Une enveloppe valide et autorisee est routee vers l'agent qui declare la capacite (verifie par assertion)
- [ ] Un sender absent de l'ACL est rejete (`unauthorized`), pas route
- [ ] Une capacite sans agent declarant renvoie `no_route` au lieu de planter
- [ ] Tout tourne offline, sans cle API ni dependance

---

## Exercice 2 : Delegation de tache request/response avec correlation ID, statuts et timeout

### Objectif
Implementer le **contrat de delegation** A2A (section 3.4 du cours) cote protocole : un client soumet une tache, recoit un **correlation ID**, puis suit le **cycle de vie** (`submitted → working → completed/failed`) via des appels `get`. Tu dois prouver deux choses que beaucoup d'implementations ratent : (a) les transitions de statut sont **monotones et tracees**, et (b) une tache qui depasse son **deadline** bascule proprement en `failed`/`timed_out` au lieu de rester bloquee.

### Consigne
1. Definis un enum `Status` avec `SUBMITTED`, `WORKING`, `COMPLETED`, `FAILED`, `TIMED_OUT`.
2. Modelise un `TaskRecord` (dataclass) : `corr_id`, `status`, `submitted_at`, `deadline`, `result`, `history: list[str]`. Chaque transition **append** dans `history`.
3. Ecris un `DelegationBroker` (le cote serveur) avec :
   - `submit(work_fn, deadline_s) -> corr_id` : cree un `TaskRecord` en `SUBMITTED`, genere un correlation ID unique, et **stocke** `work_fn` (un callable qui simule le travail, sans thread) ;
   - `step(corr_id)` : avance la tache d'un cran — passe `SUBMITTED → WORKING`, puis a l'appel suivant execute `work_fn` et passe `WORKING → COMPLETED` (ou `FAILED` si `work_fn` leve) ; mais **avant** chaque transition, si l'horloge logique a depasse le `deadline`, bascule en `TIMED_OUT` ;
   - `get(corr_id) -> dict` : renvoie l'etat courant (statut + history).
   - Utilise une **horloge logique injectable** (`now: Callable[[], float]`) pour rendre le timeout deterministe, sans `time.sleep`.
4. Cote client, ecris `delegate_and_poll(broker, work_fn, deadline_s, now)` qui soumet, recupere le `corr_id`, boucle sur `step`/`get` jusqu'a un etat terminal, et retourne le `TaskRecord` final.
5. **Prouve par assertions** : un travail rapide passe `submitted → working → completed` (history exacte) ; un `work_fn` qui leve donne `failed` ; une tache dont le deadline est depasse (horloge avancee a la main) finit en `timed_out` sans jamais executer `work_fn`.

### Criteres de reussite
- [ ] Chaque tache recoit un correlation ID unique, renvoye a la soumission
- [ ] L'history enregistre la sequence exacte des transitions `submitted → working → completed`
- [ ] Un `work_fn` qui leve fait passer la tache en `failed` (et pas `completed`)
- [ ] Le timeout est deterministe (horloge logique injectee) et bascule en `timed_out`
- [ ] Une tache `timed_out` n'a jamais execute `work_fn` (verifie par un compteur d'effets)
- [ ] Aucun thread reel, aucun `time.sleep`, execution offline et deterministe

---

## Exercice 3 : Handshake versionne + negociation de capacites entre client et agent

### Objectif
Implementer le **handshake** d'entree d'un dialogue A2A : avant d'envoyer la moindre tache, un client et un agent doivent s'accorder sur (a) une **version de protocole** commune et (b) le **sous-ensemble de capacites** que les deux supportent. C'est l'etape de negociation qui rend deux agents heterogenes interoperables — ou qui echoue **proprement** si l'intersection est vide.

### Consigne
1. Modelise un agent et un client comme deux profils declaratifs :
   ```python
   {"versions": ["1.0", "1.1"], "capabilities": {"optimize_routes", "estimate_eta"}}
   ```
2. Ecris `negotiate(client_profile, server_profile) -> dict` qui :
   - calcule la **version commune la plus elevee** (intersection des `versions`, on prend la max selon un ordre de version) ; si l'intersection est vide → `{"ok": False, "reason": "version_mismatch"}` ;
   - calcule l'**intersection des capacites** ; si vide → `{"ok": False, "reason": "no_common_capability"}` ;
   - sinon → `{"ok": True, "version": <v>, "capabilities": <set trie>}`.
3. Ecris une fonction de comparaison de versions `version_key("1.10") -> (1, 10)` (split sur `.`, tuple d'entiers) pour que `"1.10" > "1.9"` soit vrai (piege classique de la comparaison lexicographique de versions).
4. Apres un handshake reussi, le client ne doit pouvoir demander **que** les capacites negociees : ecris `dispatch(session, capability)` qui leve `ValueError` si `capability` n'est pas dans l'ensemble negocie.
5. **Prouve par assertions** : deux profils compatibles negocient la version max commune et la bonne intersection ; un mismatch de version renvoie `version_mismatch` ; une intersection de capacites vide renvoie `no_common_capability` ; `version_key` ordonne correctement `"1.10"` devant `"1.9"` ; demander une capacite hors session leve `ValueError`.

### Criteres de reussite
- [ ] `negotiate` retourne la version commune **la plus elevee**, pas la premiere trouvee
- [ ] `version_key` traite `"1.10" > "1.9"` correctement (pas de comparaison lexicographique naive)
- [ ] Un mismatch de version renvoie `version_mismatch`, une intersection de capacites vide renvoie `no_common_capability`
- [ ] Apres handshake, demander une capacite hors de l'ensemble negocie leve `ValueError`
- [ ] Les profils sont declaratifs (versions + capacites), aucun couplage au framework
- [ ] Execution offline, deterministe, sans dependance
