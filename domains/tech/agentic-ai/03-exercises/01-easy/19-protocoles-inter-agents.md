# Exercices — Protocoles inter-agents (J19)

---

## Exercice 1 : Etendre l'Agent Card avec des metadonnees de confiance

### Objectif
Comprendre la structure de l'Agent Card A2A et y ajouter des informations de trust qui permettent a un orchestrateur de verifier l'identite d'un agent avant de lui deleguer une tache.

### Consigne
En partant du code de `02-code/19-protocoles-inter-agents.py` :

1. Ajoute un champ `trust_metadata` a la dataclass `AgentCard` :
   - `issuer: str` — l'organisation qui a emis/signe cette card (ex: `"acme-corp.example.com"`)
   - `issued_at: str` — date ISO 8601 d'emission
   - `expires_at: str` — date ISO 8601 d'expiration
   - `signature: str` — une chaine "signature" (mock : SHA-256 des champs name+url+issued_at)
2. Ajoute une methode `is_valid(self) -> bool` a `AgentCard` qui verifie que :
   - `expires_at` est superieur a la date actuelle
   - `signature` correspond bien au SHA-256 attendu (mock verification)
3. Ajoute une methode `verify_trust(card_dict: dict) -> tuple[bool, str]` (fonction libre) qui prend un agent card dict et retourne `(True, "ok")` ou `(False, "raison")`.
4. Teste avec 3 cartes : une valide, une expiree, une avec signature falsifiee.
5. Affiche clairement le verdict pour chaque carte.

### Criteres de reussite
- [ ] `AgentCard.to_dict()` inclut `trustMetadata` avec les 4 champs
- [ ] `is_valid()` retourne `False` pour une carte expiree
- [ ] `is_valid()` retourne `False` pour une signature incorrecte
- [ ] `verify_trust()` retourne `(True, "ok")` pour la carte valide
- [ ] Les 3 cas sont testes et le verdict s'affiche

---

## Exercice 2 : Ajouter un etat `input-required` au cycle de vie des taches

### Objectif
Implementer l'etat `input-required` du cycle de vie A2A, qui permet a un agent de suspendre une tache et de demander une clarification au client avant de continuer.

### Consigne
1. Sous-classe `AgentServer` en `ClarifyingAgentServer` :
   - Dans `_execute`, si le texte de la tache contient le mot `"ambiguous"`, passe d'abord en etat `input-required` avec un message `"Veuillez preciser le nombre de vehicules disponibles."`
   - Attends une reponse du client (simule avec un `threading.Event` et une methode `provide_clarification(task_id, text)`)
   - Apres reception de la clarification, reprends le traitement et passe en `completed`
2. Cote client, implementer le flow :
   - Envoyer une tache ambigue
   - Detecter l'etat `input-required` dans les events
   - Appeler `provide_clarification(task_id, "3 vehicules disponibles")`
   - Attendre la completion
3. Afficher chaque transition d'etat au fil de l'execution.

### Criteres de reussite
- [ ] L'etat `input-required` apparait dans le flux d'evenements
- [ ] La tache ne passe pas en `completed` sans clarification
- [ ] Apres `provide_clarification`, la tache reprend et se complete
- [ ] L'historique de la tache montre les 4 transitions : submitted → working → input-required → working → completed
- [ ] Aucune cle API ni connexion reseau n'est necessaire

---

## Exercice 3 : Registre d'agents et routage par skill

### Objectif
Implementer un **registre d'agents** qui permet a un orchestrateur de decouvrir dynamiquement quel agent possede le skill requis, et de lui router la tache.

### Consigne
1. Cree une classe `AgentRegistry` :
   - `register(server: AgentServer)` : enregistre un serveur et indexe ses skills
   - `find_by_skill(skill_id: str) -> list[AgentServer]` : retourne les serveurs qui ont ce skill
   - `find_best(skill_id: str) -> AgentServer | None` : retourne le premier serveur disponible
2. Cree deux agents specialistes :
   - `RouteOptimizerServer` (deja dans le code de base) avec skill `optimize_routes`
   - Un `WeatherAgentServer` avec skill `get_weather` qui, dans `_execute`, retourne une meteo fictive pour une ville extraite du texte de la tache
3. Cree un `SmartOrchestrator` qui :
   - Recoit une tache en texte libre
   - Detecte le skill requis avec une heuristique simple (mots-cles : "route"/"optimize" → `optimize_routes` ; "weather"/"meteo" → `get_weather`)
   - Consulte le registre pour trouver l'agent competent
   - Delegue la tache et retourne le resultat
4. Teste avec 3 taches : une de routing, une de meteo, une ambigue (afficher l'erreur de routage).

### Criteres de reussite
- [ ] `AgentRegistry.find_by_skill` retourne les bons agents
- [ ] `SmartOrchestrator` route correctement la tache de routing vers `RouteOptimizerServer`
- [ ] `SmartOrchestrator` route correctement la tache meteo vers `WeatherAgentServer`
- [ ] La tache ambigue declenche un message d'erreur clair ("no agent found for skill X")
- [ ] Le code est stdlib pur, sans reseau
