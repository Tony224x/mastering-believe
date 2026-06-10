# Exercices Medium — LangGraph fondamentaux (J5)

---

## Exercice 1 : Boucle agent-tools avec garde d'iterations

### Objectif
Implementer le pattern central de LangGraph : un cycle `agent -> tools -> agent` avec un conditional edge qui decide de boucler ou de terminer, protege par une limite d'iterations dans le state.

### Consigne
En partant du stub `MiniStateGraph` (ou de la vraie lib si installee) :

1. Definis le state :
   ```python
   class AgentState(TypedDict):
       messages: Annotated[list, add]
       iterations: int
       pending_tool: str | None
   ```
2. Cree un node `agent_node` (mock deterministe) qui :
   - Lit le dernier message
   - Si la question demande un calcul non encore fait -> retourne `{"pending_tool": "calculator", ...}`
   - Si le resultat du calcul est deja dans les messages -> produit la reponse finale et `{"pending_tool": None}`
   - Incremente `iterations` a chaque passage
3. Cree un node `tools_node` qui execute `pending_tool` (calculator via `eval` d'une expression extraite par regex) et ajoute un message `{"role": "tool", "content": ...}`
4. Ajoute un conditional edge apres `agent` : `pending_tool != None -> tools`, sinon `-> END`
5. Ajoute la garde : si `iterations >= 5`, route vers END quoi qu'il arrive, avec un message "[MAX ITERATIONS]"
6. Teste : "Compute 12 * 34 then stop" (2 passages agent), et un mock volontairement casse qui redemande toujours le meme outil (la garde doit declencher)

### Criteres de reussite
- [ ] Le graph contient un cycle agent -> tools -> agent
- [ ] Le conditional edge route vers tools puis vers END au bon moment
- [ ] La question test termine en exactement 2 passages agent + 1 passage tools
- [ ] Le scenario "agent casse" s'arrete via la garde avec le message [MAX ITERATIONS]
- [ ] Le state final contient le message tool ET la reponse finale

---

## Exercice 2 : Node de validation avec retry-loop

### Objectif
Construire un pattern production courant : un node generateur suivi d'un node validateur qui renvoie vers un node correcteur quand la sortie est invalide, avec un nombre de retries borne dans le state.

### Consigne
1. State : `draft: str`, `errors: Annotated[list, add]`, `retries: int`, `validated: bool`
2. Node `generator` (mock) : produit un draft volontairement invalide au premier passage (ex: JSON sans la cle `"summary"`), valide au second
   - Astuce : le mock se base sur `retries` pour changer de comportement
3. Node `validator` : verifie 3 regles **locales** (sans LLM) :
   - Le draft est un JSON parseable
   - Il contient les cles `summary` et `confidence`
   - `confidence` est un float entre 0 et 1
   - Ecrit la liste des violations dans `errors` et `validated: True/False`
4. Node `fixer` : construit un prompt de correction avec les erreurs (mock : retourne le draft corrige), incremente `retries`
5. Routing : `generator -> validator`, puis conditional : `validated -> END`, `not validated et retries < 2 -> fixer -> validator`, `retries >= 2 -> END` avec un draft d'erreur
6. Teste les 3 chemins : valide du premier coup, valide apres 1 fix, jamais valide (le mock retourne toujours un draft casse)

### Criteres de reussite
- [ ] La validation est 100% locale et teste les 3 regles
- [ ] Le chemin "invalide -> fixer -> validator -> END" fonctionne
- [ ] Les erreurs de chaque round sont accumulees dans `errors` (reducer add)
- [ ] La limite de 2 retries est respectee sur le cas "jamais valide"
- [ ] Les 3 scenarios de test passent avec des asserts

---

## Exercice 3 : Human-in-the-loop — interrupt avant action sensible

### Objectif
Implementer le mecanisme d'interrupt de LangGraph dans le stub : le graph se met en pause avant un node sensible, un humain inspecte/modifie le state, puis l'execution reprend.

### Consigne
1. Etends `MiniStateGraph`/`MiniCompiled` pour supporter `compile(interrupt_before=["sensitive"])` :
   - Quand le node suivant est dans `interrupt_before`, `invoke` retourne le state courant avec une cle speciale `"__interrupted__": "sensitive"` au lieu d'executer le node
   - Ajoute une methode `resume(state)` qui reprend l'execution **a partir du node interrompu** (en l'executant cette fois)
2. Construis un graph : `START -> draft_email -> sensitive (send_email) -> END`
   - `draft_email` ecrit `{"email_to": "boss@corp.com", "email_body": "..."}` dans le state
   - `send_email` ajoute `{"sent": True, "sent_to": state["email_to"]}`
3. Scenario A (approve) : invoke -> interrompu -> l'humain inspecte -> resume sans modification -> email envoye
4. Scenario B (edit) : invoke -> interrompu -> l'humain change `email_to` en `"assistant@corp.com"` -> resume -> l'email part vers la nouvelle adresse
5. Scenario C (reject) : invoke -> interrompu -> l'humain n'appelle jamais resume et ajoute `{"aborted": True}` — verifie que `sent` est absent du state

### Criteres de reussite
- [ ] `invoke` s'arrete AVANT d'executer le node sensible (verifiable : `sent` absent)
- [ ] Le state retourne contient `__interrupted__` avec le nom du node
- [ ] `resume` reprend exactement au node interrompu (pas depuis le debut)
- [ ] Le scenario B montre que la modification humaine est prise en compte
- [ ] Les 3 scenarios passent avec des asserts
