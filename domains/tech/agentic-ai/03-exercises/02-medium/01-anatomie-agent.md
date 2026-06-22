# Exercices Medium — Anatomie d'un Agent (J1)

---

## Exercice 1 : Agent avec function calling natif (structured output)

### Objectif
Passer du parsing regex (fragile) au function calling natif de l'API OpenAI — la methode utilisee en production.

### Consigne
Reimplemente l'agent ReAct en utilisant le **function calling** (tool_use) de l'API OpenAI au lieu du parsing texte :

1. Formate les outils au format `tools` de l'API OpenAI (schema JSON standard)
2. Envoie les outils dans le parametre `tools` de l'appel API
3. Parse la reponse : si `tool_calls` est present, execute l'outil et renvoie le resultat avec `role: "tool"`
4. Si pas de `tool_calls`, c'est la reponse finale
5. Ajoute un outil `finish` special ou detecte quand le LLM repond directement sans appeler d'outil

Le system prompt n'a plus besoin de decrire le format ReAct — l'API gere le format.

**Mode simule** : cree des reponses simulees qui imitent la structure de reponse avec `tool_calls` (JSON).

### Criteres de reussite
- [ ] L'agent utilise le parametre `tools` de l'API au lieu de decrire les outils dans le prompt
- [ ] Le parsing est base sur la structure JSON de la reponse, pas sur du regex
- [ ] L'historique des messages utilise `role: "tool"` pour les resultats d'outils
- [ ] Le meme probleme multi-etapes ("25 * 47 + current time") est resolu correctement
- [ ] Code plus robuste que la version regex (pas de casse si le LLM change de formulation)

---

## Exercice 2 : Agent avec memoire de travail (working memory)

### Objectif
Implementer une working memory qui persiste entre les etapes — le "scratchpad" de l'agent.

### Consigne
Ajoute un mecanisme de working memory a l'agent :

1. Definis un outil `note_to_self` qui permet a l'agent de sauvegarder une note structuree :
   ```json
   {"key": "user_preference", "value": "Prefers Celsius"}
   ```
2. Definis un outil `recall` qui permet a l'agent de recuperer une note par cle
3. Stocke les notes dans un dict Python (pas besoin de persistance)
4. **A chaque etape**, injecte un resume de la working memory dans le prompt :
   ```
   Working Memory:
   - user_preference: Prefers Celsius
   - intermediate_result: GDP France = 44400 USD
   ```
5. Teste avec un scenario multi-etapes ou l'agent doit se souvenir d'une info intermediaire :
   "Calculate 25 * 47, then add 100 to the result, then multiply by 2"
   (L'agent doit noter le resultat intermediaire pour l'utiliser a l'etape suivante)

### Criteres de reussite
- [ ] L'outil `note_to_self` stocke des paires cle-valeur accessibles entre les etapes
- [ ] L'outil `recall` retrouve une note par sa cle
- [ ] La working memory est affichee dans le prompt a chaque iteration
- [ ] L'agent resout correctement le probleme a 3 etapes en utilisant sa memoire
- [ ] Les notes sont visibles dans la trace d'execution

---

## Exercice 3 : Multi-agent basique — Router + Specialists

### Objectif
Implementer le pattern multi-agent le plus simple : un routeur qui delegue a des agents specialises.

### Consigne
Cree un systeme avec 3 agents :

1. **Router Agent** : recoit la question utilisateur, decide quel specialiste appeler
   - Outils : `delegate_to_math`, `delegate_to_research`, `respond_directly`
   - System prompt : "You are a router. Analyze the question and delegate to the right specialist."

2. **Math Agent** : resout les problemes mathematiques
   - Outils : `calculator`
   - System prompt optimise pour le calcul

3. **Research Agent** : repond aux questions factuelles
   - Outils : `search`
   - System prompt optimise pour la recherche

Implementation :
- Chaque agent est une instance de `react_agent` avec ses propres outils et system prompt
- Le router appelle les specialistes via des fonctions Python (pas de communication reseau)
- Le resultat du specialiste est retourne au router qui formate la reponse finale

Teste avec :
- "What is 123 * 456?" → devrait router vers Math Agent
- "What is the capital of Guinea?" → devrait router vers Research Agent
- "Hello, how are you?" → devrait repondre directement

### Criteres de reussite
- [ ] Le router identifie correctement le type de question (math, research, general)
- [ ] Chaque specialiste a son propre system prompt et ses propres outils
- [ ] La trace affiche clairement : Router → delegation → Specialist → reponse
- [ ] Le systeme fonctionne en mode simule (hardcoded responses pour chaque agent)
- [ ] Le code est structure : chaque agent est configurable via des parametres
