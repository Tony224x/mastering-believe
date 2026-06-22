# Exercices Medium — Memory & State (J3)

---

## Exercice 1 : Hybrid memory avec summarization adaptative

### Objectif
Implementer une strategie hybride intelligente qui adapte la frequence de summarization selon l'activite de la conversation.

### Consigne
Cree une classe `AdaptiveHybridMemory` qui :

1. **Detecte l'intensite de la conversation** :
   - Si les messages arrivent rapidement (< 5 messages entre chaque summarization trigger), utilise un seuil bas (`summary_threshold = 15`)
   - Si la conversation est lente (> 20 messages entre triggers), utilise un seuil haut (`summary_threshold = 40`)
2. **Priorite les messages importants** :
   - Les messages utilisateur sont toujours gardes plus longtemps que les messages assistant
   - Les messages contenant des "marqueurs d'importance" (mots-cles : "important", "budget", "deadline", "remember", "n'oublie pas") sont proteges de la summarization — ils restent dans le buffer recent
3. **Double-summary** : garde deux niveaux de summary :
   - `executive_summary` : 1-2 phrases, les faits les plus critiques (preferences, contraintes)
   - `detailed_summary` : paragraphe plus complet avec le contexte des echanges precedents

Teste avec un scenario de 30+ messages qui inclut :
- Des messages "importants" (avec marqueurs)
- Des messages banals ("Ok", "Merci", "Je vois")
- Un moment ou l'utilisateur donne une contrainte critique au message 5, et la redemande au message 28

### Criteres de reussite
- [ ] Les messages marques "importants" sont gardes dans le buffer meme quand d'autres sont supprimes
- [ ] Le `executive_summary` est toujours < 200 tokens
- [ ] Le `detailed_summary` preserve plus de contexte que l'executive
- [ ] La contrainte du message 5 est accessible au message 28 (via le summary ou le buffer)
- [ ] Le seuil de summarization s'adapte a l'intensite de la conversation
- [ ] Le token budget total (summaries + buffer) reste sous le max_tokens configure

---

## Exercice 2 : Vector memory avec recency weighting et TTL

### Objectif
Implementer un vector store avec ponderation temporelle et expiration automatique des memoires.

### Consigne
Etends la classe `VectorMemory` pour ajouter :

1. **Recency weighting** : le score de recherche combine similarite cosinus + bonus temporel
   ```
   final_score = similarity * (1 - recency_weight) + recency_score * recency_weight
   ```
   Ou `recency_score` = 1.0 pour une memoire d'il y a 0s, decroit vers 0.0 avec le temps (decay exponentiel, demi-vie configurable)

2. **TTL (Time-to-Live)** : chaque memoire a un TTL optionnel en secondes
   - A l'ajout : `store(text, metadata, ttl=3600)` (expire dans 1h)
   - A la recherche : les memoires expirees sont automatiquement exclues
   - Methode `cleanup_expired()` pour supprimer physiquement les memoires expirees

3. **Importance scoring** : chaque memoire a un score d'importance (0.0 - 1.0) mis a jour automatiquement
   - Chaque fois qu'une memoire est retournee dans un resultat de recherche, son importance augmente (+0.1, cap a 1.0)
   - Les memoires jamais consultees voient leur importance decroitre avec le temps
   - Le score final de recherche integre aussi l'importance :
   ```
   final_score = similarity * 0.5 + recency * 0.3 + importance * 0.2
   ```

Teste avec :
- 10 memoires ajoutees a des timestamps differents (simuler avec un `created_at` custom)
- Recherche qui montre que les memoires recentes sont favorisees
- TTL qui expire 3 memoires → elles n'apparaissent plus dans les resultats
- Importance qui augmente pour une memoire frequemment consultee

### Criteres de reussite
- [ ] Le recency weighting favorise les memoires recentes a similarite egale
- [ ] Le TTL exclut automatiquement les memoires expirees des resultats
- [ ] `cleanup_expired()` supprime physiquement les memoires expirees et retourne le nombre supprime
- [ ] L'importance augmente quand une memoire est retournee dans un search
- [ ] Le score final combine les 3 facteurs (similarite, recency, importance)
- [ ] Les poids sont configurables a l'initialisation

---

## Exercice 3 : State machine avec reducers et time-travel

### Objectif
Implementer le pattern immutable state + reducers pour un agent, avec la capacite de time-travel debugging.

### Consigne
Cree un systeme complet :

1. **Immutable State** :
   ```python
   @dataclass(frozen=True)
   class AgentState:
       messages: tuple[dict, ...] = ()
       working_memory: tuple[tuple[str, Any], ...] = ()  # (key, value) pairs
       iteration: int = 0
       total_tokens: int = 0
       status: str = "running"  # "running" | "paused" | "done" | "error"
   ```

2. **Actions** (types d'evenements) :
   - `ADD_MESSAGE` : ajouter un message (role, content)
   - `SET_MEMORY` : ecrire dans la working memory (key, value)
   - `INCREMENT` : incrementer l'iteration et les tokens
   - `FINISH` : marquer comme done
   - `ERROR` : marquer comme error avec un message

3. **Reducer** : une fonction pure `reduce(state, action) -> new_state`

4. **State History** : un `StateHistory` qui enregistre chaque (state, action) pair
   - `push(state, action)` : enregistrer
   - `get_state_at(step)` : retourner l'etat a l'etape N
   - `replay_from(step)` : retourner tous les etats depuis l'etape N
   - `diff(step_a, step_b)` : montrer ce qui a change entre deux etapes
   - `find_action(predicate)` : trouver la premiere action qui satisfait un predicat

5. **Simule 8 etapes d'agent** avec des actions variees, puis :
   - Affiche l'etat a l'etape 3
   - Montre le diff entre etape 2 et etape 5
   - Trouve l'action qui a provoque une erreur
   - "Branch" : charge l'etat de l'etape 4, applique des actions differentes, compare les deux branches

### Criteres de reussite
- [ ] L'etat est reellement immutable (frozen dataclass, aucune mutation)
- [ ] Le reducer est une fonction pure (pas d'effets de bord)
- [ ] Chaque action produit un nouvel etat sans modifier l'ancien
- [ ] `get_state_at(step)` retourne l'etat exact a cette etape
- [ ] `diff(a, b)` montre les champs qui ont change entre deux etapes
- [ ] Le branching fonctionne : deux branches differentes a partir du meme checkpoint
- [ ] L'historique est complet : on peut reconstituer n'importe quel etat intermediaire
