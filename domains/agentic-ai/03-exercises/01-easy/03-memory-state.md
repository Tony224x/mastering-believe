# Exercices Faciles — Memory & State (J3)

---

## Exercice 1 : Token-aware sliding window

### Objectif
Comprendre comment le token counting influence la gestion de la memoire de conversation.

### Consigne
En partant du `SlidingWindowMemory` dans `02-code/03-memory-state.py` :

1. Modifie la classe pour ajouter une methode `get_stats() -> dict` qui retourne :
   - `total_messages_seen` : nombre total de messages ajoutes depuis la creation
   - `current_window_size` : nombre de messages dans la fenetre actuelle
   - `current_token_count` : nombre de tokens dans la fenetre
   - `dropped_messages` : nombre de messages supprimes
   - `avg_tokens_per_message` : moyenne de tokens par message dans la fenetre
2. Cree un scenario de test avec 15 messages de tailles variees (certains courts "Ok", d'autres longs avec 200+ caracteres)
3. Configure la fenetre a `max_messages=8, max_tokens=300`
4. Affiche les stats apres chaque message ajoute pour visualiser comment la fenetre evolue

### Criteres de reussite
- [ ] `get_stats()` retourne les 5 metriques correctement
- [ ] Les messages longs ejectent plus de messages courts que prevu (effet du token budget)
- [ ] L'affichage montre clairement l'evolution de la fenetre etape par etape
- [ ] Le `avg_tokens_per_message` change quand un message long est ajoute

---

## Exercice 2 : Working memory avec types valides

### Objectif
Comprendre l'importance de la validation dans la working memory d'un agent.

### Consigne
Etends la classe `WorkingMemory` pour ajouter un systeme de typage simple :

1. Ajoute une methode `set_typed(key, value, expected_type, source)` qui verifie le type avant de stocker :
   - Si `value` n'est pas du type `expected_type`, leve une `TypeError` avec un message clair
   - Types supportes : `str`, `int`, `float`, `list`, `dict`, `bool`
2. Ajoute une methode `get_typed(key, expected_type)` qui verifie le type au moment de la lecture
3. Teste avec des cas valides et invalides :
   - `set_typed("budget", 500, int)` → OK
   - `set_typed("budget", "five hundred", int)` → TypeError
   - `get_typed("budget", str)` → TypeError (stocke comme int, lu comme str)

### Criteres de reussite
- [ ] `set_typed` rejette les valeurs du mauvais type avec un message explicite
- [ ] `get_typed` verifie le type au moment de la lecture
- [ ] Les messages d'erreur indiquent le type attendu et le type recu
- [ ] Au moins 3 cas valides et 3 cas invalides testes
- [ ] La methode `set` originale (non-typee) fonctionne toujours

---

## Exercice 3 : Serialisation de checkpoint

### Objectif
Comprendre ce qu'il faut sauvegarder et restaurer pour reprendre l'execution d'un agent.

### Consigne
1. Cree un `AgentState` avec les champs suivants :
   - `messages` (list), `working_memory` (dict), `iteration` (int), `task` (str), `tools_used` (list de str)
2. Ecris une fonction `save_checkpoint(state, filepath)` qui serialise l'etat en JSON
3. Ecris une fonction `load_checkpoint(filepath) -> AgentState` qui reconstruit l'etat
4. Teste le cycle complet :
   - Cree un etat avec des donnees realistes (5 messages, 3 cles en working memory)
   - Sauvegarde en JSON
   - Charge depuis le JSON
   - Verifie que TOUS les champs sont identiques (assert)
5. Teste un cas d'erreur : que se passe-t-il si le fichier JSON est corrompu ? Gere l'erreur proprement.

### Criteres de reussite
- [ ] Le JSON sauvegarde est lisible et contient tous les champs
- [ ] Le cycle save → load → compare ne perd aucune donnee
- [ ] Le chargement d'un fichier corrompu retourne une erreur claire (pas un crash)
- [ ] Le code utilise `dataclass` ou `Pydantic` pour le state (pas un dict nu)
