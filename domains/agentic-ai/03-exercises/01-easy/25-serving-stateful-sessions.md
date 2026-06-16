# Exercices — Serving stateful & sessions (J25)

---

## Exercice 1 : TTL et expiration des sessions

### Objectif
Comprendre comment les sessions peuvent expirer automatiquement pour liberer des ressources, et implementer un mecanisme de nettoyage analogique a `EXPIRE` dans Redis ou `DELETE … WHERE updated_at < NOW() - INTERVAL`.

### Consigne
En partant de `02-code/25-serving-stateful-sessions.py` :

1. Cree une classe `TTLSQLiteCheckpointer` qui etend `SQLiteCheckpointer` avec :
   - Un parametre `ttl_seconds: float` (ex : 60.0 pour une session qui expire apres 60 secondes sans activite)
   - Une methode `purge_expired() -> int` qui supprime de la base tous les checkpoints dont le `created_at` du dernier step est plus ancien que `ttl_seconds`, et retourne le nombre de threads supprimes
   - `load` doit retourner `None` si le dernier checkpoint est expire (au lieu de le retourner)
2. Cree 3 sessions :
   - `session-fresh` : mise a jour "maintenant"
   - `session-old` : creee avec `created_at = time.time() - 120` (simulee comme ancienne)
   - `session-borderline` : `created_at = time.time() - 59` (limite non atteinte)
3. Appelle `purge_expired()` avec `ttl_seconds=60` et verifie :
   - `session-old` est supprimee
   - `session-fresh` et `session-borderline` survivent
   - `load("session-old")` retourne `None` apres la purge
4. Affiche un rapport : nombre supprime, sessions restantes

### Criteres de reussite
- [ ] `TTLSQLiteCheckpointer` etend `SQLiteCheckpointer` sans dupliquer le schema SQL
- [ ] `purge_expired()` retourne le nombre exact de threads supprimes
- [ ] `load` retourne `None` pour une session expiree
- [ ] Les 2 sessions non-expirees sont toujours accessibles apres la purge
- [ ] Le rapport affiche les bonnes valeurs

---

## Exercice 2 : Isolation multi-utilisateur

### Objectif
S'assurer qu'un utilisateur ne peut pas acceder a la session d'un autre, meme s'il connait le `thread_id`.

### Consigne
1. Cree une classe `SecureSessionManager` qui enveloppe `SessionManager` et ajoute un champ `owner_id` a chaque `Checkpoint.state` :
   - `create_session(user_id: str) -> str` : genere un `thread_id` (UUID v4), sauvegarde un checkpoint initial avec `state["owner_id"] = user_id`, retourne le `thread_id`
   - `process_turn(thread_id: str, user_id: str, message: str) -> str` : verifie que `state["owner_id"] == user_id` avant de traiter ; si non, leve une `PermissionError` avec le message `"access denied: thread owned by another user"`
2. Cree 2 utilisateurs : `alice` et `bob`
3. Alice cree une session ; Bob essaie d'envoyer un message dans la session d'Alice
4. Verifie que la `PermissionError` est levee
5. Verifie qu'Alice peut toujours envoyer un message dans sa propre session

### Criteres de reussite
- [ ] `create_session` genere un UUID non-predictible
- [ ] `process_turn` leve `PermissionError` si `user_id` ne correspond pas a l'owner
- [ ] Le message d'erreur contient `"access denied"`
- [ ] Alice peut toujours utiliser sa session apres la tentative de Bob
- [ ] Bob peut creer sa propre session et l'utiliser sans erreur

---

## Exercice 3 : Detecter le drift avec un comparateur de distributions

### Objectif
Approfondir la detection de drift en comparant non seulement le taux de succes moyen mais aussi la variance — un taux stable a 80% peut cacher une bimodalite (certains types de requetes tombent a 0%, d'autres restent a 100%).

### Consigne
1. Etends `OnlineDriftMonitor` pour creer `AdvancedDriftMonitor` :
   - Stocke en plus du succes/echec le type de requete (`query_type: str`)
   - Methode `success_rate_by_type() -> dict[str, float]` : taux de succes par type sur la fenetre courante
   - Methode `detect_type_drift(baseline_by_type: dict[str, float], threshold: float = 0.15) -> list[str]` : retourne les types dont le taux a baisse de plus de `threshold` (ex. 0.15 = 15 pp) par rapport a la baseline
2. Simule une periode saine avec 3 types : `"faq"` (95%), `"support"` (80%), `"complex"` (70%)
3. Simule une degradation : `"complex"` tombe a 30%, les autres restent stables
4. Verifie que `detect_type_drift` signale uniquement `"complex"` comme derive
5. Affiche le rapport complet : taux par type, types derives, taux global

### Criteres de reussite
- [ ] `success_rate_by_type` calcule correctement par type sur la fenetre courante
- [ ] `detect_type_drift` identifie `"complex"` comme le seul type derive
- [ ] `"faq"` et `"support"` ne sont PAS signales comme derives
- [ ] La fenetre glissante (maxlen) s'applique aussi par type
- [ ] Le rapport affiche les taux avant et apres degradation
