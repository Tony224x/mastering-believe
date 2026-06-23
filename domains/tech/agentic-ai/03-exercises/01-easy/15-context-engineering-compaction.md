# Exercices — Context engineering (J15)

---

## Exercice 1 : Ajouter une politique de retention au ContextManager

### Objectif
Comprendre que la compaction brute (garder les N derniers messages) peut perdre des informations critiques (le goal initial). Implementer une politique de retention selective.

### Consigne
En partant de `02-code/15-context-engineering-compaction.py` :

1. Cree une classe `RetentionPolicy` avec une methode `should_keep(message: dict) -> bool` qui retourne `True` si le message doit etre preserve malgre la compaction.
2. Implemente deux criteres de retention :
   - Le premier message de type `"user"` (le goal original) est toujours conserve
   - Tout message dont le contenu contient le mot-cle `"[CRITICAL]"` est toujours conserve
3. Modifie le comportement de `ContextManager._compact` pour passer les messages marques comme "a conserver" directement dans la liste resultante, APRES le resume et AVANT les messages recents.
4. Teste avec un historique de 12 messages dont :
   - 1 premier message user "goal: [CRITICAL] never delete external API credentials"
   - 1 message au milieu marque `[CRITICAL]` contenant une decision importante
   - 10 messages ordinaires
5. Verifie qu'apres compaction, le message goal et le message `[CRITICAL]` sont bien dans le contexte resultant.

### Criteres de reussite
- [ ] `RetentionPolicy.should_keep` identifie correctement les messages a garder
- [ ] Apres compaction, le message goal (premier user) est present
- [ ] Apres compaction, le message `[CRITICAL]` est present
- [ ] Les messages ordinaires sont compactes en resume
- [ ] Le test affiche clairement le nombre de messages avant et apres compaction

---

## Exercice 2 : Mesure du context rot par scoring de coherence

### Objectif
Detecter le context rot empiriquement : mesurer si un agent "derive" de son objectif original au fil des tours.

### Consigne
1. Cree une fonction `coherence_score(goal: str, recent_action: str) -> float` :
   - Retourne un score entre 0.0 et 1.0
   - Score = proportion de mots du `goal` (apres stopwords) presents dans `recent_action`
   - Stopwords a ignorer : `{"le", "la", "les", "de", "du", "des", "un", "une", "et", "ou", "en", "a", "the", "of", "a", "to", "in", "is", "it"}`
2. Cree une classe `ContextRotMonitor` :
   - `__init__(goal: str, window: int = 5)` : fenetre glissante de N derniers scores
   - `record(action: str) -> float` : enregistre une action, retourne le score de coherence
   - `rot_detected(threshold: float = 0.3) -> bool` : retourne True si la moyenne des scores sur la fenetre est sous le seuil
   - `summary() -> dict` : retourne `{"scores": [...], "mean": ..., "rot": ...}`
3. Simule un agent qui derive :
   - Tours 1-3 : actions coherentes avec le goal "analyze Flask security vulnerabilities"
   - Tours 4-6 : actions completement hors sujet ("listing python packages", "checking git history", "reading README")
4. Montre que `rot_detected()` passe de `False` a `True` au fil des tours.

### Criteres de reussite
- [ ] `coherence_score` retourne 1.0 quand `recent_action` contient tous les mots du goal
- [ ] `coherence_score` retourne 0.0 quand aucun mot ne correspond
- [ ] `ContextRotMonitor.rot_detected` retourne `False` sur les 3 premiers tours
- [ ] `ContextRotMonitor.rot_detected` retourne `True` apres les 3 tours hors sujet
- [ ] Le summary affiche les scores individuels et la moyenne

---

## Exercice 3 : Budget adaptatif avec escalade a l'orchestrateur

### Objectif
Implementer le pattern d'escalade : quand un sous-agent approche de l'epuisement de son budget, il signale a l'orchestrateur qu'il a besoin de plus de ressources ou qu'il doit interrompre.

### Consigne
En reutilisant `TokenBudget` de `02-code/15-context-engineering-compaction.py` :

1. Cree une classe `BudgetEvent` avec les champs `agent: str`, `kind: str` (parmi `"warning"`, `"critical"`, `"exhausted"`), `remaining_pct: float`, `message: str`.
2. Cree une classe `BudgetEventBus` :
   - `subscribe(handler: Callable[[BudgetEvent], None]) -> None`
   - `publish(event: BudgetEvent) -> None` : notifie tous les abonnes
3. Cree une classe `MonitoredBudget` qui enveloppe `TokenBudget` :
   - `consume(agent: str, tokens: int) -> bool` : appelle le vrai `consume`, puis publie un evenement si le seuil change :
     - < 30% restant → publie `"warning"`
     - < 10% restant → publie `"critical"`
     - < 0% restant → publie `"exhausted"`
   - Chaque seuil n'est publie qu'une seule fois par agent (eviter le spam)
4. Cree un orchestrateur simple `OrchestratorLogger` qui s'abonne au bus et loggue chaque evenement recu.
5. Simule 3 agents qui consomment progressivement leur budget et montre les evenements en temps reel.

### Criteres de reussite
- [ ] Les 3 types d'evenements (`warning`, `critical`, `exhausted`) sont emis
- [ ] Chaque type d'evenement n'est emis qu'une seule fois par agent
- [ ] L'orchestrateur recoit et loggue les evenements avec le nom de l'agent
- [ ] Le cycle complet fonctionne sur 3 agents avec des profils de consommation differents
- [ ] La simulation s'affiche avec des messages clairs indiquant quel agent a declenche quel evenement
