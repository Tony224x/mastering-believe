# Exercices — Verifiers & self-improvement (J17)

---

## Exercice 1 : Implementer un verifier de parite

### Objectif
Comprendre la distinction ORM/PRM en construisant deux verifiers complementaires sur un probleme simple : verifier si une expression arithmetique a une valeur paire ou impaire.

### Consigne
En partant de `02-code/17-verifiers-self-improvement.py` :

1. Cree une classe `ParityVerifier` (ORM) avec une methode `score(answer: str) -> float` :
   - Retourne `1.0` si le dernier nombre dans `answer` est **pair**
   - Retourne `0.0` si le dernier nombre est **impair**
   - Retourne `0.5` si le parsing echoue (reponse ambigue)
2. Cree une classe `ParityProcessVerifier` (PRM) avec une methode `score_step(step: str) -> float` :
   - Retourne `1.0` si l'etape contient un sous-resultat pair (ex: `"4 * 3 = 12"`)
   - Retourne `0.4` si l'etape contient un sous-resultat impair
   - Retourne `0.2` si aucun sous-resultat n'est detecte
3. Teste les deux verifiers sur 5 expressions :
   - `"step1: 3 + 5 = 8"` → pair
   - `"step1: 3 + 4 = 7"` → impair
   - `"answer: 12"` → pair
   - `"answer: 7"` → impair
   - `"answer: ??"` → ambigu
4. Affiche, pour chaque expression, le score ORM et le score PRM cote a cote

### Criteres de reussite
- [ ] `ParityVerifier.score` retourne `1.0` pour les reponses finales paires
- [ ] `ParityVerifier.score` retourne `0.0` pour les reponses finales impaires
- [ ] `ParityVerifier.score` retourne `0.5` sur une reponse non-parseable
- [ ] `ParityProcessVerifier.score_step` detecte correctement les sous-resultats pairs/impairs
- [ ] L'affichage montre les 5 cas avec les deux scores

---

## Exercice 2 : Comparer best-of-N et weighted majority

### Objectif
Mesurer empiriquement la difference de robustesse entre best-of-N et weighted majority en presence d'un verifier bruite.

### Consigne
1. Copie les fonctions `best_of_n` et `weighted_majority` de `02-code/17-verifiers-self-improvement.py`
2. Cree un generateur `noisy_generator(target, extra_context="") -> str` :
   - Avec probabilite `0.5` : retourne `f"answer: {int(target)}"` (correct)
   - Avec probabilite `0.3` : retourne `f"answer: {int(target) + random.randint(1, 5)}"` (proche mais faux)
   - Avec probabilite `0.2` : retourne `f"answer: {random.randint(1, 100)}"` (completement faux)
3. Cree un verifier bruite `NoisyOutcomeVerifier(target, noise=0.15)` :
   - `score(answer) -> float` : score exact + bruit gaussien `N(0, noise)`, clamp `[0, 1]`
4. Lance `best_of_n` et `weighted_majority` sur le meme target (`target=24`) avec `n=10`, repete `20 fois` chaque experience
5. Affiche la moyenne et l'ecart-type du score final pour les deux methodes

### Criteres de reussite
- [ ] `noisy_generator` respecte les 3 probabilites declarees
- [ ] `NoisyOutcomeVerifier` ajoute du bruit gaussien et clamp en `[0, 1]`
- [ ] L'experience est repete 20 fois pour chaque methode
- [ ] La moyenne et l'ecart-type sont affiches pour les deux methodes
- [ ] Le commentaire final explique en une phrase pourquoi l'une est plus robuste

---

## Exercice 3 : Lessons store avec expiration

### Objectif
Etendre le lessons store de `SelfImprovingAgent` avec un mecanisme d'expiration : les lecons trop anciennes ou peu observees sont ecartees.

### Consigne
En partant de `02-code/17-verifiers-self-improvement.py` :

1. Modifie la structure `Lesson` (ou cree une classe equivalente) pour ajouter :
   - `last_seen: str` (ISO datetime, mis a jour a chaque occurrence)
   - `ttl_days: int` (time-to-live en jours, defaut = 7)
2. Cree une fonction `filter_lessons(lessons: list[Lesson], now: str, min_confidence: float = 0.4) -> list[Lesson]` :
   - Exclut les lecons dont `(now - last_seen).days > ttl_days`
   - Exclut les lecons dont `confidence < min_confidence`
   - Retourne la liste filtree triee par `confidence` decroissant
3. Teste avec 5 lecons :
   - 2 lecons fraıches et confiantes → gardees
   - 1 lecon ancienne (8 jours) et confiante → expiree
   - 1 lecon fraıche mais peu confiante (`confidence=0.2`) → filtree
   - 1 lecon ancienne ET peu confiante → expiree + filtree
4. Affiche le nombre de lecons avant et apres filtrage, avec la raison d'exclusion pour chaque lecon exclue

### Criteres de reussite
- [ ] Le champ `last_seen` est pris en compte pour l'expiration temporelle
- [ ] Le filtrage par `min_confidence` est independant du filtrage par TTL
- [ ] La fonction retourne les lecons triees par `confidence` decroissant
- [ ] Les 5 lecons de test produisent exactement 2 lecons survivantes
- [ ] La raison d'exclusion est imprimee pour chaque lecon exclue
