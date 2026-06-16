# Exercices — Inference engineering (J24)

---

## Exercice 1 : Sampling contraint sur un enum

### Objectif

Comprendre pourquoi le masquage de tokens garantit une sortie valide la ou un simple `temperature` ne le fait pas.

### Consigne

En partant de `02-code/24-inference-engineering.py` :

1. Cree une classe `WeightedEnumDecoder` qui, comme `TokenMaskedEnumDecoder`, n'autorise qu'un ensemble de valeurs (`allowed`), mais qui **echantillonne** parmi les tokens autorises proportionnellement a leurs logits (softmax sur les seuls tokens autorises), au lieu de prendre l'argmax.
2. Ajoute un parametre `seed` pour rendre l'echantillonnage deterministe.
3. Verifie sur 1000 tirages que **100 % des sorties** sont dans `allowed`, meme quand un token hors-grammaire a le plus gros logit.
4. Affiche la distribution empirique des tokens tires.

### Criteres de reussite

- [ ] `WeightedEnumDecoder` n'emet jamais un token hors `allowed` (verifie sur 1000 tirages)
- [ ] L'echantillonnage suit approximativement le softmax des logits autorises
- [ ] Le `seed` rend les tirages reproductibles
- [ ] La distribution empirique est affichee

---

## Exercice 2 : Cascade routing avec verifier

### Objectif

Aller au-dela du routing binaire : tenter d'abord le modele faible, et n'**escalader** vers le modele fort que si un verifier juge la reponse insuffisante.

### Consigne

1. Cree une classe `CascadeRouter` reutilisant `ModelSpec`.
2. Sa methode `call(query, verifier, est_in, est_out)` doit :
   - Toujours appeler d'abord le `weak` model (compter son cout)
   - Passer la "reponse" mock au `verifier(query) -> bool`
   - Si `verifier` retourne `False`, appeler le `strong` model (compter son cout en plus)
3. Implemente un `verifier` mock qui renvoie `False` pour les requetes contenant un mot-cle de complexite (reutilise `COMPLEX_KEYWORDS`).
4. Sur un batch de 50 requetes (40 simples, 10 complexes), compare le cout total :
   - `all-strong`
   - routing simple (J24, `ModelRouter`)
   - cascade routing
5. Affiche un tableau des 3 strategies (cout + nombre d'appels weak/strong).

### Criteres de reussite

- [ ] `CascadeRouter` appelle toujours le weak, puis le strong seulement si le verifier echoue
- [ ] Le cout cascade inclut bien le double appel sur les requetes escaladees
- [ ] Le tableau compare les 3 strategies
- [ ] La cascade coute plus cher que le routing simple sur les cas escalades mais reste sous l'all-strong dans ce batch

---

## Exercice 3 : Cache de prefixe avec TTL et LRU

### Objectif

Rendre le `PromptCache` realiste : les entrees expirent (TTL) et le cache a une capacite bornee (eviction LRU).

### Consigne

1. Cree une classe `TTLLRUCache` inspiree de `PromptCache` avec :
   - `__init__(self, capacity: int, ttl: float, read_discount: float = 0.10)`
   - une horloge **injectable** `now: Callable[[], float]` (pour tester sans `time.sleep`)
2. `call(prefix, prefix_tokens, suffix_tokens)` doit :
   - Considerer une entree expiree (age > ttl) comme un **miss** et la rafraichir
   - Sur un hit, mettre l'entree en tete (most-recently-used)
   - Sur un miss avec cache plein, evincer l'entree la moins recemment utilisee
3. Teste 3 scenarios et verifie les stats :
   - Hit chaud (deux appels rapprochés sur le meme prefixe) -> 1 hit
   - Expiration (deuxieme appel apres `ttl`) -> 2 miss
   - Eviction (capacity=2, 3 prefixes distincts, puis re-appel du premier) -> miss

### Criteres de reussite

- [ ] Une entree plus vieille que `ttl` est traitee comme un miss
- [ ] L'eviction LRU retire bien l'entree la moins recemment utilisee
- [ ] L'horloge injectable permet de tester sans `sleep`
- [ ] Les 3 scenarios affichent des stats correctes (hits/miss attendus)
