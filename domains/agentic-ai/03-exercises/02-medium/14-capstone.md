# Exercices Medium — Capstone (J14)

> Ces exercices ETENDENT le capstone `AcmeResearcher` (`02-code/14-capstone.py`).
> Ne reimplemente pas le systeme entier : ajoute/durcis une brique a la fois.
> Les solutions fournies embarquent un mini-AcmeResearcher autonome pour tourner offline.

---

## Exercice 1 : Cache semantique de findings + reuse cross-requete

### Objectif
Ajouter une nouvelle capacite au capstone : un cache de findings qui evite de relancer le researcher (retriever + LLM) pour des requetes semantiquement proches — economie de cout et de latence reelle en prod.

### Consigne
1. Cree un `FindingsCache` qui stocke, par requete, les `findings` produits par le `ResearcherAgent`
2. **Match semantique (mock)** : deux requetes sont "proches" si leur overlap de tokens significatifs (apres retrait des stopwords courants) depasse un seuil (ex: 0.6). Reutilise le cache si match
3. Branche le cache AVANT l'appel au researcher dans le flow `AcmeResearcher.run` :
   - cache hit → on saute le researcher (et son cout), on logge `cache_hit`
   - cache miss → on appelle le researcher, on stocke le resultat
4. Le `state` gagne un champ `cache_status: "hit" | "miss"`
5. Teste avec 3 requetes : "What is the revenue of Acme in 2025?", puis "Acme 2025 revenue?" (proche → hit), puis "Who are Acme customers?" (different → miss)
6. Verifie que le cout de la 2e requete est inferieur a la 1re (researcher saute) et que le cout total est moindre qu'avec 3 miss

### Criteres de reussite
- [ ] `FindingsCache` matche les requetes semantiquement proches (overlap de tokens)
- [ ] Un cache hit saute l'appel au researcher (cout reduit verifiable)
- [ ] Le `state` expose `cache_status`
- [ ] La requete differente produit un miss et appelle le researcher
- [ ] Le cout total avec cache < cout total sans cache (mesure)

---

## Exercice 2 : Budget adaptatif par complexite de requete + early-exit

### Objectif
Durcir le controle de budget du capstone : au lieu d'un plafond fixe, allouer le budget selon la complexite estimee de la requete, et permettre un early-exit si la qualite est deja suffisante avant d'epuiser les etapes.

### Consigne
1. Ajoute un `complexity_estimator(query) -> str` qui classe la requete en `simple` / `composed` / `deep` selon des heuristiques (nombre de "and"/virgules, mots-cles comme "compare", "analyze", longueur)
2. Mappe la complexite a un budget : `simple → 0.1$`, `composed → 0.3$`, `deep → 0.6$`. Le `BudgetTracker` est initialise avec ce budget
3. **Early-exit** : apres l'etape `analyze`, evalue si les findings couvrent deja les keywords attendus (mock : si la confiance >= seuil). Si oui, on peut sauter une eventuelle etape de re-recherche et aller directement au writer
4. Le `state` expose `allocated_budget`, `complexity`, `early_exit: bool`
5. Teste 3 requetes de complexites differentes : verifie que le budget alloue varie, qu'une requete simple early-exit, et qu'aucune ne depasse son budget alloue

### Criteres de reussite
- [ ] La complexite est estimee et mappee a un budget different
- [ ] Le `BudgetTracker` reçoit le budget adapte a la complexite
- [ ] L'early-exit se declenche quand la qualite est deja suffisante
- [ ] Le `state` expose complexity / allocated_budget / early_exit
- [ ] Aucune requete ne depasse son budget alloue (verdict != budget_exceeded)

---

## Exercice 3 : Citations verifiables + detection de claim non-source

### Objectif
Ajouter une eval supplementaire au capstone qui verifie que chaque affirmation du rapport final est **sourcee** par un `doc_id` present dans les findings — defense anti-hallucination orientee production.

### Consigne
1. Modifie le `writer` (mock) pour qu'il produise un rapport ou chaque affirmation est suivie d'une citation `[doc_id]` (ex: "Acme made 820k euros [k4].")
2. Cree un `CitationEvaluator` qui :
   - Extrait toutes les citations `[xxx]` du rapport
   - Verifie que chaque `doc_id` cite existe bien dans les `findings` du state (pas de citation fantome)
   - Detecte les affirmations factuelles SANS citation (phrases avec un chiffre ou un nom propre mais pas de `[...]`) → `uncited_claims`
   - Retourne `{cited_claims, uncited_claims, phantom_citations, grounding_ratio, passed}`
3. `passed` si `grounding_ratio >= 0.8` ET `phantom_citations == 0`
4. Teste avec : un rapport bien source (passe), un rapport avec une citation fantome (`[k99]` inexistant → fail), un rapport avec une affirmation chiffree non sourcee (fail)

### Criteres de reussite
- [ ] Le writer produit des rapports avec citations `[doc_id]`
- [ ] `CitationEvaluator` detecte les citations fantomes (doc_id absent des findings)
- [ ] Les affirmations chiffrees non sourcees sont reperees
- [ ] Le `grounding_ratio` est calcule correctement
- [ ] Les 3 cas (source / fantome / non-source) donnent les bons verdicts
