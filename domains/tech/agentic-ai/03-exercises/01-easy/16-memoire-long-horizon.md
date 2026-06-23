# Exercices — Memoire long-horizon (J16)

> Chaque exercice s'appuie sur `02-code/16-memoire-long-horizon.py`. Lis ce fichier avant de commencer.

---

## Exercice 1 : Scoring de pertinence personnalise

### Objectif
Comprendre l'impact de chaque composante du scoring (recence, importance, similarite) en faisant varier les poids.

### Consigne
En utilisant `RelevanceScorer` et `MemoryEntry` de `02-code/16-memoire-long-horizon.py` :

1. Cree 4 `MemoryEntry` representant des scenarios contrastes :
   - Un episode tres recent (< 1h) mais peu important (importance=0.2)
   - Un episode ancien (> 7 jours) mais tres important (importance=0.9)
   - Un episode moyen en age (24h) avec contenu tres similaire a ta query
   - Un episode tres recent (< 30 min) et contenu totalement hors-sujet
2. Definis une query : `"user preference format CSV report"`
3. Score chaque entree avec trois configurations de poids :
   - Config A : `(0.5, 0.1, 0.4)` — recence dominante
   - Config B : `(0.1, 0.7, 0.2)` — importance dominante
   - Config C : `(0.2, 0.2, 0.6)` — similarite dominante (defaut Generative Agents)
4. Affiche un tableau comparatif : entree x configuration, avec le score total et le winner par configuration

### Criteres de reussite
- [ ] Les 4 entrees ont des ages et importances bien contrastes
- [ ] Les 3 configurations de `RelevanceScorer` sont creees separement
- [ ] Le tableau compare clairement les 4 x 3 = 12 scores
- [ ] Le winner (entree avec le score le plus eleve) est different selon la config
- [ ] Explication en commentaire de pourquoi chaque config favorise une entree differente

---

## Exercice 2 : Consolidation manuelle episodique → semantique

### Objectif
Implementer et observer la consolidation : transformer des episodes repetitifs en faits semantiques.

### Consigne
En partant de `EpisodicMemory` et `SemanticMemory` de `02-code/16-memoire-long-horizon.py` :

1. Cree une `EpisodicMemory` et insere 8 episodes :
   - 3 episodes sur le theme "CSV" (ex : "user requested CSV", "CSV export succeeded", "user thanked for CSV report")
   - 3 episodes sur le theme "paiement" (ex : "payment 503 error", "payment retry succeeded", "user reported payment delay")
   - 2 episodes sans theme clair (ex : "user said hello", "agent started session")
2. Ecris une fonction `manual_consolidate(episodic: EpisodicMemory, semantic: SemanticMemory, theme: str, min_count: int = 2) -> MemoryEntry | None` qui :
   - Filtre les episodes contenant `theme` dans leur contenu
   - Si `len(episodes_filtres) >= min_count`, cree un fait semantique dans `semantic` avec :
     - `key = f"consolidated_{theme}"`
     - `content = f"[{len(episodes_filtres)} episodes] Theme '{theme}': " + dernier_episode.content`
     - `importance = min(1.0, 0.5 + 0.1 * len(episodes_filtres))`
   - Sinon retourne `None`
3. Appelle la fonction pour les themes "CSV", "payment" et "hello"
4. Affiche les faits crees et verifie que "hello" ne produit pas de fait (trop peu d'episodes)

### Criteres de reussite
- [ ] `manual_consolidate` est implementee et appele correctement
- [ ] Le fait "CSV" est cree depuis 3 episodes
- [ ] Le fait "payment" est cree depuis 3 episodes
- [ ] Le theme "hello" ne produit pas de fait (< 2 episodes)
- [ ] `importance` varie selon le nombre d'episodes consolides
- [ ] Le contenu du fait mentionne le nombre d'episodes source

---

## Exercice 3 : MemGPT paging — visualiser les evictions

### Objectif
Observer concretement le mecanisme de page_in / page_out en simulant un contexte sature.

### Consigne
En utilisant `HierarchicalMemory` de `02-code/16-memoire-long-horizon.py` (la limite `MAX_MAIN_CONTEXT = 5`) :

1. Observe 8 evenements un par un avec des importances variees (importances entre 0.2 et 0.9)
2. Apres chaque `observe()`, appelle `show_main_context()` et affiche quel slot est occupe
3. Apres les 8 observations, effectue un `page_in("payment error critical", k=2)` pour recharger des episodes pertinents
4. Avant et apres le `page_in`, affiche le contenu du main context pour montrer le changement
5. Ajoute une assertion verifiant que `len(hmem._main_context) <= MAX_MAIN_CONTEXT` a tout moment

### Criteres de reussite
- [ ] 8 observations sont inserees avec des importances contrastees
- [ ] Le main context ne depasse jamais `MAX_MAIN_CONTEXT = 5` (assertion presente)
- [ ] L'affichage montre clairement les slots avant et apres `page_in`
- [ ] Au moins un episode pertinent a "payment error" est present apres `page_in`
- [ ] Commentaire expliquant quelle entree a ete evictee lors du dernier `page_out` et pourquoi
