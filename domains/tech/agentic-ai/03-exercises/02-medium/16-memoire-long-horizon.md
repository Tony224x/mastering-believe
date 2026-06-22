# Exercices Medium — Memoire long-horizon (J16)

---

## Exercice 1 : Store de memoire avec scoring salience + recence + retrieval rank

### Objectif
Aller au-dela du scoring "fige" du module : construire un store de memoire qui combine **recence**, **importance/salience** et **similarite** dans un score unique, puis prouver que pour une query donnee, les souvenirs les plus pertinents **remontent en tete** du classement — meme quand un souvenir tres recent mais hors-sujet essaie de polluer le top-K (le piege classique du retrieval naif "dernier ecrit = plus pertinent").

### Consigne
Tu peux importer `RelevanceScorer`, `MemoryEntry`, `fake_embed`, `cosine_similarity` de `02-code/16-memoire-long-horizon.py`, ou tout reembarquer dans ta solution (offline, sans cle API) :

1. Implemente une classe `SalienceStore` qui stocke des `MemoryEntry` et expose :
   - `add(content, importance, created_at) -> MemoryEntry`
   - `retrieve(query, k, t_now) -> list[(score, MemoryEntry)]` qui renvoie le top-K **trie par score decroissant**, ou `score = w_rec * recency + w_imp * importance + w_sim * similarity`.
2. Le scoring doit **normaliser la similarite** entre 0 et 1 (cosinus sur les `fake_embed`) et utiliser une recence en **decroissance exponentielle** (`exp(-lambda * heures)`).
3. Cree un corpus d'au moins 6 souvenirs contrastes : 2 tres pertinents pour la query (similarite forte), 1 ancien mais tres important (salience elevee), 1 tres recent mais totalement hors-sujet (faible importance), 2 de bruit moyen.
4. Pour la query `"user preference export format CSV"`, prouve **par assertion** que :
   - le top-1 est l'un des souvenirs reellement pertinents (pas le recent hors-sujet),
   - le souvenir recent hors-sujet **n'est pas** dans le top-2,
   - le souvenir ancien-mais-important survit dans le top-K grace a sa salience.
5. Affiche le tableau `rang | score | recency | importance | similarity | content`.

### Criteres de reussite
- [ ] `SalienceStore.retrieve` renvoie un top-K trie par score decroissant
- [ ] Le score combine recence (exp decay), importance et similarite normalisee
- [ ] Le top-1 est un souvenir pertinent, prouve par assertion
- [ ] Le souvenir recent-mais-hors-sujet est exclu du top-2 (assertion)
- [ ] Le souvenir ancien-mais-important est present dans le top-K grace a sa salience (assertion)
- [ ] Tout tourne offline, deterministe, sans dependance ni cle API

---

## Exercice 2 : Passe de consolidation — dedup de near-duplicates + decay des entrees stale

### Objectif
Implementer la **consolidation** du module (section 5) sous l'angle hygiene du store : une passe qui (a) **fusionne les near-duplicates** (souvenirs quasi identiques en contenu) en un seul, et (b) **oublie les entrees stale** (recence sous seuil) sauf si leur importance les protege (importance shield, flash-card Q5). Tu dois prouver que le nombre d'entrees **diminue** tout en gardant les souvenirs importants.

### Consigne
1. Reutilise `MemoryEntry` / `fake_embed` / `cosine_similarity`. Construis un store avec au moins 10 entrees dont :
   - 3 near-duplicates du meme fait (ex : "user prefers CSV exports", "the user likes CSV export format", "user wants CSV not PDF") — similarite cosinus elevee entre eux,
   - 2 entrees stale et peu importantes (anciennes, importance < 0.3),
   - 1 entree stale **mais critique** (ancienne, importance >= 0.8),
   - 4 entrees normales variees.
2. Ecris `dedup(entries, sim_threshold=0.6) -> list[MemoryEntry]` qui regroupe les entrees dont la similarite cosinus depasse `sim_threshold` et n'en garde **qu'un representant par cluster** : celui de **plus grande importance** (a importance egale, le plus recent). Le representant herite de l'importance max du cluster et garde une trace du nombre de doublons fusionnes (ex : champ/tag `merged_count`).
3. Ecris `forget(entries, t_now, recency_threshold=0.05, lambda_=0.02) -> list[MemoryEntry]` qui supprime toute entree dont `effective_score = max(recency, importance * 0.5)` est sous `recency_threshold` (le shield d'importance protege les entrees critiques).
4. Ecris `consolidate(entries, t_now) -> list[MemoryEntry]` qui enchaine `dedup` puis `forget`.
5. Prouve **par assertion** :
   - apres `dedup`, les 3 near-duplicates sont reduits a 1 entree (count global baisse),
   - apres `forget`, les 2 entrees stale peu importantes disparaissent,
   - l'entree stale **critique** survit (importance shield),
   - le store final a strictement moins d'entrees qu'au depart.

### Criteres de reussite
- [ ] `dedup` fusionne les near-duplicates au-dessus du seuil de similarite et garde le representant le plus important
- [ ] Le representant trace le nombre de doublons fusionnes
- [ ] `forget` supprime les entrees stale peu importantes
- [ ] L'entree stale critique survit grace au shield d'importance (assertion)
- [ ] Le nombre d'entrees final est strictement inferieur a l'initial (assertion)
- [ ] Execution offline, deterministe, sans dependance

---

## Exercice 3 : Reflection — deriver des faits semantiques traçables depuis l'episodique

### Objectif
Implementer la **reflection** de style Generative Agents (section 5.2) : un processus qui analyse le stream episodique et **derive des faits semantiques de plus haut niveau**, en gardant la **traçabilite** vers les episodes sources. Tu dois prouver que chaque insight derive **reference bien les episodes** dont il provient (pas un fait sorti de nulle part).

### Consigne
1. Modelise un `@dataclass SemanticFact` avec au minimum : `content: str`, `importance: float`, `source_ids: list[str]`, `confidence: float`.
2. Cree un stream de >= 8 episodes (chacun avec un `id` court et un `content`), regroupables par theme via un mot-cle commun :
   - 3 episodes autour de "CSV" (l'utilisateur reclame/remercie pour du CSV),
   - 2 episodes autour de "payment" (erreurs/retries de paiement),
   - 3 episodes divers sans theme partage.
3. Ecris `reflect(episodes, min_support=2) -> list[SemanticFact]` qui :
   - groupe les episodes par leur **mot-cle saillant** (token non-trivial le plus frequent, en ignorant des stopwords),
   - pour chaque groupe de taille `>= min_support`, cree un `SemanticFact` :
     - `content` = un fait generalise mentionnant le theme et le nombre d'episodes,
     - `source_ids` = la liste des `id` des episodes du groupe,
     - `confidence` croissante avec le support (ex : `min(0.95, 0.5 + 0.1 * len(groupe))`),
     - `importance` = moyenne des importances du groupe, legerement boostee.
4. Prouve **par assertion** :
   - un fait "CSV" est derive avec **au moins 3** `source_ids`, tous presents dans le stream,
   - un fait "payment" est derive avec **2** sources,
   - les themes uniques (apparaissant 1 seule fois) **ne produisent pas** de fait,
   - tous les `source_ids` de tous les faits derives correspondent a des `id` reellement existants dans le stream (traçabilite verifiee).

### Criteres de reussite
- [ ] `SemanticFact` porte `content`, `importance`, `source_ids`, `confidence`
- [ ] `reflect` groupe par mot-cle saillant et exige un support minimal
- [ ] Le fait "CSV" reference >= 3 episodes sources existants (assertion)
- [ ] Le fait "payment" reference exactement 2 sources (assertion)
- [ ] Les themes a episode unique ne generent aucun fait (assertion)
- [ ] La confiance croit avec le support et tous les `source_ids` existent reellement (traçabilite verifiee)
- [ ] Execution offline, deterministe, sans dependance
