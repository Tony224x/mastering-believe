# Exercices Hard — Memoire long-horizon (J16)

---

## Exercice 1 : Pipeline complet de memoire long-horizon sur une simulation multi-session

### Objectif
Cabler bout-en-bout le flux de memoire du module (section 6) : **ecriture episodique → consolidation/oubli → reflection semantique → retrieval au moment du recall**, le tout sur une simulation de plusieurs sessions etalees dans le temps. Tu dois prouver les deux proprietes qui font la difference entre un agent amnesique et un agent long-horizon : (a) le **recall s'ameliore** quand la reflection a tourne (l'agent retrouve un fait consolide qu'il ne pouvait pas inferer d'un seul episode), et (b) le **store reste borne** (l'oubli + la consolidation empechent la croissance illimitee).

### Consigne
Construis un `LongHorizonMemory` qui orchestre les trois etages (episodique, semantique, consolidation) :

1. **Ecriture** : `observe(content, importance, t)` ajoute un episode horodate au stream episodique.
2. **Consolidation periodique** : `consolidate(t_now)` doit, en une passe :
   - **fusionner les near-duplicates** episodiques (similarite cosinus sur `fake_embed` au-dessus d'un seuil),
   - **oublier** les episodes stale et peu importants (`effective_score = max(recency, importance*0.5)` sous seuil, importance shield actif),
   - **deriver des faits semantiques** par reflection (groupes d'episodes partageant un mot-cle saillant, `>= min_support`), avec `source_ids` traçables.
3. **Retrieval au recall** : `recall(query, k, t_now)` score sur **l'union** {episodes restants} ∪ {faits semantiques} avec recence + importance + similarite, et renvoie le top-K trie.
4. **Simulation multi-session** : joue >= 3 sessions a des timestamps croissants (ex : J0, J3, J10). Chaque session ecrit plusieurs episodes ; entre les sessions, appelle `consolidate`. Inclus volontairement un theme **repete sur plusieurs sessions** (ex : "user prefers CSV") et du bruit ephemere.
5. **Prouve par assertions** :
   - **Recall ameliore** : pour la query liee au theme repete, fais un `recall` AVANT toute consolidation (le top renvoie au mieux un episode brut) et un `recall` APRES consolidation, et montre qu'apres consolidation un **fait semantique consolide** (decontextualise) apparait dans le top-K — un signal que l'episode unique ne donnait pas.
   - **Store borne** : apres la simulation complete + consolidation finale, le nombre **total d'unites memoire** (episodes restants + faits semantiques) est **strictement inferieur** au nombre total d'episodes bruts ecrits — la consolidation/oubli a bien compresse.
   - **Traçabilite** : au moins un fait derive a des `source_ids` non vides, tous correspondant a des episodes reellement ecrits.

### Criteres de reussite
- [ ] Le pipeline enchaine ecriture → consolidation (dedup + oubli) → reflection → retrieval
- [ ] La simulation couvre >= 3 sessions a des timestamps croissants avec un theme repete
- [ ] AVANT consolidation, le recall du theme ne remonte qu'un episode brut ; APRES, un fait semantique consolide apparait dans le top-K (assertion)
- [ ] Le total d'unites memoire final est strictement inferieur au nombre d'episodes bruts ecrits (assertion : store borne)
- [ ] Au moins un fait derive a des `source_ids` valides et traçables (assertion)
- [ ] L'importance shield empeche l'oubli d'un episode critique meme tres ancien (assertion)
- [ ] Execution offline, deterministe, sans cle API ni dependance

---

## Exercice 2 : Politique d'eviction sous budget de capacite (hybride recence/frequence/salience)

### Objectif
Implementer une **garbage collection / eviction** de la memoire sous **budget de capacite dur** (section 5.3 + paging MemGPT). Quand le store atteint sa capacite max, il faut evincer la "moins utile" des entrees en combinant **recence** (LRU), **frequence d'acces** (LFU) et **salience** (importance). Tu dois prouver que (a) la capacite est **toujours respectee**, et (b) les souvenirs **importants ne sont jamais evinces** (protected set), meme s'ils sont vieux et rarement accedes.

### Consigne
Construis un `BoundedMemory(capacity)` avec une politique d'eviction hybride :

1. **Entree** : chaque souvenir porte `content`, `importance`, `created_at`, `last_access`, `access_count`.
2. **Acces** : `get(content_or_id)` (ou `touch(...)`) met a jour `last_access = t_now` et incremente `access_count` — c'est le signal LRU+LFU.
3. **Insertion bornee** : `add(content, importance, t_now)` ajoute l'entree ; **si** la taille depasse `capacity`, declenche `_evict(t_now)` jusqu'a revenir sous la capacite.
4. **Score d'eviction hybride** : `evict_score = w_rec * recency + w_freq * norm_freq + w_sal * importance` (plus le score est **bas**, plus l'entree est candidate a l'eviction). `norm_freq` normalise `access_count` (ex : `access_count / (max_access + 1)`).
5. **Protected set** : toute entree avec `importance >= protect_threshold` (ex : 0.8) est **immunisee** contre l'eviction — elle ne peut jamais etre choisie comme victime. (Si **toutes** les entrees sont protegees et qu'on depasse la capacite, leve une `MemoryError` explicite plutot que d'evincer une entree critique.)
6. **Scenario de test** :
   - capacite = 5, `protect_threshold = 0.8`.
   - Insere une entree **critique** (importance 0.95) tres tot, ne la touche **plus jamais** (vieille + jamais accedee).
   - Insere 4 entrees normales, puis spamme des insertions jusqu'a forcer plusieurs evictions.
   - Accede frequemment a une entree "chouchou" (importance moyenne) pour la maintenir en vie via la frequence.
7. **Prouve par assertions** :
   - `len(store) <= capacity` apres **chaque** insertion (invariant de capacite),
   - l'entree critique (importance 0.95) est **toujours presente** a la fin, malgre son age et son zero acces (protected set),
   - l'entree "chouchou" frequemment accedee survit a une entree de meme importance jamais accedee (la frequence compte),
   - une entree banale, ancienne et jamais accedee a bien ete **evincee**,
   - le cas "store plein de seules entrees protegees + nouvelle insertion" leve `MemoryError`.

### Criteres de reussite
- [ ] Chaque entree porte recence, frequence d'acces et salience (importance)
- [ ] `len(store) <= capacity` est un invariant verifie apres chaque insertion (assertion)
- [ ] Le score d'eviction combine recence (LRU), frequence (LFU) et salience
- [ ] Le protected set garantit qu'une entree critique (importance >= seuil) n'est jamais evincee (assertion)
- [ ] Une entree frequemment accedee survit a une entree equivalente jamais accedee (assertion)
- [ ] Le cas "tout protege + overflow" leve `MemoryError` au lieu d'evincer une entree critique
- [ ] Execution offline, deterministe, sans dependance ni cle API
