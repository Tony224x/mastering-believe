# Exercices Medium — Context engineering & compaction (J15)

---

## Exercice 1 : Compaction hierarchique par blocs tematiques

### Objectif
Depasser le sliding window du module (qui resume "tout sauf les N derniers") : construire une compaction **hierarchique** qui regroupe les messages par **phase** (exploration, implementation, tests...) et produit **un resume par bloc**, tout en garantissant que les messages epingles ("pinned") survivent. C'est la strategie 2 du cours (hierarchical summary) couplee a la strategie 3 (selective keep).

### Consigne
Tu peux reembarquer un mini `estimate_tokens` dans ta solution (1 token ~ 4 chars) ou l'importer du module 02-code.

1. Modelise un message comme un dict `{"role": ..., "content": ..., "phase": ..., "pinned": bool}`. La cle `phase` est une etiquette libre (ex : `"exploration"`, `"implementation"`, `"tests"`).
2. Ecris une fonction `group_by_phase(messages: list[dict]) -> dict[str, list[dict]]` qui conserve l'**ordre d'apparition** des phases (pas un tri alphabetique : utilise l'ordre de premiere apparition).
3. Ecris `summarize_block(phase: str, msgs: list[dict]) -> dict` qui retourne un message resume `{"role": "system", "content": "[SUMMARY phase=...] ...", "phase": phase, "pinned": False}`. Le resume doit mentionner la phase, le nombre de messages condenses, et l'objet du dernier message du bloc.
4. Ecris `hierarchical_compact(messages: list[dict]) -> list[dict]` qui :
   - extrait d'abord **tous** les messages `pinned` (ils ne sont jamais resumes), dans leur ordre d'origine,
   - resume chaque bloc de phase **non-pinned** en un seul message via `summarize_block`,
   - retourne `[messages pinned...] + [un resume par phase...]`, l'ordre des resumes suivant l'ordre des phases.
5. Teste avec un historique d'au moins 12 messages couvrant 3 phases, dont 2 messages `pinned` (ex : le goal initial et une contrainte critique).

### Criteres de reussite
- [ ] `group_by_phase` respecte l'ordre de premiere apparition des phases (verifie par assertion)
- [ ] Apres compaction, **tous** les messages `pinned` sont presents et inchanges
- [ ] Il y a exactement **un** message resume par phase non-pinned
- [ ] Le nombre total de tokens apres compaction est **strictement inferieur** au nombre avant (assertion)
- [ ] Le nombre de messages apres compaction = (nb pinned) + (nb de phases distinctes), verifie par assertion
- [ ] Execution offline, deterministe, sans dependance reseau

---

## Exercice 2 : Gestionnaire d'offloading automatique vers un VirtualFS

### Objectif
Implementer l'offloading **automatique** (section 4 du cours) : tout resultat de tool dont la taille depasse un seuil est deporte sur un VirtualFS et remplace dans le contexte par un **placeholder** court, tout en restant **recuperable** a la demande. Tu dois prouver que les tokens en-contexte restent **bornes** meme quand on injecte des resultats enormes.

### Consigne
1. Implemente (ou reutilise) un `VirtualFS` minimal avec `write(path, content)`, `read(path) -> str | None`, `list_files() -> list[str]`.
2. Ecris un `OffloadingManager` :
   - `__init__(self, vfs, token_threshold: int = 200)` : seuil au-dessus duquel un resultat est offloade.
   - `add_tool_result(self, tool_name: str, content: str) -> str` : si `estimate_tokens(content) > token_threshold`, ecrit le contenu dans le VFS sous un chemin **unique** (ex : `f"tool_results/{tool_name}_{compteur}.txt"`), pousse dans le contexte un **placeholder** court du type `[OFFLOADED -> <path> (<n> tokens on disk)]`, et retourne ce placeholder. Sinon, pousse le contenu tel quel dans le contexte et le retourne.
   - `context_tokens(self) -> int` : somme des tokens des entrees actuellement **en contexte** (placeholders + petits resultats).
   - `retrieve(self, path: str) -> str | None` : recupere depuis le VFS le contenu offloade.
3. Le placeholder doit etre **petit** : impose-toi un plafond (ex : <= 40 tokens par placeholder) verifie par assertion.
4. Simule l'arrivee de 6 resultats de tools dont au moins 3 sont enormes (plusieurs milliers de chars) et 2 petits.
5. **Prouve par assertions** :
   - apres avoir injecte des resultats totalisant des dizaines de milliers de tokens "sur disque", le `context_tokens()` reste sous un plafond raisonnable (ex : < 500),
   - chaque gros resultat est bien **recuperable** integralement via `retrieve(path)` (round-trip exact),
   - les petits resultats (sous le seuil) restent **inline** dans le contexte (pas offloades).

### Criteres de reussite
- [ ] Tout resultat > seuil est offloade vers le VFS et remplace par un placeholder
- [ ] Tout resultat <= seuil reste inline dans le contexte
- [ ] `context_tokens()` reste borne (< plafond) malgre des dizaines de milliers de tokens stockes sur disque (assertion)
- [ ] Chaque contenu offloade est recuperable a l'identique via `retrieve` (round-trip verifie)
- [ ] Le placeholder respecte un plafond de tokens (assertion)
- [ ] Execution offline, deterministe, sans dependance

---

## Exercice 3 : Harnais de comparaison compaction proactive vs reactive

### Objectif
Mesurer empiriquement la difference entre compaction **reactive** (on compacte quand `tokens > seuil`) et **proactive** (on compacte preventivement si `tokens_actuels + tokens_estimes_du_prochain_tour > limite`), comme decrit en section 3.3 du cours. Le but : montrer chiffres a l'appui que la proactive evite des depassements (overflows) que la reactive subit.

### Consigne
1. Ecris un simulateur de run deterministe : une liste de tours, chaque tour ajoute un certain nombre de tokens au contexte (utilise une sequence fixe, ex : des tailles croissantes pour simuler des resultats de tools de plus en plus gros).
2. Implemente `run_reactive(turns, limit, keep_tail) -> dict` :
   - on ajoute les tokens du tour,
   - **apres** ajout, si `tokens > limit`, on compacte (on remplace le contexte par un resume fixe de `summary_tokens` tokens + on garde les `keep_tail` derniers tours),
   - on compte un **overflow** chaque fois que, juste apres l'ajout et **avant** la compaction, `tokens > limit` (le depassement a eu lieu, le LLM aurait deja vu un contexte trop grand).
3. Implemente `run_proactive(turns, limit, keep_tail, summary_tokens)` :
   - **avant** d'ajouter le tour, on estime `tokens_actuels + cout_estime_du_tour` ; si ca depasse `limit`, on compacte **d'abord** (resume + keep_tail), puis on ajoute,
   - on compte un overflow seulement si, malgre la compaction preventive, le contexte depasse encore la limite apres ajout.
4. Fais tourner les deux sur **le meme** scenario et retourne pour chacun `{"compactions": ..., "overflows": ..., "peak_tokens": ..., "final_tokens": ...}`.
5. **Prouve par assertions** :
   - la proactive a **strictement moins** (ou au pire egal) d'overflows que la reactive sur un scenario ou la reactive en subit au moins 1,
   - les deux finissent sous la limite a la fin (le contexte est viable),
   - la proactive ne fait pas un nombre absurde de compactions (borne raisonnable).

### Criteres de reussite
- [ ] `run_reactive` compte au moins 1 overflow sur le scenario de test
- [ ] `run_proactive` a un nombre d'overflows <= celui de la reactive, et strictement inferieur sur le scenario choisi (assertion)
- [ ] Les deux strategies finissent avec un contexte sous la limite
- [ ] Le rapport expose `compactions`, `overflows`, `peak_tokens`, `final_tokens` pour chaque strategie
- [ ] Le scenario est identique pour les deux (comparaison equitable)
- [ ] Execution offline, deterministe, sans dependance reseau
