# Exercices Medium — Inference at Scale

---

## Exercice 1 : Budget memoire GPU — KV cache vs poids

### Objectif
Chiffrer ce qui rentre vraiment sur un GPU : poids quantises + KV cache, et en deduire le batch max.

### Consigne
Tu sers Llama-3-70B sur une H100 80 Go.

**Donnees :**
- Poids fp16 : 140 Go ; int8 : 70 Go ; int4 : 35 Go
- KV cache : ~320 Ko par token
- Tu vises des sequences de 4000 tokens (prompt + generation)
- Overhead runtime (activations, framework) : reserve 10 Go

**Questions :**
1. En int8 (70 Go) sur 80 Go, combien de Go restent pour le KV cache apres avoir reserve 10 Go d'overhead ? Conclusion ?
2. En int4 (35 Go), combien de Go pour le KV cache ? Combien de sequences de 4000 tokens tiennent en parallele (KV par sequence = 4000 * 320 Ko) ?
3. Pourquoi le batch max (nombre de sequences simultanees) est-il dicte par le KV cache et non par les poids ?
4. PagedAttention alloue le KV par pages de 16 tokens non-contigues. Pourquoi ca augmente le nombre de sequences simultanees vs une allocation contigue qui reserve 4000 tokens d'avance ?
5. Tu actives le KV cache en int8 (moitie de la taille). Comment ca change le nombre de sequences simultanees en int4 ?

### Criteres de reussite
- [ ] int8 : 80 - 70 - 10 = 0 Go pour le KV -> int8 ne laisse quasi RIEN, batch ~impossible sur 1 H100
- [ ] int4 : 80 - 35 - 10 = 35 Go pour le KV ; KV/seq = 4000 * 320 Ko ≈ 1.28 Go -> ~27 sequences
- [ ] Le batch max est limite par le KV cache (croit avec batch * longueur), pas par les poids (fixes)
- [ ] PagedAttention evite de sur-reserver pour les seq courtes -> plus de seq tiennent (moins de gaspillage)
- [ ] KV int8 -> KV/seq ≈ 0.64 Go -> ~54 sequences (~2x plus)

---

## Exercice 2 : Continuous batching vs static — throughput et utilisation GPU

### Objectif
Quantifier pourquoi le continuous batching multiplie le throughput sur des sequences de longueurs inegales.

### Consigne
Un GPU traite un batch. Les sequences ont des longueurs de generation tres inegales.

**Donnees (static batching, batch=4) :**
- Seq A : 1000 tokens ; Seq B, C, D : 50 tokens chacune
- 1 step = 1 token genere pour toutes les seq actives du batch
- Temps par step = 20 ms (constant tant qu'au moins 1 slot actif)

**Questions :**
1. En **static batching**, on attend que la plus longue (1000 tokens) finisse. Combien de steps au total ? Combien de "slot-steps" utiles vs gaspilles (B,C,D finissent a 50) ?
2. Calcule l'utilisation moyenne des 4 slots sur la duree totale (slot-steps utiles / slot-steps disponibles).
3. En **continuous batching**, des qu'une seq finit (B,C,D a 50), on insere une nouvelle seq dans le slot libre. Sur une charge soutenue, a combien tend l'utilisation des slots ?
4. Si 1000 nouvelles requetes de 50 tokens attendent dans la queue, le continuous batching peut-il les absorber pendant que A genere ses 1000 tokens ? Explique le mecanisme.
5. Quel pre-requis technique rend le continuous batching possible (gestion memoire) ?

### Criteres de reussite
- [ ] Static : 1000 steps. Slot-steps utiles = 1000 + 50 + 50 + 50 = 1150 ; disponibles = 4 * 1000 = 4000
- [ ] Utilisation static ≈ 1150 / 4000 ≈ 29% (3 slots quasi vides 95% du temps)
- [ ] Continuous batching : utilisation tend vers ~100% (slots remplis en continu)
- [ ] Oui : les slots liberes par B,C,D accueillent des nouvelles requetes pendant que A continue -> throughput x3+
- [ ] Pre-requis : PagedAttention (KV cache pagine non-contigu) pour inserer/retirer des seq sans realloc

---

## Exercice 3 : Routing semantique + prefix caching — economie de cout

### Objectif
Chiffrer l'impact combine du routing par tier et du prefix caching sur la facture.

### Consigne
Un produit LLM recoit 1M requetes/jour. Sans optimisation, tout va au gros modele.

**Donnees :**
- Gros modele : $5 / 1M tokens (moyenne in+out ponderee)
- Petit modele (tier nano) : $0.3 / 1M tokens
- Chaque requete = 3000 tokens in (dont 2500 de system prompt partage) + 500 tokens out
- Distribution reelle du trafic : 60% trivial (routable vers nano), 30% mid (gros modele OK), 10% complexe (gros modele obligatoire)
- Prefix caching : les 2500 tokens de system prompt, s'ils sont caches, sont factures 10% du prix normal

**Questions :**
1. Cout/jour baseline (tout au gros modele, pas de cache) : 1M * 3500 tokens * $5/1M.
2. Avec routing seul : 60% nano + 40% gros modele. Calcule le cout/jour (approxime in+out au meme prix par tier).
3. Avec prefix caching seul (pas de routing) : les 2500 tokens system passent a 10% du prix. Recalcule le cout/jour du gros modele.
4. Avec routing + prefix caching combines sur la part gros-modele : estime le cout/jour total. Quelle reduction vs baseline ?
5. Quel est le risque d'un routeur mal calibre (envoie du complexe vers le nano) ? Quelle metrique surveiller ?

### Criteres de reussite
- [ ] Baseline : 1M * 3500 * 5/1e6 = $17500/jour
- [ ] Routing : 60% * (1M*3500*0.3/1e6) + 40% * (1M*3500*5/1e6) = $630 + $7000 = $7630/jour
- [ ] Prefix caching seul : par requete 2500 system a 10% + 1000 non-caches plein tarif -> (2500*0.1 + 1000) = 1250 tokens-equivalents -> $6250/jour
- [ ] Routing + caching (cache applique a la part gros-modele) : ~$3130/jour, soit ~82% de reduction vs baseline
- [ ] Routeur mal calibre -> qualite degradee sur le complexe ; surveiller le taux d'escalade / fallback et la satisfaction par tier
