# Exercices Hard — Jour 16 : Mixture of Experts

---

## Exercice 7 : Token-choice vs expert-choice routing

### Objectif

Implementer les deux paradigmes de routing (token choisit ses experts vs expert choisit ses tokens) et caracteriser leur compromis fondamental : debordement d'expert vs drop de token.

### Consigne

1. **Setup** : un batch de scores de routing `scores (B, N)` = softmax de logits aleatoires (`B=512` tokens, `N=8` experts). On compare deux strategies a budget de compute egal.

2. **Token-choice (top-k)** : chaque token choisit ses `k=2` experts (les 2 plus grands scores). Implementer le dispatch avec un buffer de capacite `C = CF * B * k / N`. Mesurer :
   - le taux de **tokens droppes** (token dont un slot ne rentre dans aucun buffer) ;
   - la distribution de charge reelle par expert (min, max, ecart-type).

3. **Expert-choice (top-M)** : chaque **expert** choisit ses `M = B * k / N` tokens preferes (ses M plus grands scores dans la colonne). Implementer. Mesurer :
   - le taux de tokens **non selectionnes** (token qu'aucun expert n'a pris) ;
   - le nombre de slots traites par expert (doit etre exactement M -> equilibre parfait par construction).

4. **Comparaison rigoureuse** sur 3 distributions de scores :
   - uniforme (scores ~ aleatoires) ;
   - skewed (quelques experts tres attractifs) ;
   - quasi-collapse (1 expert domine).
   Pour chacune, tabuler : drop token-choice, tokens-non-vus expert-choice, equilibre de charge.

5. **Analyse** :
   - Confirmer : token-choice peut **deborder** un expert populaire (drops si CF trop bas), expert-choice garantit l'equilibre mais **laisse des tokens orphelins**.
   - Quel token est sacrifie dans chaque cas (un token "ambivalent" sans expert clair vs un token en exces sur un expert populaire) ?
   - Relier a la solution DeepSeek-V3 (auxiliary-loss-free balancing via biais par expert) : en quoi est-ce un 3e compromis ?

### Criteres de reussite

- [ ] Token-choice (top-k + buffer) et expert-choice (top-M) sont implementes correctement
- [ ] Expert-choice donne un equilibre de charge parfait (M par expert) par construction
- [ ] Token-choice deborde sous CF bas / routage skewed (drop chiffre)
- [ ] Expert-choice laisse des tokens non selectionnes (chiffre) quand le routage est skewed
- [ ] L'analyse compare *quel* token est sacrifie dans chaque schema et situe l'approche DeepSeek-V3
- [ ] Code numpy, seed, modulaire, commente WHY

---

## Exercice 8 : Specialisation des experts — mesurer ce qu'ils apprennent

### Objectif

Construire un mini-MoE entrainable (numpy, gradient manuel) sur une tache jouet a structure latente connue, puis mesurer si les experts se specialisent reellement et sur quoi (lien avec l'idee fausse n°3 du cours).

### Consigne

1. **Tache jouet a clusters** : generer des donnees `x` issues de `G=4` clusters gaussiens distincts dans `R^d` (`d=16`), avec une cible `y` = fonction lineaire **differente par cluster** (chaque cluster a sa propre matrice cible `W_g`). Le cluster d'origine de chaque point est la **structure latente** que l'on connait mais qu'on ne donne PAS au modele.

2. **Mini-MoE entrainable** : `N=4` experts lineaires (`expert_i(x) = x @ Wi`), routeur softmax `W_g`, top-1 routing (hard pour le forward, mais on entraine le routeur via un soft-routing differentiable pour les gradients). Loss MSE + load balancing loss.

3. **Entrainer** ~500 steps en gradient descent manuel (numpy). Tracer la loss.

4. **Mesurer la specialisation** apres training :
   - Pour chaque cluster latent `g`, quelle est la distribution des experts choisis par ses tokens ? (matrice de confusion cluster -> expert)
   - Le routeur a-t-il appris a envoyer (approximativement) un cluster -> un expert ? Calculer un score de "purete" (chaque expert recoit-il majoritairement un seul cluster ?).

5. **Contre-mesure de l'idee fausse** : montrer que sans load balancing loss, le routeur peut collapse (1-2 experts traitent plusieurs clusters), et que la qualite finale (MSE) en patit vs le cas equilibre.

6. **Analyse** :
   - Cette tache jouet a une structure latente PROPRE (clusters separes) -> la specialisation par "domaine" emerge. Relier a la nuance du cours : sur du langage reel, Mixtral specialise plutot par patterns syntaxiques, DeepSeek-V3 (fine-grained) plus par domaine. Pourquoi la granularite change la nature de la specialisation ?
   - Que se passerait-il si le nombre d'experts `N` etait < nombre de clusters `G` ?

### Criteres de reussite

- [ ] Donnees a `G` clusters avec cible lineaire distincte par cluster
- [ ] Mini-MoE entrainable (forward + backward manuels en numpy) qui converge (loss baisse)
- [ ] Matrice de confusion cluster -> expert calculee ; score de purete chiffre
- [ ] Avec load balancing, le routeur tend vers une affectation ~1 cluster / 1 expert
- [ ] Sans load balancing, collapse demontre + MSE degradee chiffree
- [ ] L'analyse relie granularite/specialisation au cours et discute le cas `N < G`
- [ ] Code numpy, seed, commente WHY (gradients inclus)

---

## Exercice 9 : Comptabilite complete d'un MoE + tradeoff communication all-to-all

### Objectif

Construire un modele de cout end-to-end d'un MoE distribue : params totaux/actifs, VRAM, FLOPs, ET le cout de communication all-to-all qui peut annuler les gains (cf section 7 du cours).

### Consigne

1. **Comptabilite params** (SwiGLU realiste, 3 matrices par expert) : ecrire `moe_accounting(layers, d_model, d_ff, N, k, vocab, n_shared=0)` qui calcule, en incluant un eventuel **shared expert** toujours actif (DeepSeek-style) :
   - params totaux (attention partagee + N experts FFN SwiGLU + shared + router + embeddings) ;
   - params actifs par token (`attention + (k + n_shared) experts + router`) ;
   - ratio de sparsite.
   Valider sur Mixtral-like (N=8, k=2 -> ~47B total) et DeepSeek-V3-like (N=256, k=8, 1 shared, d_ff fin -> ~671B total / ~37B actifs, ordre de grandeur).

2. **VRAM** : a poids FP16 (2 octets/param), calculer la VRAM des poids pour les 2 modeles. Confirmer l'idee fausse n°2 : le MoE prend la VRAM de **tous** les experts (pas seulement k).

3. **FLOPs/token** : `2 * params_actifs` (regle standard). Comparer au dense de meme params totaux. Donner le gain compute.

4. **Cout communication all-to-all** : modeliser le temps de communication par layer MoE sur un cluster de `P` GPUs (expert parallelism, experts repartis sur les GPUs). Pour chaque token, 2 all-to-all transferent `d_model` floats. Etant donne une bande passante interconnect `BW` (ex NVLink 600 GB/s, Ethernet 25 GB/s) :
   - `comm_bytes_per_layer = 2 * tokens * d_model * 2 octets` (FP16) ;
   - `t_comm = comm_bytes / BW`.
   Comparer `t_comm` au temps de compute `t_compute = FLOPs_layer / GPU_throughput` (ex H100 ~1e15 FLOP/s). Calculer la fraction de temps en communication.

5. **Balayage interconnect** : pour `BW in {600, 200, 50, 25} GB/s`, donner la fraction du temps total passee en all-to-all. Retrouver le chiffre du cours (~20-50% sur les gros MoE distribues, pire sans NVLink).

6. **Analyse** : conclure sur quand le MoE est rentable (cf section 8). Pourquoi DeepSeek a eu besoin de DualPipe + communication FP8 ? A quel point l'infra (NVLink/IB) conditionne le gain MoE ?

### Criteres de reussite

- [ ] `moe_accounting` gere SwiGLU (3 matrices), le shared expert et le router
- [ ] Les ordres de grandeur Mixtral (~47B) et DeepSeek-V3 (~671B/37B) sont retrouves
- [ ] VRAM des poids chiffree pour les 2 ; confirme que MoE ne reduit PAS la VRAM
- [ ] FLOPs/token = 2 * actifs ; gain compute vs dense equivalent chiffre
- [ ] Le modele de cout all-to-all donne la fraction comm/total et son explosion a basse bande passante
- [ ] L'analyse relie le verdict au cours (infra-dependant, DualPipe/FP8)
- [ ] Code numpy, commente WHY
