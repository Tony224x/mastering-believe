# Exercices Hard — Jour 21 : Mechanistic interpretability

---

## Exercice 7 : Causal mediation complet — carte de recovery (layer x position)

### Objectif

Aller au-dela du patching ponctuel du tier medium : construire un toy decoder ou un **fait** vit a une couche/position *connue* du residual stream, puis produire la **carte causale complete** `(layer x position)` du recovery par activation patching (denoising). Le point central du cours : montrer **numeriquement** que correlation (probe eleve partout) != causalite (un seul site fait remonter le logit cible). On asserte que le site porteur depasse 0.5 de recovery et que les sites non pertinents sont ~0.

### Consigne

1. **Toy decoder traceable avec un fait localise** : construire en numpy un decoder additif a `n_layers` couches (residual stream `h += contrib`). A une position `P*` et a partir d'une couche `L*` precises, on injecte une **direction-fait** `d_fact` (la colonne d'unembedding du token cible `t_clean`). Pour le run `corrupted`, on injecte a la place la direction d'un autre token `t_corr`. Tout le reste (les autres positions, le bruit) est identique entre les deux runs et n'apporte aucune info sur le fait.

2. **Run clean & corrupted** : enregistrer le tenseur d'activations `acts[run][layer][position]` (shape residual `d_model` par cellule). Verifier que `clean` predit `t_clean` en top-1 et `corrupted` predit `t_corr` en top-1 (via `softmax(h_final @ W_U)`).

3. **Patching denoising sur toute la grille** : pour **chaque** `(L, P)` :
   - repartir du run `corrupted`,
   - copier l'activation clean `acts["clean"][L][P]` dans le residual a `(L, P)`,
   - propager le forward jusqu'a la fin **a partir de ce point** (les couches `> L` re-tournent avec l'activation patchee),
   - mesurer le **recovery normalise** : `(logit_clean_patched - logit_corr) / (logit_clean - logit_corr)` sur le logit du token `t_clean`. Recovery = 1.0 -> on a totalement restaure le fait ; 0.0 -> aucun effet.

4. **Carte causale** : afficher la grille `(layers x positions)` des recoveries. **Asserter** :
   - le site porteur `(L*, P*)` (et les couches `>= L*` a `P*`) a un recovery `> 0.5`,
   - les positions `P != P*` ont un recovery `~ 0` (`< 0.1`),
   - les couches `< L*` a `P*` (avant l'injection) ont aussi un recovery `~ 0`.

5. **Probe vs patching (le piege du cours)** : entrainer un linear probe a detecter `t_clean` vs `t_corr` a chaque couche, **a la position P\*** ET **a une position distractrice** `P_d` ou l'on aura volontairement *copie* `d_fact` dans le residual **sans** la router vers la sortie (une feature presente mais causalement inerte). Montrer que le probe est **eleve aux deux positions** (l'info est *presente*), alors que le patching n'a un effet **qu'a P\***. C'est la demonstration numerique correlation != causalite.

### Criteres de reussite

- [ ] Le toy decoder injecte le fait a un `(L*, P*)` connu et `clean/corrupted` donnent bien `t_clean`/`t_corr` en top-1
- [ ] Le patching denoising re-tourne le forward a partir du site patche (pas juste une projection finale)
- [ ] La carte `(layers x positions)` de recovery est affichee
- [ ] **Assertions** : recovery `> 0.5` au site porteur, `< 0.1` aux positions/couches non pertinentes
- [ ] Le probe est eleve a `P*` ET a la position distractrice inerte, mais le patching n'a d'effet qu'a `P*` (correlation != causalite, demontre numeriquement)
- [ ] Code commente (POURQUOI normaliser le recovery, POURQUOI le probe peut mentir)

---

## Exercice 8 : Induction head en VRAIE attention Q/K/V + phase transition

### Objectif

Depasser deux raccourcis : la *fonction* I/O du `02-code` (un `for`/`if` algorithmique) **et** le matching "softmax sur un canal" du tier medium. Ici on implemente le circuit a 2 layers comme de **vraies tetes d'attention** avec matrices `Q`, `K`, `V`, `W_O` : une **previous-token head** (layer 1) ecrit l'identite du token precedent dans le residual via OV, et une **induction head** (layer 2) fait un *prefix matching* QK sur ce signal pour copier. Puis on demontre la **phase transition** d'Olsson 2022 : en augmentant la force des poids prev-token + induction, l'accuracy de copie in-context **saute** brutalement.

### Consigne

1. **Setup token + embeddings** : vocab `V`, embedding `E (V, d_model)`, unembedding `W_U (d_model, V)`. On reserve explicitement des **sous-espaces** du residual : un bloc "token courant", un bloc "previous-token signal" (ecrit par la head 1), pour pouvoir router proprement. Positions encodees (positional embedding simple) car la prev-token head a besoin de la notion de position `t-1`.

2. **Layer 1 — previous-token head (attention reelle)** :
   - `Q1, K1` construits pour que la position `t` attende fortement la position `t-1` (biais positionnel : `score[t, s]` maximal pour `s = t-1`). Softmax causal (masque les positions futures).
   - `V1, W_O1` : la valeur transportee est l'embedding du token a la position attendue, ecrite dans le bloc "previous-token signal" du residual de la position `t`.
   - Resultat : `residual[t]` contient maintenant "le token precedent etait `tokens[t-1]`".

3. **Layer 2 — induction head (attention reelle)** :
   - `Q2` lit le **token courant** (bloc token courant de `residual[t]`), `K2` lit le **previous-token signal** ecrit par la head 1 (bloc dedie de chaque `residual[s]`). Le score QK est donc maximal a la position `s` dont le *previous token* == token courant. C'est le **prefix matching**.
   - `V2, W_O2` : copie l'embedding du token a la position `s` (qui est `tokens[s]`, c.-a-d. le token *suivant* l'occurrence precedente) vers la sortie -> on predit `tokens[s]`.
   - Logits finaux : `residual_final @ W_U`.

4. **Validation** : sur des sequences repetees `[A B C D | A B C ...]`, asserter que `argmax(logits[-1])` predit le bon token d'induction (le token qui suivait la derniere occurrence du token courant). Comparer a la baseline "copie naive du token courant" (qui echoue).

5. **Phase transition** : parametrer la force du circuit par un scalaire `strength` (gain commun applique a `W_O1` et a la matrice QK de la head 2). Faire varier `strength` de ~0 a une grande valeur, mesurer l'accuracy de copie in-context sur un batch de sequences. **Tracer (afficher) la courbe accuracy vs strength** et asserter qu'elle est quasi-aleatoire a faible strength puis **saute** vers ~1.0 au-dela d'un seuil. Relier explicitement a Olsson 2022 (emergence brutale du circuit pendant le training).

6. **Analyse** : pourquoi le mecanisme est-il *l'attention* (QK = ou regarder, OV = quoi copier) et non la fonction `if` du `02-code` ? Pourquoi faut-il composer **2** heads via le residual stream (la head 2 ne peut matcher que parce que la head 1 a *ecrit* le previous-token signal) ?

### Criteres de reussite

- [ ] Les deux heads sont de vraies attentions (matrices `Q/K/V/W_O`, scores = `QK^T/sqrt(d)`, softmax causal), pas un `if` ni un simple "softmax sur un canal"
- [ ] La head 1 ecrit le previous-token signal dans un sous-espace dedie du residual, la head 2 le lit en `K` (composition via residual stream)
- [ ] `argmax(logits[-1])` predit le bon token d'induction (assertion) et bat la copie naive
- [ ] La courbe accuracy vs `strength` est affichee et montre une transition nette (assertion : faible a `strength` bas, ~1.0 a `strength` haut)
- [ ] Contraste explicite QK/OV (mecanisme attentionnel) vs raccourci fonctionnel du `02-code`
- [ ] Code commente (POURQUOI 2 layers, role du residual stream comme bus de communication, lien phase transition / Olsson 2022)

---

## Exercice 9 : TopK SAE (Gao/OpenAI 2024) recupere PLUS de features que le L1 naif

### Objectif

Le `02-code` montre un SAE L1 naif qui recupere peu de features et a beaucoup de dead features. Ici on construit un ground-truth **controle** (features sparses superposees dans un espace bas), on entraine **les deux** SAE (L1 naif et TopK), et on demontre — **par construction** — que TopK recupere **au moins autant** de features ground-truth (match par cosine) **et** a **moins** de dead features. On garde le caveat honnete (toy), mais l'inegalite doit tenir.

### Consigne

1. **Ground-truth controle** : generer `m` features atomiques (directions unitaires `D_true (m, d_model)`, quasi-orthogonales) et des codes **sparses** `c (n, m)` (chaque exemple active `k_true` features parmi `m`, magnitude positive). Les activations observees = `acts = c @ D_true` dans `R^d_model` avec `d_model < m` -> superposition reelle et controlee (on connait la verite terrain).

2. **SAE L1 naif (baseline `02-code`)** : encoder `f = ReLU(acts @ W_enc + b_enc)`, decoder `acts_rec = f @ W_dec + b_dec`, loss = `MSE + lambda * ||f||_1`, normalisation des colonnes de `W_dec` a chaque step. `n_sae >= m`. Reproduire fidelement le pattern du `02-code` (shrinkage L1 -> dead features).

3. **SAE TopK (Gao 2024)** : meme archi, mais **pas de L1**. A chaque forward, ne garder que les **K plus grandes** activations de `f` par exemple (les autres a zero) -> `f_topk`. La sparsite est imposee *exactement* par K (pas par une penalite qui retrecit les magnitudes). Backprop seulement a travers les entrees survivantes (le masque top-K stoppe le gradient des autres). Choisir `K ≈ k_true`.

4. **Anti dead-features (au choix, au moins un)** :
   - init du decoder alignee sur des directions de donnees, et/ou
   - **resampling** des features mortes (ou un terme de penalite explicite type "aux loss" simplifie) pour reviver les unites jamais actives — exactement la classe de fixes que TopK + tricks visent.

5. **Mesure du recovery (cosine matching)** : pour chaque feature SAE, la feature ground-truth la plus alignee (cosine entre colonne de `W_dec` et ligne de `D_true`). Compter (a) features ground-truth **uniques** recuperees (cosine `> seuil`, ex 0.5, et active rate non nul), (b) **dead features** (active rate ~ 0). Faire un **matching 1-1** (chaque ground-truth comptee une seule fois, prendre le meilleur match disponible) pour ne pas surestimer.

6. **Assertions (l'inegalite doit tenir)** :
   - `recovered_topk >= recovered_l1` (TopK recupere au moins autant de ground-truth),
   - `dead_topk <= dead_l1` (TopK a au plus autant de dead features),
   - bonus shrinkage : la magnitude moyenne reconstruite des features actives est **plus proche de 1** (moins sous-estimee) pour TopK que pour L1.

7. **Caveat honnete** : comme le `02-code`, conclure que c'est un toy ; le but est de *reproduire le mecanisme* (TopK evite le shrinkage L1 et le dead-feature problem), pas de prouver un resultat a l'echelle.

### Criteres de reussite

- [ ] Ground-truth controle : `m` features sparses, `k_true` actives/exemple, superposees dans `d_model < m` (verite terrain connue)
- [ ] SAE L1 naif correct (ReLU + L1 + normalisation colonnes decoder), reproduit shrinkage + dead features
- [ ] SAE TopK correct (masque top-K par exemple, pas de L1, gradient stoppe hors top-K)
- [ ] Au moins un fix dead-feature implemente (resampling / init data-aligned)
- [ ] Matching 1-1 par cosine, comptage des ground-truth recuperees et des dead features
- [ ] **Assertions** : `recovered_topk >= recovered_l1` ET `dead_topk <= dead_l1` (inegalite tenant par construction)
- [ ] Effet shrinkage mesure (magnitude reconstruite plus proche de 1 pour TopK) + caveat toy honnete
- [ ] Code commente (POURQUOI L1 cree shrinkage + dead features, POURQUOI TopK les evite, POURQUOI normaliser W_dec)
