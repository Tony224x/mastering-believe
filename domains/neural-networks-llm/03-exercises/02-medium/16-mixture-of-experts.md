# Exercices Medium — Jour 16 : Mixture of Experts

---

## Exercice 4 : Forward MoE complet + comparaison FLOPs dense vs sparse

### Objectif

Implementer une couche MoE de bout en bout (routeur top-k + dispatch + somme ponderee) et mesurer empiriquement l'economie de FLOPs par token face a un FFN dense de capacite equivalente.

### Consigne

1. Implementer en numpy une `MoELayer` (cf code du Jour 16) avec `N` experts FFN a 2 matrices (`W_up (d_model, d_ff)`, `W_dn (d_ff, d_model)`, ReLU au milieu), un routeur `W_g (d_model, N)`, et un routing **top-k** avec renormalisation des poids.

2. Implementer aussi un **FFN dense de reference** dont le `d_ff_dense` est dimensionne pour avoir **autant de parametres totaux** que les `N` experts reunis (`d_ff_dense = N * d_ff`). C'est la comparaison honnete : meme capacite de stockage.

3. Faire passer un batch de `B=32` tokens (`d_model=64`, `d_ff=128`, `N=8`, `k=2`) dans les deux. Verifier que les shapes de sortie sont identiques.

4. **Compter les FLOPs** d'une passe forward par token (compter une multiply-add comme 1 FLOP ; on ignore le routeur, negligeable) :
   - dense : `2 * d_model * d_ff_dense * 2` (up + down) ;
   - MoE : `k * (2 * d_model * d_ff * 2)` (seulement k experts s'activent) + routeur `d_model * N`.
   - Donner le ratio FLOPs_dense / FLOPs_MoE.

5. **Compter les params** des deux (experts + routeur vs FFN dense). Verifier que les params totaux sont du meme ordre, mais que les params **actifs** par token du MoE valent `k/N` de ceux du dense.

6. Faire varier `k in {1, 2, 4, 8}` (a N=8 fixe). Tracer (ou imprimer) le ratio de sparsite (total/actif) en fonction de k. A `k=N`, que devient le MoE ?

### Criteres de reussite

- [ ] La `MoELayer` route top-k, renormalise les poids et somme les sorties des k experts
- [ ] Le FFN dense de reference a `d_ff_dense = N * d_ff` (meme capacite)
- [ ] Les sorties ont la meme shape `(B, d_model)`
- [ ] Ratio FLOPs : MoE fait ~`N/k` fois moins de FLOPs que le dense equivalent (ici ~4x a k=2, N=8)
- [ ] Params actifs MoE = `k/N` * params experts ; ratio sparsite affiche pour chaque k
- [ ] A `k=N`, le MoE active tous les experts -> redevient un dense (sparsite = 1, aucun gain)
- [ ] Code numpy, seed, commente WHY

---

## Exercice 5 : Expert collapse et load balancing loss en action

### Objectif

Reproduire le collapse du routeur (2-3 experts gagnent tout le trafic) sur quelques steps de training simule, puis montrer que la load balancing loss de Shazeer le previent.

### Consigne

1. Setup : `N=8` experts, top-1 routing, `B=256` tokens (`d_model=32`). Routeur `W_g (d_model, N)` initialise petit.

2. Implementer la `load_balancing_loss(top_k_indices, full_probs, N)` du cours :
   - `f_i` = fraction de tokens routes vers l'expert i ;
   - `P_i` = proba softmax moyenne de l'expert i ;
   - `L_aux = N * sum_i (f_i * P_i)`.

3. **Reproduire le collapse** : faire une boucle de ~50 steps ou, a chaque step, on pousse le routeur a renforcer son expert favori (gradient grossier : on ajoute un petit increment aux logits de l'expert le plus charge). Sans regularisation, mesurer la **fraction de trafic recue par le top-1 expert** au fil des steps. Elle doit grimper vers ~1.0 (collapse).

4. **Ajouter la loss auxiliaire** : a chaque step, soustraire `alpha * gradient(L_aux)` (approxime numeriquement, ou un terme qui penalise les logits de l'expert surcharge). Avec `alpha ~ 0.01-0.1`, montrer que la distribution de charge reste **proche de l'uniforme** (`f_i ~ 1/N`).

5. Calculer et afficher, dans les deux scenarios (avec/sans aux loss) :
   - l'entropie de la distribution de charge `f` (haute = equilibre, basse = collapse) ;
   - le nombre d'experts "morts" (`f_i ~ 0`).

6. Analyse : pourquoi `f_i` seul n'est pas differentiable, et pourquoi le produit `f_i * P_i` resout ce probleme ? (relier au cours)

### Criteres de reussite

- [ ] `load_balancing_loss` correcte (=1.0 en uniforme, =N en collapse total)
- [ ] Sans regularisation, la fraction du top-1 expert grimpe vers ~1.0 (collapse reproductible)
- [ ] Avec aux loss, la charge reste proche de l'uniforme (entropie elevee, peu/pas d'experts morts)
- [ ] L'entropie de `f` et le nombre d'experts morts sont chiffres dans les 2 cas
- [ ] L'analyse explique : `f_i` passe par argmax (non differentiable) ; `f_i * P_i` accroche le gradient de `P_i` (differentiable) a la distribution dure
- [ ] Code numpy, seed, commente WHY

---

## Exercice 6 : Capacity factor et token dropping

### Objectif

Modeliser le buffer fixe par expert (capacity factor) et mesurer le taux de tokens droppes en fonction du CF et du desequilibre de routage.

### Consigne

1. Implementer `dispatch_with_capacity(assignments, N, capacity)` qui, etant donne l'expert choisi par chaque token (top-1), remplit le buffer de chaque expert **jusqu'a `capacity`** dans l'ordre des tokens, et **drop** (marque comme non traite) les tokens en exces une fois le buffer plein.

2. `capacity = ceil(CF * tokens_par_batch / N)` (formule du cours). Implementer le calcul.

3. **Scenario equilibre** : `B=512` tokens routes **uniformement** sur `N=8` experts (assignments tires uniformement). Pour `CF in {1.0, 1.25, 1.5, 2.0}`, mesurer le **taux de drop** (% de tokens non traites).

4. **Scenario desequilibre** : meme batch mais routage **skewed** (ex : 50% des tokens vont aux 2 premiers experts, le reste reparti). Refaire la mesure du taux de drop par CF.

5. Comparer les deux scenarios :
   - Confirmer qu'a CF=1.0, meme l'uniforme drop ~1-3% (effet du remplissage discret/aleatoire, comme le NOTE du cours).
   - Montrer que le desequilibre fait exploser le drop a bas CF.
   - Montrer que monter CF reduit le drop **mais double la memoire allouee** (capacite x2).

6. Analyse : pourquoi a l'**inference** on ne drop generalement pas (cf cours) ? Quel est le tradeoff exact entre CF, memoire VRAM et qualite au training ?

### Criteres de reussite

- [ ] `dispatch_with_capacity` remplit les buffers et drop l'exces correctement
- [ ] `capacity = ceil(CF * B / N)` implemente
- [ ] Scenario equilibre : drop faible (~1-3%) a CF=1.0, ~0% a CF>=1.5
- [ ] Scenario desequilibre : drop eleve a bas CF, qui chute quand CF monte
- [ ] Le tradeoff memoire (capacite proportionnelle a CF) est explicite
- [ ] L'analyse explique l'absence de drop a l'inference (pas de buffer fixe) + tradeoff CF/VRAM/qualite
- [ ] Code numpy, seed, commente WHY
