# Exercices Medium — Jour 17 : State Space Models

---

## Exercice 4 : Selectivite — pourquoi S4 echoue et Mamba reussit

### Objectif

Implementer la tache "selective copying" du cours et montrer empiriquement qu'un SSM non-selectif (B, C fixes) laisse fuir les filler tokens, alors qu'un SSM selectif (B, C fonction de l'input) les supprime.

### Consigne

1. **Tache** : generer une sequence de `N=64` tokens dont une fraction (~25%) sont des **data tokens** (valeur aleatoire) et le reste des **filler** (valeur 0), avec un `flag` par token (1=data, 0=filler). Le modele ideal reproduit les data et supprime les filler.

2. **SSM non-selectif** : recurrence `h_t = A h_{t-1} + B x_t`, `y_t = C h_t` avec `A` stable (diagonale dans (0,1)), `B, C` **fixes**. Implementer en numpy.

3. **SSM selectif (Mamba-style)** : `B_t = B_base * gate(flag_t)` et `C_t = C_base * gate(flag_t)`, ou `gate` ferme l'entree sur les filler (gate=0 si filler). Implementer.

4. **Mesurer le SNR** (signal-to-noise) : pour chaque variante, calculer
   - le "leak" = `mean(|y|)` sur les positions filler (devrait etre ~0 pour un bon modele) ;
   - le "signal" = `mean(|y|)` sur les positions data ;
   - le SNR = signal / leak.
   Le SSM selectif doit avoir un SNR nettement superieur.

5. **Faire varier la decay de A** (`decay in {0.5, 0.85, 0.95, 0.99}`) : comment le leak du non-selectif evolue-t-il quand A retient plus longtemps ? Pourquoi le non-selectif souffre-t-il plus a forte memoire ?

6. **Analyse** : expliquer pourquoi rendre `B, C, Delta` fonction de `x_t` est l'equivalent fonctionnel d'un gate dynamique / d'une attention, mais en O(N). Relier a la note du cours (la selectivite agit via `Delta_t` sur `Ā = exp(Delta_t * A)`).

### Criteres de reussite

- [ ] La tache selective copying est generee (data + filler + flags)
- [ ] Les deux SSM (non-selectif, selectif) sont implementes en numpy
- [ ] Le SSM selectif a un SNR (signal/leak) nettement superieur au non-selectif
- [ ] Le balayage de `decay` montre que le non-selectif fuit davantage a forte memoire
- [ ] L'analyse relie selectivite -> gate dynamique -> equivalent attention en O(N), via Delta_t
- [ ] Code numpy, seed, commente WHY

---

## Exercice 5 : Benchmark de complexite — SSM lineaire vs attention quadratique

### Objectif

Mesurer empiriquement les courbes de scaling SSM (lineaire-ish) vs self-attention (quadratique) en fonction de la longueur de sequence et retrouver le facteur ~N quand N quadruple.

### Consigne

1. Implementer une `attention_forward(x, d_model)` jouet (single-head : projeter en Q/K/V, scores `Q K^T`, softmax, sortie) — cout O(N^2 d).

2. Implementer un `ssm_forward(x, A, B, C)` via le mode convolutionnel (construire le kernel puis convoluer).

3. **Chronometrer** les deux pour `N in [128, 512, 2048, 8192]`. Afficher un tableau `N | t_ssm | t_attn | ratio`.

4. **Verifier le scaling** : quand `N` est multiplie par 4,
   - l'attention devrait croitre ~16x (quadratique) ;
   - le SSM devrait croitre ~4x (lineaire) — ou un peu plus si la convolution numpy est directe (O(N^2)) et le kernel build naif (O(N D^2)).

5. **Honnetete intellectuelle** : ecrire explicitement le DISCLAIMER du cours — `np.convolve` est une convolution **directe** O(N^2), pas une FFT, donc ce micro-bench **sous-estime** les vrais SSM. Refaire la convolution avec **FFT** (`np.fft`) et montrer que le scaling du SSM s'ameliore (plus proche de O(N log N)).

6. **Memoire** : calculer (pas mesurer) le pic memoire theorique de la matrice d'attention `N x N` en FP16 pour `N in {2048, 32768, 131072}` (cf table du cours) vs la memoire d'etat fixe du SSM. Conclure sur le mur quadratique.

### Criteres de reussite

- [ ] Attention et SSM (mode conv) implementes et chronometres
- [ ] Le tableau N | t_ssm | t_attn | ratio est affiche
- [ ] Le scaling quadratique de l'attention (~x16 quand N x4) est observe
- [ ] Le DISCLAIMER np.convolve est present ; la version FFT ameliore le scaling SSM
- [ ] Le pic memoire theorique de l'attention N^2 vs l'etat fixe SSM est chiffre (cf table du cours)
- [ ] Code numpy, seed, commente WHY

---

## Exercice 6 : Hybride Transformer + Mamba — le ratio 1/8 de Jamba

### Objectif

Simuler une stack hybride (layers Mamba + rares layers d'attention) sur une tache de recall associatif et montrer qu'ajouter quelques layers d'attention recupere le recall que le Mamba pur rate.

### Consigne

1. **Tache MQAR-like (Multi-Query Associative Recall)** : generer des paires cle->valeur dispersees dans une sequence, puis poser des requetes "quelle valeur pour la cle X ?". Mesurer l'accuracy de recall.

2. **Bloc Mamba simplifie** : un SSM selectif (cf Exercice 4) qui compresse l'historique dans un etat de taille fixe `D`. Implementer une "stack" de `L` layers Mamba (chaque layer applique le SSM puis un residual).

3. **Bloc attention simplifie** : une attention causale qui peut acceder directement a toute la sequence (recall exact via les scores). Implementer.

4. **Trois architectures** a `L=8` layers :
   - **A** : 8 layers Mamba (pure SSM) ;
   - **B** : 7 Mamba + 1 attention (ratio Jamba) ;
   - **C** : 8 layers attention (pure transformer).
   Mesurer l'accuracy de recall MQAR-like de chacune en fonction du **nombre de paires** a memoriser (densite croissante).

5. **Reproduire le phenomene** : pure SSM decroche quand la densite de paires augmente (l'etat fixe sature) ; ajouter 1 layer d'attention recupere une grande partie du recall ; le pure transformer plafonne au mieux.

6. **Analyse** :
   - Pourquoi 1 layer d'attention sur 8 suffit a recuperer ~la qualite recall ? (relier au cours)
   - Quel est le cout (memoire/compute) ajoute par cette 1 layer d'attention vs le gain de recall ?
   - Pourquoi le pure SSM sature : taille d'etat fixe = compresseur lossy (idee fausse n°4).

### Criteres de reussite

- [ ] La tache MQAR-like (paires cle->valeur + requetes) est generee
- [ ] Bloc Mamba (SSM selectif, etat fixe) et bloc attention (acces global) implementes
- [ ] Les 3 architectures (pure SSM, hybride 7+1, pure attention) sont evaluees
- [ ] Pure SSM decroche quand la densite de paires monte ; l'hybride recupere le recall
- [ ] L'analyse relie le ratio 1/8 de Jamba au compromis recall/throughput du cours
- [ ] Code numpy, seed, commente WHY
