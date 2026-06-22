# Exercices Medium — Jour 8 : Pretraining & Tokenization

---

## Exercice 4 : BPE complet — train, encode, decode, et invariant de round-trip

### Objectif

Reimplementer le BPE de `02-code/08-pretraining-tokenization.py` (train + encode) et le COMPLETER avec un `decode` exact, puis prouver l'invariant fondamental : `decode(encode(texte)) == texte`.

### Consigne

En te basant sur `02-code/08-pretraining-tokenization.py` :

1. Reprendre `pre_tokenize`, `get_pair_stats`, `merge_pair`, `train_bpe`, `encode_word`.

2. Implementer `decode_tokens(tokens)` : reconstruire le mot d'origine a partir de la liste de tokens BPE. Indice : concatener les tokens et retirer le marqueur `</w>` final (le remplacer par un espace de fin de mot).

3. **Invariant de round-trip** : pour un ensemble de mots (vus ET non vus au training), verifier que `decode_tokens(encode_word(w)) == w` pour TOUS. C'est la propriete qui garantit qu'un tokenizer est *lossless* (aucune information perdue).

4. **Couverture vs taille de vocab** : entrainer le BPE avec un nombre croissant de merges (`0, 5, 10, 20, 40`). Pour chaque taille de vocab, mesurer la **longueur moyenne de sequence** (nb de tokens par mot) sur un corpus de test. Tracer (en ASCII ou en chiffres) la courbe : plus de merges -> sequences plus courtes mais vocab plus gros. C'est le tradeoff central de la tokenization.

5. **Determinisme** : montrer que l'ordre des merges est crucial — appliquer les merges dans le DESORDRE donne une tokenization differente (et potentiellement incoherente). Pourquoi l'encodeur DOIT respecter l'ordre d'apprentissage ?

### Criteres de reussite

- [ ] `decode_tokens` reconstruit le mot exact (gestion du `</w>`)
- [ ] Round-trip `decode(encode(w)) == w` verifie pour mots vus ET non vus
- [ ] La longueur moyenne de sequence DECROIT quand le nombre de merges augmente
- [ ] Le tradeoff vocab vs longueur est quantifie (chiffres concrets)
- [ ] La demonstration que l'ordre des merges change le resultat est faite et expliquee

---

## Exercice 5 : Cross-entropy, bits-per-token et perplexite

### Objectif

Implementer les 3 metriques de qualite d'un modele de langage — cross-entropy, bits-per-token et perplexite — et comprendre leurs relations exactes.

### Consigne

1. Entrainer un **modele bigramme** sur un petit corpus de caracteres : `P(c_t | c_{t-1})` estime par comptage + lissage de Laplace (`+1`). Stocker une matrice `(vocab, vocab)` de probabilites conditionnelles normalisees.

2. Implementer les metriques sur un texte de test (suite de tokens) :
   - `cross_entropy_nats = - (1/N) * sum log P(c_t | c_{t-1})` (en nats, base e).
   - `bits_per_token = cross_entropy_nats / log(2)` (conversion nats -> bits).
   - `perplexity = exp(cross_entropy_nats)`.

3. **Verifier les relations** :
   - `perplexity = 2 ** bits_per_token` (identite exacte).
   - Pour un modele uniforme sur `V` symboles : `perplexity == V` et `bits_per_token == log2(V)`. Le verifier numeriquement.
   - La perplexite s'interprete comme le "nombre effectif de choix" a chaque position : un modele parfait sur un texte deterministe a une perplexite de 1.

4. **Bigramme vs uniforme** : montrer que le bigramme entraine a une perplexite SIGNIFICATIVEMENT plus basse que le modele uniforme sur le meme texte de test. De combien (en %) ?

5. **Effet du lissage** : tester sans lissage (`+0`) — que se passe-t-il si une transition du test n'a jamais ete vue au training ? (Indice : `P = 0` -> `log(0) = -inf` -> perplexite infinie.) Pourquoi le lissage est-il indispensable ?

### Criteres de reussite

- [ ] Le bigramme avec lissage de Laplace produit des distributions valides (lignes sommant a 1)
- [ ] `perplexity == 2 ** bits_per_token` verifie (ecart < 1e-9)
- [ ] Modele uniforme : `perplexity == V` et `bits_per_token == log2(V)` verifies
- [ ] Le bigramme a une perplexite nettement < uniforme (gain quantifie)
- [ ] Sans lissage, une transition inconnue donne perplexite infinie ; le lissage l'evite

---

## Exercice 6 : Scaling laws — Chinchilla et l'allocation optimale du compute

### Objectif

Implementer le calcul de l'optimum Chinchilla et l'utiliser pour repondre a la question : "etant donne un budget de compute, comment repartir entre taille du modele et quantite de donnees ?"

### Consigne

D'apres le cours (`01-theory/08-pretraining-tokenization.md`, section scaling laws) :

1. **Loi de compute** : le compute d'entrainement (en FLOPs) est approxime par `C ≈ 6 * N * D` ou `N` = nb de parametres, `D` = nb de tokens. Implementer `compute_flops(N, D)`.

2. **Regle Chinchilla** : l'optimum compute correspond a `D ≈ 20 * N` (environ 20 tokens par parametre). Implementer `chinchilla_optimal(C)` qui, pour un budget `C` donne, renvoie `(N_opt, D_opt)` en resolvant `C = 6 * N * (20 * N)` -> `N = sqrt(C / 120)`, puis `D = 20 * N`.

3. **Verifier sur des cas reels** :
   - GPT-3 : N=175e9, D=300e9 -> ratio D/N ≈ 1.7 (SOUS-entraine vs 20).
   - Chinchilla : N=70e9, D=1.4e12 -> ratio D/N == 20 (optimal).
   - Calculer le compute de chacun et confirmer que Chinchilla, a compute COMPARABLE, utilise un modele plus petit mais plus de donnees.

4. **Frontiere iso-compute** : pour un budget fixe `C`, tracer (en chiffres) plusieurs couples `(N, D)` respectant `6*N*D = C` et montrer que la loss modelisee `L(N, D) = E + A/N^alpha + B/D^beta` (forme parametrique de Chinchilla) est MINIMISEE au point `D ≈ 20*N`. Utiliser des valeurs typiques (`A=406, B=410, alpha=0.34, beta=0.28, E=1.69`).

5. **Inference-aware** : LLaMA-3-8B utilise ~1875 tokens/param, BIEN au-dela de l'optimum Chinchilla. Expliquer (calcul a l'appui) pourquoi : Chinchilla minimise le compute de TRAINING, mais en production c'est le compute d'INFERENCE (proportionnel a N) qui domine sur des milliards de requetes.

### Criteres de reussite

- [ ] `compute_flops` et `chinchilla_optimal` sont corrects (N = sqrt(C/120))
- [ ] Les ratios D/N de GPT-3 (~1.7) et Chinchilla (20) sont retrouves
- [ ] La frontiere iso-compute est exploree et la loss est minimale vers D ≈ 20*N
- [ ] Le calcul confirme : a compute egal, le modele Chinchilla-optimal est plus petit que GPT-3
- [ ] L'explication inference-aware est correcte (training one-shot vs inference repetee)
