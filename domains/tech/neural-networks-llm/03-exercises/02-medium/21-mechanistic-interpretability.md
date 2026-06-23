# Exercices Medium — Jour 21 : Mechanistic interpretability

---

## Exercice 4 : Activation patching — le test causal from scratch

### Objectif

Implementer le protocole d'activation patching (causal mediation, Vig/Meng) sur un modele jouet et produire une **carte causale** (quel layer/position porte l'info), pour depasser la simple correlation du probing.

### Consigne

1. **Toy model traceable** : construire un petit "transformer-like" en numpy (quelques couches, residual stream additif) dont la sortie depend d'une information injectee a une position/couche connue (un "fait"). Deux inputs : `clean` (donne la reponse A) et `corrupted` (donne la reponse B).

2. **Run clean & corrupted** : enregistrer toutes les activations intermediaires `acts[layer][position]` des deux runs. Verifier que clean -> A (top-1) et corrupted -> B (top-1).

3. **Patching (denoising)** : pour chaque `(layer L, position P)` :
   - re-run le `corrupted`,
   - mais remplacer l'activation a `(L, P)` par celle du run `clean`,
   - mesurer le **recovery** : de combien la sortie revient vers A (ex : `(logit_A_patched - logit_A_corrupted) / (logit_A_clean - logit_A_corrupted)`).

4. **Carte causale** : afficher une grille `(layers x positions)` des % de recovery. Identifier le/les `(L, P)` ou patcher restaure A -> c'est la ou l'info est portee **causalement**.

5. **Comparer a un probe** : entrainer un linear probe pour detecter l'info a chaque couche. Montrer qu'un probe peut etre eleve a une couche **sans** que patcher cette couche ait un effet causal (correlation != causalite). Discuter denoising vs noising (cf cours, Heimersheim & Nanda 2024).

### Criteres de reussite

- [ ] Le toy model est traceable (activations enregistrables) et clean/corrupted donnent A/B
- [ ] Le patching remplace bien l'activation cible et mesure un recovery normalise
- [ ] La carte causale identifie un point chaud `(L, P)` coherent avec l'injection
- [ ] Le contraste probe (correlation) vs patching (causalite) est demontre numeriquement
- [ ] Code commente (POURQUOI normaliser le recovery, POURQUOI denoising != noising)

---

## Exercice 5 : Induction head — circuit a 2 layers et phase transition

### Objectif

Reproduire l'induction head (Olsson 2022) au-dela de la simulation I/O du cours : montrer la composition **previous-token head -> induction head**, et observer l'emergence (l'accuracy in-context qui monte avec la longueur de contexte).

### Consigne

1. **Tache de copie in-context** : generer des sequences `[... A B ... A] -> ?` ou la bonne reponse est `B` (le token qui suivait la derniere occurrence du token courant). Construire un jeu de test varie (motifs repetes, longueurs differentes).

2. **Implementer le circuit en 2 etapes explicites** (numpy, pas juste un lookup) :
   - **Previous-token head** : pour chaque position `t`, ecrire dans un "canal" du residual l'identite du token `t-1` (one-hot ou embedding).
   - **Induction head** : a la derniere position, faire un *prefix matching* — chercher la position `t'` ou le canal "previous token" = token courant, puis copier `token[t'+1]`. Exprimer ce match comme une attention (scores = produit scalaire, softmax) plutot qu'un `if`.

3. **Mesurer l'accuracy** du circuit sur le jeu de test. Comparer a une baseline naive (copier le token courant) et a un "bigram" (token le plus frequent apres le courant).

4. **Effet de la longueur de contexte** : faire varier le nombre de repetitions du motif `{1, 2, 3, 4}`. L'accuracy in-context monte-t-elle (plus d'exemples = matching plus fiable) ? Relier a l'in-context learning.

5. **Analyse** : pourquoi faut-il **2** layers (composition) et pas 1 ? Que fait le residual stream comme "memoire" entre les deux heads ?

### Criteres de reussite

- [ ] Le previous-token head et l'induction head sont deux etapes distinctes et composees
- [ ] Le matching est exprime comme une attention (scores + softmax), pas un simple `if`
- [ ] Le circuit bat les baselines (copie naive, bigram) sur la copie in-context
- [ ] L'effet "plus de contexte -> meilleure copie" est mesure
- [ ] Code commente (POURQUOI 2 layers, role du residual stream comme canal de communication)

---

## Exercice 6 : Sparse autoencoder (SAE) minimal — L1 vs TopK

### Objectif

Entrainer un mini-SAE (L1) sur des activations superposees, mesurer le taux de recuperation mono-semantique et les **dead features**, puis comparer a un SAE **TopK** (Gao/OpenAI 2024) qui evite le shrinkage L1.

### Consigne

1. **Generer des activations superposees** : reprendre le setup superposition (5 features sparses packees dans 2 dims via un AE tied-weights, comme PART 4 du code du jour) et collecter les hidden activations (dim 2).

2. **SAE L1 from scratch** : encoder `f = ReLU(h @ W_enc + b_enc)`, decoder `h_rec = f @ W_dec + b_dec`, loss = `MSE + lambda * ||f||_1`, avec normalisation des colonnes de `W_dec` a chaque step. `n_sae = 8 > 5`.

3. **Mesurer** : pour chaque feature SAE, l'originale la plus alignee (cosine) et son **active rate**. Compter (a) features originales recuperees (cosine > 0.5 et active), (b) **dead features** (active rate ≈ 0).

4. **SAE TopK** : remplacer la penalite L1 par un keep-**top-K** activations par exemple (mettre a zero toutes sauf les K plus grandes de `f`), sans terme L1. Re-entrainer.

5. **Comparer L1 vs TopK** : reconstruction loss, nombre de dead features, taux de recuperation. Le cours dit que TopK **evite le shrinkage** (L1 sous-estime les magnitudes) et stabilise. Le verifier : comparer la magnitude moyenne des features actives reconstruites.

### Criteres de reussite

- [ ] Le SAE L1 est correct (ReLU, L1, normalisation colonnes decoder) et reproduit le faible taux de recovery + dead features du cours
- [ ] Le SAE TopK est implemente (masque top-K, pas de L1)
- [ ] Les deux sont compares sur recon loss / dead features / recovery
- [ ] L'effet shrinkage L1 (magnitudes sous-estimees) vs TopK est mesure
- [ ] Code commente (POURQUOI L1 cree du shrinkage et des dead features, POURQUOI normaliser W_dec)
