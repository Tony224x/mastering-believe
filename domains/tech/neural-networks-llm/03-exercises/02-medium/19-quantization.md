# Exercices Medium — Jour 19 : Quantization

---

## Exercice 4 : Quantization par groupe (group-wise) from scratch

### Objectif

Implementer la quantization per-group g=128 (le sweet spot GPTQ/AWQ/NF4) et mesurer le gain face au per-tensor et au per-channel.

### Consigne

1. Generer une matrice de poids "realiste" : `W` de shape `(512, 512)`, gaussienne, mais avec **3 colonnes outliers** dont l'echelle est 30x plus grande que les autres (simule l'heterogeneite reelle d'un LLM).

2. Implementer trois quantizers symetriques INT4 (signed, plage [-7, 7]) en numpy :
   - `quantize_per_tensor(W)` : un seul scale global
   - `quantize_per_channel(W, axis)` : un scale par ligne
   - `quantize_per_group(W, group_size=128)` : decouper **chaque ligne** en blocs de `group_size` elements, un scale par bloc. Gerer le padding si la ligne n'est pas divisible.

3. Pour chacun, calculer et comparer :
   - MSE de reconstruction
   - MSE relative (`MSE / mean(W^2)`)
   - Overhead memoire en bits/poids (nombre de scales FP16 ajoutes / nombre de poids)

4. **Analyse** :
   - Le per-group bat-il le per-channel ? De combien ?
   - Quel est le compromis erreur/overhead de chaque granularite ?
   - Faire varier `group_size` dans {32, 64, 128, 256} et tracer (afficher) MSE vs overhead. Ou est le coude ?

### Criteres de reussite

- [ ] Les 3 quantizers sont corrects (round + clip + scale par granularite)
- [ ] Le per-group gere le padding pour des lignes non divisibles par `group_size`
- [ ] L'ordre attendu est respecte : per-tensor (pire) > per-channel > per-group (meilleur)
- [ ] L'overhead est chiffre en bits/poids et croit quand `group_size` diminue
- [ ] L'analyse identifie `group_size=128` comme un bon compromis (faible MSE, ~3% overhead)
- [ ] Code commente avec le POURQUOI de chaque etape

---

## Exercice 5 : Outliers et migration SmoothQuant-style

### Objectif

Reproduire le phenomene "un outlier casse INT8 naif" puis implementer la migration de magnitude SmoothQuant (`Y = X@W = (X/s) @ (s*W)`).

### Consigne

1. Construire une matrice `X` de shape `(256, 256)`, gaussienne d'ecart-type ~0.5, puis injecter **0.1% d'outliers** de magnitude 50 a des positions aleatoires.

2. **Baseline naif** : quantizer `X` en INT8 symetrique per-tensor. Mesurer la MSE **sur les 99.9% d'entrees normales uniquement** (exclure les outliers du calcul d'erreur — c'est la qualite qui compte vraiment).

3. **SmoothQuant-like** : pour chaque colonne `j`, calculer `s_j = max(|X[:, j]|)^alpha` avec `alpha=0.5`. Diviser `X` par `s` (per-colonne), quantizer le `X_smoothed`, dequantizer, puis re-multiplier par `s`. Mesurer la MSE sur les entrees normales.

4. **Verification d'invariance** : montrer numeriquement que `(X/s) @ (s*W) == X @ W` (au float-error pres) pour une matrice `W` aleatoire. C'est ce qui rend l'astuce gratuite : le scale se compense.

5. **Balayage de alpha** : faire varier `alpha` dans {0.0, 0.25, 0.5, 0.75, 1.0}. `alpha=0` = pas de migration (= naif), `alpha=1` = migration totale vers les poids. Quel `alpha` minimise l'erreur **conjointe** poids+activations (en supposant que `W` aussi doit etre quantize) ? Pourquoi le midpoint 0.5 est-il un bon defaut ?

### Criteres de reussite

- [ ] Les outliers sont injectes a ~0.1% des positions, magnitude 50
- [ ] La MSE est mesuree **hors outliers** (sur les entrees normales)
- [ ] SmoothQuant-like reduit la MSE sur les entrees normales de plusieurs ordres de grandeur vs naif
- [ ] L'invariance `(X/s)@(s*W) == X@W` est verifiee numeriquement (erreur < 1e-4)
- [ ] Le balayage de `alpha` montre que les extremes (0 ou 1) sont sous-optimaux quand poids ET activations sont quantizes
- [ ] Code commente

---

## Exercice 6 : NF4 from scratch et comparaison avec INT4 lineaire

### Objectif

Construire un codebook NF4 (NormalFloat 4-bit) a partir des quantiles d'une gaussienne et montrer qu'il bat INT4 lineaire sur des poids gaussiens (le coeur de QLoRA).

### Consigne

1. **Construire le codebook NF4** (16 niveaux) :
   - Calculer 16 quantiles d'une `N(0,1)` : `levels = sqrt(2) * erfinv(2*p - 1)` pour `p = (i + 0.5)/16`, `i = 0..15`.
   - `erfinv` n'est pas dans numpy : implementer l'approximation de Winitzki (cf code du Jour 19) OU une approche par dichotomie sur `erf` (numpy n'a pas `erf` non plus -> approximer `erf` par une formule, ou utiliser la CDF via une table). Documenter le choix.
   - Normaliser pour que `max(|levels|) = 1`.
   - Verifier : les niveaux sont plus **denses pres de 0** et plus **espaces dans les queues**.

2. **Quantizer per-block (block=64)** :
   - `quantize_nf4(W)` : par bloc de 64, `abs_max = max(|bloc|)`, `scaled = bloc / abs_max` (dans [-1, 1]), puis snapper chaque valeur au niveau le plus proche du codebook (argmin de distance).
   - `dequantize_nf4(codes, abs_max)` : `levels[codes] * abs_max`.

3. **Baseline INT4 lineaire per-block (block=64)** : meme structure mais codebook lineaire `{-7..7}/7 * abs_max`.

4. **Comparer sur des poids gaussiens** `W` de shape `(1024, 1024)` :
   - MSE NF4 vs MSE INT4 lineaire.
   - NF4 doit gagner (~10-25% selon le block size).

5. **Robustesse a la distribution** : refaire la comparaison sur une distribution **uniforme** `U(-1, 1)`. Qui gagne maintenant ? Pourquoi NF4 perd son avantage sur de l'uniforme ? (Indice : NF4 est optimise pour le gaussien.)

### Criteres de reussite

- [ ] Le codebook NF4 a 16 niveaux, denses pres de 0, normalises a [-1, 1]
- [ ] La methode `erfinv`/quantile est documentee et numeriquement raisonnable
- [ ] Le quantizer per-block gere le padding
- [ ] Sur des poids gaussiens, NF4 a une MSE plus faible que INT4 lineaire
- [ ] Sur de l'uniforme, l'avantage NF4 disparait (voire s'inverse) — et l'analyse l'explique
- [ ] Code commente avec le POURQUOI de chaque etape
