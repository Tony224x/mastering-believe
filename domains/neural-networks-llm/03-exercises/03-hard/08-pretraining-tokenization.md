# Exercices Hard — Jour 8 : Pretraining & Tokenization

---

## Exercice 7 : Entropie de Shannon, compression et borne theorique d'un tokenizer

### Objectif

Relier la tokenization a la theorie de l'information : mesurer l'entropie d'ordre 0 et d'ordre 1 d'un corpus, construire un code de Huffman, et montrer que la cross-entropy d'un modele de langage est une borne sur la compressibilite du texte.

### Consigne

1. **Entropie d'ordre 0** : pour un corpus, calculer `H_0 = -sum p(c) log2 p(c)` (en bits/symbole), ou `p(c)` est la frequence empirique de chaque caractere. C'est le nombre minimal de bits/symbole d'un code sans memoire.

2. **Codage de Huffman** : implementer Huffman from scratch (file de priorite via `heapq`, fusion des 2 noeuds les moins frequents). Encoder le corpus et mesurer la longueur moyenne reelle `L_huffman` (bits/symbole). Verifier l'encadrement de Shannon : `H_0 <= L_huffman < H_0 + 1`.

3. **Entropie d'ordre 1 (conditionnelle)** : calculer `H_1 = -sum p(c_{t-1}, c_t) log2 p(c_t | c_{t-1})`. Montrer que `H_1 <= H_0` (conditionner sur le contexte ne peut que reduire l'incertitude). Quantifier le gain en bits.

4. **Lien LM <-> compression** : un modele de langage qui atteint une cross-entropy de `X` bits/token peut, via codage arithmetique, compresser le texte a ~`X` bits/token. Reprendre le bigramme de l'Ex. 5 medium, calculer ses bits/token sur un test, et comparer a `H_0` (ordre 0) et a un modele uniforme. Montrer la hierarchie : `uniforme >= H_0 >= bigramme >= entropie reelle du langage`.

5. Question : pourquoi dit-on que "mieux predire = mieux compresser" et que la course aux LLMs est, au fond, une course a la compression du savoir humain ? (Indice : Shannon — la longueur de code optimale d'un symbole de proba `p` est `-log2 p`.)

### Criteres de reussite

- [ ] `H_0` calcule correctement (bits/symbole)
- [ ] Huffman implemente from scratch, encadrement `H_0 <= L_huffman < H_0 + 1` verifie
- [ ] `H_1 <= H_0` verifie, gain en bits quantifie
- [ ] La hierarchie uniforme >= H_0 >= bigramme est demontree numeriquement
- [ ] L'explication LM = compression est correcte (longueur optimale = -log2 p)

---

## Exercice 8 : Estimation des coefficients d'une scaling law par regression (loi de puissance)

### Objectif

A partir de "mesures" synthetiques de loss en fonction de la taille du modele, retrouver par regression les coefficients d'une loi de puissance (comme l'ont fait Kaplan et Hoffmann), et extrapoler.

### Consigne

1. **Generer des donnees synthetiques** : supposer la vraie loi `L(N) = E + A / N^alpha` avec `E=1.7, A=400, alpha=0.34`. Generer des points `(N_i, L_i)` pour `N` allant de `1e6` a `1e10` (echelle log), avec un petit bruit gaussien multiplicatif (~2%) pour simuler la mesure.

2. **Estimer alpha et A** (composante de puissance) : sur la partie `A/N^alpha` (en soustrayant une estimation de `E`), passer en log-log : `log(L - E) = log(A) - alpha * log(N)`. C'est une DROITE. Faire une regression lineaire des moindres carres (formules fermees `slope`/`intercept`, pas de sklearn) pour retrouver `alpha` (pente) et `A` (intercept).
   - Comme `E` est inconnu, balayer une grille de `E` candidats et garder celui qui donne le meilleur ajustement lineaire en log-log (R^2 max). Tu dois retrouver `E ≈ 1.7, alpha ≈ 0.34, A ≈ 400` (tolerance ~10%).

3. **Qualite de l'ajustement** : calculer le `R^2` du fit final et l'erreur relative sur chaque coefficient retrouve vs verite terrain.

4. **Extrapolation** : utiliser la loi ajustee pour PREDIRE la loss d'un modele 10x plus gros que le plus grand point d'entrainement. Comparer a la vraie loi. C'est exactement ce que font les labs : ajuster sur de petits modeles, extrapoler le comportement des gros (pour decider d'investir des millions avant d'entrainer).

5. Question : pourquoi les scaling laws sont-elles des lois de PUISSANCE (lineaires en log-log) et pas exponentielles ? Que signifie un `alpha` plus grand ? (Indice : `alpha` = vitesse de decroissance de la loss avec la taille — plus il est grand, plus chaque doublement de `N` rapporte.)

### Criteres de reussite

- [ ] Donnees synthetiques generees selon `L(N) = E + A/N^alpha` avec bruit
- [ ] Regression lineaire en log-log implementee (formules fermees) + balayage de `E`
- [ ] Coefficients retrouves a ~10% pres (`E≈1.7, alpha≈0.34, A≈400`)
- [ ] `R^2` du fit calcule et eleve (> 0.98)
- [ ] Extrapolation a 10x faite et coherente avec la vraie loi ; role de `alpha` explique
