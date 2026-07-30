# Exercices Hard — Jour 19 : Quantization

---

## Exercice 7 : GPTQ-lite from scratch (error compensation via Hessienne)

### Objectif

Implementer une version minimale mais fidele de GPTQ : quantizer une couche lineaire colonne par colonne en compensant l'erreur sur les colonnes restantes via l'inverse de la Hessienne, et battre le round-to-nearest (RTN).

### Consigne

1. **Setup** : une couche lineaire `W` de shape `(d_out, d_in)` (ex 64x256) et un dataset de calibration `X` de shape `(n_calib, d_in)` (ex 512x256), gaussien. La sortie cible est `Y = X @ W.T`.

2. **Hessienne** : `H = 2 * X.T @ X` de shape `(d_in, d_in)`. Ajouter un **dampening** sur la diagonale pour la stabilite : `H += lambda * mean(diag(H)) * I` avec `lambda ≈ 0.01` (sinon `H` est mal conditionnee). Calculer `H_inv = inv(H)`.

3. **RTN baseline** : quantizer `W` directement en INT4 symetrique per-channel (round-to-nearest, pas de compensation). Mesurer l'erreur de reconstruction de sortie `||Y - Y_q||^2`.

4. **GPTQ-lite** : pour chaque colonne `j` de `W` (de gauche a droite) :
   - Quantizer la colonne `j` (round vers INT4 + dequantize) -> erreur `e_j = W[:, j] - W_q[:, j]`.
   - Propager cette erreur aux colonnes restantes `j+1..d_in` proportionnellement a `H_inv[j, j+1:] / H_inv[j, j]` (regle OBS). Concretement, mettre a jour `W[:, k] += e_j * (H_inv[j, k] / H_inv[j, j])` pour `k > j`.
   - Verrouiller la colonne `j` (elle ne bouge plus).
   - Mesurer l'erreur de sortie finale `||Y - Y_gptq||^2`.

5. **Comparaison** : GPTQ-lite doit avoir une erreur de sortie **significativement plus faible** que RTN, a memes bits (INT4). Afficher le ratio.

6. **Analyse** :
   - Pourquoi compenser l'erreur sur les colonnes suivantes aide-t-il ? (la sortie depend de la **combinaison** des colonnes via `X`)
   - Que se passe-t-il si on enleve le dampening ? (`H` singuliere ou mal conditionnee)
   - Effet de l'ordre des colonnes : essayer un ordre par importance decroissante (`act-order` : trier par `diag(H)` decroissant). Gain ?

### Criteres de reussite

- [ ] La Hessienne `H = 2 X.T X` + dampening est correcte et inversible
- [ ] RTN baseline implemente et mesure l'erreur de sortie
- [ ] GPTQ-lite quantize colonne par colonne avec compensation via `H_inv`
- [ ] GPTQ-lite bat RTN sur l'erreur de sortie a memes bits (ratio affiche)
- [ ] L'effet du dampening et de `act-order` est teste et explique
- [ ] Code commente avec le POURQUOI (lien OBS/OBQ)

---

## Exercice 8 : Courbe perplexite-proxy vs bits/poids

### Objectif

Reproduire en miniature **la** figure du cours (qualite vs bits/poids) sur un modele jouet, et retrouver le plateau Q5/Q4 + le cliff sous Q3.

### Consigne

1. **Modele jouet end-to-end** : construire un petit "modele" qui mappe un input vers une distribution sur un vocab via une matrice d'embedding `E (vocab, d)` et une matrice de sortie. Concretement : un mini-MLP `x -> ReLU(x @ W1) @ W2 -> logits` avec `W1 (d_in, d_h)`, `W2 (d_h, vocab)`, initialise pour produire des predictions non-triviales sur un petit dataset synthetique. Calculer la **cross-entropy** de reference (FP32) sur ce dataset : c'est ton "perplexite-proxy" baseline.

2. **Quantizer les poids a differents bit-widths** : implementer un quantizer per-group (g=64) symetrique parametre par `n_bits` dans {8, 6, 5, 4, 3, 2}. Pour chaque bit-width, quantizer `W1` et `W2`, recalculer la cross-entropy sur le meme dataset.

3. **Tracer (afficher) la courbe** : cross-entropy (ou perplexite = exp(CE)) en fonction de bits/poids. Inclure aussi le point "Q2 sans groupes" (per-tensor) pour montrer la chute supplementaire sans granularite.

4. **Verifier les phenomenes du cours** :
   - Plateau quasi-plat de 8 a 4 bits (perte negligeable).
   - Coude visible entre 4 et 3 bits.
   - Cliff brutal a 2 bits.

5. **Effet "taille du modele"** : refaire avec un `d_h` 4x plus grand (modele "plus gros"). Le gros modele encaisse-t-il mieux la quantization agressive (perte relative plus faible a 3-4 bits) ? Expliquer le lien avec la redondance interne.

### Criteres de reussite

- [ ] Le modele jouet produit une cross-entropy de reference non-triviale en FP32
- [ ] Le quantizer per-group parametre par `n_bits` est correct pour tous les bit-widths testes
- [ ] La courbe perplexite-proxy vs bits/poids est affichee
- [ ] Le plateau (8->4), le coude (4->3) et le cliff (3->2) sont observables
- [ ] L'experience "gros modele encaisse mieux" est faite et l'effet (meme faible) est discute
- [ ] Code commente

---

## Exercice 9 : Double quantization (QLoRA) — quantizer les scales

### Objectif

Implementer la double quantization de QLoRA : on quantize les poids per-block, puis on **quantize les scales eux-memes**, et on chiffre l'economie reelle en bits/poids.

### Consigne

1. **Premiere couche (poids)** : quantizer une matrice `W (1024, 1024)` gaussienne en NF4 (ou INT4 per-block) avec `block_size = 64`. Cela produit un tableau de `n_blocks = W.size / 64` scales FP32 (un par bloc).

2. **Compter le cout sans double quant** :
   - 4 bits par poids (les codes)
   - + 32 bits (FP32) par bloc de 64 poids pour le scale -> `32/64 = 0.5` bit/poids
   - Total : `4.5` bits/poids effectif.

3. **Deuxieme couche (double quant)** : quantizer les **scales** eux-memes :
   - Regrouper les scales en blocs de 256 (seconde couche de blocs).
   - Quantizer chaque scale en 8 bits (FP8-like : ici un INT8 + un scale-de-scale FP32 par bloc-de-256 suffit pour la pedagogie).
   - Compter le nouveau cout des scales : `8 bits par scale / 64 poids` + `32 bits par bloc-de-256-scales / (256*64 poids)`.

4. **Chiffrer l'economie** :
   - Bits/poids avant double quant : 4.5.
   - Bits/poids apres : `4 + 8/64 + 32/(256*64)` = ? (doit donner ~4.127 bits/poids).
   - Comparer avec le chiffre du QLoRA paper (~0.373 bit/poids economise -> de ~4.5 a ~4.127).

5. **Mesurer la degradation** : la double quantization introduit-elle une erreur **supplementaire** notable sur la reconstruction de `W` (par rapport au simple NF4) ? Mesurer `MSE(W, W_dequant_simple)` vs `MSE(W, W_dequant_double)`. L'idee de QLoRA : l'erreur ajoutee par la quantization des scales est negligeable car les scales varient peu.

6. **Analyse** : pourquoi quantizer les scales coute-t-il si peu en erreur ? (les scales sont peu nombreux, lisses, et leur erreur relative se propage faiblement). Quand la double quant deviendrait-elle dangereuse ? (blocs tres petits -> beaucoup de scales -> leur quantization compte plus).

### Criteres de reussite

- [ ] La quantization NF4/INT4 per-block (b=64) produit le bon nombre de scales
- [ ] Le cout sans double quant est correctement chiffre a 4.5 bits/poids
- [ ] La double quantization des scales (blocs de 256, 8 bits) est implementee
- [ ] Le cout apres double quant est ~4.127 bits/poids (calcul detaille)
- [ ] La MSE supplementaire due a la double quant est mesuree et jugee negligeable
- [ ] L'analyse explique POURQUOI l'economie est quasi gratuite

---

## Exercice 10 : MXFP4, carte de precision mixte, et le vrai argument du QAT (Kimi K3, 2026)

### Objectif

Implementer un format **MX (Microscaling)**, constater qu'il est *moins* precis que NF4 a nombre de bits egal, et comprendre pourquoi il gagne quand meme — puis mesurer l'ecart train/inference que le QAT existe pour supprimer.

### Consigne

Un format MX, c'est trois choses fixees par la spec OCP : un bloc de **32** elements, **une** echelle partagee contrainte a etre une **puissance de deux** (E8M0 : 8 bits d'exposant, zero mantisse), et un mini-format flottant par element. Pour MXFP4 c'est **FP4 E2M1**, dont les 8 magnitudes representables sont `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`.

**Partie A — implementer MXFP4**

1. Ecrire `round_to_fp4_e2m1(x)` : arrondi au plus proche element de la table de magnitudes, signe conserve.

2. Ecrire `quantize_mx(x, elem_max=6.0, block=32)` :
   ```
   amax        = max|bloc|
   shared_exp  = floor(log2(amax)) - floor(log2(elem_max))
   scale       = 2 ** shared_exp
   q           = round_to_fp4(clip(bloc / scale, -elem_max, elem_max))
   ```
   Pourquoi `floor` et non `round` sur l'exposant ? Que se passerait-il si l'echelle etait arrondie vers le haut ?

3. **Comparaison a bloc egal.** Sur `x ~ N(0, 1)` de shape `(1024, 1024)`, comparer la MSE relative de MXFP4, de l'INT4 lineaire par bloc et de NF4 — les trois avec `block = 32`, pour que la comparaison porte sur le **format** et non sur la quantite de metadonnees d'echelle autorisee.

4. Expliquer le classement obtenu. Deux causes distinctes : (a) l'echelle en puissance de deux gaspille jusqu'a un facteur 2 de dynamique quand `amax` tombe juste au-dessus d'une puissance de deux ; (b) NF4 place en plus ses 16 niveaux aux quantiles d'une gaussienne. Si MXFP4 est le moins precis des trois, sur quel autre axe gagne-t-il ? (Indice : que coute la dequantification quand l'echelle est un simple decalage de bits ?)

**Partie B — la carte de precision**

5. Kimi K3 ne quantifie pas tout. Reconstituer le budget memoire d'une couche MoE avec `d = 7168`, `latent = 3584`, `d_ff = 3072`, `896` experts routes et `2` experts partages, en comptant 2 matrices par expert :
   - experts routes en MXFP4 : bits effectifs par poids = `4 + 8/32 = 4.25`
   - experts partages, projections latentes, routeur : BF16 (16 bits)

   Reporter le total en Go et le ratio face au tout-BF16.

6. Quelle fraction des parametres representent les experts routes ? En deduire pourquoi laisser le routeur et les experts partages en haute precision ne coute presque rien — et pourquoi ce sont precisement eux qu'il faut proteger.

7. Les activations, elles, sont en **MXFP8 E4M3**, dont le maximum representable est **448**. Relier ce chiffre a l'exercice 9 du jour 9 (SiTU-GLU). Quelle contrainte le format d'activation impose-t-il a la fonction d'activation ?

**Partie C — l'ecart que le QAT supprime**

8. Construire une tete de sortie jouet : `h` de shape `(256, 512)`, `W` de shape `(512, 4096)`. Calculer les logits avec `W` en pleine precision ("entrainement") et avec `quantize_mx(W)` ("inference"), puis les distributions softmax correspondantes.

9. Des poids aleatoires donnent un softmax presque plat, ou deux politiques quelconques se ressemblent. **Balayer une echelle de logits** dans `{1, 3, 6, 10}` et reporter, pour chaque valeur : la probabilite top-1 moyenne (a quel point la politique est confiante), la `KL(p_train || p_infer)`, l'accord top-1 et le recouvrement top-5.

10. Constater que l'accord top-1 et le recouvrement top-5 **ne bougent pas** avec l'echelle, alors que la KL grandit. Pourquoi ? Que faut-il en conclure sur la lecture de ces trois metriques ?

11. **L'argument central.** En generation de texte, cet ecart est une question de qualite. En RL, c'est une question de **correction** : les rollouts sont echantillonnes depuis le moteur d'inference, mais le gradient est calcule contre les poids d'entrainement. Expliquer en quoi l'hypothese *on-policy* de l'algorithme devient fausse, et pourquoi faire tourner le QAT pendant tout le post-training (SFT **et** RL) rend cet ecart nul **par construction**.

12. **Honnetete methodologique.** Cette experience mesure l'ecart ; elle n'entraine rien, donc elle ne demontre pas que le QAT recupere la qualite perdue. Formuler explicitement ce qui est mesure ici et ce qui reste une affirmation du papier.

### Criteres de reussite

- [ ] `round_to_fp4_e2m1` et `quantize_mx` sont corrects ; le `floor` est justifie (un arrondi vers le haut de l'exposant ferait clipper le maximum du bloc)
- [ ] A bloc egal (32), le classement obtenu est NF4 < INT4 lineaire < MXFP4 en MSE relative — MXFP4 est bien le **moins** precis
- [ ] L'avantage de MX est correctement identifie : echelle = decalage de bits, bloc et format figes par la spec, donc dequantification integrable au datapath du Tensor Core. Precision par bit echangee contre debit.
- [ ] Budget memoire : experts routes ~99 % des parametres, total ~10.8 Go contre ~39.7 Go en tout-BF16 (~3.7x)
- [ ] Le lien MXFP8 E4M3 (max 448) -> necessite d'une activation bornee est fait explicitement
- [ ] Le balayage de l'echelle des logits montre une KL croissante a accord top-1 constant, et l'explication est donnee (une remise a l'echelle ne peut pas reordonner des logits)
- [ ] L'argument RL est correctement enonce : sans QAT, la politique qui echantillonne et la politique qui apprend ne sont pas la meme, donc l'update est silencieusement off-policy
- [ ] La limite de l'experience est enoncee sans contournement
