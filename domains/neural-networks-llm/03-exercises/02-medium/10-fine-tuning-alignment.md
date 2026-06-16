# Exercices Medium — Jour 10 : Fine-tuning & Alignment

---

## Exercice 4 : DPO from scratch — forward, gradient et mini training loop

### Objectif

Implementer la loss DPO en NumPy avec son gradient analytique, puis montrer qu'une mini policy converge vers les preferences SANS reward model ni RL.

### Consigne

1. Reprendre la formulation DPO du code du Jour 3 (`02-code/10-fine-tuning-alignment.py`) :
   ```
   margin = beta * [(logp_θ(y_w) - logp_ref(y_w)) - (logp_θ(y_l) - logp_ref(y_l))]
   L = -log σ(margin)
   ```

2. Implementer une `TinyPolicy` en NumPy : un simple vecteur de logits `z` (taille `vocab`) avec `log_softmax`. La policy de reference est une copie figee des logits initiaux.

3. Implementer **forward + gradient analytique** de la loss DPO par rapport aux logits de la policy :
   - Rappel : `d(-log σ(m))/dm = -(1 - σ(m)) = σ(m) - 1`
   - Puis chaîner via `d(margin)/d(logp_θ) = +beta` (chosen) et `-beta` (rejected)
   - Et `d(log_softmax(z)[i])/dz[j] = δ_ij - softmax(z)[j]`
   - Verifier le gradient analytique par **difference finie** (l'ecart doit etre < 1e-5)

4. Entrainer la policy (SGD, lr=0.5, 50 steps) pour preferer `chosen=token 0` a `rejected=token 4`. A chaque step, afficher `p(chosen)`, `p(rejected)`, `loss`, `margin`.

5. Analyser :
   - La probabilite de chosen monte-t-elle ? Celle de rejected baisse-t-elle ?
   - Que devient le `margin` au fil du training ? Pourquoi la loss ne tend-elle jamais exactement vers 0 ?
   - Si on augmente `beta` (0.1 → 1.0), le training est-il plus rapide ou plus instable ?

### Criteres de reussite

- [ ] La loss DPO est implementee avec une version numeriquement stable de `-log σ`
- [ ] Le gradient analytique passe le test de difference finie (ecart < 1e-5)
- [ ] Le training converge : `p(chosen)` monte, `p(rejected)` baisse
- [ ] L'effet du `beta` est analyse (amplitude du gradient)
- [ ] Le code est commente avec le POURQUOI (notamment l'init de la reference figee)

---

## Exercice 5 : LoRA — forward, backward et fusion des poids

### Objectif

Implementer un `LoRALinear` complet en NumPy (forward ET backward), verifier l'equivalence avec le full fine-tuning sur le sous-espace de rang r, et implementer la fusion des adaptateurs.

### Consigne

1. Reprendre la classe `LoRALinearNumpy` du code (W gele, A entrainable, B init a zero, `scale = alpha/r`).

2. Ajouter la **backward pass** : etant donne `dL/dy` (de shape `(batch, out)`), calculer `dL/dA` et `dL/dB`.
   - Rappel forward : `y = x @ W.T + (x @ A.T) @ B.T * scale`
   - Pose `h = x @ A.T` (shape `(batch, r)`)
   - `dL/dB = scale * dL/dy.T @ h` (shape `(out, r)`)
   - `dL/dA = scale * (dL/dy @ B).T @ x` (shape `(r, in)`)
   - W n'est PAS mis a jour (gele) — verifier qu'aucun gradient ne le touche

3. Verifier les gradients de A et B par difference finie.

4. **Fusion** : implementer `merge()` qui calcule `W_merged = W + scale * (B @ A)`. Verifier que `forward_merged(x)` (avec W_merged et sans adaptateur) donne exactement le meme resultat que `forward(x)` (avec W gele + adaptateur). Pourquoi la fusion ne coute-t-elle RIEN a l'inference ?

5. **Rang effectif** : entrainer LoRA (descente de gradient simple) pour approximer une cible `Delta = W_target - W` de rang faible. Montrer que :
   - Si `Delta` est de rang <= r, LoRA peut le capturer parfaitement
   - Si `Delta` est de rang > r, LoRA capture sa meilleure approximation de rang r (lien avec la SVD tronquee)

6. Analyser : pourquoi initialise-t-on B=0 et pas A=0 ? Que se passe-t-il si les deux sont random ?

### Criteres de reussite

- [ ] La backward pass de A et B est correcte (test difference finie < 1e-5)
- [ ] W ne recoit jamais de gradient (gele)
- [ ] La fusion donne un resultat identique (ecart < 1e-10) au forward avec adaptateur
- [ ] La demo de rang effectif montre que LoRA capture un Delta de rang <= r
- [ ] L'explication B=0 vs A=0 est correcte (au step 0, le modele LoRA = modele base)

---

## Exercice 6 : Comparer les variantes d'alignment (DPO vs IPO vs SimPO)

### Objectif

Comprendre concretement comment quelques variantes recentes modifient la loss de preference, en les implementant cote a cote sur les memes donnees.

### Consigne

On note `pi_logratio_w = logp_θ(y_w) - logp_ref(y_w)` et de meme pour `l`.

1. Implementer 3 losses sur le meme batch de log-probs synthetiques :
   - **DPO** : `L = -log σ(beta * (pi_logratio_w - pi_logratio_l))`
   - **IPO** : `L = (pi_logratio_w - pi_logratio_l - 1/(2*beta))^2` (regression au lieu de classification)
   - **SimPO** : pas de reference ! `L = -log σ(beta/|y_w| * logp_θ(y_w) - beta/|y_l| * logp_θ(y_l) - gamma)`
     (longueur-normalisee + marge `gamma`)

2. Sur un batch de 8 paires synthetiques (genere des log-probs aleatoires plausibles), calculer la loss moyenne de chaque methode.

3. **Sensibilite au reward hacking par la longueur** : construire un cas ou `y_w` est TRES long (donc `logp_θ(y_w)` tres negatif a cause de la somme sur beaucoup de tokens) mais devrait gagner. Montrer que :
   - DPO (somme non normalisee) peut etre biaise par la longueur
   - SimPO (normalise par la longueur) corrige ce biais

4. **IPO et l'overfitting** : montrer que quand `pi_logratio_w - pi_logratio_l` devient tres grand, la loss DPO sature (gradient → 0) alors que IPO continue de pousser (regression vers une cible fixe). Discuter : pourquoi IPO est plus robuste a l'overfitting des preferences deterministes ?

5. Faire un tableau recap : pour chaque methode, indiquer (a) besoin d'une reference oui/non, (b) classification vs regression, (c) gestion de la longueur, (d) un avantage et un inconvenient.

### Criteres de reussite

- [ ] Les 3 losses sont implementees avec leurs formules correctes
- [ ] Le cas "y_w tres long" montre le biais de longueur de DPO et la correction de SimPO
- [ ] La saturation du gradient DPO vs la non-saturation d'IPO est demontree numeriquement
- [ ] Le tableau recap est complet et exact (reference, classif/regression, longueur)
- [ ] L'analyse relie chaque variante a la faiblesse de DPO qu'elle corrige
