# Exercices Hard — Jour 10 : Fine-tuning & Alignment

---

## Exercice 7 : Deriver DPO depuis l'objectif RLHF contraint en KL

### Objectif

Reconstruire la derivation theorique qui transforme l'objectif RLHF (maximiser une reward sous contrainte KL) en la loss DPO, puis valider numeriquement chaque etape de la derivation.

### Consigne

**Partie A — derivation papier (a ecrire dans des commentaires/docstrings).**

1. Partir de l'objectif RLHF :
   ```
   max_π  E_{x, y~π}[ r(x, y) ] - β * KL(π(·|x) || π_ref(·|x))
   ```
   Montrer que la solution optimale est :
   ```
   π*(y|x) = (1/Z(x)) * π_ref(y|x) * exp(r(x, y) / β)
   ```
   ou `Z(x) = Σ_y π_ref(y|x) exp(r(x,y)/β)` est la fonction de partition.
   (Indice : poser le Lagrangien, deriver par rapport a π(y|x), utiliser le multiplicateur pour la contrainte Σπ=1.)

2. Inverser cette relation pour exprimer la reward en fonction des policies :
   ```
   r(x, y) = β * log(π*(y|x) / π_ref(y|x)) + β * log Z(x)
   ```

3. Injecter cette expression dans le modele de preference de **Bradley-Terry** :
   ```
   P(y_w > y_l | x) = σ(r(x, y_w) - r(x, y_l))
   ```
   Montrer que le terme `β log Z(x)` **s'annule** (il ne depend pas de y), ce qui donne la loss DPO sans avoir besoin d'estimer Z(x) ni d'entrainer un reward model.

**Partie B — validation numerique.**

4. Sur un petit espace discret (vocab de 6 actions, une seule "prompt"), choisir une reward arbitraire `r` et une `π_ref` arbitraire. Calculer numeriquement :
   - `π*` via la formule fermee de l'etape 1 (verifier que c'est bien une distribution : somme = 1)
   - Verifier que `π*` maximise reellement l'objectif RLHF : comparer son score a celui de 1000 distributions aleatoires (perturbations de π*)
5. Verifier l'etape 2 : recalculer `r` a partir de `π*` et `π_ref` (a la constante `β log Z` pres) et montrer que les **differences** `r(y_w) - r(y_l)` sont exactes (independantes de Z).
6. Verifier l'etape 3 : montrer que la proba de preference Bradley-Terry calculee avec les rewards "vraies" et avec les rewards "reconstruites depuis les policies" sont identiques.

### Criteres de reussite

- [ ] La derivation des etapes 1-3 est ecrite clairement et correctement
- [ ] L'annulation de `β log Z(x)` est explicitement justifiee
- [ ] `π*` calculee numeriquement est une distribution valide et maximise l'objectif RLHF (vs perturbations aleatoires)
- [ ] Les differences de reward reconstruites depuis les policies matchent les vraies (a la constante pres)
- [ ] La proba Bradley-Terry est identique avec les deux jeux de rewards (ecart < 1e-10)

---

## Exercice 8 : Memoire d'entrainement — full FT vs LoRA vs QLoRA (modele d'allocation)

### Objectif

Construire un modele quantitatif de l'empreinte memoire GPU d'un fine-tuning, de l'expliquer composant par composant (poids, gradients, etats Adam, activations), et l'utiliser pour decider quelle methode tient sur quelle GPU.

### Consigne

1. Implementer une fonction `memory_breakdown(n_params, method, dtype_bytes, ...)` qui renvoie un dictionnaire avec, en octets :
   - **weights** : `n_params * dtype_bytes`
   - **gradients** : seulement pour les params ENTRAINABLES
   - **optimizer states (Adam)** : 2 moments (m, v), classiquement en fp32 (4 bytes), pour les params entrainables uniquement
   - **master weights** (si entrainement mixte fp16 + master fp32) : copie fp32 des params entrainables
   - Couvrir 3 methodes :
     - `full` : tous les params entrainables, mixed precision (poids fp16, gradients fp16, Adam fp32, master fp32)
     - `lora` : poids de base fp16 GELES (pas de gradient/Adam dessus), seuls les adaptateurs (≈ `2*d*r*n_layers*4_projections`) sont entrainables
     - `qlora` : poids de base en 4-bit (0.5 byte/param) GELES, adaptateurs entrainables en fp16/Adam fp32

2. Appliquer le modele a **LLaMA 7B** (7e9 params, suppose tout dans les couches lineaires pour simplifier ; adaptateurs LoRA r=16 sur Q,K,V,O, 32 couches, d=4096) :
   - Donner le breakdown des 3 methodes en GB
   - Verifier l'ordre de grandeur attendu : full FT ≈ 70-80 GB (ne tient PAS sur une seule A100 80GB avec les activations), LoRA ≈ 16-18 GB, QLoRA ≈ 5-7 GB

3. **Activations** : ajouter un terme grossier d'activations `≈ batch * seq_len * d_model * n_layers * c` (avec `c` une constante ~ 2 bytes * facteur). Montrer comment le **gradient checkpointing** echange du compute contre de la memoire (divise les activations stockees par ~sqrt(n_layers)). Recalculer si full FT tient avec checkpointing.

4. **Decision** : ecrire une fonction `fits(method, gpu_memory_gb)` et produire un tableau : pour {full, lora, qlora} x {A100 40GB, A100 80GB, RTX 4090 24GB, H100 80GB}, indiquer OUI/NON. Commenter quelle config permet de fine-tuner un 7B sur une 4090 grand public.

5. **Trade-off qualite** : discuter (en commentaire) pourquoi QLoRA perd tres peu de qualite malgre le 4-bit (la quantization 4-bit NF4 + double quantization preserve l'essentiel ; les adaptateurs en pleine precision compensent).

### Criteres de reussite

- [ ] Le breakdown distingue clairement weights / gradients / Adam / master / activations
- [ ] Les gradients et l'Adam ne sont comptes QUE sur les params entrainables
- [ ] Les ordres de grandeur LLaMA 7B sont coherents (full ~70-80 GB, LoRA ~16-18 GB, QLoRA ~5-7 GB)
- [ ] L'effet du gradient checkpointing sur les activations est modelise et chiffre
- [ ] Le tableau `fits` est correct et la conclusion "QLoRA sur 4090" est justifiee
- [ ] L'explication du faible cout qualite de QLoRA est correcte
