# Exercices Medium — Jour 10 : Fine-tuning & Alignment

---

## Exercice 4 : LoRA from scratch — init, forward, merge

### Objectif

Implementer une couche LoRA complete en NumPy et verifier ses 3 proprietes contractuelles : transparence a l'init, equivalence apres merge, economie de parametres.

### Consigne

1. Implementer :

```python
class LoRALinear:
    """y = x @ W.T + (alpha / r) * (x @ A.T) @ B.T
    W: (d_out, d_in) GELE. A: (r, d_in) init N(0, 0.02). B: (d_out, r) init ZEROS."""
    def forward(self, x): ...
    def merge(self):   # retourne W_merged = W + (alpha/r) * B @ A
    def trainable_params(self): ...
```

2. Verifier les 3 proprietes (d_in=64, d_out=64, r=8, alpha=16, seed fixe) :
   - **Transparence a l'init** : `forward(x) == x @ W.T` EXACTEMENT (B=0 → l'adapter ne fait rien). Question piege en commentaire : pourquoi B=0 et pas A=0, ou les deux a 0 ? (Les deux a 0 → gradient de A nul a jamais ; A=0 et B random marche aussi mais la convention est B=0 ; A et B random → le modele part perturbe.)
   - **Equivalence post-merge** : apres avoir simule un entrainement (remplir A et B de valeurs aleatoires), `x @ merge().T == forward(x)` a 1e-10 pres
   - **Economie** : trainable = `r * (d_in + d_out)` ; calculer le ratio vs full fine-tuning pour (4096, 4096, r=16) → ~0.78%

3. Verifier que le rang de `B @ A` est exactement r (via `np.linalg.matrix_rank`) pour r=8 < d, et expliquer en une phrase pourquoi la mise a jour est de rang AU PLUS r.

4. Tableau : r ∈ {1, 4, 16, 64} → params entrainables et % du full FT pour une matrice 4096x4096.

### Criteres de reussite

- [ ] Transparence a l'init exacte (difference == 0, pas juste petite)
- [ ] Merge == forward a 1e-10 apres "entrainement" simule
- [ ] La reponse au piege de l'init (B=0 vs A=0 vs les deux) est correcte et complete
- [ ] Le rang de BA est verifie == r
- [ ] Le tableau des ratios est correct (r=16 sur 4096² → 0.78%)

---

## Exercice 5 : La loss DPO a la main puis en code

### Objectif

Calculer la loss DPO sur des valeurs imposees (a la main d'abord), puis verifier en code chaque comportement attendu — comprendre ce que DPO optimise VRAIMENT.

### Consigne

Rappel : `L_DPO = -log sigmoid(beta * [(logp_pol(y_w) - logp_ref(y_w)) - (logp_pol(y_l) - logp_ref(y_l))])`
ou y_w = reponse preferee (chosen), y_l = rejetee.

1. **A la main** (puis verifier en code, tolerance 1e-6) avec beta=0.1 :
   - cas A : logp_pol(y_w)=-10, logp_ref(y_w)=-10, logp_pol(y_l)=-10, logp_ref(y_l)=-10 → marge=0, L = -log(0.5) ≈ 0.6931
   - cas B : logp_pol(y_w)=-8, logp_ref(y_w)=-10, logp_pol(y_l)=-12, logp_ref(y_l)=-10 → marge=0.1*(2-(-2))=0.4, L = -log sigmoid(0.4) ≈ 0.5130
   - cas C (la politique s'est degradee) : logp_pol(y_w)=-12, logp_pol(y_l)=-8, refs a -10 → L ≈ 0.9130

2. Implementer `dpo_loss(logp_pol_w, logp_pol_l, logp_ref_w, logp_ref_l, beta)` vectorisee, verifier les 3 cas.

3. **Gradient par differences finies** sur logp_pol_w et logp_pol_l (eps=1e-6) pour le cas B :
   - verifier `dL/dlogp_pol_w < 0` (la loss pousse a AUGMENTER la logprob du chosen)
   - verifier `dL/dlogp_pol_l > 0` (et a DIMINUER celle du rejected)
   - verifier que les deux gradients ont la meme magnitude (|grad_w| == |grad_l| a 1e-9 — symetrie de la marge)

4. **Effet de beta** : pour la marge brute du cas B, tracer L et |gradient| pour beta ∈ {0.01, 0.1, 0.5, 1.0, 5.0}. Verifier : le gradient (en valeur absolue) par rapport a la marge sature quand beta*marge est grand (sigmoid saturee) — afficher le facteur `sigmoid(-beta*marge)` qui module le gradient.

5. **Implicit reward** : calculer `r_hat = beta * (logp_pol - logp_ref)` pour chosen et rejected du cas B et verifier que la marge de reward implicite = 0.4. Commenter : DPO entraine un reward model implicite sans en entrainer un explicitement.

### Criteres de reussite

- [ ] Les 3 calculs a la main correspondent au code a 1e-6
- [ ] Les signes ET la symetrie des gradients sont verifies numeriquement
- [ ] La modulation du gradient par `sigmoid(-beta * marge)` est exhibee (les exemples deja bien classes recoivent un petit gradient)
- [ ] Le reward implicite est calcule et l'interpretation est correcte
- [ ] Le role de beta (force du rappel a la reference / sharpness) est explique avec les valeurs a l'appui

---

## Exercice 6 : Reward model Bradley-Terry sur preferences synthetiques

### Objectif

Entrainer un reward model minimal a partir de paires de preferences — la brique centrale de RLHF — et verifier qu'il retrouve le reward sous-jacent.

### Consigne

1. Generer les donnees (seed fixe) :
   - "reponses" = vecteurs de features x ∈ R^6, 400 paires (x_a, x_b) tirees N(0,1)
   - vrai reward cache : `r*(x) = w_true . x` avec `w_true = [2, -1, 0.5, 0, 1.5, -0.5]`
   - label de preference echantillonne selon Bradley-Terry : `P(a > b) = sigmoid(r*(x_a) - r*(x_b))` (labels BRUITES, pas deterministes — c'est realiste)

2. Implementer le reward model lineaire `r(x) = w . x` et la loss : `L = -mean(log sigmoid(r(x_pref) - r(x_rej)))`. Deriver le gradient a la main (commentaire) : `dL/dw = -mean((1 - sigmoid(dr)) * (x_pref - x_rej))`, l'implementer, et entrainer par gradient descent (lr=0.1, 500 steps) sur 320 paires de train.

3. Evaluer sur 80 paires de test :
   - accuracy de preference (le RM ordonne-t-il comme les labels ?) — attendu > 75% (les labels sont bruites, 100% est impossible)
   - accuracy contre le VRAI reward (ordonne-t-il comme r* ?) — attendu > 90%
   - cosine similarity entre w appris et w_true > 0.95

4. Verifier l'**invariance par translation** du reward : ajouter une constante c a tous les rewards ne change pas la loss (verifier numeriquement) → le reward Bradley-Terry n'est identifiable qu'a une constante pres. Commenter pourquoi c'est sans consequence pour RLHF (seules les differences comptent).

5. Bonus : courbe accuracy(test) en fonction du nombre de paires de train {20, 50, 100, 320} — combien de comparaisons faut-il pour un signal fiable ?

### Criteres de reussite

- [ ] Les labels sont echantillonnes (bruites), pas pris au argmax — et le commentaire explique pourquoi
- [ ] Le gradient analytique est correct (verifie par differences finies sur quelques coordonnees, < 1e-6)
- [ ] Accuracy vs labels > 75% ET accuracy vs vrai reward > 90% sur le test
- [ ] cosine(w, w_true) > 0.95
- [ ] L'invariance par translation est demontree numeriquement et commentee
