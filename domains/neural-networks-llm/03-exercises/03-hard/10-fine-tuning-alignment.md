# Exercices Hard — Jour 10 : Fine-tuning & Alignment

---

## Exercice 7 : Fine-tuning LoRA de bout en bout sur un modele gele

### Objectif

Derouler le pipeline LoRA complet : pre-entrainer un petit modele, le geler, l'adapter a une nouvelle tache en n'entrainant QUE les adapters — et prouver que les poids de base n'ont pas bouge d'un bit.

### Consigne

1. **"Pre-training"** : entrainer un MLP 2 couches (1 → 32 → 1, tanh, NumPy from scratch) a regresser `f_A(x) = sin(3x)` sur x ∈ [-2, 2] (800 steps, MSE < 0.01). C'est notre "modele de base".

2. **Tache cible** : `f_B(x) = sin(3x + 1.2) * 0.8` (proche mais differente — un "domain shift").
   Mesurer la MSE du modele de base sur la tache B (elle doit etre mauvaise, > 0.1).

3. **LoRA** : geler W1, W2 (copies de sauvegarde + jamais de mise a jour). Ajouter des adapters `A1 (r x 1), B1 (32 x r)` et `A2 (r x 32), B2 (1 x r)` avec r=2, B init a 0. Forward : `h = tanh(x @ (W1 + s*B1@A1).T + b1)` etc. (s = alpha/r, alpha=4).
   Deriver le backward pour A et B uniquement (chain rule a travers la couche — les gradients de W existent mais ne sont pas appliques). Verifier les gradients de A1, B1, A2, B2 par differences finies (< 1e-5).

4. **Entrainement LoRA** sur la tache B (1500 steps, lr=0.05) :
   - MSE finale sur B < 0.02
   - les poids de base sont BIT-IDENTIQUES a la sauvegarde (`np.array_equal`, pas une tolerance)
   - params entrainables / params totaux < 25% (petit modele : le ratio est moins spectaculaire qu'en vrai — le calculer aussi pour une couche 4096x4096 r=8 en commentaire)

5. **Le test anti-oubli** : apres l'entrainement LoRA, re-evaluer le modele SANS les adapters (B@A non merges, juste ignores) sur la tache A : la MSE doit etre EXACTEMENT celle d'avant le fine-tuning. Comparer avec un full fine-tuning sur B (entrainer une copie sans rien geler) : sa MSE sur A doit s'etre fortement degradee (oubli catastrophique). Tableau final : `modele | MSE tache A | MSE tache B`.

### Criteres de reussite

- [ ] Le pre-training atteint MSE < 0.01 sur A
- [ ] Gradient check des 4 adapters < 1e-5
- [ ] LoRA atteint MSE < 0.02 sur B avec les poids de base bit-identiques (np.array_equal)
- [ ] Le test anti-oubli passe : base+adapters-retires == base original sur A ; le full FT montre l'oubli (MSE sur A degradee d'un facteur > 10)
- [ ] Le tableau final met en evidence le benefice "multi-adapters, un seul modele de base"
- [ ] Execution < 30 s

---

## Exercice 8 : Mini-DPO — aligner une politique tabulaire

### Objectif

Implementer la boucle DPO complete sur une politique softmax jouet : observer la montee des chosen, la descente des rejected, et le role de beta dans le rappel a la reference.

### Consigne

1. **Setup** : 4 prompts, vocabulaire de 8 reponses possibles. La politique est une table de logits `(4, 8)` ; `pi(y|x) = softmax(logits[x])`. Init : logits ~ N(0, 0.1). La reference = copie GELEE de l'init.

2. **Preferences** : pour chaque prompt x, 6 paires (y_w, y_l) tirees selon un vrai classement cache des reponses (ex : reward cache `r*(x, y)` aleatoire fixe ; y_w = la meilleure des 2 reponses tirees, sans bruit cette fois).

3. **DPO** : la logprob d'une reponse est `log softmax(logits[x])[y]`. Implementer la loss DPO batch sur toutes les paires et son gradient sur la table de logits :
   - gradient analytique : pour chaque paire, `dL/dlogits[x] = -beta * sigmoid(-beta*marge) * (d_logp_w/d_logits - d_logp_l/d_logits)` avec `d_logp_y/d_logits[x] = onehot(y) - softmax(logits[x])`
   - verifier le gradient par differences finies (< 1e-6) sur quelques entrees
   - entrainer 300 steps (lr=0.5) pour beta=0.1

4. **Verifications post-entrainement** :
   - la marge moyenne de reward implicite `beta*[(logp_pol-logp_ref)(y_w) - (...)(y_l)]` a augmente de façon monotone (croissante sur des checkpoints tous les 50 steps)
   - pour chaque prompt : `p(y_w)` moyen a augmente, `p(y_l)` moyen a baisse vs la reference
   - l'accuracy de classement des paires (marge > 0) atteint 100% sur ce probleme jouet

5. **Etude de beta** : reentrainer avec beta ∈ {0.05, 0.5, 2.0} (meme seed, memes paires) et mesurer le KL moyen `KL(pi || pi_ref)` par prompt en fin d'entrainement.
   - Attendu et a verifier : beta plus grand → la MEME marge cible est atteinte avec un KL plus FAIBLE (beta est le taux de change marge/KL). Afficher le tableau beta | marge finale | KL final.
   - Commenter le cas pathologique beta→0 : la politique s'eloigne sans limite de la reference (reward hacking du pauvre).

### Criteres de reussite

- [ ] Le gradient analytique DPO passe le check par differences finies (< 1e-6)
- [ ] La marge implicite croit de maniere monotone sur les checkpoints
- [ ] p(chosen) ↑ et p(rejected) ↓ par rapport a la reference pour tous les prompts
- [ ] Accuracy de classement 100% en fin d'entrainement
- [ ] Le tableau beta/marge/KL montre la relation attendue (KL decroissant en beta) et le commentaire sur beta→0 est correct
- [ ] Execution < 30 s
