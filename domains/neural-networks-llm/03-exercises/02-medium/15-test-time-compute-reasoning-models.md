# Exercices Medium — Jour 15 : Test-time compute & reasoning models

---

## Exercice 4 : Courbe de scaling de la self-consistency

### Objectif

Mesurer empiriquement le gain de la self-consistency en fonction de K et retrouver le rendement decroissant typique des courbes test-time compute.

### Consigne

1. Implementer un `noisy_solver(problem, noise)` (numpy uniquement) qui, comme le code du Jour 15, repond `(a + b) % m` mais produit une erreur **plausible** avec probabilite `noise` (off-by-one, oubli du modulo, inversion).

2. Implementer `self_consistency(problem, k)` = majority vote sur k echantillons.

3. Generer 500 problemes aleatoires `(a, b, m=97)`. Pour `k in [1, 3, 5, 10, 20, 40, 80]`, mesurer l'accuracy moyenne du majority vote.

4. Tracer (ou imprimer en ASCII) la courbe accuracy vs k. Identifier :
   - le gain marginal entre k=1 et k=5 ;
   - le gain marginal entre k=40 et k=80 ;
   - le **plateau** approximatif.

5. Calculer le **cout** : si k=40 coute 40x le compte d'inference de k=1, quel est le ratio gain/cout ? Conclusion sur le sweet spot.

6. Varier `noise` (0.2, 0.4, 0.6). A noise=0.6 (le modele se trompe plus souvent qu'il n'a raison sur la reponse correcte exacte), la self-consistency aide-t-elle encore ? Expliquer.

### Criteres de reussite

- [ ] L'accuracy croit avec k puis sature (~0.6 a k=1 vers ~0.9+ a k=40 pour noise=0.4)
- [ ] Le gain marginal s'effondre apres ~k=20-40 (rendement decroissant clair)
- [ ] Le ratio gain/cout est calcule et montre que les grands k sont inefficients
- [ ] A noise tres eleve, le vote n'aide plus (voire degrade) car la masse des erreurs depasse celle de la bonne reponse
- [ ] Le code est numpy, deterministe (seed), commente WHY

---

## Exercice 5 : GRPO sur un bandit, avec et sans normalisation par std

### Objectif

Implementer GRPO from scratch sur un bandit a K bras et mesurer l'impact de la normalisation `(r - mean) / std` sur la vitesse et la stabilite de convergence.

### Consigne

1. Reprendre le bandit du code Jour 15 : 4 bras, `TRUE_REWARDS = [0.1, 0.3, 0.8, 0.5]`, reward bruitee `gauss(0, sigma)`.

2. Implementer la boucle GRPO :
   - policy = softmax(logits) sur 4 actions ;
   - echantillonner un groupe de G trajectoires ;
   - `advantage = (r - mean) / std` ;
   - gradient policy : pour chaque action a echantillonnee, `logits[i] += lr * adv * ((i==a) - probs[i])`.

3. Comparer **3 variantes** sur 300 steps, meme seed :
   - **A** : advantage brut `r - mean` (sans diviser par std) ;
   - **B** : advantage normalise `(r - mean) / std` (GRPO standard) ;
   - **C** : pas de baseline du tout, `advantage = r` (REINFORCE naif).

4. Pour chacune, tracer la proba du bras optimal (bras 2) au fil des steps. Comparer la vitesse de convergence et la variance des updates.

5. Faire varier `sigma` (0.05, 0.3, 0.8). Montrer qu'a `sigma` eleve, la normalisation par std (variante B) devient nettement plus stable que A et C. Expliquer pourquoi.

6. Question d'analyse : pourquoi la normalisation par std rend-elle l'algorithme robuste a l'echelle des rewards (ex : rewards dans [0,1] vs [0,100]) ?

### Criteres de reussite

- [ ] Les 3 variantes convergent vers le bras 2 (mais pas a la meme vitesse/stabilite)
- [ ] La variante C (sans baseline) a la plus haute variance et converge le plus lentement
- [ ] A sigma eleve, la variante B (std-normalisee) est visiblement plus stable
- [ ] L'explication : diviser par std re-echelle l'advantage en "z-score", rendant le pas d'update independant de l'amplitude absolue des rewards
- [ ] Code numpy, seed fixe, courbes lisibles

---

## Exercice 6 : Budget de thinking tokens et overthinking

### Objectif

Modeliser la courbe accuracy vs thinking-budget et implementer un controleur qui choisit le budget optimal par difficulte, pour eviter l'overthinking sur les taches faciles.

### Consigne

1. Implementer `reasoning_accuracy(budget_tokens, difficulty)` qui modelise (cf code Jour 15) :
   - une croissance en `log10(budget)` jusqu'a un sweet spot dependant de la difficulte ;
   - un plateau (`saturation = 0.55 + 0.40 * difficulty`) ;
   - une **legere decroissance** au-dela du sweet spot (le modele "se perd" sur les problemes faciles).

2. Tabuler accuracy pour `budget in [100, 500, 2000, 8000, 32000]` et `difficulty in [0.1, 0.5, 0.9]`. Reproduire le phenomene : sur facile, le pic est tot puis ca redescend ; sur dur, le plateau est plus haut et plus tardif.

3. Implementer `optimal_budget(difficulty, candidate_budgets)` qui retourne le budget maximisant l'accuracy.

4. Implementer un **routeur cout-conscient** : etant donne un cout lineaire en tokens (`cost = budget * price`), trouver le budget qui maximise `accuracy - lambda * cost` pour quelques valeurs de lambda. Montrer que le budget optimal **diminue** quand lambda (sensibilite au cout) augmente.

5. Calculer, sur un mix realiste (70% faciles, 30% durs), l'economie de tokens d'un routeur adaptatif vs un budget fixe a 32000 tokens pour tout, a accuracy globale comparable.

6. Analyse : citer 2 mecanismes produit concrets (cf theorie) pour eviter l'overthinking en prod.

### Criteres de reussite

- [ ] La table reproduit le pic-puis-declin sur facile et le plateau eleve sur dur
- [ ] `optimal_budget` renvoie un budget plus petit pour les taches faciles
- [ ] Le routeur cout-conscient reduit le budget quand lambda augmente
- [ ] L'economie de tokens du routeur adaptatif est chiffree (vs budget fixe max)
- [ ] Les 2 mecanismes anti-overthinking sont cites (ex : `budget_tokens` cap, router/classifieur de query en amont)
- [ ] Code numpy, commente WHY
