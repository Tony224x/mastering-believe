# Exercices Faciles — Jour 20 : Distillation, donnees synthetiques & SLMs

---

## Exercice 1 : Distillation de logits — softmax a temperature a la main

### Objectif

Comprendre ce que le student "apprend en plus" quand on distille la distribution complete du teacher (dark knowledge) plutot que la seule classe gagnante, et le role de la temperature `T`.

### Consigne

Le teacher produit, pour un input, les **logits** suivants sur 4 classes :

`z_teacher = [4.0, 2.0, 1.0, 0.5]`

1. **Softmax T=1** : calculer `p = softmax(z_teacher)`. Quelle classe gagne ? Quelle proba ?

2. **Softmax a temperature** : la distillation utilise `softmax(z / T)`. Calculer la distribution pour `T = 2` et `T = 4`.
   - Que se passe-t-il sur les **classes perdantes** (2, 3, 4) quand `T` monte ? Leur proba monte-t-elle ou descend-elle ?
   - Pourquoi une distribution **plus douce** (T eleve) transmet-elle plus d'information au student qu'un one-hot `[1, 0, 0, 0]` ?

3. **Dark knowledge** : a `T=1`, le rapport `p(classe 2) / p(classe 3)` encode une info ("la classe 2 ressemble plus a la bonne reponse que la classe 3"). Calculer ce rapport a `T=1` et a `T=4`. Lequel preserve mieux la **structure relative** des perdantes ?

4. **Hard label vs soft label** : si le student n'apprenait que le one-hot `[1,0,0,0]`, quelle info perdrait-il ? Donner un exemple concret (ex : "chat" vs "tigre" vs "voiture").

### Criteres de reussite

- [ ] `softmax(z, T=1)` ≈ `[0.78, 0.105, 0.039, 0.024]` (classe 1 gagne a ~78%)
- [ ] Quand `T` monte, la distribution se **lisse** : la classe 1 baisse, les perdantes montent
- [ ] Le rapport `p2/p3` reste ≈ `exp((2-1)/T)` : `e^1 ≈ 2.72` a T=1, `e^0.25 ≈ 1.28` a T=4 (la structure relative est conservee, juste compressee)
- [ ] Comprehension : le soft label transmet la similarite entre classes (dark knowledge) que le hard label efface

---

## Exercice 2 : KL divergence teacher -> student

### Objectif

Implementer la loss reellement utilisee en distillation de logits (`KL(teacher || student)`) et voir qu'elle est minimale quand le student copie le teacher.

### Consigne

Teacher (fixe) : `z_teacher = [3.0, 1.0, 0.2]`. On considere 3 students candidats (leurs logits) :

- `z_A = [3.0, 1.0, 0.2]` (copie exacte)
- `z_B = [2.0, 1.5, 1.0]` (distribution plus plate)
- `z_C = [5.0, 0.0, -2.0]` (distribution plus piquee, sur-confiante)

1. Calculer `p_teacher = softmax(z_teacher / T)` et `p_student = softmax(z_student / T)` avec `T = 2` pour chaque student.

2. Calculer la **KL divergence** `KL(p_teacher || p_student) = sum_i p_teacher[i] * log(p_teacher[i] / p_student[i])`.

3. Classer A, B, C par KL croissante. Lequel est le meilleur student ? Pourquoi `KL >= 0` toujours, et `= 0` seulement si les distributions sont identiques ?

4. **Loss combinee (rappel cours)** : en pratique on melange `loss = alpha * KL(teacher, student) * T^2 + (1 - alpha) * CE(hard_label, student)`. A quoi sert le terme `CE` sur le vrai label ? Pourquoi le facteur `T^2` ?

### Criteres de reussite

- [ ] Student A donne `KL ≈ 0` (copie exacte) -> meilleur
- [ ] Student B (trop plat) et C (trop piquee) ont une `KL > 0`
- [ ] `KL >= 0` est compris (inegalite de Gibbs), `= 0` ssi distributions egales
- [ ] Comprehension : le terme `CE` ancre le student sur la vraie reponse (le teacher peut se tromper) ; `T^2` rescale le gradient car les logits ont ete divises par `T`

---

## Exercice 3 : Economie de la distillation — break-even

### Objectif

Refaire le calcul "distiller ou rester sur API frontier ?" du cours et trouver le volume de tokens au-dela duquel la distillation est rentable.

### Consigne

Donnees (chiffres jouets coherents avec le cours) :
- **API frontier** : `0.50 $ / 1M tokens` de sortie.
- **Distillation** : cout fixe one-shot = `30 000 $` (generation dataset + training).
- **Self-host du SLM distille** : `0.02 $ / 1M tokens` (inference sur infra propre, hors maintenance).

1. **Cout API** sur `V` millions de tokens/mois sur 12 mois : `cout_api = 12 * V * 0.50`.

2. **Cout distillation** sur 12 mois : `cout_distill = 30000 + 12 * V * 0.02`.

3. **Break-even** : resoudre `cout_api = cout_distill` pour `V`. Donner le volume mensuel (en millions de tokens) au-dela duquel distiller est gagnant.

4. **Tracer (a la main ou en tete)** : pour `V = 1M`, `V = 10M`, `V = 100M` tokens/mois, qui gagne ? Recouper avec le seuil "10-50M tokens/mois" du cours.

5. **Cout cache** : le calcul ci-dessus ignore la **maintenance** (equipe ML). Si on ajoute `5000 $/mois` de maintenance, le break-even monte ou descend ? Recalculer.

### Criteres de reussite

- [ ] `cout_api = 6 * V` (k$), `cout_distill = 30 + 0.24 * V` (k$, V en millions/mois)
- [ ] Break-even sans maintenance : `V ≈ 30 / 5.76 ≈ 5.2M tokens/mois` (ordre de grandeur du cours)
- [ ] Avec maintenance (`+60 k$/an`), le break-even **monte** (la distillation coute plus cher) -> `V ≈ 90 / 5.76 ≈ 15.6M`
- [ ] Comprehension : sous le seuil, rester sur API + caching ; au-dessus, distiller. Les couts caches (maintenance) repoussent le seuil
