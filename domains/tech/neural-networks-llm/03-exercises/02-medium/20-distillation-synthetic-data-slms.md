# Exercices Medium — Jour 20 : Distillation, donnees synthetiques & SLMs

---

## Exercice 4 : Loss de distillation KL complete (logits) from scratch

### Objectif

Implementer en numpy la vraie loss de distillation de logits (Hinton 2015) : `loss = alpha * T^2 * KL(teacher_soft, student_soft) + (1-alpha) * CE(hard_label, student)`, et montrer que distiller bat l'entrainement sur hard labels seuls.

### Consigne

1. **Setup jouet** : un probleme de classification a `K = 5` classes. Generer un teacher "expert" : une matrice de logits `Z_teacher` de shape `(n, K)` pour `n = 500` exemples, telle que la classe correcte gagne mais avec une **structure de similarite** entre classes (ex : classe 0 ressemble a classe 1). Les vrais labels `y` (hard) sont l'argmax bruite du teacher.

2. **Student** : un classifieur lineaire `softmax(X @ W)` from scratch (X = features synthetiques de dim `d`), entraine par descente de gradient.

3. Implementer deux entrainements du **meme** student :
   - **Hard only** : loss = cross-entropy sur `y` (one-hot).
   - **Distillation** : loss = `alpha * T^2 * KL(softmax(Z_teacher/T) || softmax(student_logits/T)) + (1-alpha) * CE(y, student)` avec `T = 3`, `alpha = 0.7`.

4. **Evaluer** sur un hold-out : accuracy ET une metrique de **calibration** (ex : la distribution du student est-elle plus proche de celle du teacher ?). Le student distille doit generaliser au moins aussi bien, et imiter la structure du teacher.

5. **Ablation temperature** : balayer `T` dans `{1, 2, 3, 5, 10}`. Tracer (afficher) l'accuracy. Trop bas (T=1) -> on perd la dark knowledge ; trop haut -> la distribution devient quasi uniforme et le signal se noie. Identifier le sweet spot.

### Criteres de reussite

- [ ] `KL`, `softmax` a temperature et `CE` implementes correctement (numpy, softmax stable)
- [ ] Le facteur `T^2` est present (compense le `1/T^2` du gradient des soft logits)
- [ ] Le student distille >= hard-only en accuracy hold-out
- [ ] L'ablation `T` montre une courbe en cloche (sweet spot intermediaire ~2-3)
- [ ] Code commente avec le POURQUOI de chaque etape (notamment dark knowledge et `T^2`)

---

## Exercice 5 : Pipeline de donnees synthetiques — l'impact du filtrage

### Objectif

Reproduire et instrumenter le pipeline du cours (`02-code/20`) : generer du synthetique, filtrer en cascade, et **mesurer** combien chaque etage de filtrage compte pour la qualite finale du student.

### Consigne

1. **Teacher bruite** : reprendre l'idee du code du jour (classification sentiment jouet, ou ta propre tache). Le teacher genere des exemples avec un **taux de bruit** controlable `p_noise` (label retourne avec proba `p_noise`).

2. **Filtres en cascade** : implementer (a) rule-based (longueur/format), (b) un "LLM-judge" simule (coherence texte/label), (c) dedup (hash de set de mots). Compter le rendement (% d'exemples passant) a chaque etage.

3. **Ablation des filtres** : entrainer 4 students :
   - seeds seuls (baseline),
   - seeds + synth **brut** (sans filtre),
   - seeds + synth **filtre** (cascade complete),
   - seeds + synth filtre **mais avec p_noise eleve** (teacher mediocre).
   Comparer l'accuracy hold-out.

4. **Courbe volume vs qualite** : pour le synth filtre, faire varier le nombre d'exemples generes par seed `{2, 5, 10, 20}`. Le gain sature-t-il ? A volume egal, la **qualite** (filtrage) bat-elle le **volume** (brut) ?

5. **Contamination** : injecter volontairement 2-3 exemples de l'eval set dans le train et mesurer l'accuracy gonflee artificiellement. Montrer pourquoi un eval set **jamais publie** est indispensable.

### Criteres de reussite

- [ ] Le teacher a un `p_noise` parametrable ; le pipeline mesure le rendement par etage
- [ ] Les 4 students sont compares ; "synth filtre" > "synth brut" quand `p_noise` est non negligeable
- [ ] La courbe volume montre une saturation (qualite > volume au-dela d'un seuil)
- [ ] La demo de contamination montre une accuracy artificiellement gonflee
- [ ] Code commente (POURQUOI le filtrage est l'etape #1 du cours)

---

## Exercice 6 : Distillation de sequences (SFT sur outputs teacher)

### Objectif

Implementer la methode **dominante en 2026** (Type 2 du cours) : pas d'acces aux logits, on SFT le student sur les *reponses completes* du teacher. Montrer l'effet du **mode collapse** quand le teacher genere a temperature trop basse.

### Consigne

1. **Tache jouet sequentielle** : apprendre a continuer une mini-grammaire (ex : sequences de tokens suivant une regle simple, ou completer une operation). Le "teacher" est une fonction-regle qui produit la bonne continuation, avec une **temperature** de generation simulee (diversite des reponses).

2. **Generation du dataset** : pour chaque prompt seed, le teacher genere `K` completions a temperature `temp`. A `temp` basse, les completions sont quasi identiques ; a `temp` haute, elles sont diverses.

3. **Student** : un modele de sequence minimal (n-gram / petit MLP bag-of-tokens -> next token, from scratch) entraine par SFT (cross-entropy next-token) sur les paires `(prompt, completion_teacher)`.

4. **Mesurer le mode collapse** : entrainer le student sur un dataset genere a `temp = 0.1` (peu diversifie) vs `temp = 1.0` (diversifie). Mesurer :
   - la **diversite** des sorties du student (nombre de continuations distinctes sur un set de prompts),
   - l'**accuracy** (la continuation est-elle valide selon la regle ?).

5. **Analyse** : pourquoi `temp` trop basse cote teacher -> student qui repete une seule facon de repondre (mode collapse) ? Quel est le compromis diversite/qualite ? Relier au piege #2 du cours.

### Criteres de reussite

- [ ] La distillation de sequences est implementee SANS acces aux logits (SFT sur outputs)
- [ ] Le teacher a une temperature de generation controlable
- [ ] Le student entraine sur `temp=0.1` montre une diversite de sortie plus faible (mode collapse)
- [ ] Le compromis diversite (temp haute) vs fidelite (temp basse) est mesure et discute
- [ ] Code commente (POURQUOI varier la temperature evite le mode collapse)
