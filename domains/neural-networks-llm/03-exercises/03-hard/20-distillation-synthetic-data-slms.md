# Exercices Hard — Jour 20 : Distillation, donnees synthetiques & SLMs

---

## Exercice 7 : Distillation de logits (Hinton 2015) from scratch — dark knowledge & temperature

### Objectif

Implementer **de bout en bout** la distillation de logits (Type 1 du cours) sur un vrai mini-classifieur entraine par descente de gradient, et **prouver experimentalement** la "dark knowledge" : un student entraine sur les cibles douces du teacher (`T > 1`) recupere mieux la **structure relative des mauvaises reponses** du teacher qu'un student entraine sur des hard labels one-hot — alors que les deux voient les memes inputs.

C'est plus dur que l'easy/medium : ici il n'y a **pas** de numpy. Tu implementes softmax stable, cross-entropy, KL et la descente de gradient d'un classifieur lineaire multi-classes **a la main** (listes Python + `math`), puis tu instrumentes la dynamique d'entrainement pour montrer la decroissance de `KL(teacher || student)`.

### Consigne

1. **Teacher fige avec une structure de similarite.** Construire un teacher "expert" sur `K = 4` classes dont la matrice de poids encode une **proximite** entre certaines classes (ex : classe 0 et classe 1 sont semantiquement proches — "chat"/"tigre" — tandis que classe 3 est isolee — "voiture"). Le teacher, sur un input, doit donc produire une distribution ou la bonne classe gagne **mais ou la 2e place est systematiquement la classe soeur**. C'est ca la dark knowledge : l'ordre relatif des perdantes (`p(soeur) > p(lointaine)`).

2. **Dataset.** Generer `n` features synthetiques `x` (dimension `d`), calculer les logits teacher, et fabriquer :
   - les **soft targets** `softmax(z_teacher / T)` (avec `T > 1`, ex `T = 3`),
   - les **hard labels** = argmax du teacher (one-hot), **avec un peu de bruit d'etiquetage** (le teacher n'est pas parfait — cf. les 5 % de bruit du `02-code`).

3. **Student lineaire from scratch.** Un classifieur `logits = W @ x + b` (W de shape `(K, d)`), softmax, entraine par SGD. Implementer **deux** entrainements du **meme** student (memes inits, meme seed, memes x) :
   - **Hard** : loss = cross-entropy sur le label one-hot.
   - **Soft (distillation)** : loss = `T^2 * KL(softmax(z_teacher/T) || softmax(z_student/T))`. Le facteur `T^2` compense le `1/T^2` du gradient des soft logits (cf. cours).

4. **Tracer la dynamique.** Pour le student soft, logger `KL(teacher || student)` (a `T`) **a chaque epoch**. Elle doit **decroitre de facon (quasi) monotone** : verifier que `KL_final < KL_initial` et que la suite est globalement decroissante (autoriser un petit bruit SGD).

5. **Metrique "dark knowledge" hold-out.** Sur un **hold-out** jamais vu en train, definir le **dark-knowledge recovery** : la fraction d'exemples ou le student reproduit le **bon ordre des perdantes** du teacher, c.-a-d. `rank_student(2e classe du teacher) < rank_student(3e classe du teacher)` (la 2e place du student tombe sur la classe soeur, pas sur une lointaine). Montrer que **soft > hard** sur cette metrique — c'est la preuve que le soft target transmet une info que le one-hot efface, **a accuracy comparable**.

6. **Ablation temperature.** Balayer `T` dans `{1, 2, 3, 5, 10, 20}` cote distillation, **moyenne sur plusieurs seeds**. Une `T` tres haute aplatit la cible vers l'uniforme et **degrade** (le signal de classe se noie) — c'est l'inegalite robuste a asserter. Caveat honnete attendu : avec un student **lineaire** (capacite tres limitee), la "cloche" classique de Hinton (sweet spot intermediaire ~2-3) n'est que **partielle** ; une `T` moderee (1-2) fait deja aussi bien. Documenter ce caveat plutot que de forcer une cloche artificielle (esprit "honest caveat" du domaine).

### Criteres de reussite

- [ ] softmax stable, cross-entropy, KL et SGD multi-classes implementes **a la main** (stdlib, sans numpy)
- [ ] le teacher encode une vraie structure de similarite (classe soeur systematiquement 2e)
- [ ] le facteur `T^2` est present dans la loss de distillation
- [ ] `KL(teacher || student)` **decroit** au fil des epochs (assert `KL_final < KL_initial`)
- [ ] le student soft **bat** le student hard sur la metrique dark-knowledge recovery hold-out (et colle mieux a la distribution teacher : KL hold-out plus basse)
- [ ] l'ablation `T` (moyenne multi-seed) montre qu'une `T` tres haute degrade ; caveat honnete documente sur la cloche partielle d'un student lineaire

---

## Exercice 8 : Pipeline de distillation par donnees synthetiques — ablation de filtres & tradeoff qualite/quantite honnete

### Objectif

Etendre le pipeline du `02-code` en une experience **rigoureuse et honnete** : generer depuis un teacher bruite, faire tourner la **cascade complete de filtres** (rule-based + LLM-judge simule + dedup minhash + contamination check), entrainer le student, puis **ablater chaque filtre un par un** pour **quantifier sa contribution** a l'accuracy hold-out. Conclusion attendue (et assertee) : le pipeline filtre **bat** le pipeline brut, et la contamination est **detectee** quand on l'injecte volontairement.

C'est plus dur que le medium (Exercice 5) : ici on ne se contente pas de comparer "filtre vs brut" — on fait une **ablation leave-one-filter-out** (retirer un seul etage a la fois) pour attribuer un delta d'accuracy a chaque filtre, et on assure la **robustesse statistique** en moyennant sur plusieurs seeds (sinon le verdict est du bruit).

### Consigne

1. **Teacher bruite parametrable.** Reprendre l'idee sentiment du `02-code` : `teacher_generate(seed, n, p_noise)` ou `p_noise` est la proba de retourner le label. Prendre un `p_noise` **non negligeable** (ex 0.20-0.30) pour que les filtres aient un travail reel a faire.

2. **Cascade de filtres** (reutiliser/adapter les fonctions du `02-code`) :
   - `rule_based_ok` (longueur, format, presence de ponctuation),
   - `llm_judge` (coherence texte/label : detecte les labels retournes par le teacher),
   - `minhash_dedup` (set-of-words hash),
   - `contamination_check` (Jaccard vs l'eval set).

3. **Ablation leave-one-filter-out.** Definir le pipeline complet `{rule, judge, dedup}` puis, pour chaque filtre `f`, construire le pipeline **sans `f`** uniquement, entrainer un student, et mesurer la **chute d'accuracy** par rapport au pipeline complet. Le filtre dont le retrait fait le plus chuter l'accuracy est le plus precieux (avec ce teacher bruite, ce sera le **judge**, car il neutralise le bruit d'etiquetage). Inclure aussi le pipeline **sans aucun filtre** (synth brut) comme borne basse.

4. **Robustesse multi-seed.** Repeter l'experience sur `>= 5` seeds et **moyenner** les accuracies (le SGD + le tirage des donnees sont stochastiques). Reporter la moyenne. **Asserter** sur les **moyennes**, pas sur un run unique.

5. **Tradeoff qualite/quantite honnete.** Faire varier le **volume** genere par seed `n ∈ {5, 10, 20, 40}` et comparer, **a volume egal de generations**, le student "filtre" vs "brut". Montrer que (a) le filtrage gagne quand le teacher est bruite, et (b) le gain du volume **sature** : doubler les generations brutes n'achete presque rien une fois le bruit accumule. Garder la nuance du `02-code` ("plus de donnees brutes peut parfois aider a court terme, mais le bruit s'accumule") — ne pas sur-vendre.

6. **Contamination injectee.** Injecter volontairement 2-3 exemples de l'eval set dans le pool synthetique **avant** le contamination check. Asserter que `contamination_check` en detecte **au moins autant** que ce qui a ete injecte (>= n_injected), et montrer l'accuracy artificiellement gonflee si on **n'avait pas** filtre la contamination. C'est l'illustration du piege #1 du cours.

### Criteres de reussite

- [ ] teacher avec `p_noise` parametrable, cascade `rule + judge + dedup + contamination` operationnelle
- [ ] ablation **leave-one-filter-out** : un delta d'accuracy attribue a chaque filtre (judge le plus precieux ici)
- [ ] resultats **moyennes sur >= 5 seeds** ; les assertions portent sur les moyennes
- [ ] **assert** : pipeline filtre (moyenne) **>** pipeline brut (moyenne) sur l'eval hold-out
- [ ] **assert** : la contamination injectee est detectee (`detected >= n_injected`)
- [ ] courbe volume vs qualite montrant la saturation ; ton **honnete** (pas de sur-claim), code commente

---

## Exercice 9 : Sequence distillation — teacher-forcing vs on-policy & exposure bias

### Objectif

Demontrer, sur une tache **sequentielle** jouet, le phenomene d'**exposure bias** : un student distille en **pur teacher-forcing** (il ne voit jamais ses propres tokens pendant l'entrainement) **compose ses erreurs** sur un rollout multi-pas (auto-regressif a l'inference), alors qu'une etape **on-policy** (le student genere, le teacher re-score ses propres etats) reduit cette accumulation d'erreur. C'est la justification de "on-policy + RL" du cours (section 5, "MiniLM-R1" d'Apple).

C'est le plus subtil des trois : on construit un **vrai rollout auto-regressif** ou l'erreur a l'etape `t` change l'etat d'entree de l'etape `t+1`. On reste un **toy fidele** (modele de transition tabulaire / petit next-token from scratch), et — dans l'esprit "honest caveat" du `02-code` — on **modelise explicitement** le mecanisme de propagation d'erreur plutot que de monter une boucle RL lourde. On asserte l'**inegalite qualitative** : `erreur_rollout(on-policy) < erreur_rollout(teacher-forcing)`.

### Consigne

1. **Tache sequentielle a etats.** Definir un automate jouet : un alphabet de `V` tokens et une **regle de transition** deterministe (teacher) `next = rule(state)` ou `state` est le dernier token (ou un petit contexte). La tache du student : predire le prochain token. La cle pedagogique : a l'inference, le student **consomme ses propres sorties** (`state_{t+1} = student_output_t`), donc une erreur a l'etape `t` decale tous les etats suivants.

2. **Distribution des etats : train vs rollout.** En **teacher-forcing**, le student n'est entraine que sur des etats **issus du teacher** (la trajectoire correcte). A l'inference, des qu'il se trompe une fois, il atterrit sur un etat **jamais vu en train** (hors-distribution) — il ne sait pas s'en remettre. C'est l'exposure bias. Construire l'experience pour que cette **mismatch de distribution** soit visible.

3. **Variante on-policy.** Ajouter une phase ou le student **genere** des etats (rollout) et le **teacher re-score** ces etats visites par le student (`(state_student, rule(state_student))`), puis on entraine le student aussi sur ces paires. Maintenant le student a vu des etats **dans sa propre distribution**, y compris des etats "post-erreur" — il apprend a se **recuperer**.

4. **Metrique : error compounding sur rollout multi-pas.** Pour chaque variante (TF pur vs +on-policy), lancer `R` rollouts de longueur `L` depuis des prefixes de depart, en mode auto-regressif (le student consomme ses sorties). Mesurer :
   - le **taux d'erreur par pas**, et surtout
   - la **probabilite de derailment** : fraction de rollouts qui finissent hors-trajectoire a l'etape `L` (l'erreur composee).
   Montrer que la TF pure derive plus vite (courbe d'erreur cumulee qui s'envole), l'on-policy reste plus pres de la trajectoire.

5. **Assert l'inegalite qualitative.** `derailment_rate(on-policy) < derailment_rate(teacher-forcing)` (et/ou erreur cumulee a `L` pas plus faible en on-policy). Si, sur un toy donne, l'ecart venait a etre instable, **modeliser explicitement** la borne : l'erreur composee en TF croit en `O(eps * L^2)` (Ross et al., DAgger) vs `O(eps * L)` en on-policy — et asserter l'inegalite sur cette borne analytique en complement de la mesure empirique. Documenter le caveat honnetement (comme le `02-code` documente ses 5 % de bruit teacher).

6. **Relier au cours.** Expliquer pourquoi le pur SFT sur outputs teacher (Type 2, dominant en 2026) **souffre quand meme** de l'exposure bias, et pourquoi DeepSeek/Apple ajoutent une phase on-policy/RL (Type 3) pour les taches longues (reasoning multi-pas).

### Criteres de reussite

- [ ] tache sequentielle avec **rollout auto-regressif** ou l'erreur a `t` decale l'etat a `t+1` (vrai error compounding, pas un proxy plat)
- [ ] variante **on-policy** : le student genere, le teacher re-score les etats visites par le student, on re-entraine dessus
- [ ] metrique de **derailment / erreur cumulee** mesuree sur `R` rollouts de longueur `L`
- [ ] **assert** : `derailment(on-policy) < derailment(teacher-forcing)` (empirique) **et** borne analytique `O(eps L^2)` vs `O(eps L)` assertee
- [ ] caveat honnete documente (toy fidele, pas une boucle RL complete), lien explicite avec section 5 du cours
- [ ] code commente avec le POURQUOI (exposure bias = mismatch train/inference des distributions d'etats)
