# Exercices Faciles — Jour 21 : Mechanistic interpretability

---

## Exercice 1 : Logit lens a la main — entropie qui chute

### Objectif

Comprendre concretement ce que mesure le logit lens : la distribution sur le vocab projetee a chaque couche, et voir l'entropie tomber de "uniforme" vers "piquee" au fil de la profondeur.

### Consigne

On a un vocab de 4 tokens. Pour 3 couches successives, le residual stream projete par `W_U` donne ces **logits** :

- Layer 1 : `[0.1, 0.0, -0.1, 0.05]`  (quasi plat)
- Layer 2 : `[1.5, 0.2, 0.1, 0.3]`
- Layer 3 : `[5.0, 0.5, 0.2, 0.1]`  (pique sur le token 0)

1. **Softmax** par couche : calculer `p_L = softmax(logits_L)`.

2. **Entropie** par couche : `H = -sum_i p[i] * log(p[i])`. L'entropie max sur 4 tokens est `log(4) ≈ 1.386`.

3. Verifier que `H` **decroit** layer 1 -> 3. Quelle couche est la plus proche de l'uniforme ? Laquelle est la plus "decidee" ?

4. **Lecture mech interp** : si le bon token (token 0) etait deja top-1 en layer 2 mais disparaissait en layer 3, que soupconnerait-on ? (Indice : *suppression circuit*, cf cours.)

### Criteres de reussite

- [ ] Layer 1 : `p ≈ [0.27, 0.25, 0.22, 0.26]`, entropie ≈ 1.38 (≈ uniforme)
- [ ] Layer 3 : `p ≈ [0.97, 0.01, 0.01, 0.01]`, entropie << 1 (piquee)
- [ ] L'entropie decroit strictement de L1 a L3
- [ ] Comprehension : entropie qui chute = le modele "se decide" en profondeur ; un token qui apparait puis disparait = circuit de suppression

---

## Exercice 2 : Probing — correlation n'est pas causalite

### Objectif

Distinguer probing (l'info est-elle *presente* ?) d'activation patching (l'info est-elle *utilisee* ?), le piege #1 du cours.

### Consigne

On entraine un linear probe sur les hidden states d'une couche L pour predire "le sujet est-il singulier ou pluriel". Resultats :

| Tache | Accuracy du probe |
|---|---|
| Vraie tache (singulier/pluriel) | 92% |
| Control task (labels aleatoires) | 88% |

1. **Selectivity** (Hewitt & Liang 2019) : `selectivity = acc_vraie - acc_control`. La calculer. Que vaut-elle ici ? Est-ce rassurant ou suspect ?

2. **Probe trop puissant** : pourquoi un control a 88% (sur des labels *aleatoires* !) est-il alarmant ? Que dit-il sur la capacite du probe a "memoriser" plutot qu'a "decoder une vraie feature" ?

3. **Lineaire obligatoire** : pourquoi le cours insiste-t-il sur un probe **lineaire** et non un MLP ? (Indice : un MLP non-lineaire "trouve toujours quelque chose".)

4. **Le saut causal** : meme avec une selectivity excellente (ex : 92% vraie, 50% control), pourquoi cela ne prouve-t-il **pas** que le modele *utilise* cette info pour sa prediction ? Quelle technique faut-il pour la causalite ?

### Criteres de reussite

- [ ] `selectivity = 92 - 88 = 4 points` -> tres faible -> **suspect** (le probe memorise)
- [ ] Un control eleve signale un probe trop expressif ou des activations "trop riches"
- [ ] Le probe lineaire borne l'expressivite -> conclusions interpretables
- [ ] Comprehension : probing = correlation ; seul l'**activation patching** (substituer l'activation et mesurer l'effet sur la sortie) etablit la causalite

---

## Exercice 3 : Superposition — pourquoi 5 features dans 2 dims ?

### Objectif

Comprendre le calcul de capacite qui explique pourquoi un neurone code plusieurs features (polysemy) et le role de la sparsity.

### Consigne

1. **Le probleme** : on veut representer `n_features = 5` directions dans un espace de `n_hidden = 2` dimensions.
   - Combien de directions **orthogonales** tient-on au maximum dans R^2 ? (Reponse : 2.)
   - Donc combien de features au minimum doivent **partager** des directions (se superposer) ?

2. **Le pentagone** : Elhage 2022 montre qu'avec une forte sparsity, le modele place les 5 features comme les 5 sommets d'un pentagone regulier dans R^2.
   - Quel angle separe deux features **adjacentes** sur un pentagone ? (`360 / 5`)
   - Le cosinus de cet angle (l'interference entre 2 features adjacentes) : `cos(72°) ≈ ?`

3. **Role de la sparsity** : avec `sparsity = 0.9`, une feature est active avec proba `0.1`. Quelle est la proba que **deux** features donnees soient actives **en meme temps** (independance) ? Pourquoi une faible co-activation rend la superposition "presque sans cout" ?

4. **Consequence pour mech interp** : si un neurone individuel est une combinaison de plusieurs features superposees, pourquoi l'interpretation "1 neurone = 1 concept" est-elle fausse ? Quelle est la solution du cours (section SAE) ?

### Criteres de reussite

- [ ] R^2 tient 2 directions orthogonales ; au moins 3 des 5 features doivent se superposer
- [ ] Angle adjacent du pentagone = `72°`, `cos(72°) ≈ 0.309` (interference faible mais non nulle)
- [ ] Co-activation de 2 features = `0.1 * 0.1 = 0.01` (1%) -> l'interference est rarement "payee"
- [ ] Comprehension : sous sparsity, stocker `n > d` features est rationnel ; un neurone melange donc plusieurs features -> il faut les SAE pour les **demixer**
