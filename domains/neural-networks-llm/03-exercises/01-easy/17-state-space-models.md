# Exercices Faciles — Jour 17 : State Space Models

---

## Exercice 1 : Recurrence SSM a la main

### Objectif

Derouler a la main la recurrence d'un SSM lineaire `h_t = A h_{t-1} + B x_t`, `y_t = C h_t` pour internaliser le mode recurrent (inference, style RNN).

### Consigne

On prend un SSM **scalaire** (etat de dimension D=1) pour faire le calcul a la main :

```
A = 0.5   (scalaire, |A| < 1 -> stable)
B = 1.0
C = 2.0
h_0 = 0
x = [1, 0, 0, 1]   (4 tokens)
```

1. Calculer `h_1, h_2, h_3, h_4` en appliquant `h_t = A * h_{t-1} + B * x_t` step par step.

2. Calculer `y_1, y_2, y_3, y_4` avec `y_t = C * h_t`.

3. **Stabilite** : refaire le calcul avec `A = 1.5` (instable). Que deviennent `h_t` et `y_t` ? Pourquoi exige-t-on `|A| < 1` ?

4. **Memoire** : a quel point le token `x_1=1` influence-t-il encore `y_4` (avec A=0.5) ? Calculer la contribution de `x_1` a `h_4`. Pourquoi dit-on que l'etat est un "compresseur lossy" de l'historique ?

5. Question conceptuelle : combien de memoire faut-il pour calculer `y_t` en mode recurrent, independamment de la longueur de la sequence ? (Indice : que stocke-t-on entre deux steps ?)

### Criteres de reussite

- [ ] h = [1.0, 0.5, 0.25, 1.125] ; y = [2.0, 1.0, 0.5, 2.25] (avec A=0.5)
- [ ] Avec A=1.5 : h croit (1, 1.5, 2.25, 4.375...), y explose -> instabilite
- [ ] Contribution de x_1 a h_4 = A^3 * B * x_1 = 0.5^3 = 0.125 (s'efface exponentiellement)
- [ ] L'etat de taille fixe ne peut garder qu'une trace decroissante du passe -> lossy
- [ ] Memoire recurrente = O(D) (on ne garde que h, taille fixe), independante de N

---

## Exercice 2 : Equivalence recurrent / convolutionnel

### Objectif

Verifier la propriete miracle des SSM : la recurrence se deroule en une **convolution** par un kernel `K = (CB, CAB, CA^2 B, ...)`, ce qui rend le training parallelisable.

### Consigne

Reprendre le SSM scalaire de l'Exercice 1 : `A=0.5, B=1.0, C=2.0`, `x=[1, 0, 0, 1]`.

1. Construire le **kernel SSM** `K` de longueur 4 : `K[k] = C * A^k * B`. Donner `K[0], K[1], K[2], K[3]`.

2. Appliquer la **convolution causale** : `y_t = sum_{k=0..t-1} K[k] * x[t-k]` (avec indices a partir de 1). Calculer `y_1, y_2, y_3, y_4`.

3. **Comparer** avec les `y_t` obtenus en mode recurrent a l'Exercice 1. Sont-ils identiques ? (ils doivent l'etre exactement)

4. Question conceptuelle : pourquoi le mode convolutionnel est-il parallelisable au training alors que le mode recurrent est sequentiel ? (Indice : la convolution peut se calculer par FFT sur toute la sequence d'un coup ; la recurrence depend de `h_{t-1}`.)

5. **Bonus** : ecrire `ssm_kernel(A, B, C, N)` et `ssm_conv(x, K)` en numpy (ou `np.convolve`) et verifier numeriquement l'egalite avec une boucle recurrente.

### Criteres de reussite

- [ ] K = [2.0, 1.0, 0.5, 0.25] (= C*A^k*B = 2*0.5^k)
- [ ] Convolution : y = [2.0, 1.0, 0.5, 2.25] (= les memes que le mode recurrent)
- [ ] Les deux modes donnent exactement le meme y
- [ ] L'explication : conv = parallele (FFT, pas de dependance temporelle) ; recurrence = sequentielle (h_t depend de h_{t-1})
- [ ] Bonus : code numpy verifie l'egalite (diff ~ 1e-12)

---

## Exercice 3 : SSM vs RNN vs Transformer — la table de complexite

### Objectif

Savoir placer SSM, RNN et attention sur les axes complexite/parallelisme/memoire, et choisir le bon backbone selon le workload (cf le cadre mental du cours).

### Consigne

1. Remplir le tableau suivant (compute training, memoire inference par step, parallelisme training, recall associatif) :

| Architecture | Compute training | Memoire/step inference | Parallelisable au training ? | Recall associatif dense |
|---|---|---|---|---|
| RNN classique (LSTM) | ? | ? | ? | ? |
| Transformer (attention) | ? | ? | ? | ? |
| SSM lineaire (Mamba) | ? | ? | ? | ? |

2. Pour une sequence de `N=100 000` tokens, classer les 3 architectures par memoire de l'attention/etat (du moins gourmand au plus gourmand). Justifier avec les ordres de grandeur du cours.

3. **Pourquoi un RNN classique n'est PAS parallelisable** alors qu'un SSM lineaire l'est ? (Indice : la non-linearite dans la recurrence — cf idee fausse n°2 du cours.)

4. Pour chacun de ces workloads, choisir le backbone le plus adapte (cf cadre mental section 9) :
   - (a) modele audio sur sequences de 200k samples ;
   - (b) agent de code qui doit retrouver une definition de classe 40k tokens plus haut ;
   - (c) LM production avec contexte 256k ET throughput critique.

### Criteres de reussite

- [ ] RNN : compute O(N), memoire/step O(1), NON parallelisable (non-linearite dans la recurrence), recall faible
- [ ] Transformer : compute O(N^2), memoire/step O(N) (KV cache croit), parallelisable, recall excellent
- [ ] SSM : compute O(N log N) au training (FFT), memoire/step O(1), parallelisable, recall faible/moyen
- [ ] A N=100k : SSM (etat fixe) < RNN (etat fixe) << Transformer (matrice N^2 / KV cache)
- [ ] L'explication : la non-linearite (tanh/gate LSTM) dans la recurrence RNN interdit le mode conv ; le SSM est lineaire en h -> parallelisable
- [ ] Choix : (a) pure Mamba/SSM, (b) Transformer (recall dense), (c) hybride Jamba-like
