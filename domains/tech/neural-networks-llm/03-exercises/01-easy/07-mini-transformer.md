# Exercices Faciles — Jour 7 : Mini-Transformer (Capstone)

---

## Exercice 1 : Calcul d'attention a la main sur 2 tokens

### Objectif

Faire tourner un mini-GPT a la main sur un exemple minimal (2 tokens, 1 tete) pour voir toutes les pieces bouger ensemble.

### Consigne

Soit un mini-GPT avec :
- `vocab_size = 4` (tokens : `{A, B, C, D}`)
- `n_embed = 2`
- `n_head = 1`
- `block_size = 2`

Les parametres simplifies :

```
Token embedding table E (4, 2) :
  A: [1, 0]
  B: [0, 1]
  C: [1, 1]
  D: [-1, 0]

Positional embedding (2, 2) :
  pos 0: [0.1, 0.0]
  pos 1: [0.0, 0.1]

Attention projections (2 -> 2) :
  W_Q = [[1, 0], [0, 1]]   (identite)
  W_K = [[1, 0], [0, 1]]   (identite)
  W_V = [[1, 0], [0, 1]]   (identite)
  W_O = [[1, 0], [0, 1]]   (identite)
```

On ignore (pour simplifier) : LayerNorm, FFN, residuals.

Input : la sequence `[A, B]` (token ids `[0, 1]`).

1. **Embeddings** :
   - Calculer `token_emb(A) + pos_emb(0)` = ?
   - Calculer `token_emb(B) + pos_emb(1)` = ?
   - Ecrire la matrice `X` de shape `(2, 2)`.

2. **Calcul de Q, K, V** :
   - `Q = X @ W_Q`, `K = X @ W_K`, `V = X @ W_V`
   - Avec `W_Q = W_K = W_V = I`, donner Q, K, V.

3. **Scores d'attention** :
   - Calculer `scores = Q @ K^T` (matrice 2x2).
   - Diviser par `sqrt(d_head) = sqrt(2) ≈ 1.414`.

4. **Masque causal** :
   - Ecrire la matrice de masque causale 2x2 (1 = autorise, 0 = bloque).
   - Remplacer les positions bloquees par `-inf` dans la matrice de scores.

5. **Softmax** :
   - Appliquer softmax sur chaque ligne.
   - Observer : pour la ligne 0 (token A), quelle est la distribution ? Pour la ligne 1 (token B) ?

6. **Output** :
   - Calculer `output = weights @ V` (shape 2x2).

7. **Interpretation** : pour le token A (position 0), son output depend-il de B ? Pourquoi ? C'est l'effet du masque causal.

### Criteres de reussite

- [ ] Embeddings : X = `[[1.1, 0.0], [0.0, 1.1]]` (approximativement)
- [ ] Scores avant masque : approximativement diagonaux (car les tokens sont orthogonaux)
- [ ] Apres masque causal, le score (0, 1) devient -inf
- [ ] Softmax ligne 0 : `[1.0, 0.0]` (token A ne voit que lui-meme)
- [ ] Softmax ligne 1 : deux valeurs sommant a 1 (token B voit A et lui-meme)
- [ ] L'explication du masque : A ne peut pas regarder B car B est "dans le futur" pour A

---

## Exercice 2 : Modifier le mini-GPT pour utiliser 4 tetes au lieu de 2

### Objectif

Comprendre comment le nombre de tetes affecte les dimensions internes.

### Consigne

Le mini-GPT du cours est configure avec :
```python
n_embed = 48
n_head = 4
```
Donc `head_dim = 48 / 4 = 12`.

1. **Calculer les dimensions** pour une variante avec `n_head = 8` au lieu de 4 :
   - Nouvelle valeur de `head_dim` ?
   - Est-ce un entier ? (Si non, qu'est-ce que ca signifie et quelle est la contrainte ?)

2. **Comparer les parametres** de l'attention pour les deux configs :
   - Avec `n_head = 4` : combien de params dans l'attention (W_Q, W_K, W_V, W_O) ?
   - Avec `n_head = 8` : combien de params ?
   - Les deux ont-ils le meme nombre de params ?

3. **Explication** : donner l'intuition derriere ce resultat. (Indice : les projections Q, K, V sont `n_embed → n_embed`, independamment du nombre de tetes.)

4. **Modification de code** : ecrire la modification a apporter a la config du mini-GPT (juste les 1-2 lignes).

5. **Question conceptuelle** : si deux configs (4 heads vs 8 heads) ont le MEME nombre de parametres, laquelle est generalement meilleure ? Y a-t-il un tradeoff ?

6. **Bonus** : a ton avis, pourquoi GPT-3 utilise 96 tetes (et GPT-2 en utilise 12-25) alors que le cout est identique ? Quel est l'avantage de plus de tetes ?

### Criteres de reussite

- [ ] `head_dim` pour 8 tetes = 6 (= 48 / 8)
- [ ] La contrainte : `n_embed` doit etre divisible par `n_head`
- [ ] Les deux configs ont exactement le MEME nombre de parametres d'attention (4 * n_embed^2 = 4 * 48^2 = 9216, sans biais)
- [ ] L'explication : les projections sont `n_embed → n_embed` dans les deux cas, la difference est juste comment on "reshape" le resultat
- [ ] La modification : changer `n_head=4` en `n_head=8` dans la config (et `n_embed` doit rester divisible)
- [ ] Plus de tetes = plus de sous-espaces d'attention differents, donc plus de capacite de "specialisation" par tete, mais chaque tete a moins de dimensions pour elle

---

## Exercice 3 : Temperature sampling — effet sur la generation

### Objectif

Comprendre comment la temperature controle le tradeoff determinisme vs diversite en generation.

### Consigne

Soit les logits d'un modele pour 5 tokens :
```
logits = [2.0, 1.0, 0.5, 0.0, -1.0]
```

1. **Temperature = 1.0 (standard)** :
   - Calculer `softmax(logits)`. Approximation : `exp(2)=7.39, exp(1)=2.72, exp(0.5)=1.65, exp(0)=1.0, exp(-1)=0.37`.
   - Somme = ?
   - Probabilites = ?
   - Quel token est le plus probable ?

2. **Temperature = 0.5 (plus pointu)** :
   - Calculer `logits / 0.5 = [4.0, 2.0, 1.0, 0.0, -2.0]`.
   - Applique softmax. `exp(4)=54.6, exp(2)=7.39, exp(1)=2.72, exp(0)=1.0, exp(-2)=0.135`.
   - Probabilites ?
   - Observation : le token le plus probable a-t-il augmente ou diminue sa probabilite ?

3. **Temperature = 2.0 (plus plat)** :
   - Calculer `logits / 2.0 = [1.0, 0.5, 0.25, 0.0, -0.5]`.
   - Applique softmax. `exp(1)=2.72, exp(0.5)=1.65, exp(0.25)=1.28, exp(0)=1.0, exp(-0.5)=0.61`.
   - Probabilites ?
   - Observation : le token le plus probable a-t-il augmente ou diminue sa probabilite ?

4. **Greedy vs sampling** :
   - Le greedy prend toujours le token avec la plus haute proba. Quel token sera toujours choisi ici ?
   - Pourquoi c'est problematique pour la generation de texte long ? (Indice : pense a la diversite et aux boucles).

5. **Cas limites** :
   - Que se passe-t-il quand `T → 0` ?
   - Que se passe-t-il quand `T → infini` ?

6. **Cas pratique** : quelle temperature choisirais-tu pour :
   - Generer du code (besoin de precision) ?
   - Generer un poeme (besoin de creativite) ?
   - Repondre a une question factuelle ?

### Criteres de reussite

- [ ] T=1.0 : probas approx `[0.57, 0.21, 0.13, 0.08, 0.03]`, token 0 a 57%
- [ ] T=0.5 : probas approx `[0.83, 0.11, 0.04, 0.015, 0.002]`, token 0 a 83% (plus concentre)
- [ ] T=2.0 : probas approx `[0.36, 0.22, 0.17, 0.13, 0.08]`, token 0 a 36% seulement (plus plat)
- [ ] Greedy choisit toujours le token 0 — deterministe, sujet a boucles
- [ ] T=0 → one-hot (equivalent greedy), T=inf → uniforme
- [ ] Code : T basse (~0.1-0.3), poeme : T haute (~0.9-1.2), question factuelle : T basse (~0.1-0.5)
