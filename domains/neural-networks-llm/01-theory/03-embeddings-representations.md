# Jour 3 — Embeddings & Representations : comment les reseaux comprennent le sens

> **Temps estime** : 5h | **Prerequis** : Jour 2 (MLP, forward/backward, optimizers)

---

## 1. Pourquoi la representation est TOUT en deep learning

### Le principe fondamental : garbage in, garbage out

Un MLP parfaitement optimise avec le meilleur optimizer du monde ne fera **rien** si les donnees d'entree sont mal representees. La facon dont tu encodes tes donnees est plus importante que l'architecture du reseau.

**Analogie** : imagine que tu veux enseigner a un enfant ce qu'est un "chat". Tu peux soit :
- Lui donner un numero d'identifiant : "chat = 42" — il ne peut rien en deduire
- Lui donner une description riche : "4 pattes, fourrure, moustaches, ronronne, 3kg" — il peut generaliser

Le reseau de neurones est comme l'enfant : il a besoin d'une representation **riche et structuree** pour apprendre.

### Ce que doit capturer une bonne representation

1. **Similarite** : les concepts proches doivent avoir des representations proches
2. **Structure** : les relations entre concepts doivent etre encodees geometriquement
3. **Compacite** : la representation doit etre dense (pas de dimensions inutiles)
4. **Generalisation** : elle doit capturer les proprietes essentielles, pas le bruit

---

## 2. One-hot encoding : pourquoi c'est nul

### Le concept

On represente chaque mot (ou categorie) par un vecteur binaire avec un seul 1 :

```
Vocabulaire : [chat, chien, roi, reine, homme, femme]

chat  = [1, 0, 0, 0, 0, 0]
chien = [0, 1, 0, 0, 0, 0]
roi   = [0, 0, 1, 0, 0, 0]
reine = [0, 0, 0, 1, 0, 0]
homme = [0, 0, 0, 0, 1, 0]
femme = [0, 0, 0, 0, 0, 1]
```

### Les 3 problemes fondamentaux

**Probleme 1 — Aucune semantique**

La similarite cosinus entre TOUS les paires de vecteurs one-hot est 0 :

```
cos(chat, chien) = 0
cos(chat, roi)   = 0
cos(roi, reine)  = 0
```

Pour le reseau, "chat" est aussi different de "chien" que de "roi". Aucune notion de proximite semantique.

**Probleme 2 — Sparse et inefficace**

Avec un vocabulaire de 50 000 mots, chaque vecteur a 50 000 dimensions avec un seul 1. C'est du gaspillage : 99.998% des valeurs sont des zeros.

```
Vocabulaire V = 50 000 mots
Taille d'un vecteur one-hot = 50 000 floats = 200 KB par mot
Matrice de poids pour la premiere couche : 50 000 x d_hidden
Avec d_hidden = 512 : 25.6 millions de parametres juste pour l'entree !
```

**Probleme 3 — Curse of dimensionality**

En haute dimension (50 000), les points deviennent equidistants. La notion de "voisin proche" perd son sens. Le reseau a besoin exponentiellement plus de donnees pour couvrir cet espace.

### Le constat

Il faut une representation ou :
- Les mots similaires ont des vecteurs proches
- La dimension est fixe et raisonnable (50-1024, pas 50 000)
- Les relations semantiques sont capturees geometriquement

C'est exactement ce que font les **embeddings**.

---

## 3. Word2Vec from scratch : l'idee geniale

### "A word is known by the company it keeps" — J.R. Firth (1957)

L'hypothese distributionnelle : le sens d'un mot est defini par les mots qui l'entourent.

```
"Le chat dort sur le tapis"
"Le chien dort sur le canape"

"chat" et "chien" apparaissent dans les memes contextes
  → ils doivent avoir des representations similaires
```

Word2Vec (Mikolov et al., 2013) transforme cette intuition en un algorithme d'apprentissage simple et elegant.

### 3.1 Skip-gram : predire le contexte a partir du mot central

**Tache** : etant donne un mot central, predire les mots qui l'entourent.

```
Phrase : "Le roi gouverne le royaume avec sagesse"

Mot central : "gouverne" (window = 2)
Contexte : ["roi", "le", "royaume", "avec"]

Paires d'entrainement (centre, contexte) :
  (gouverne, roi)
  (gouverne, le)
  (gouverne, royaume)
  (gouverne, avec)
```

**Architecture** (etonnamment simple) :

```
Input              Hidden              Output
(one-hot V)        (embedding d)       (proba V)

  V dims              d dims              V dims
  ┌───┐              ┌───┐              ┌───┐
  │ 0 │              │   │              │p_1│
  │ 0 │    W_in      │   │    W_out     │p_2│
  │ 1 │ ────────→    │   │ ────────→    │...│
  │ 0 │  (V x d)     │   │  (d x V)    │p_V│
  │ 0 │              │   │              │   │
  └───┘              └───┘              └───┘

W_in  : (V, d) — matrice d'embedding d'entree
W_out : (d, V) — matrice de prediction du contexte
```

**Forward pass** :

```
1. Le mot central est encode en one-hot : x = [0, 0, ..., 1, ..., 0]  (taille V)
2. Le vecteur d'embedding est extrait : h = W_in[mot_idx]  (taille d)
   (c'est juste un lookup — multiplier one-hot par W revient a selectionner une ligne)
3. Le score de chaque mot du vocabulaire : z = h @ W_out  (taille V)
4. La distribution de probabilites : p = softmax(z)  (taille V)
5. Loss : cross-entropy entre p et le vrai mot de contexte
```

**Ce qu'on apprend** : les lignes de W_in sont les embeddings. Apres entrainement, W_in[chat] et W_in[chien] seront proches car ils predisent les memes mots de contexte.

### 3.2 CBOW : predire le mot central a partir du contexte

CBOW (Continuous Bag of Words) fait l'inverse : les mots de contexte predisent le mot central.

```
Contexte : ["roi", "le", "royaume", "avec"]
                         ↓
                  Predire : "gouverne"
```

**Architecture** :

```
1. Encoder chaque mot de contexte avec W_in
2. Faire la moyenne des embeddings : h = mean(W_in[roi], W_in[le], W_in[royaume], W_in[avec])
3. Predire le mot central : z = h @ W_out, p = softmax(z)
```

**CBOW vs Skip-gram** :

| | Skip-gram | CBOW |
|---|---|---|
| Tache | Centre → contexte | Contexte → centre |
| Mots rares | Meilleur (chaque occurrence genere des paires) | Moins bon (noye dans la moyenne) |
| Vitesse | Plus lent (plus de paires par mot) | Plus rapide |
| Usage | Standard pour la qualite | Standard pour la vitesse |

### 3.3 Negative sampling : l'astuce qui rend tout possible

**Le probleme** : le softmax sur V mots est un goulot d'etranglement.

```
softmax(z_i) = exp(z_i) / Σ_{j=1}^{V} exp(z_j)

Avec V = 50 000 mots : il faut calculer 50 000 exponentielles a CHAQUE forward pass,
pour CHAQUE paire d'entrainement. C'est O(V) par sample.

Avec des millions de paires et V = 50 000 : impraticable.
```

**La solution — Negative Sampling** : au lieu de predire le bon mot parmi TOUS les mots, on reformule le probleme en classification binaire.

Pour chaque paire (centre, contexte) positive :
1. C'est une paire VRAIE → label = 1
2. On tire k mots aleatoires (negativs) → label = 0

```
Paire positive : (gouverne, royaume) → label = 1

Paires negatives (k=5) :
  (gouverne, banane)   → label = 0
  (gouverne, ciment)   → label = 0
  (gouverne, violon)   → label = 0
  (gouverne, parking)  → label = 0
  (gouverne, Jupiter)  → label = 0
```

**Nouvelle loss** (pour une paire positive et k negatives) :

```
L = -log(σ(v_contexte · v_centre)) - Σ_{neg} log(σ(-v_neg · v_centre))

σ = sigmoid
v_centre  = embedding du mot central (ligne de W_in)
v_contexte = embedding du mot de contexte (colonne de W_out)
v_neg     = embeddings des mots negatifs
```

**Intuition** : on veut que le produit scalaire soit GRAND pour les vraies paires et PETIT pour les fausses paires. Le sigmoid transforme ca en probabilite.

**Complexite** : O(k) au lieu de O(V). Avec k = 5-20 et V = 50 000, c'est 2 500x a 10 000x plus rapide.

**Comment choisir les negatifs ?** Pas uniformement ! On echantillonne proportionnellement a `freq(mot)^{3/4}`. L'exposant 3/4 donne plus de chance aux mots rares (vs leur frequence brute) tout en sur-representant les mots frequents par rapport a l'uniforme.

```
Mot "le"    : freq = 0.05  → freq^0.75 = 0.0188
Mot "chat"  : freq = 0.0001 → freq^0.75 = 0.000316

Ratio brut : 500x
Ratio ajuste : 59x  (le mot rare a plus de chances d'etre echantillonne)
```

### 3.4 Ce qui emerge : analogies vectorielles

Le resultat le plus spectaculaire de Word2Vec : les **relations semantiques deviennent des operations vectorielles**.

```
vec("roi") - vec("homme") + vec("femme") ≈ vec("reine")

Interpretation geometrique :
  Le vecteur (roi - homme) capture la direction "royaute"
  Ajouter cette direction a "femme" → "reine"

  homme ─────────────→ roi
    |                    |
    | vec "genre"        | vec "genre"
    |                    |
  femme ─────────────→ reine
         vec "royaute"
```

D'autres analogies qui emergent :

```
Paris - France + Italie ≈ Rome        (capitales)
walked - walking + swimming ≈ swam    (conjugaison)
big - bigger + cold ≈ colder          (comparatif)
```

**Clusters semantiques** : dans l'espace d'embedding, les mots s'organisent en groupes thematiques.

```
Espace 2D (projection) :

      [roi] [reine]
      [prince] [princesse]
                                [voiture]
  [homme] [femme]               [moto]
  [garcon] [fille]              [velo]
                                [bus]
          [chat] [chien]
          [lion] [tigre]
```

Les mots de royaute sont proches entre eux. Les animaux forment un cluster. Les vehicules aussi. Et les relations homme/femme sont des translations paralleles.

---

## 4. Espaces vectoriels : mesurer la similarite

### 4.1 Similarite cosinus

La mesure standard pour comparer des embeddings :

```
cos(A, B) = (A · B) / (||A|| * ||B||)

A · B     = Σ a_i * b_i         (produit scalaire)
||A||     = √(Σ a_i^2)          (norme L2)
```

**Proprietes** :
- Resultat entre -1 et 1
- 1 = meme direction (tres similaire)
- 0 = orthogonal (pas de relation)
- -1 = directions opposees (antonymes parfois)

**Pourquoi cosinus et pas distance euclidienne ?**

La similarite cosinus mesure l'**angle** entre les vecteurs, pas leur magnitude. C'est crucial car :

```
Cas 1 : vecteurs de magnitudes differentes mais meme direction
  A = [1, 2, 3]
  B = [2, 4, 6]    (B = 2*A)

  cos(A, B) = 1.0       ← identiques en direction
  dist(A, B) = ||A|| = 3.74  ← tres differents en distance !

Cas 2 : meme magnitude mais directions differentes
  A = [1, 0, 0]
  B = [0, 1, 0]

  cos(A, B) = 0.0       ← orthogonaux
  dist(A, B) = √2 = 1.41  ← relativement proches en distance
```

En NLP, la direction d'un embedding capture le **sens** (quels concepts sont actives), tandis que la magnitude capture la **frequence** ou l'**intensite**. On veut comparer le sens, donc on utilise le cosinus.

### 4.2 Distance euclidienne

```
dist(A, B) = √(Σ (a_i - b_i)^2)
```

**Quand l'utiliser** :
- Apres normalisation L2 (tous les vecteurs sur la sphere unite) : distance euclidienne et cosinus sont equivalents
- Pour les espaces ou la magnitude a du sens (pixels, coordonnees spatiales)
- Pour le clustering avec k-means (qui minimise la distance euclidienne)

### 4.3 En pratique

```
Recherche semantique (RAG, similarity search) → cosinus
k-means clustering → euclidienne (ou cosinus apres normalisation)
Classification (MLP apres embedding) → pas de mesure explicite, le MLP apprend
Retrieval avec FAISS/Annoy → cosinus ou dot product
```

**Astuce** : si on normalise tous les vecteurs a norme 1, alors :
```
dist(A, B)^2 = 2 - 2 * cos(A, B)
```
Les deux mesures donnent le meme classement. C'est ce que font la plupart des systemes de retrieval en pratique.

---

## 5. Au-dela de Word2Vec : chaque innovation resout un probleme

### 5.1 GloVe — Global Vectors (Pennington et al., 2014)

**Probleme de Word2Vec** : il utilise seulement des fenetres locales. Les statistiques globales de co-occurrence sont perdues.

**Idee de GloVe** : construire une matrice de co-occurrence globale, puis la factoriser.

```
Matrice de co-occurrence X (V x V) :
  X[i][j] = nombre de fois que le mot j apparait dans le contexte du mot i

         chat  chien  dort  mange  os
  chat     -     5     12    8     1
  chien    5     -     10    9     7
  dort    12    10      -    0     0
  mange    8     9      0    -     3
  os       1     7      0    3     -
```

**Objectif** : trouver des vecteurs w_i et w_j tels que :
```
w_i · w_j + b_i + b_j ≈ log(X[i][j])
```

Le produit scalaire de deux embeddings doit approximer le log de leur co-occurrence.

**Avantage** : combine les statistiques globales (matrice complete) avec l'efficacite de l'entrainement (SGD sur les paires non-nulles).

### 5.2 FastText — embeddings de sous-mots (Bojanowski et al., 2017)

**Probleme de Word2Vec et GloVe** : un mot inconnu (out-of-vocabulary / OOV) n'a pas d'embedding. "electroencephalogramme" est traite comme completement inconnu.

**Idee de FastText** : representer chaque mot comme la somme de ses n-grams de caracteres.

```
Mot : "orange"
N-grams (n=3) : <or, ora, ran, ang, nge, ge>

Embedding("orange") = emb(<or) + emb(ora) + emb(ran) + emb(ang) + emb(nge) + emb(ge>) + emb(orange)
```

**Avantage** : meme un mot jamais vu peut avoir un embedding raisonnable si ses sous-mots sont connus.

```
"electroencephalogramme" = emb(ele) + emb(lec) + emb(ect) + emb(ctr) + ... + emb(mme)

Les sous-mots comme "electro", "encephalo", "gramme" sont partages avec d'autres mots
  → l'embedding capture la morphologie
```

### 5.3 ELMo — Embeddings contextuels (Peters et al., 2018)

**Probleme de Word2Vec, GloVe et FastText** : un mot a UN SEUL embedding, quel que soit le contexte.

```
"La banque du fleuve est inondee"    → banque = rive
"La banque centrale monte les taux"  → banque = institution financiere

Word2Vec : meme vecteur pour "banque" dans les deux cas !
```

**Idee d'ELMo** : utiliser un BiLSTM (reseau recurrent bidirectionnel) pour generer l'embedding en fonction du contexte.

```
Phrase : "La banque du fleuve"
         ←←←←←←←←←←←←←←←←←←   (backward LSTM)
         →→→→→→→→→→→→→→→→→→→   (forward LSTM)

Embedding("banque") = f(etats caches du LSTM au mot "banque")

Ce vecteur est DIFFERENT si le contexte est "fleuve" vs "centrale"
```

**Revolution** : c'est le debut des embeddings contextuels. ELMo a ouvert la voie a BERT et GPT.

### Timeline des innovations

```
2013 — Word2Vec        Embeddings statiques, fenetre locale
         ↓  (probleme : pas de stats globales)
2014 — GloVe           Factorisation de matrice de co-occurrence
         ↓  (probleme : mots inconnus)
2017 — FastText         Embeddings de sous-mots (n-grams)
         ↓  (probleme : un seul embedding par mot, pas de contexte)
2018 — ELMo            Embeddings contextuels (BiLSTM)
         ↓  (probleme : BiLSTM pas assez puissant, pas parallelisable)
2018 — BERT/GPT        Transformers (attention is all you need)
         ↓  (on y vient au Jour 5)
```

---

## 6. Embeddings dans les LLMs modernes

### Token embeddings + position embeddings

Dans un Transformer (GPT, BERT, LLaMA...), l'entree est la somme de deux embeddings :

```
Embedding final = Token Embedding + Position Embedding

Token Embedding  : le "sens" du token (comme Word2Vec, mais learnable)
Position Embedding : l'"adresse" du token dans la sequence
```

**Pourquoi des position embeddings ?** Un Transformer traite tous les tokens en parallele (pas sequentiellement comme un LSTM). Sans information de position, "le chat mange la souris" et "la souris mange le chat" auraient la meme representation.

### La lookup table = matrice de poids learnable

L'embedding n'est PAS un algorithme complexe. C'est juste une matrice de poids :

```
Vocabulaire V = 50 000 tokens
Dimension d = 768 (BERT-base)

Embedding matrix E : (V, d) = (50 000, 768)
  → 38.4 millions de parametres

Pour le token "chat" (id = 4287) :
  embedding = E[4287]   ← juste un lookup de ligne !
```

C'est exactement comme le W_in de Word2Vec, sauf que :
1. La matrice est initialisee aleatoirement (pas pre-entrainee avec des paires)
2. Elle est apprise end-to-end avec le reste du modele (backprop a travers tout)
3. La taille est beaucoup plus grande (des millions de tokens possibles)

### Dimensions typiques

| Modele | V (vocab size) | d (embedding dim) | Params embedding |
|---|---|---|---|
| BERT-base | 30 522 | 768 | 23.4M |
| GPT-2 | 50 257 | 768 | 38.6M |
| GPT-3 | 50 257 | 12 288 | 617M |
| LLaMA-2-7B | 32 000 | 4 096 | 131M |
| LLaMA-3-8B | 128 256 | 4 096 | 525M |

**Observation** : la matrice d'embedding represente souvent 5-15% des parametres totaux du modele. C'est une des plus grandes "tables de lookup" au monde.

### Comment le gradient atteint l'embedding

```
Forward :
  token_id = 42
  embedding = E[42]           ← lookup (pas de multiplication)
  ... passe a travers attention, FFN, etc.
  loss = cross_entropy(output, target)

Backward :
  ∂loss/∂E[42] = gradient qui arrive de la couche suivante
  E[42] -= lr * ∂loss/∂E[42]  ← on met a jour SEULEMENT la ligne 42 !

Les lignes pour les tokens absents du batch ne sont pas modifiees.
C'est une mise a jour sparse — tres efficace.
```

---

## 7. Applications : pourquoi les embeddings sont la base de tout

### 7.1 Recherche semantique (la base du RAG)

```
1. Encoder tous les documents en embeddings (avec un modele comme text-embedding-3)
2. Encoder la question de l'utilisateur en embedding
3. Trouver les documents les plus proches (similarite cosinus)
4. Les passer au LLM comme contexte

Question : "Comment nourrir un chat ?"
Embedding question ───→ cos similarity ───→ Top-3 documents les plus proches
                                            ↓
                                   LLM genere la reponse
```

Sans embeddings de qualite, le RAG ne peut pas trouver les bons documents.

### 7.2 Clustering

Regrouper des textes par theme sans labels :

```
Embeddings de 10 000 tickets de support
         ↓
    k-means (k=20)
         ↓
Cluster 1 : problemes de paiement
Cluster 2 : questions de livraison
Cluster 3 : bugs techniques
...
```

### 7.3 Classification

Les embeddings comme features d'entree pour un classifieur :

```
Texte → embedding (1536 dims) → MLP [1536, 256, num_classes] → classe

Beaucoup plus efficace que d'utiliser des features artisanales (bag-of-words, TF-IDF)
```

### 7.4 Recommandation

Meme principe que la recherche semantique, mais avec des items :

```
Embedding(utilisateur) ~ Embedding(produit_aime)

Recommander : les produits dont l'embedding est le plus proche de l'utilisateur
```

### Le point cle

Les embeddings sont le **langage commun** entre le monde reel (texte, images, audio) et le monde mathematique (vecteurs, matrices, gradients). Tout systeme d'IA moderne passe par cette etape de transformation en vecteurs denses.

---

## 8. Flash Cards — Active Recall

### Q1 : Pourquoi one-hot encoding est une mauvaise representation pour les mots ? (3 raisons)

<details>
<summary>Reponse</summary>

1. **Aucune semantique** : la similarite cosinus entre toutes les paires de vecteurs one-hot est 0. "Chat" et "chien" sont aussi differents que "chat" et "roi".

2. **Sparse et inefficace** : avec V = 50 000 mots, chaque vecteur a 50 000 dimensions avec un seul 1. 99.998% de zeros.

3. **Curse of dimensionality** : en haute dimension, tous les points deviennent equidistants. Il faut exponentiellement plus de donnees.

La solution : des embeddings denses (50-1024 dims) ou les mots similaires ont des vecteurs proches.

</details>

### Q2 : Quelle est la difference entre Skip-gram et CBOW ? Lequel est meilleur pour les mots rares ?

<details>
<summary>Reponse</summary>

- **Skip-gram** : etant donne le mot CENTRAL, predire les mots de CONTEXTE
- **CBOW** : etant donne les mots de CONTEXTE, predire le mot CENTRAL

Skip-gram est meilleur pour les mots rares car chaque occurrence d'un mot genere plusieurs paires d'entrainement (une par mot de contexte). Avec CBOW, le mot rare est noye dans la moyenne des embeddings de contexte.

</details>

### Q3 : Pourquoi utilise-t-on le negative sampling au lieu du softmax complet ?

<details>
<summary>Reponse</summary>

Le softmax complet est en O(V) car il faut calculer `exp(z_j)` pour CHAQUE mot du vocabulaire (ex: V = 50 000) a chaque forward pass.

Le negative sampling reformule le probleme en classification binaire : on entraine sur la paire positive + k paires negatives (k = 5-20). Complexite O(k) au lieu de O(V).

Les mots negatifs sont echantillonnes selon `freq(mot)^{3/4}` pour donner plus de chances aux mots rares tout en sur-representant les mots frequents.

</details>

### Q4 : Pourquoi utilise-t-on la similarite cosinus plutot que la distance euclidienne pour comparer des embeddings ?

<details>
<summary>Reponse</summary>

La similarite cosinus mesure l'**angle** entre les vecteurs (la direction), pas leur **magnitude**.

En NLP, la direction d'un embedding capture le **sens** (quels concepts sont actives), tandis que la magnitude capture la **frequence/intensite**. Deux embeddings de meme sens mais de magnitudes differentes doivent etre consideres comme similaires.

Cas ou la distance euclidienne convient : apres normalisation L2 (sphere unite), ou quand la magnitude a du sens (coordonnees spatiales, pixels).

Astuce : si tous les vecteurs sont normalises, `dist(A,B)^2 = 2 - 2*cos(A,B)` — les deux mesures sont equivalentes.

</details>

### Q5 : Comment fonctionne l'embedding dans un LLM moderne ? Qu'est-ce que la matrice d'embedding ?

<details>
<summary>Reponse</summary>

L'embedding d'un LLM est une **matrice de poids learnable** E de taille (V, d), ou V = taille du vocabulaire et d = dimension d'embedding.

Pour un token d'id `i`, l'embedding est simplement `E[i]` — un lookup de ligne (pas de multiplication matricielle).

L'embedding final est la somme de :
- **Token embedding** : le sens du token (ligne de la matrice E)
- **Position embedding** : l'adresse du token dans la sequence

Le gradient ne met a jour que les lignes correspondant aux tokens du batch (mise a jour sparse). La matrice E represente typiquement 5-15% des parametres du modele.

</details>
