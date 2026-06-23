# Exercices Faciles — Jour 3 : Embeddings & Representations

---

## Exercice 1 : One-hot vs embeddings — calcul a la main

### Objectif

Comprendre concretement pourquoi one-hot est une mauvaise representation et pourquoi les embeddings denses resolvent le probleme.

### Consigne

1. Soit le vocabulaire : ["paris", "lyon", "france", "chat", "chien"]

   a. Ecrire les 5 vecteurs one-hot. Calculer la similarite cosinus entre TOUTES les paires (10 paires). Que constate-t-on ?

   b. Maintenant, inventer des embeddings denses de dimension 3 qui capturent la semantique. Par exemple :
   ```
   paris   = [0.9, 0.1, 0.8]   (ville, animal, france)
   lyon    = [0.8, 0.1, 0.7]
   france  = [0.5, 0.0, 1.0]
   chat    = [0.0, 0.9, 0.1]
   chien   = [0.1, 0.8, 0.1]
   ```

   c. Recalculer les 10 paires de similarite cosinus avec ces embeddings. Quelles paires sont maintenant les plus proches ? Est-ce semantiquement coherent ?

2. Calculer le nombre de parametres pour encoder 50 000 mots :
   - En one-hot : dimension du vecteur ?
   - En embedding dense (d=256) : taille de la matrice d'embedding ?
   - Quel est le ratio de compression ?

### Criteres de reussite

- [ ] Les 10 paires one-hot ont toutes similarite = 0 (sauf auto-similarite)
- [ ] Les embeddings denses donnent : cos(paris, lyon) > cos(paris, chat)
- [ ] Le ratio de compression one-hot vs embedding est calcule (50000 vs 256 dims)
- [ ] L'explication est claire : one-hot ne capture aucune relation, embeddings capturent la proximite semantique

---

## Exercice 2 : Construire les paires Skip-gram a la main

### Objectif

Savoir generer les paires d'entrainement (centre, contexte) pour Word2Vec Skip-gram, qui sont la matiere premiere de l'apprentissage.

### Consigne

Soit la phrase : "le chat mange du poisson frais"

1. Pour une **window size = 1** :
   - Pour chaque mot de la phrase, lister les paires (mot_central, mot_contexte)
   - Combien de paires au total ?

2. Pour une **window size = 2** :
   - Meme exercice. Combien de paires au total ?
   - Quels mots apparaissent dans le plus de paires ? Pourquoi ?

3. Pour une **window size = 3** :
   - Meme exercice.
   - Que se passe-t-il pour les mots au debut et a la fin de la phrase ?

4. Observer : quand on augmente la fenetre, quel type de relations semantiques est mieux capture ? (syntaxiques proches vs semantiques lointaines)

5. Avec ces 3 corpus :
   ```
   Corpus A : "le chat dort" + "le chien dort"
   Corpus B : "le chat mange" + "le chien joue"
   ```
   Pour chaque corpus (window=1), lister les paires. Pourquoi "chat" et "chien" auraient des embeddings plus proches dans le corpus A ?

### Criteres de reussite

- [ ] Les paires pour window=1 sont correctes (mots aux bords ont moins de paires)
- [ ] Le nombre de paires augmente avec la window size
- [ ] L'observation sur fenetre petite (syntaxe) vs grande (semantique) est juste
- [ ] L'explication corpus A vs B montre l'hypothese distributionnelle en action

---

## Exercice 3 : Similarite cosinus vs distance euclidienne

### Objectif

Comprendre quand utiliser chaque mesure et pourquoi le cosinus est standard en NLP.

### Consigne

Soit 4 vecteurs :
```
A = [1, 2, 3]
B = [2, 4, 6]       (meme direction que A, magnitude 2x)
C = [3, 2, 1]       (direction differente)
D = [0.1, 0.2, 0.3] (meme direction que A, magnitude 10x plus petite)
```

1. Calculer la similarite cosinus pour TOUTES les 6 paires : (A,B), (A,C), (A,D), (B,C), (B,D), (C,D)

2. Calculer la distance euclidienne pour les memes 6 paires

3. Classer les paires de la plus similaire a la moins similaire selon :
   - Le cosinus
   - La distance euclidienne
   Les classements sont-ils les memes ?

4. Expliquer : pourquoi A et D sont "identiques" en cosinus mais "tres differents" en distance euclidienne ? Quel est l'impact en NLP ?

5. Cas pratique : si on cherche des documents similaires a une requete dans un systeme RAG, pourquoi le cosinus est-il preferable ?

6. **Bonus** : normaliser les 4 vecteurs a norme 1. Recalculer distances euclidiennes et cosinus. Que constate-t-on ?

### Criteres de reussite

- [ ] Les 6 cosinus sont corrects (cos(A,B) = cos(A,D) = 1.0)
- [ ] Les 6 distances euclidiennes sont correctes
- [ ] Les classements divergent : cosinus groupe A,B,D ensemble (meme direction), euclidienne les separe
- [ ] L'explication NLP est claire : direction = sens, magnitude = frequence
- [ ] Bonus : apres normalisation, euclidienne et cosinus donnent le meme classement
