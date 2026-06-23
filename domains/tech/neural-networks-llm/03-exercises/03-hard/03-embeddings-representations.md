# Exercices Hard — Jour 3 : Embeddings & Representations

---

## Exercice 7 : Implementer GloVe from scratch et comparer avec Word2Vec

### Objectif

Comprendre la difference fondamentale entre Word2Vec (predictive) et GloVe (count-based / matrix factorization) en implementant les deux et en comparant les embeddings resultants.

### Consigne

1. **Construire la matrice de co-occurrence** a partir du corpus du Jour 3 :
   - Pour chaque paire (mot_i, mot_j) dans une fenetre de taille 2, incrementer `X[i][j]`
   - Ponderer par la distance : 1/distance (un mot a distance 1 compte plus qu'a distance 2)
   - La matrice est symetrique : `X[i][j] = X[j][i]`
   - Afficher la matrice pour les 10 mots les plus frequents

2. **Implementer GloVe** (numpy uniquement) :
   - Objectif : trouver W et b tels que `w_i . w_j + b_i + b_j ≈ log(X[i][j])` pour les paires avec X[i][j] > 0
   - Loss : `L = Σ_{i,j: X[i][j]>0} f(X[i][j]) * (w_i . w_j + b_i + b_j - log(X[i][j]))^2`
   - Fonction de ponderation : `f(x) = (x / x_max)^0.75 si x < x_max, sinon 1`
     (les paires tres frequentes ne doivent pas dominer)
   - Optimiser avec Adam (lr=0.05) pendant 200 epochs

3. **Comparer Word2Vec vs GloVe** sur les memes metriques :
   - Similarite cosinus entre paires cles (roi/reine, chat/chien, homme/femme)
   - Qualite des analogies (roi - homme + femme)
   - Similarite intra-groupe vs inter-groupe
   - Temps d'entrainement

4. **Analyser** :
   - GloVe utilise les statistiques GLOBALES (matrice complete), Word2Vec utilise des fenetres LOCALES. Quel impact sur les embeddings ?
   - Sur un petit corpus, lequel est meilleur ? Et sur un gros corpus ?
   - Combiner les deux : prendre la moyenne de l'embedding GloVe et Word2Vec pour chaque mot. Est-ce mieux ?

5. **Visualiser** : projeter les embeddings GloVe et Word2Vec en 2D avec PCA et les afficher cote a cote. Les clusters sont-ils similaires ?

### Criteres de reussite

- [ ] La matrice de co-occurrence est correcte (symetrique, ponderee par distance)
- [ ] GloVe est implemente avec la bonne loss et la fonction de ponderation f(x)
- [ ] L'optimisation converge (loss diminue)
- [ ] La comparaison est faite sur au moins 3 metriques
- [ ] L'analyse explique les differences theoriques et les confirme empiriquement
- [ ] La visualisation PCA montre les clusters des deux methodes
- [ ] Bonus : l'ensemble GloVe+Word2Vec est teste

---

## Exercice 8 : Subword embeddings (FastText) from scratch + gestion des mots inconnus

### Objectif

Implementer le mecanisme de sous-mots de FastText pour resoudre le probleme des mots hors vocabulaire (OOV) — un probleme critique en production.

### Consigne

1. **Implementer la decomposition en n-grams de caracteres** :
   ```python
   def get_ngrams(word, min_n=3, max_n=6):
       """
       Retourne les n-grams de caracteres pour un mot.
       Ajoute les marqueurs < (debut) et > (fin).

       Exemple: "chat" → ["<ch", "cha", "hat", "at>", "<cha", "chat", "hat>", "<chat", "chat>", "<chat>"]
       + le mot complet comme n-gram special
       """
       pass
   ```

2. **Implementer FastText Skip-gram** (numpy uniquement) :
   - L'embedding d'un mot = somme des embeddings de ses n-grams
   - Matrice d'embedding : (n_total_ngrams, d) — chaque n-gram unique a son propre vecteur
   - Forward : `v_word = sum(W_ngrams[ngram_ids])`
   - Le reste est identique a Word2Vec Skip-gram avec negative sampling
   - Entrainer sur le corpus du Jour 3

3. **Tester les mots inconnus (OOV)** :
   - Apres entrainement, calculer l'embedding de mots JAMAIS VUS dans le corpus :
     - "chateau" (si absent), "royaute", "animal", "chatte", "chiens"
   - L'embedding = somme des n-grams connus (ceux vus pendant l'entrainement)
   - Trouver les mots les plus similaires a ces mots OOV

4. **Comparer Word2Vec standard vs FastText** :
   - Pour les mots IN-vocabulary : la qualite des embeddings est-elle comparable ?
   - Pour les mots OUT-of-vocabulary : Word2Vec n'a rien, FastText a un embedding. Montrer la difference.
   - Impact de min_n et max_n sur la qualite

5. **Analyse morphologique** :
   - Montrer que des mots avec des prefixes/suffixes communs ont des embeddings plus proches avec FastText qu'avec Word2Vec
   - Exemples : "gouverne" / "gouvernement", "roi" / "royal"

6. **Benchmark memoire** : comparer le nombre total de parametres Word2Vec vs FastText. FastText a plus de parametres (un embedding par n-gram), mais gere les OOV. Le trade-off en vaut-il la peine ?

### Criteres de reussite

- [ ] La decomposition en n-grams est correcte (avec marqueurs < et >)
- [ ] FastText est implemente correctement (somme des n-grams)
- [ ] Le training converge
- [ ] Les mots OOV recoivent des embeddings raisonnables (proches de mots semantiquement similaires)
- [ ] La comparaison Word2Vec vs FastText est quantitative (metriques, pas juste "ca marche")
- [ ] L'analyse morphologique montre l'avantage de FastText pour les mots derives
- [ ] Le benchmark memoire est chiffre
