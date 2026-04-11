# Exercices Medium — Jour 3 : Embeddings & Representations

---

## Exercice 4 : Word2Vec CBOW from scratch

### Objectif

Implementer la variante CBOW (Continuous Bag of Words) de Word2Vec et comparer avec Skip-gram.

### Consigne

1. Reprendre le corpus du code du Jour 3 (ou un corpus similaire avec des groupes semantiques clairs)

2. Implementer CBOW from scratch (numpy uniquement) :
   - Pour chaque mot central, la prediction se fait a partir de la **moyenne** des embeddings de contexte
   - Forward pass : `h = mean(W_in[ctx_words])`, `z = h @ W_out.T`, `p = softmax(z)` (ou negative sampling)
   - Backward pass avec gradient correct a travers la moyenne
   - Utiliser negative sampling (k=5)

3. Entrainer sur le corpus pendant 100 epochs avec les memes hyperparametres que le Skip-gram du code

4. Comparer les resultats :
   - Similarite cosinus entre les memes paires de mots (roi/reine, chat/chien, homme/femme)
   - Temps d'entrainement CBOW vs Skip-gram
   - Qualite des analogies

5. Analyser : pourquoi Skip-gram est-il generalement meilleur pour les mots rares ? Illustrer avec un mot qui apparait peu dans le corpus.

### Criteres de reussite

- [ ] CBOW est implemente correctement (moyenne des embeddings de contexte)
- [ ] Le gradient a travers la moyenne est correct (chaque mot de contexte recoit 1/n du gradient)
- [ ] Le training converge (loss diminue)
- [ ] La comparaison Skip-gram vs CBOW est faite sur les memes metriques
- [ ] L'explication sur les mots rares est illustree par un exemple concret
- [ ] Le code est commente avec WHY

---

## Exercice 5 : Negative sampling — implementer et analyser les alternatives

### Objectif

Comprendre en profondeur pourquoi le negative sampling fonctionne et comment la distribution d'echantillonnage impacte la qualite des embeddings.

### Consigne

1. Implementer 3 strategies de negative sampling et comparer :
   - **Uniforme** : chaque mot a la meme probabilite d'etre echantillonne
   - **Proportionnel a la frequence** : `p(w) = freq(w) / total_freq`
   - **Standard Word2Vec** : `p(w) = freq(w)^0.75 / sum(freq^0.75)`

2. Pour chaque strategie, entrainer un Skip-gram avec les memes hyperparametres (100 epochs, k=5)

3. Mesurer pour chaque strategie :
   - Loss finale
   - Similarite intra-groupe (royaute, animaux, genre) vs inter-groupe
   - Qualite de l'analogie roi - homme + femme

4. Implementer le **softmax complet** (pas de negative sampling) comme baseline. Comparer :
   - Temps d'execution par epoch
   - Qualite des embeddings

5. Analyser :
   - Pourquoi l'uniforme est mauvais ? (indice : les mots frequents comme "le", "de" polluent l'entrainement)
   - Pourquoi l'exposant 0.75 et pas 0.5 ou 1.0 ?
   - A partir de quel k (nombre de negatives) la qualite sature ?

6. **Bonus** : tracer la courbe qualite (similarite intra-groupe) vs k pour k = [1, 2, 5, 10, 20, 50]. Trouver le sweet spot.

### Criteres de reussite

- [ ] Les 3 strategies de sampling sont implementees correctement
- [ ] Le softmax complet fonctionne comme baseline
- [ ] La comparaison de temps montre que negative sampling est beaucoup plus rapide
- [ ] La strategie `freq^0.75` donne les meilleurs resultats
- [ ] L'analyse explique POURQUOI avec des arguments quantitatifs
- [ ] Le bonus montre la saturation de qualite autour de k=5-15

---

## Exercice 6 : Embedding d'un vocabulaire custom + recherche semantique

### Objectif

Construire un mini-systeme de recherche semantique end-to-end, de l'entrainement des embeddings a la requete.

### Consigne

1. Creer un corpus thematique de 50+ phrases dans un domaine specifique (ex: cuisine, sport, technologie). Le corpus doit contenir des synonymes, des mots proches, et des mots sans rapport.

2. Entrainer un Word2Vec Skip-gram sur ce corpus (reprendre le code du Jour 3)

3. Implementer un **index de recherche semantique** :
   ```python
   class SemanticIndex:
       def __init__(self, documents, embeddings, word2idx):
           # Encoder chaque document comme la moyenne des embeddings de ses mots
           pass

       def search(self, query, top_k=3):
           # Encoder la requete, trouver les documents les plus proches (cosinus)
           pass
   ```

4. L'embedding d'un document = **moyenne ponderee** des embeddings de ses mots :
   - Ponderation TF-IDF : les mots rares dans le corpus mais presents dans le document comptent plus
   - Ignorer les mots hors vocabulaire

5. Tester avec 10 requetes et verifier que les resultats sont semantiquement pertinents

6. Comparer avec une recherche par **mots-cles exacts** (baseline). Montrer un cas ou la recherche semantique trouve un document pertinent que la recherche exacte rate (parce que les mots sont differents mais le sens est le meme).

### Criteres de reussite

- [ ] Le corpus est suffisamment riche (50+ phrases, groupes semantiques)
- [ ] L'embedding de document utilise la moyenne ponderee TF-IDF
- [ ] La classe SemanticIndex fonctionne correctement
- [ ] Au moins 7/10 requetes retournent des resultats pertinents
- [ ] Un cas concret montre l'avantage de la recherche semantique vs mots-cles
- [ ] Le code est structure et commente
