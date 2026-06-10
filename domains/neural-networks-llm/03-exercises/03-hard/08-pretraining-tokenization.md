# Exercices Hard — Jour 8 : Pre-training & Tokenization

---

## Exercice 7 : WordPiece vs BPE — implementer les deux criteres de merge

### Objectif

Implementer le critere de merge de WordPiece (score par vraisemblance) a cote de celui de BPE (frequence brute) et exhiber un corpus ou ils divergent — comprendre qu'un tokenizer est un choix statistique, pas un detail.

### Consigne

1. Factoriser un trainer generique `train_tokenizer(corpus, n_merges, scoring)` ou `scoring` est une fonction `(pair_freq, freq_left, freq_right) -> score` :
   - BPE : `score = pair_freq`
   - WordPiece : `score = pair_freq / (freq_left * freq_right)` (vraisemblance mutuelle : favorise les paires dont les elements n'apparaissent QUE ensemble)
   - `freq_left/right` = frequence du token (en tant que symbole courant) dans le corpus segmente actuel
   - Tie-break deterministe identique pour les deux

2. Entrainer les deux sur le corpus :
   `["hugging"]*10 + ["hugs"]*5 + ["hug"]*4 + ["bun"]*12 + ["bug"]*3 + ["qatar"]*2`
   avec 8 merges, et afficher cote a cote la sequence des merges.

3. Analyser :
   - identifier au moins un merge ou les deux algorithmes choisissent une paire DIFFERENTE au meme rang
   - expliquer pourquoi : ex. `("q", "a")` — "q" n'apparait que dans "qatar", donc `freq("q") = freq("qa")` → score WordPiece maximal (1/freq_right), alors que sa frequence brute est trop faible pour BPE
   - verifier numeriquement les scores de la paire divergente sous les deux criteres

4. Implementer l'encodage WordPiece par **longest-match-first** (l'algorithme reel de BERT, different de l'application des merges) : decouper un mot en cherchant le plus long prefixe present dans le vocab, prefixer `##` pour les sous-mots non initiaux. Tester sur "hugging" et "unhugging" (avec fallback [UNK] si un caractere manque).

5. Comparer les segmentations des deux tokenizers sur les mots du corpus + 2 mots inconnus, dans un tableau.

### Criteres de reussite

- [ ] Le trainer est factorise (une seule boucle de merge, deux fonctions de scoring)
- [ ] Une divergence de merge est exhibee avec les scores des deux criteres calcules et affiches
- [ ] L'explication "WordPiece favorise les paires exclusives, BPE les paires frequentes" est demontree sur l'exemple
- [ ] L'encodage longest-match-first est correct (gere ##, [UNK], et le cas mot entier dans le vocab)
- [ ] Les deux runs sont deterministes et le tableau comparatif est affiche

---

## Exercice 8 : Tokenizer Unigram — segmentation Viterbi

### Objectif

Implementer le coeur du tokenizer Unigram (SentencePiece) : trouver la segmentation de probabilite maximale par programmation dynamique, et la valider contre une recherche exhaustive.

### Consigne

1. Vocabulaire impose (log-probabilites unigram) :

```python
vocab = {
    "h": -5.0, "u": -5.0, "g": -5.0, "s": -4.5, "b": -5.5, "n": -5.0,
    "hu": -3.5, "ug": -3.6, "gs": -4.8, "bu": -4.0, "un": -4.2,
    "hug": -2.2, "bun": -2.5, "hugs": -2.0, "ny": -5.5, "y": -6.0,
}
```

   La probabilite d'une segmentation = somme des log-probs de ses tokens. Le meilleur decoupage maximise cette somme.

2. Implementer `viterbi_segment(word, vocab)` :
   - `best[i]` = meilleur log-prob pour segmenter `word[:i]`, `back[i]` = indice de debut du dernier token
   - recurrence : `best[i] = max over j<i of (best[j] + logp(word[j:i]))` pour `word[j:i]` dans le vocab
   - retourner (segmentation, log-prob totale) ; retourner None si insegmentable

3. Implementer `brute_force_segment(word, vocab)` qui enumere TOUTES les segmentations possibles (recursif) et retourne la meilleure. Verifier que Viterbi == brute force (segmentation ET score) sur : "hug", "hugs", "bun", "bunny", "hughug", "bug" (insegmentable ? verifier), et 20 mots aleatoires de longueur <= 10 construits sur l'alphabet du vocab.

4. Verifier sur "hugs" que la segmentation `["hugs"]` (-2.0) bat `["hug", "s"]` (-6.7) et `["hu", "gs"]` (-8.3) — afficher le classement des 3.

5. Complexite : compter les appels a la recurrence pour des mots de longueur 4, 8, 12 et verifier la croissance ~O(L * max_token_len) du Viterbi contre l'explosion exponentielle (2^(L-1) decoupages) du brute force.

6. Question (commentaire) : pourquoi Unigram permet-il le "subword regularization" (echantillonner des segmentations sous-optimales pendant le training) la ou BPE est deterministe ?

### Criteres de reussite

- [ ] Viterbi et brute force donnent la MEME segmentation et le MEME score sur tous les cas testes
- [ ] Le cas insegmentable est gere proprement (None, pas une exception)
- [ ] Le classement des 3 segmentations de "hugs" est correct et affiche
- [ ] Les comptages demontrent l'ecart de complexite DP vs exhaustif
- [ ] La reponse sur le subword regularization est correcte (Unigram definit une DISTRIBUTION sur les segmentations, BPE une procedure deterministe)
