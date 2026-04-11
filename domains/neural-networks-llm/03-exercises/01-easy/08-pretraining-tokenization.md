# Exercices Faciles — Jour 8 : Pre-training & Tokenization

---

## Exercice 1 : BPE a la main sur un petit corpus

### Objectif

Executer mentalement l'algorithme BPE pour bien comprendre son fonctionnement etape par etape.

### Consigne

Soit le corpus :
```
"hug hug hug pug pug bun bun bunny bunny bunny"
```

1. **Pre-tokenization** : ajouter le marqueur de fin de mot `</w>` et decouper chaque mot en caracteres. Ecrire la frequence de chaque mot.

2. **Vocabulaire initial** : lister tous les tokens (caracteres + `</w>`).

3. **Iteration 1** : compter toutes les paires adjacentes dans le corpus. Quelle est la paire la plus frequente ? Quel est le token cree ?

4. **Iteration 2** : apres le premier merge, recompter les paires. Quelle est maintenant la plus frequente ?

5. **Iteration 3** : meme exercice.

6. Apres 3 merges, ecrire le vocabulaire final. Combien de tokens contient-il ?

7. **Encoder un nouveau mot** : comment "bug" est-il tokenise avec les merges appris ? Et "hungry" (mot non vu) ?

### Criteres de reussite

- [ ] Les frequences initiales sont correctes (5 mots distincts)
- [ ] Les 3 merges sont identifies correctement et dans le bon ordre
- [ ] Le vocabulaire final contient caracteres initiaux + tokens merges
- [ ] L'encodage de "bug" applique les merges dans l'ordre d'apprentissage
- [ ] Pour "hungry", tu comprends qu'on retombe sur des caracteres individuels pour les parties inconnues

---

## Exercice 2 : CLM vs MLM — ecriture de la loss

### Objectif

Savoir ecrire la loss pour CLM (GPT) et MLM (BERT) sur une phrase concrete, et comprendre pourquoi CLM a un signal plus dense.

### Consigne

Soit la phrase tokenisee : `[The, cat, sits, on, the, mat]` (6 tokens).

Supposons que le modele predit des logits a chaque position. On note `p_i[j]` la probabilite predite que le token a la position `i` soit le token `j` du vocabulaire.

1. **CLM (GPT-style)** :
   - Quel est l'input et le target a chaque position ? (indice : predire t+1 sachant t)
   - Ecrire la loss comme une somme de log-probas sur quelles positions ?
   - Combien de termes dans la somme ?

2. **MLM (BERT-style)** : supposons que les positions 2 (`sits`) et 4 (`the`) sont masquees, donc l'input devient `[The, cat, [MASK], on, [MASK], mat]`.
   - Sur quelles positions la loss est-elle calculee ?
   - Combien de termes dans la somme ?
   - Quelle est la proportion de tokens qui contribuent a la loss ?

3. **Span corruption (T5-style)** : supposons qu'on masque le span `[sits, on]` (positions 2-3) avec le sentinel `<X>`.
   - Quel est l'input de l'encoder ?
   - Quel est le target du decoder ?
   - A quoi sert le sentinel ?

4. **Analyse** : si chaque token coute 1 forward pass pour etre "entraine", combien de "forward utiles" CLM fournit-il sur 1000 tokens ? Et MLM ? Conclusion sur l'efficacite d'entrainement ?

### Criteres de reussite

- [ ] CLM : 5 termes dans la loss (positions 1 a 5, chacune predit la suivante)
- [ ] MLM : 2 termes dans la loss (seulement les positions masquees), soit 33% du signal
- [ ] Span corruption : encoder input = `[The, cat, <X>, the, mat]`, decoder target = `[<X>, sits, on, <Y>]`
- [ ] Conclusion : CLM fournit environ 6x plus de signal par token que MLM (100% vs ~15%)

---

## Exercice 3 : Scaling laws — calcul de l'optimum Chinchilla

### Objectif

Savoir appliquer la loi Chinchilla pour dimensionner un modele et un dataset.

### Consigne

Rappel de Chinchilla : pour un compute optimal, il faut environ **20 tokens par parametre**. Le compute en FLOPs pour entrainer un modele transformer est approximativement :
```
C ≈ 6 × N × D
```
ou `N` = nombre de parametres et `D` = nombre de tokens d'entrainement. Le facteur 6 vient de (forward + backward) × (matmul + activations).

1. **Cas 1 — GPT-3** : 175B parametres, 300B tokens.
   - Calculer le compute (en FLOPs).
   - Calculer le ratio D/N.
   - Est-ce proche de l'optimum Chinchilla (20) ? Sinon, le modele est-il sur-entraine ou sous-entraine ?

2. **Cas 2 — Chinchilla** : 70B parametres, 1.4T tokens.
   - Calculer le compute.
   - Calculer le ratio D/N.
   - Comparer au compute de GPT-3. Qui utilise le plus de FLOPs ?

3. **Cas 3 — Dimensionnement** : tu as un budget de `C = 1e22 FLOPs`.
   - Selon Chinchilla, quelle est la valeur optimale de N ? De D ?
   - Indice : utilise N = D / 20 et C = 6 × N × D. Remplace et resous.

4. **Cas 4 — LLaMA 3** : 8B parametres, 15T tokens.
   - Calculer le ratio D/N.
   - Ce ratio est 94x plus grand que l'optimum Chinchilla. Pourquoi Meta a-t-il fait ce choix (aider en pensant au **cout d'inference** en production) ?

5. **Bonus** : si tu doubles ton compute, selon Chinchilla, combien dois-tu augmenter `N` et `D` chacun ?

### Criteres de reussite

- [ ] Cas 1 : C ≈ 3.15e23 FLOPs, ratio D/N ≈ 1.71, GPT-3 est SOUS-ENTRAINE
- [ ] Cas 2 : C ≈ 5.88e23 FLOPs, ratio D/N = 20, Chinchilla est au point optimal
- [ ] Cas 3 : avec C = 1e22, N ≈ 9.1B, D ≈ 183B tokens (resolution de l'equation)
- [ ] Cas 4 : ratio D/N ≈ 1875. Meta sur-entraine pour reduire le cout d'inference a long terme
- [ ] Bonus : doubler C implique `N × √2` et `D × √2` (chacun multiplie par 1.41)
