# Exercices Medium — Jour 14 : Capstone (extensions de mini-LLaMA)

> Ces exercices ETENDENT le mini-LLaMA de `02-code/14-capstone.py`. Le code de
> reference est en PyTorch ; les solutions fournies implementent les memes idees
> en NumPy (entrainables sans framework) pour que tu puisses tout faire tourner.
> Tu peux aussi etendre directement le fichier PyTorch si torch est installe.

---

## Exercice 4 : Entrainer le mini-LLaMA sur un petit corpus + courbe de loss

### Objectif

Passer du mini-LLaMA "poids aleatoires" (qui genere du charabia) a un modele REELLEMENT entraine sur un petit corpus, et tracer la courbe de loss pour verifier l'apprentissage.

### Consigne

1. Reprendre le mini-LLaMA (`02-code/14-capstone.py`) et le `CharTokenizer`.

2. Construire un petit corpus de caracteres (quelques phrases repetees, ou un court texte) et le decouper en batches de sequences `(batch, seq_len)` avec les cibles decalees de 1 (next-token prediction).

3. **Boucle d'entrainement** :
   - Forward → logits `(batch, seq_len, vocab)`
   - Loss = cross-entropy next-token sur toute la sequence
   - Backward + optimizer (AdamW si PyTorch ; en NumPy, implementer le forward/backward d'un mini-LM trainable)
   - Logger la loss a chaque epoch

4. **Courbe de loss** : afficher (ou tracer en ASCII / matplotlib) la loss au fil des epochs. Elle doit DESCENDRE nettement.

5. **Generation avant/apres** : generer un echantillon AVANT entrainement (charabia) et APRES (le modele doit reproduire des motifs du corpus). Comparer.

6. Analyser :
   - La loss converge-t-elle vers une valeur basse ? Que vaudrait la loss d'un modele parfait sur ce corpus (lien avec l'entropie du corpus) ?
   - Le modele overfit-il (corpus minuscule) ? Comment le verrais-tu ?

### Criteres de reussite

- [ ] Le corpus est correctement tokenise et batche (cibles decalees de 1)
- [ ] La boucle d'entrainement tourne et la loss DESCEND nettement
- [ ] La courbe de loss est affichee/tracee
- [ ] La generation apres entrainement reproduit des motifs du corpus (pas du charabia)
- [ ] L'analyse loss/entropie/overfitting est correcte

---

## Exercice 5 : Verifier que le KV cache donne les MEMES logits que le forward complet

### Objectif

Prouver empiriquement que la generation avec KV cache produit exactement les memes logits que le forward "complet" (re-passe sur toute la sequence a chaque step), et mesurer le speedup.

### Consigne

1. Reprendre l'attention du mini-LLaMA (GQA + RoPE + KV cache).

2. **Mode A — sans cache (reference)** : pour generer le token a la position `t`, re-passer le modele sur TOUTE la sequence `[0..t]` et prendre les logits de la derniere position.

3. **Mode B — avec cache** : prefill sur le prompt, puis decode token par token en n'injectant que le NOUVEAU token (start_pos correct pour RoPE et l'indexation du cache).

4. **Equivalence** : pour une sequence donnee, comparer a chaque step les logits du mode A et du mode B. L'ecart doit etre < 1e-4 (idealement < 1e-6). Si ce n'est pas le cas, c'est souvent un bug de positions RoPE ou d'indexation du cache — debugger.

5. **Speedup** : timer la generation de N tokens avec et sans cache. Montrer que le cache est plus rapide et que l'ecart grandit avec la longueur.

6. **Piege RoPE** : expliquer pourquoi `start_pos` doit etre l'absolu (pas 0) lors du decode. Que se passe-t-il si on oublie d'incrementer `start_pos` ? (les logits divergent du mode sans cache)

### Criteres de reussite

- [ ] Le mode sans cache (reference) et le mode avec cache sont implementes
- [ ] Les logits des deux modes sont identiques a chaque step (ecart < 1e-4)
- [ ] Le speedup du cache est mesure et grandit avec la longueur
- [ ] L'importance de `start_pos` pour RoPE et l'indexation est expliquee
- [ ] Un test "start_pos oublie" montre la divergence (preuve par l'absurde)

---

## Exercice 6 : Perplexite avant/apres entrainement + weight tying

### Objectif

Implementer la perplexite (la metrique standard des LMs) et l'utiliser pour quantifier l'amelioration apportee par l'entrainement, puis mesurer l'effet du weight tying sur le nombre de parametres.

### Consigne

1. **Perplexite** : implementer `perplexity(model, tokens)` :
   ```
   PPL = exp( -(1/N) * sum_i log p(token_i | token_<i) )
   ```
   C'est l'exponentielle de la cross-entropy moyenne (en nats). PPL = nombre moyen de choix "equivalents" que le modele hesite a chaque position.

2. **Avant/apres** : mesurer la perplexite du mini-LLaMA :
   - Avant entrainement (poids aleatoires) → PPL proche de `vocab_size` (le modele est uniforme)
   - Apres entrainement → PPL nettement plus basse
   - Sur le corpus de train ET sur un petit corpus de test (le modele generalise-t-il ?)

3. **Borne** : montrer que PPL d'un modele uniforme = `vocab_size`. Le verifier numeriquement avec des logits a zero (softmax uniforme).

4. **Weight tying** : implementer le partage de poids entre `token_embedding` et `lm_head` (les deux sont `(vocab, d_model)`). Mesurer :
   - L'economie de parametres (`vocab_size * d_model`)
   - Le % du total que ca represente sur le mini-modele
   - L'impact sur la perplexite (le partage degrade-t-il, ou pas, vu que les deux apprennent ensemble ?)

5. Analyser : pourquoi la perplexite est-elle une meilleure metrique que la "loss brute" pour comparer des modeles ? Pourquoi le weight tying est-il (presque) gratuit en qualite tout en economisant des params ?

### Criteres de reussite

- [ ] `perplexity` est correctement implementee (exp de la cross-entropy moyenne)
- [ ] PPL avant entrainement ≈ vocab_size, PPL apres nettement plus basse
- [ ] La borne "uniforme = vocab_size" est verifiee numeriquement
- [ ] Le weight tying economise `vocab_size * d_model` params (chiffre)
- [ ] L'analyse PPL vs loss et l'interet du weight tying sont corrects
