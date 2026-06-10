# Exercices Medium — Jour 7 : Mini-Transformer (Capstone)

---

## Exercice 4 : Tokenizer caractere + get_batch — l'alignement input/target

### Objectif

Implementer la brique data du mini-GPT : tokenizer caractere et generateur de batches, en verifiant l'alignement decale qui definit l'entrainement autoregressif.

### Consigne

1. Implementer un tokenizer caractere :

```python
class CharTokenizer:
    def __init__(self, text): ...   # vocab = sorted(set(text))
    def encode(self, s) -> list[int]: ...
    def decode(self, ids) -> str: ...
```

   Verifier le round-trip `decode(encode(s)) == s` sur un texte avec accents et ponctuation, et que le vocab est deterministe (trie).

2. Implementer `get_batch(data, block_size, batch_size, rng)` qui retourne `x, y` de shape `(batch_size, block_size)` avec **y decale de 1** : `y[b, t] == data[i_b + t + 1]`.

3. Verifier l'alignement, le coeur de l'exercice :
   - pour chaque batch : `x[b, 1:] == y[b, :-1]` (les targets sont les inputs decales)
   - `y[b, t]` est bien "le caractere qui suit `x[b, :t+1]`" dans le texte original (verifier en re-decodant)
   - les indices de depart ne depassent jamais `len(data) - block_size - 1` (pas d'out-of-bounds)

4. Question d'efficacite (verifier par le code) : un batch `(B, T)` fournit combien d'exemples de prediction au modele ? (Reponse attendue : B*T, un par position — c'est ce qui rend l'entrainement autoregressif si efficace.) Compter explicitement les paires (contexte, target) distinctes d'un batch.

### Criteres de reussite

- [ ] Round-trip exact du tokenizer, vocab deterministe
- [ ] `x[b, 1:] == y[b, :-1]` verifie sur 100 batches aleatoires
- [ ] Le test de re-decodage confirme que chaque target est le caractere suivant dans le texte
- [ ] Aucun indice hors limites sur 1000 tirages avec un petit dataset (len proche de block_size)
- [ ] Le comptage B*T exemples/batch est explicite et correct

---

## Exercice 5 : Top-k, top-p et temperature from scratch

### Objectif

Implementer les 3 strategies de sampling utilisees par tous les LLMs en production, avec leurs proprietes exactes.

### Consigne

Sur le vecteur de logits impose `logits = [3.0, 2.5, 1.0, 0.5, -1.0, -2.0]` (vocab de 6 tokens) :

1. `sample_temperature(logits, T)` : softmax de `logits / T`. Verifier :
   - T=1.0 → softmax standard
   - T→0 (utiliser T=0.01) : la probabilite du token argmax > 0.999 (converge vers greedy)
   - T=2.0 : l'entropie de la distribution est STRICTEMENT superieure a celle de T=1.0

2. `top_k_filter(logits, k)` : garde les k logits les plus hauts, met les autres a `-inf`, renormalise. Verifier :
   - exactement k tokens ont une probabilite > 0
   - les k tokens gardes sont les bons (comparer aux indices tries)
   - les probabilites relatives ENTRE les tokens gardes sont inchangees (ratio p_i/p_j identique avant/apres filtrage, tolerance 1e-9)

3. `top_p_filter(logits, p)` : garde le plus petit ensemble de tokens dont la proba cumulee >= p. Verifier :
   - pour p=0.9 : l'ensemble garde est minimal (retirer le dernier token garde ferait passer sous 0.9)
   - cas limite p=1.0 : tous les tokens gardes ; p tres petit (0.01) : seul l'argmax reste
   - la somme des probabilites renormalisees vaut 1 (1e-9)

4. Echantillonner 10 000 tokens avec chaque strategie (seed fixe) et verifier que les frequences empiriques correspondent aux probabilites theoriques (ecart max < 2 points de %).

### Criteres de reussite

- [ ] Les 3 fonctions operent sur les LOGITS (pas les probabilites) — l'ordre temperature → filtrage → softmax est correct et documente
- [ ] Tous les tests de proprietes passent avec les tolerances indiquees
- [ ] Le cas top-p minimal est verifie algorithmiquement (pas a l'oeil)
- [ ] Les frequences empiriques collent a la theorie (< 2 pts d'ecart)
- [ ] Un commentaire explique quand utiliser top-k vs top-p (distribution plate vs piquee)

---

## Exercice 6 : Debugger une boucle de generation cassee

### Objectif

Corriger 3 bugs classiques dans une fonction `generate()` — les memes que tout le monde ecrit la premiere fois qu'il implemente un GPT.

### Consigne

Le code suivant contient **3 bugs**. Le modele `model(ids)` retourne les logits `(T, vocab)` pour chaque position.

```python
def generate_buggy(model, prompt_ids, n_new, block_size, temperature=1.0):
    ids = list(prompt_ids)
    for _ in range(n_new):
        context = ids                          # BUG ? (que se passe-t-il quand len(ids) > block_size ?)
        logits = model(context)
        next_logits = logits[0]                # BUG ? (quelle position predit le token suivant ?)
        probs = softmax(next_logits)
        probs = probs / temperature            # BUG ? (la temperature s'applique a quoi ?)
        probs = probs / probs.sum()
        ids.append(int(np.argmax(probs)))
    return ids
```

1. Identifier les 3 bugs, decrire le symptome de chacun (crash ? sortie degeneree ? parametre sans effet ?)
2. Ecrire `generate_fixed` : contexte tronque aux `block_size` derniers tokens, logits de la DERNIERE position, temperature appliquee aux LOGITS avant softmax
3. Construire un "modele" jouet deterministe pour tester (ex : un bigram count-based sur un petit texte, wrappe pour exposer l'interface `(T,) -> (T, vocab)`), et verifier :
   - `generate_fixed` en greedy reproduit exactement la sequence attendue calculee a la main
   - avec `len(prompt) > block_size`, la version corrigee ne crashe pas et n'utilise que les block_size derniers tokens (verifier par construction d'un cas ou ca change la prediction)
   - la temperature a un effet mesurable sur la diversite (nombre de tokens distincts generes sur 50 runs a T=0.1 vs T=2.0)

### Criteres de reussite

- [ ] Bug 1 identifie : contexte non tronque → hors du positional encoding / crash ou degradation au-dela de block_size
- [ ] Bug 2 identifie : `logits[0]` predit le token APRES le premier caractere, pas apres le dernier → sortie incoherente
- [ ] Bug 3 identifie : diviser les PROBABILITES par T puis renormaliser n'a AUCUN effet (invariance du softmax) — demontre numeriquement
- [ ] `generate_fixed` reproduit la sequence greedy de reference exactement
- [ ] Le test de troncature et le test de diversite passent
