# Exercices Medium — Jour 11 : Inference optimisee

---

## Exercice 4 : KV cache vs naif — equivalence numerique + scaling du speedup

### Objectif

Implementer une attention causale AVEC et SANS KV cache en NumPy, prouver qu'elles produisent EXACTEMENT la meme sortie, puis mesurer comment le speedup grandit avec la longueur.

### Consigne

1. Reprendre `attention_no_cache` et `AttentionWithCache` du code (`02-code/11-inference-optimisee.py`).

2. **Equivalence** : pour une sequence donnee, generer pas a pas avec le cache et comparer la sortie du dernier token a celle de l'attention naive (relancee sur toute la sequence). L'ecart doit etre < 1e-10 a chaque step. Pourquoi cette equivalence est-elle exacte (et pas juste approximative) ?

3. **Comptage de FLOPs** : ecrire une fonction qui compte (analytiquement) les FLOPs de l'attention par token genere :
   - Naif : `O(t^2 * d)` pour le token t (recalcul sur toute la sequence)
   - Cache : `O(t * d)` pour le token t
   - Tracer le ratio naif/cache pour t = 100, 1000, 10000

4. **Cout cumule** : pour generer une sequence de n tokens depuis un prompt vide :
   - Naif : `sum_{t=1..n} t^2 ≈ n^3/3`
   - Cache : `sum_{t=1..n} t ≈ n^2/2`
   - Verifier numeriquement le ratio pour n=1000 (attendu ~ 2n/3 ≈ 667)

5. **Mesure de temps** : timer la generation naive vs cache pour 50 puis 200 nouveaux tokens. Montrer que le speedup MESURE grandit avec la longueur (coherent avec la theorie).

6. Analyser : pourquoi le cache transforme un cout total `O(n^3)` en `O(n^2)` ? Quel est le prix paye (memoire) ?

### Criteres de reussite

- [ ] Les deux attentions donnent un resultat identique (ecart < 1e-10) a chaque step
- [ ] Le comptage de FLOPs montre un ratio croissant en t
- [ ] Le ratio cumule pour n=1000 est verifie numeriquement (~600-700)
- [ ] Le speedup mesure augmente entre 50 et 200 tokens
- [ ] L'explication O(n^3) -> O(n^2) et le cout memoire sont corrects

---

## Exercice 5 : Quantization int8 / int4, per-tensor vs per-channel

### Objectif

Implementer la quantization symetrique en NumPy, comparer int8 vs int4 et per-tensor vs per-channel, et mesurer l'erreur sur un vrai matmul.

### Consigne

1. Implementer `quantize(W, bits, mode)` ou :
   - `bits` ∈ {8, 4} : range `[-(2^(bits-1)-1), 2^(bits-1)-1]` (ex: int8 → [-127,127], int4 → [-7,7])
   - `mode` ∈ {"per_tensor", "per_channel"} : une seule scale globale, ou une scale par ligne (channel de sortie)
   - Retourner `(q, scale)` puis `dequantize`

2. Construire une matrice de poids realiste `W (256, 256)` avec quelques **outliers** (qq valeurs 10x plus grandes que le reste — c'est ce qui casse la quantization en pratique).

3. Pour chaque combinaison (bits, mode), mesurer :
   - L'erreur de reconstruction moyenne et max sur W
   - L'erreur relative sur un vrai matmul `x @ W` (avec un x aleatoire)

4. **Effet des outliers** : montrer que per-tensor souffre des outliers (le max global gonfle la scale et ecrase les petites valeurs), alors que per-channel isole le degat dans les lignes concernees.

5. **int8 vs int4** : montrer que passer de 8 a 4 bits multiplie l'erreur par ~16 (le pas de quantization passe de `max/127` a `max/7`). Quand int4 est-il quand meme acceptable ?

6. Analyser le trade-off memoire/qualite : int8 = /4, int4 = /8, mais per-channel ajoute un cout (une scale par ligne). Est-ce negligeable ?

### Criteres de reussite

- [ ] La quantization symetrique int8 et int4 est correcte (range, scale, clip)
- [ ] Per-channel donne une erreur strictement plus faible que per-tensor en presence d'outliers
- [ ] int4 a une erreur ~16x celle d'int8 (verifie numeriquement, ordre de grandeur)
- [ ] L'erreur sur le matmul est mesuree, pas seulement sur les poids
- [ ] Le cout des scales per-channel est chiffre et juge negligeable

---

## Exercice 6 : Speculative decoding — simuler le draft + verify

### Objectif

Simuler la boucle speculative decoding (un petit modele "draft" propose K tokens, le gros modele "target" les verifie en un pass) et comprendre quand le gain est reel.

### Consigne

On modelise deux "modeles" comme des distributions sur un vocab : `p_target` (gros, lent) et `q_draft` (petit, rapide, parfois faux).

1. Implementer la **boucle speculative** standard (Leviathan et al., 2023) :
   - Le draft propose K tokens `x_1..x_K` (echantillonnes selon q)
   - Le target calcule `p(x_i)` pour tous en UN pass
   - Pour chaque i, accepter `x_i` avec proba `min(1, p(x_i)/q(x_i))`
   - Au premier rejet en position j : echantillonner un token de correction depuis la distribution residuelle `normalize(max(0, p - q))`, puis stopper
   - Si tous acceptes : echantillonner un token bonus depuis `p`

2. **Correction du sampling** : verifier empiriquement (sur 100000 tirages) que la distribution des tokens produits par speculative decoding est **identique** a un sampling direct depuis `p_target`. C'est la propriete cle : pas de perte de qualite.

3. **Acceptance rate** : mesurer le nombre moyen de tokens acceptes par pass en fonction de la "proximite" entre draft et target (faire varier un parametre qui rapproche q de p).

4. **Modele de speedup** : si le draft est `c` fois moins cher que le target et que l'acceptance rate moyen est `alpha` tokens par pass, donner la formule du speedup theorique et la tracer en fonction de alpha pour c=0.1.

5. Analyser : pourquoi speculative decoding ne change PAS la distribution de sortie ? Pourquoi le gain s'effondre si le draft est trop different du target ?

### Criteres de reussite

- [ ] La boucle draft+verify est implementee avec accept/reject et token de correction corrects
- [ ] La distribution empirique du speculative sampling == distribution de p_target (ecart < 0.02 en L1)
- [ ] L'acceptance rate augmente quand draft se rapproche de target
- [ ] Le modele de speedup est formule et trace
- [ ] L'analyse explique la garantie d'equivalence ET la dependance a la qualite du draft
