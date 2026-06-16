# Exercices Faciles — Jour 19 : Quantization

---

## Exercice 1 : Symmetric INT8 — calcul a la main

### Objectif

Savoir quantizer un petit vecteur de poids en INT8 symetrique et mesurer l'erreur introduite.

### Consigne

Soit le vecteur de poids : `W = [0.12, -0.45, 2.10, -3.00, 0.03, 1.20]`

1. **Scale symetrique** (INT8 signed, plage utile [-127, 127]) :
   - Trouver `max_abs = max(|W|)`
   - Calculer `scale = max_abs / 127`

2. **Quantize** : pour chaque element, `q = clip(round(W / scale), -127, 127)`. Donner les entiers.

3. **Dequantize** : `W_hat = q * scale`. Donner le vecteur reconstruit.

4. **Erreur** : calculer `error = |W - W_hat|` element par element, puis l'erreur moyenne et l'erreur max.

5. **Le petit element** : `0.03` est le plus petit. Quel `q` recoit-il ? Quelle est son erreur **relative** (`|err| / |valeur|`) ? Pourquoi les petites valeurs souffrent-elles plus que les grandes en quantization per-tensor ?

6. **INT4** : refaire la quantization en INT4 signed (plage utile [-7, 7], donc `scale = max_abs / 7`). Comparer l'erreur moyenne INT8 vs INT4. Pourquoi INT4 est-il ~16x plus bruite par valeur ?

### Criteres de reussite

- [ ] `max_abs = 3.0`, `scale = 3.0/127 ≈ 0.02362`
- [ ] `q ≈ [5, -19, 89, -127, 1, 51]`
- [ ] Erreur moyenne INT8 ~0.003-0.008, erreur max ~0.012 (la moitie d'un pas de scale)
- [ ] `0.03` -> `q=1` -> `W_hat ≈ 0.0236`, erreur relative ~21% (enorme)
- [ ] INT4 : `scale = 3/7 ≈ 0.4286`, erreur moyenne ~16x plus grande (chaque pas vaut 16x plus)
- [ ] Comprehension : un seul `scale` par tenseur est dimensionne sur le `max_abs` ; les petites valeurs n'ont que quelques codes pour les representer

---

## Exercice 2 : Symmetric vs asymmetric sur une distribution skewed

### Objectif

Comprendre pourquoi on prefere l'asymetrique pour des activations post-ReLU (toujours >= 0).

### Consigne

Soit le vecteur d'activations (post-ReLU, donc positif) : `A = [0.0, 0.5, 1.2, 2.4, 3.8, 5.0]` (range [0, 5]).

1. **Symmetric INT4** ([-7, 7]) : `scale = max(|A|) / 7`. Quantize, dequantize, donne l'erreur moyenne.
   - Combien de codes entiers sont **gaspilles** ? (Indice : tous les codes negatifs -7..-1 ne sont jamais utilises car `A >= 0`.)

2. **Asymmetric INT4** (non-signe [0, 15], convention du code du Jour 19) :
   - `scale = (max(A) - min(A)) / 15`
   - `zero_point = round(0 - min(A) / scale)` puis clip dans [0, 15]
   - `q = clip(round(A / scale + zero_point), 0, 15)`
   - `A_hat = (q - zero_point) * scale`
   - Donner `scale`, `zero_point`, les `q`, et l'erreur moyenne.

3. **Comparaison** : combien de fois l'asymetrique reduit-il l'erreur ? Pourquoi ?

4. **Pour les poids** : les poids d'un LLM bien entraine sont quasi-centres sur 0. Pour eux, symmetric vs asymmetric : quel ecart attends-tu ? Pourquoi on garde quand meme symmetric pour les poids (1 parametre vs 2) ?

### Criteres de reussite

- [ ] Symmetric INT4 : `scale = 5/7 ≈ 0.714`, ~7 codes sur 15 gaspilles (la moitie negative)
- [ ] Asymmetric : `scale = 5/15 ≈ 0.333`, `zero_point = 0` (car min = 0), erreur ~2x plus faible
- [ ] L'asymetrique utilise les 16 niveaux sur [0, 5] ; le symetrique n'en utilise que 8 (cote positif)
- [ ] Pour des poids centres, l'ecart est faible -> symmetric suffit et coute 1 scale au lieu de scale+zero_point
- [ ] Comprehension : asymmetric = pas de gaspillage de plage, au prix d'un parametre de plus

---

## Exercice 3 : Granularite — per-tensor vs per-channel

### Objectif

Voir concretement pourquoi un seul scale global ecrase les colonnes a petite echelle quand les colonnes ont des magnitudes tres differentes.

### Consigne

Soit une mini-matrice 2x3 (2 lignes, 3 colonnes) ou chaque colonne a une echelle differente :

```
        col_0   col_1   col_2
ligne_0 [ 0.10    8.0    0.02 ]
ligne_1 [-0.08   -7.5    0.03 ]
```

1. **Per-tensor INT8** : un seul `scale = max(|tout|) / 127`. Quantize toute la matrice avec ce scale unique.
   - Que deviennent les valeurs de `col_0` et `col_2` (echelle ~0.1 et ~0.02) ? Vers quel `q` sont-elles arrondies ?

2. **Per-channel INT8** (1 scale par colonne) : `scale_j = max(|colonne_j|) / 127`. Quantize chaque colonne avec son propre scale.
   - Donne les 3 scales.
   - Que deviennent `col_0` et `col_2` maintenant ?

3. **Comparer l'erreur relative** sur `col_2` (la plus petite) entre per-tensor et per-channel.

4. **Cout memoire** : pour une vraie matrice 4096x4096 en INT4, combien de scales FP16 ajoute le per-channel (per-row) ? Et le per-group g=128 ? Exprime l'overhead per-group en % (rappel du cours : ~3%).

### Criteres de reussite

- [ ] Per-tensor : `scale = 8.0/127 ≈ 0.063` ; `col_2` (~0.02-0.03) -> `q=0` ou `q=±1`, quasi detruite
- [ ] Per-channel : `scale_0 ≈ 0.10/127`, `scale_1 ≈ 8.0/127`, `scale_2 ≈ 0.03/127` ; chaque colonne bien resolue
- [ ] Erreur relative sur `col_2` : ~100% en per-tensor (arrondie a 0), ~<1% en per-channel
- [ ] Per-channel 4096x4096 : 4096 scales FP16 (0.05% overhead)
- [ ] Per-group g=128 : 4096 * 32 = 131072 scales ; sur INT4 c'est 16 bits / (128*4 bits) = 3.125% overhead
