# Exercices Faciles — Jour 18 : Long context (Flash Attention, RoPE scaling)

---

## Exercice 1 : Le mur memoire — calcul a la main

### Objectif

Quantifier pourquoi l'attention vanilla explose la VRAM, et voir que le tiling (Flash Attention) ramene le pic memoire de la matrice d'attention de `O(N^2)` a `O(B^2)`.

### Consigne

La matrice de scores `S = Q @ K^T` est de taille `N x N`. En FP16, chaque element occupe 2 octets. C'est cette matrice qui doit etre materialisee en VRAM pour appliquer le softmax (attention vanilla).

1. **Pic vanilla** : pour `N` dans `{1024, 4096, 16384, 100_000, 1_000_000}`, calculer la taille de `S` en FP16 :
   - `peak_bytes = N * N * 2`
   - Convertir en MB (`/ 1024^2`) puis en GB quand c'est plus lisible. Reproduire la table du cours (1024 -> 2 MB, 1M -> ~1.9 TB).

2. **Pic tiled** : avec Flash Attention, on ne garde qu'une tuile `B x B` en SRAM (`B = 128`). Calculer `tiled_bytes = B * B * 2` et le convertir en MB. Constater qu'il est **constant** (independant de `N`).

3. **Ratio** : pour `N = 100_000`, combien de fois le pic vanilla est-il plus gros que le pic tiled ? (Indice : `(N/B)^2`.)

4. **Multiplier par les heads/couches** : un Llama 70B a ~64 heads et 80 couches. Pour `N = 1_000_000`, donner l'ordre de grandeur du pic vanilla cumule (un seul head -> tous heads x toutes couches). Pourquoi est-ce physiquement impossible sur un GPU (H100 = 80 GB) ?

### Criteres de reussite

- [ ] `N=1024 -> 2 MB`, `N=4096 -> 32 MB`, `N=16384 -> 512 MB`, `N=1M -> ~1.9 TB` (un head)
- [ ] Pic tiled constant : `128*128*2 = 32768 octets ≈ 0.031 MB`, identique pour tous les `N`
- [ ] Ratio a `N=100_000` : `(100000/128)^2 ≈ 610 000x`
- [ ] Le cumul heads x couches a `N=1M` donne des **petaoctets** -> hors de portee d'un GPU 80 GB
- [ ] Comprehension : le tiling change la **complexite memoire** (O(N^2) -> O(N)) sans changer les FLOPs

---

## Exercice 2 : Online softmax — fusionner deux blocs

### Objectif

Implementer le coeur de Flash Attention : la mise a jour en streaming des statistiques du softmax (running-max + running-sum) quand on traite les scores par blocs, et verifier qu'elle donne exactement le meme resultat qu'un softmax one-shot.

### Consigne

Le softmax stable se calcule avec `m = max(scores)` puis `sum(exp(scores - m))`. Quand on decoupe les scores en deux blocs `S1` et `S2` (vus l'un apres l'autre), on ne peut pas voir tous les scores d'un coup. Flash Attention maintient des statistiques **running** et les fusionne.

Soit deux blocs de scores 1D (un seul "query") :

```
S1 = [1.0, 3.0, 2.0]
S2 = [0.5, 4.0, 2.5]
```

1. **One-shot (reference)** : concatener `S = [S1, S2]`, calculer `m = max(S)`, `p = exp(S - m)`, `l = sum(p)`, et `softmax = p / l`.

2. **Streaming (Flash-style)** : sans concatener, traiter `S1` puis `S2` :
   - Apres `S1` : `m1 = max(S1)`, `l1 = sum(exp(S1 - m1))`.
   - Quand `S2` arrive : `m2 = max(S2)`, `m_new = max(m1, m2)`.
   - **Rescale** l'ancienne somme : `l_new = exp(m1 - m_new) * l1 + sum(exp(S2 - m_new))`.
   - Le facteur de rescale `exp(m1 - m_new)` corrige le fait qu'on avait soustrait l'ancien max `m1`.

3. **Verification** : montrer que `m_new == m` (one-shot) et `l_new == l` (one-shot) a l'erreur flottante pres.

4. **Pourquoi le rescale ?** Expliquer (en commentaire ou print) pourquoi on ne peut pas simplement faire `l1 + sum(exp(S2 - m2))` sans corriger : les exponentielles de `S1` ont ete calculees relativement a `m1`, pas a `m_new`.

### Criteres de reussite

- [ ] One-shot calcule `m`, `l`, et le softmax sur `S` concatene
- [ ] La fusion streaming reconstruit `m_new` et `l_new` sans jamais concatener
- [ ] `|m_new - m| < 1e-9` et `|l_new - l| < 1e-9`
- [ ] Le facteur `exp(m1 - m_new)` est bien applique a l'ancienne somme
- [ ] Comprehension : sans rescale, on melangerait des exp calculees avec des max differents

---

## Exercice 3 : RoPE inverse frequencies + Position Interpolation

### Objectif

Calculer les frequences inverses de RoPE, puis appliquer Position Interpolation (PI) et observer numeriquement que PI comprime **toutes** les frequences du meme facteur (`1/scale`), ce qui ecrase la resolution locale (hautes frequences).

### Consigne

RoPE rote chaque paire de dimensions par un angle `m * theta_i`, avec :

```
theta_i = 1 / base^(2i/d)   pour i = 0, 2, 4, ..., d-2   (base = 10000)
```

Position Interpolation (Meta, 2023) etend le contexte en **divisant la position effective** par `scale = L_target / L_train`, ce qui revient a diviser toutes les frequences par `scale`.

1. **rope_frequencies(d, base)** : pour `d = 64`, `base = 10000`, calculer le vecteur des `d/2 = 32` frequences inverses. Afficher la premiere (haute frequence, `i=0`) et la derniere (basse frequence, `i=62`).

2. **PI** : avec `L_train = 4096`, `L_target = 32768` (donc `scale = 8`), calculer `f_pi = f_orig / scale`.

3. **Effet haute freq vs basse freq** : calculer le ratio `f_pi / f_orig` pour la premiere paire (haute freq) et la derniere paire (basse freq). Constater qu'il vaut `1/scale = 0.125` partout. Pourquoi est-ce un probleme pour les hautes frequences (resolution locale fine) ?

4. **Interpretation des longueurs d'onde** : la longueur d'onde d'une dimension est `lambda_i = 2*pi / theta_i`. Calculer `lambda` pour la premiere et la derniere paire. La haute freq a une petite longueur d'onde (varie vite, capte la position locale), la basse freq une grande (varie lentement, capte le long-range). Confirmer numeriquement.

### Criteres de reussite

- [ ] `f_orig[0] = 1.0` (haute freq) et `f_orig[-1] ≈ 1.0/10000^(62/64) ≈ 1.3e-4` (basse freq)
- [ ] `f_pi = f_orig / 8` element par element
- [ ] Le ratio `f_pi/f_orig` vaut `0.125` pour TOUTES les paires (PI est uniforme)
- [ ] La longueur d'onde de la haute freq (`~6.28`) est tres inferieure a celle de la basse freq (`~48000`)
- [ ] Comprehension : PI comprime aussi les hautes frequences -> perte de precision locale (d'ou NTK/YaRN)
