# Exercices Medium — Jour 11 : Inference optimisee

---

## Exercice 4 : Quantization int8 from scratch — per-tensor vs per-channel

### Objectif

Implementer la quantization symetrique int8 et mesurer pourquoi la granularite per-channel est indispensable des qu'il y a des outliers.

### Consigne

1. Implementer :

```python
def quantize_symmetric(W, axis=None):
    """scale = max(|W|) / 127 (global si axis=None, par canal sinon).
    Retourne (W_int8, scale). W_int8 = round(W / scale), clip [-127, 127]."""

def dequantize(W_int8, scale): ...
```

2. Tests de base sur `W = [[0.1, -0.3], [1.5, -2.0]]` :
   - per-tensor : scale = 2.0/127 ; verifier les valeurs int8 attendues (calculees a la main : 6, -19, 95, -127)
   - round-trip : `|W - deq(quant(W))| <= scale/2 + 1e-9` element par element (l'erreur max theorique est un demi-pas)

3. **Le scenario outlier** (le cas reel des LLMs) : matrice (64, 8) ~ N(0, 0.1) avec UNE colonne multipliee par 50.
   - quantizer per-tensor (un scale global) et per-channel (un scale par colonne, axis=0)
   - mesurer l'erreur relative moyenne sur les colonnes NORMALES (hors outlier) : per-tensor doit etre catastrophique (l'outlier consomme toute la plage), per-channel doit rester fin
   - critere chiffre : erreur moyenne per-channel < erreur moyenne per-tensor / 10 sur les colonnes normales

4. Impact aval : pour `y = x @ W`, comparer `y_fp` et `y_int8` (deux granularites) : erreur relative `||y_fp - y_q|| / ||y_fp||`. Per-channel < 1%, per-tensor degrade.

5. Tableau memoire : taille de W en fp32 / fp16 / int8 / int4 (group-wise 64, compter les scales fp16 !) pour une matrice 4096x4096. Verifier que l'int4 group-wise ajoute ~3% d'overhead de scales.

### Criteres de reussite

- [ ] Les valeurs int8 du petit exemple correspondent au calcul a la main
- [ ] La borne d'erreur scale/2 est verifiee element par element
- [ ] Le scenario outlier montre un facteur >= 10 entre les deux granularites (colonnes normales)
- [ ] L'erreur sur la matmul est < 1% en per-channel
- [ ] Le tableau memoire inclut l'overhead des scales pour le group-wise

---

## Exercice 5 : Estimateur de latence — prefill compute-bound, decode memory-bound

### Objectif

Construire le petit modele de cout que tout ingenieur inference a en tete : ou passe le temps, et quand change-t-on de regime.

### Consigne

Hardware de reference type H100 : `peak_flops = 1e15` FLOP/s (fp16 dense), `bandwidth = 3.35e12` B/s.

1. Implementer :

```python
def prefill_time(n_params, n_tokens, peak_flops):
    """compute-bound: 2 * n_params * n_tokens FLOPs / peak_flops"""

def decode_time_per_token(n_params, kv_bytes_per_token_total, bytes_per_param, bandwidth, batch=1):
    """memory-bound: (poids + KV cache) a relire de la HBM pour CHAQUE token.
    Les poids sont lus UNE fois quel que soit le batch."""
```

2. Application 7B fp16 (`n_params=7e9`, 2 bytes/param) :
   - prefill de 2048 tokens : verifier ~0.029 s (2*7e9*2048 / 1e15)
   - decode batch=1 sans cache compte : verifier ~239 tok/s (3.35e12 / 14e9) — et expliquer en commentaire pourquoi on lit TOUS les poids pour UN token
   - comparer : tokens/s de prefill vs decode → ratio > 100x, c'est LE fait central de l'inference

3. **Arithmetic intensity et crossover de batch** : en decode avec batch B, on lit les poids une fois pour B tokens → `2*N*B` FLOPs pour `2*N` bytes lus, intensite = B FLOP/byte. Le hardware bascule compute-bound quand `B > peak_flops / bandwidth` (~299 pour notre H100).
   - implementer `decode_regime(batch)` qui retourne le temps memory-bound, le temps compute-bound, et le regime (le max des deux domine)
   - trouver numeriquement le batch de crossover et verifier ~299
   - tableau : B ∈ {1, 8, 64, 256, 512} | tok/s/requete | tok/s total | regime

4. Ajouter le KV cache au modele de cout : pour un 7B (32 couches, d=4096, GQA nul — MHA, fp16), `kv_bytes_per_token = 2 * 32 * 4096 * 2` = 0.5 MB/token. A quelle longueur de sequence le cache (batch=1) egale-t-il la taille des poids (14 GB) ? Verifier ~26 700 tokens — le cache devient le probleme aux longs contextes (et des batch=4, c'est ~6700 tokens par sequence).

### Criteres de reussite

- [ ] Les 3 valeurs de reference (0.029 s, 239 tok/s, ratio > 100x) sont retrouvees
- [ ] Le batch de crossover ~299 est trouve numeriquement (pas juste par la formule)
- [ ] Le tableau des regimes est correct (tok/s/requete ~constant jusqu'au crossover, puis decroit)
- [ ] La longueur d'egalite cache==poids ~26.7k tokens est calculee
- [ ] Chaque formule est justifiee en commentaire (pourquoi 2*N*T, pourquoi memory-bound, etc.)

---

## Exercice 6 : Speculative decoding — la regle d'acceptation qui preserve la distribution

### Objectif

Implementer la regle d'acceptation/rejet du speculative sampling et verifier empiriquement sa garantie magique : la distribution de sortie est EXACTEMENT celle du gros modele.

### Consigne

1. Deux distributions sur un vocab de 6 tokens (imposees) :
   - cible (gros modele) : `p = [0.4, 0.25, 0.15, 0.1, 0.07, 0.03]`
   - draft (petit modele) : `q = [0.25, 0.3, 0.2, 0.1, 0.1, 0.05]`

2. Implementer un step de speculative sampling pour UN token :

```python
def spec_sample_one(p, q, rng):
    """1. tirer x ~ q
    2. accepter avec proba min(1, p[x]/q[x])
    3. si rejet: re-tirer dans la distribution residuelle
       r = normalize(max(p - q, 0))
    Retourne (token, accepted: bool)"""
```

3. **Verification de la garantie** : sur 200 000 tirages, la distribution empirique des tokens retournes doit correspondre a `p` (distance en variation totale < 0.005). C'est contre-intuitif : on tire selon q, mais on obtient p.

4. **Taux d'acceptation** : verifier que le taux empirique correspond a la theorie `alpha = sum_x min(p[x], q[x])` (± 0.5 pt). Calculer alpha pour : draft == cible (alpha=1), draft uniforme, et le q impose.

5. **Tokens par cycle** : avec k tokens draftes par cycle et une acceptation ~iid de proba alpha, l'esperance de tokens produits par cycle est `E = (1 - alpha^(k+1)) / (1 - alpha)`. Implementer la formule + une simulation Monte Carlo du cycle (accepter jusqu'au premier rejet, +1 token de correction) et verifier l'accord (± 2%). Tableau : alpha ∈ {0.6, 0.8, 0.95} x k ∈ {2, 4, 8} → speedup potentiel si le draft est gratuit, et commentaire sur pourquoi k=8 n'apporte presque rien a alpha=0.6.

### Criteres de reussite

- [ ] La distribution residuelle est correcte (max(p-q, 0) renormalisee) — testee : elle somme a 1
- [ ] TV distance empirique vs p < 0.005 sur 200k tirages (LA garantie du papier)
- [ ] Taux d'acceptation empirique == theorique (± 0.5 pt) pour les 3 couples (p, q)
- [ ] La formule E(tokens/cycle) colle a la simulation (± 2%)
- [ ] Le tableau alpha x k est interprete (rendements decroissants en k)
