# Exercices Medium — Dynamic Programming

---

## Exercice 4 : Linear DP avec choix — House Robber II (circulaire)

### Objectif

Maitriser le DP lineaire "take / skip" sur une contrainte **circulaire** : les maisons sont en cercle, donc la premiere et la derniere sont adjacentes. Tu apprends a reduire un probleme circulaire a deux passes lineaires.

### Consigne

Tu cambriole des maisons disposees **en cercle**. Tu ne peux pas voler deux maisons adjacentes, et comme c'est un cercle, la maison `0` et la maison `n-1` sont adjacentes. Retourne le montant maximum que tu peux voler.

```python
def rob_circular(nums: list[int]) -> int:
    """
    Houses are arranged in a circle: house 0 and house n-1 are adjacent.
    Return the maximum amount you can rob without robbing two adjacent houses.
    """
    pass
```

**Indice** : un cercle = "soit on prend la maison 0 (et on exclut la derniere), soit on l'exclut". Resous deux House Robber lineaires : sur `nums[0:n-1]` et sur `nums[1:n]`, puis prends le max. Traite a part le cas `n == 1`.

### Tests

```python
assert rob_circular([2, 3, 2]) == 3            # On ne peut pas prendre maison 0 et 2 (adjacentes)
assert rob_circular([1, 2, 3, 1]) == 4         # 1 + 3
assert rob_circular([0]) == 0
assert rob_circular([5]) == 5                   # Une seule maison
assert rob_circular([1, 2]) == 2               # Deux maisons adjacentes : la plus grande
assert rob_circular([2, 7, 9, 3, 1]) == 11     # 2 + 9 (pas le 1 final, adjacent au 2)
assert rob_circular([1, 2, 3]) == 3
```

### Criteres de reussite

- [ ] Reduit le cercle a deux House Robber lineaires (`[0:n-1]` et `[1:n]`)
- [ ] Gere le cas `n == 1` separement (une seule maison, pas de voisin)
- [ ] Le House Robber lineaire interne est en O(1) espace (deux variables roulantes)
- [ ] Complexite O(n) temps, O(1) espace
- [ ] Tous les tests passent

---

## Exercice 5 : DP 2D — Longest Common Subsequence

### Objectif

Coder le DP 2D le plus classique (LCS) avec la table `dp[i][j]`, comprendre la recurrence diagonale, et savoir l'optimiser en espace O(min(m, n)).

### Consigne

Etant donne deux strings `text1` et `text2`, retourne la longueur de leur **plus longue sous-sequence commune** (LCS). Une sous-sequence n'est pas forcement contigue mais garde l'ordre.

```python
def longest_common_subsequence(text1: str, text2: str) -> int:
    """
    Return the length of the longest common subsequence of text1 and text2.
    A subsequence keeps relative order but need not be contiguous.
    """
    pass
```

**Indice** :
- Etat : `dp[i][j]` = LCS de `text1[:i]` et `text2[:j]`
- Si `text1[i-1] == text2[j-1]` : `dp[i][j] = dp[i-1][j-1] + 1` (diagonale + 1)
- Sinon : `dp[i][j] = max(dp[i-1][j], dp[i][j-1])`

### Tests

```python
assert longest_common_subsequence("abcde", "ace") == 3      # "ace"
assert longest_common_subsequence("abc", "abc") == 3
assert longest_common_subsequence("abc", "def") == 0
assert longest_common_subsequence("", "abc") == 0
assert longest_common_subsequence("abc", "") == 0
assert longest_common_subsequence("bl", "yby") == 1          # "b"
assert longest_common_subsequence("ezupkr", "ubmrapg") == 2 # "up"
assert longest_common_subsequence("oxcpqrsvwf", "shmtulqrypy") == 2  # "qr"
```

### Criteres de reussite

- [ ] Utilise une table `dp` de taille `(m+1) x (n+1)` (ou une version 1D rolling)
- [ ] La recurrence distingue le cas "caractere egal" (diagonale + 1) du cas different (max haut/gauche)
- [ ] Gere les strings vides (LCS = 0)
- [ ] Complexite O(m * n) temps, O(m * n) ou O(min(m, n)) espace
- [ ] Tous les tests passent

---

## Exercice 6 : Count DP — Decode Ways

### Objectif

Maitriser un Count DP avec des transitions conditionnelles non triviales (un ou deux caracteres a la fois, validation du mapping `A=1..Z=26`). C'est un grand classique d'entretien qui piege sur les `0`.

### Consigne

Un message contenant uniquement des chiffres est encode selon le mapping `'A' -> "1"`, `'B' -> "2"`, ..., `'Z' -> "26"`. Etant donne une string `s` de chiffres, retourne le **nombre de facons** de la decoder.

Attention : `"06"` n'est pas un decodage valide de `"6"` (pas de zero en tete), et un `'0'` isole ne se decode pas.

```python
def num_decodings(s: str) -> int:
    """
    Return the number of ways to decode a digit string using A=1..Z=26.
    Leading zeros (e.g. "06") and standalone zeros are invalid.
    """
    pass
```

**Indice** :
- `dp[i]` = nombre de decodages de `s[:i]`. Base : `dp[0] = 1`.
- Un caractere : si `s[i-1] != '0'`, `dp[i] += dp[i-1]`.
- Deux caracteres : si `s[i-2:i]` est entre `"10"` et `"26"`, `dp[i] += dp[i-2]`.

### Tests

```python
assert num_decodings("12") == 2          # "AB" (1 2) ou "L" (12)
assert num_decodings("226") == 3         # "BZ", "VF", "BBF"
assert num_decodings("0") == 0           # Un zero isole : invalide
assert num_decodings("06") == 0          # Zero en tete : invalide
assert num_decodings("10") == 1          # "J"
assert num_decodings("100") == 0         # "10" puis "0" isole : invalide
assert num_decodings("1") == 1
assert num_decodings("27") == 1          # 27 > 26 -> seulement "2","7"
assert num_decodings("2101") == 1        # "U","J","A" -> "21","0"?... seul chemin valide
assert num_decodings("11106") == 2
```

### Criteres de reussite

- [ ] `dp[i]` combine la contribution "1 caractere" et "2 caracteres" sous conditions
- [ ] Gere correctement les `0` (un `0` ne peut etre decode qu'en `"10"` ou `"20"`)
- [ ] Rejette les decodages `"00"`, `"30"`, `"40"`, ... (deuxieme chiffre 0 hors 10/20)
- [ ] Complexite O(n) temps, O(1) ou O(n) espace
- [ ] Tous les tests passent
