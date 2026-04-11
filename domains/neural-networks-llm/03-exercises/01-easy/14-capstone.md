# Exercices Faciles — Jour 14 : Capstone

---

## Exercice 1 : Paraphraser l'idee cle de l'attention en 5 lignes

### Objectif

Etre capable d'expliquer un concept fondamental en une poignee de phrases, sans jargon. C'est un exercice fondamental pour tout scientifique : si tu ne peux pas l'expliquer simplement, tu ne l'as pas compris (Feynman).

### Consigne

Ecrire une explication de **l'attention** (mechanism Transformer) en **exactement 5 lignes**, en respectant ces contraintes :

1. **Ligne 1** : le probleme que l'attention resout
2. **Ligne 2** : l'idee de base (query/key/value)
3. **Ligne 3** : le calcul central
4. **Ligne 4** : l'intuition geometrique/visuelle
5. **Ligne 5** : pourquoi c'est revolutionnaire par rapport aux alternatives

Contraintes :
- Pas de jargon non explicite (pas de "self-attention" sans dire ce que ca veut dire)
- Utilise des mots concrets quand tu peux
- Un lecteur qui n'a jamais vu l'attention doit comprendre l'essentiel
- Pas d'equation (tu peux en utiliser une mais pas obligatoire)

### Criteres de reussite

- [ ] Exactement 5 lignes (pas 4, pas 6)
- [ ] Un lecteur debutant comprend l'idee generale
- [ ] Tu mentionnes query, key, value sans les laisser comme mots-cles
- [ ] Tu expliques POURQUOI c'est mieux que RNN/LSTM (parallelisable, dependances longues)
- [ ] Tu n'utilises pas le mot "magic" ou "simply" (signes de manque de comprehension)

**Exemple de reussite** :
```
1. Dans une phrase, chaque mot a besoin de "regarder" d'autres mots pour comprendre
   son sens (ex: "il" pointe vers un nom mentionne plus tot).
2. L'attention resout ca en donnant a chaque mot 3 roles: une "question" (query),
   une "etiquette" (key), et un "contenu" (value).
3. Pour chaque mot, on compare sa query avec toutes les keys des autres mots,
   puis on prend une moyenne ponderee de leurs values selon la ressemblance.
4. Geometriquement: chaque mot "tire" une quantite d'information des autres,
   en fonction de leur similarite dans l'espace vectoriel des queries/keys.
5. Contrairement aux RNN qui traitent les mots un par un, tous les mots peuvent
   communiquer en parallele et capturer des dependances tres lointaines.
```

---

## Exercice 2 : Calculer le nombre de parametres d'un mini modele

### Objectif

Savoir faire l'estimation "a la volee" du nombre de parametres d'un transformer, une competence indispensable en entretien.

### Consigne

Soit un mini-LLaMA avec ces hyperparameters :
- `vocab_size = 32`
- `d_model = 64`
- `n_layers = 2`
- `n_heads = 4`
- `n_kv_heads = 2` (GQA avec group size 2)
- `head_dim = d_model / n_heads = 16`
- `d_ff = 256` (pour SwiGLU, ~4*d_model)

1. **Token embedding** : `vocab_size * d_model` = ?

2. **Par couche d'attention** (GQA) :
   - `W_q` : (d_model, n_heads * head_dim) = ?
   - `W_k` : (d_model, n_kv_heads * head_dim) = ?
   - `W_v` : (d_model, n_kv_heads * head_dim) = ?
   - `W_o` : (n_heads * head_dim, d_model) = ?
   - Total attention = ?

3. **Par couche FFN (SwiGLU)** : 3 matrices
   - `W_gate` : (d_model, d_ff) = ?
   - `W_up` : (d_model, d_ff) = ?
   - `W_down` : (d_ff, d_model) = ?
   - Total FFN = ?

4. **Par couche : 2 RMSNorm** : chacune a `d_model` parametres (juste le gamma)
   - Total norms = 2 * d_model = ?

5. **Total par TransformerBlock** :
   - attention + FFN + 2 norms = ?

6. **Stack de 2 couches** : 2 * (attention + FFN + 2 norms) = ?

7. **RMSNorm finale** : d_model params

8. **lm_head** : (d_model, vocab_size) = ?

9. **Grand total** : token_embedding + stack + final_norm + lm_head = ?

10. **Bonus** : si on tiait lm_head avec token_embedding (weight sharing, comme LLaMA), combien gagne-t-on ?

### Criteres de reussite

- [ ] Token embedding : 32 * 64 = 2048
- [ ] Attention par couche : 64*64 + 64*32 + 64*32 + 64*64 = 4096+2048+2048+4096 = 12288
- [ ] FFN par couche : 3 * 64 * 256 = 49152
- [ ] 2 RMSNorms par couche : 128
- [ ] Par couche total : 12288 + 49152 + 128 = 61568
- [ ] 2 couches : 123136
- [ ] Final norm : 64
- [ ] lm_head : 64 * 32 = 2048
- [ ] Grand total : 2048 + 123136 + 64 + 2048 = **127 296 params**
- [ ] Bonus : weight tying fait economiser vocab_size * d_model = 2048 (environ 1.6%)

---

## Exercice 3 : Modifier mini-LLaMA pour 8 KV heads au lieu de 2

### Objectif

Comprendre comment modifier un mini-LLaMA et predire l'impact de cette modification.

### Consigne

Dans le fichier `02-code/14-capstone.py`, le mini-LLaMA utilise `n_kv_heads=2` avec `n_heads=4` (GQA avec group size 2).

1. **Modification requise** :
   - Changer `n_kv_heads` pour le mettre a `8`.
   - Mais `n_heads=4`. Est-ce compatible ? Sinon, pourquoi ?

2. **Nouvelle configuration valide** :
   - Changer `n_heads = 8` et `n_kv_heads = 8` (donc MHA au lieu de GQA)
   - OU changer `n_heads = 8` et `n_kv_heads = 2` (toujours GQA mais plus de query heads)
   - Decider laquelle est la plus coherente avec le changement demande.

3. **Impact sur les parametres** :
   - Avec la nouvelle config `n_heads=8, n_kv_heads=8`, recalculer le nombre de parametres du modele
   - Compare a la config originale `n_heads=4, n_kv_heads=2`
   - Ou est-ce que les parametres ont augmente ?

4. **Impact sur le KV cache** :
   - KV cache = 2 * n_layers * n_kv_heads * head_dim * seq_len * batch * bytes
   - Config originale (n_kv_heads=2) : calculer pour seq=128, batch=1, fp32 (4 bytes)
   - Nouvelle config (n_kv_heads=8) : recalculer
   - Ratio ?

5. **Quand ce changement est-il pertinent ?**
   - Quel est le trade-off entre plus de KV heads et moins de KV heads ?
   - Dans quel cas prefere-t-on MHA a GQA ?

6. **Bonus — reellement modifier le code** : editer le fichier `14-capstone.py`, changer la config du `MiniLLaMA`, lancer `python 14-capstone.py`, observer le nombre de parametres affiche, verifier que ca tourne.

### Criteres de reussite

- [ ] Comprehension : avec n_heads=4, on ne peut pas avoir n_kv_heads=8 (contrainte : n_heads doit etre divisible par n_kv_heads, et n_heads >= n_kv_heads)
- [ ] Nouvelle config valide : n_heads=8, n_kv_heads=8 (MHA) OU n_heads=8, n_kv_heads=2 (GQA 4x)
- [ ] Impact sur les params : les W_q et W_o grandissent (n_heads augmente), les W_k et W_v aussi (n_kv_heads augmente)
- [ ] KV cache original : 2*2*2*16*128*1*4 = 32768 bytes = 32 KB
- [ ] KV cache nouveau : 2*2*8*16*128*1*4 = 131072 bytes = 128 KB (4x plus)
- [ ] Trade-off : plus de KV heads = plus de qualite (chaque query a sa propre K/V), mais plus de memoire et plus lent en inference
- [ ] MHA est meilleur pour la qualite quand la memoire n'est pas un probleme ; GQA est le sweet spot pour l'inference a grande echelle
