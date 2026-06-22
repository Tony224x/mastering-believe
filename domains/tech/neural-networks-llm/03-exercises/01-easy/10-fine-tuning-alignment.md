# Exercices Faciles — Jour 10 : Fine-tuning & Alignment

---

## Exercice 1 : Format SFT — construire des exemples

### Objectif

Savoir construire un dataset SFT correctement formate et comprendre ou la loss est calculee.

### Consigne

1. Prendre ces 3 "raw intents" :
   - "Explique la photosynthese simplement"
   - "Traduis 'hello' en francais"
   - "Liste 3 pays d'Europe"

2. Formatter chaque exemple avec la convention `<|user|> ... <|assistant|> ... <|end|>`. Ecrire la reponse ideale toi-meme (2-3 lignes max).

3. Pour chaque exemple, indiquer avec des marqueurs (`LOSS=1` ou `LOSS=0`) sur quels tokens la loss SFT est calculee :
   ```
   <|user|>  Explique  la  photosynthese  simplement  <|assistant|>  La  ...
   LOSS=0    LOSS=0    LOSS=0  LOSS=0     LOSS=0     LOSS=0          LOSS=1  ...
   ```

4. Expliquer pourquoi on ne calcule pas la loss sur les tokens du prompt. Quelle serait la consequence si on le faisait ?

5. **Bonus** : comment est-ce implemente en pratique ? (indice : les labels sont mis a `-100` en PyTorch)

### Criteres de reussite

- [ ] 3 exemples formattes avec les tokens speciaux
- [ ] Les tokens du prompt ont LOSS=0, les tokens de la reponse ont LOSS=1
- [ ] Explication : si on calculait la loss sur le prompt, le modele apprendrait a generer les prompts, pas a y repondre
- [ ] Bonus : en PyTorch, les labels mis a -100 sont ignores par `F.cross_entropy` (ignore_index par defaut)

---

## Exercice 2 : DPO — interpreter les signes

### Objectif

Comprendre le sens du "reward margin" dans DPO et savoir predire la direction du gradient.

### Consigne

Rappel de la loss DPO :
```
L = -log σ(β * [(logp_θ(y_w) - logp_ref(y_w)) - (logp_θ(y_l) - logp_ref(y_l))])
margin = β * [(logp_θ(y_w) - logp_ref(y_w)) - (logp_θ(y_l) - logp_ref(y_l))]
```

Pour chaque scenario, calculer le `margin` (avec β=0.1) et dire si la loss est haute ou basse :

| Scenario | logp_θ(y_w) | logp_θ(y_l) | logp_ref(y_w) | logp_ref(y_l) |
|---|---|---|---|---|
| A | -1.0 | -4.0 | -2.0 | -2.0 |
| B | -2.0 | -2.0 | -2.0 | -2.0 |
| C | -3.0 | -1.0 | -2.0 | -2.0 |
| D | -1.0 | -5.0 | -1.5 | -4.0 |

1. Pour chaque scenario, calculer `margin` avec β = 0.1.

2. Pour chaque scenario, calculer `loss = -log(sigmoid(margin))`.
   (Rappel : `sigmoid(x) = 1/(1+exp(-x))`, `log(1) = 0`, `log(0.5) = -0.69`)

3. Interpreter :
   - Scenario A : la policy est-elle en accord avec la preference humaine ? La loss est-elle basse ou haute ?
   - Scenario B : que veut dire "margin = 0" ? C'est quoi la proba sous sigmoid ?
   - Scenario C : la policy est en opposition avec la preference. Comment le gradient va-t-il pousser ?
   - Scenario D : la policy est deja tres alignee. Pourquoi le gradient est-il quand meme legerement non-zero ?

4. **Bonus** : si β = 1.0 au lieu de 0.1, comment les loss changent ? (pas besoin de recalculer toutes, juste l'intuition)

### Criteres de reussite

- [ ] Scenario A : margin = 0.1 * (1.0 - (-2.0)) = 0.3, loss ≈ 0.55 (bonne direction, loss moderee)
- [ ] Scenario B : margin = 0, sigmoid(0) = 0.5, loss = log(2) ≈ 0.69 (baseline)
- [ ] Scenario C : margin = -0.3, loss ≈ 0.85 (haute, le gradient pousse vers y_w)
- [ ] Scenario D : margin = 0.15, loss ≈ 0.62 (plus basse que scenario C mais non nulle)
- [ ] Bonus : β plus grand amplifie les differences, margin est plus extreme, les loss sont plus polarisees

---

## Exercice 3 : LoRA — compter les parametres

### Objectif

Savoir calculer la compression de parametres avec LoRA et comprendre ses avantages pratiques.

### Consigne

1. Soit une `nn.Linear(4096, 4096)`. Calculer :
   - Le nombre de parametres du full fine-tuning (uniquement les poids, ignorer le bias).
   - Le nombre de parametres LoRA pour `r = 1, 4, 8, 16, 64`. Formule : `2 * d * r` (matrices A et B).
   - Le ratio de compression pour chaque `r`.

2. Soit un LLaMA 7B dont les couches d'attention ont : `d_model = 4096`, 32 couches, et on applique LoRA sur les 4 projections Q, K, V, O.
   - Nombre total de params full fine-tuning dans les couches d'attention : `32 * 4 * 4096 * 4096`.
   - Nombre total de params LoRA avec r=16 : `32 * 4 * 2 * 4096 * 16`.
   - Ratio de compression ?

3. **Memoire GPU** : avec l'optimizer Adam (qui stocke moment 1 et moment 2), la memoire pour les gradients + Adam est environ 3x la taille des parametres (en fp32) ou 6x (en fp16 + master weights en fp32). Pour le full FT LLaMA 7B :
   - Memoire juste pour les gradients d'un layer : `4 * 4096 * 4096 * 4 bytes (fp32)` = ?
   - Memoire pour les states d'Adam : 2x la memoire des gradients
   - Total pour un layer d'attention en full FT (pas juste les params, mais gradients + Adam) ?
   - Compare a LoRA r=16 : combien de memoire pour les gradients + Adam ?

4. **Deploiement** : si tu as un modele de base LLaMA 7B (14 GB en fp16) et que tu veux deployer 50 modeles fine-tunes differents pour 50 clients :
   - Full fine-tuning : memoire totale ? (50 * 14 GB)
   - LoRA r=16 avec ~20 MB par adaptateur : memoire totale ? (14 GB + 50 * 20 MB)
   - Ratio ?

5. **Bonus** : pourquoi initialise-t-on B a zero et pas A ? Qu'est-ce qui se passerait si les deux etaient inits random ?

### Criteres de reussite

- [ ] Full FT Linear : 16 777 216 params
- [ ] LoRA r=8 : 65 536 params, ratio 256x
- [ ] LLaMA 7B attn layers full FT : ~2.15B. LoRA r=16 : ~8.4M. Ratio ~256x
- [ ] Memoire couche attention full FT : ~800 MB (params + gradients + Adam). LoRA r=16 : ~3 MB
- [ ] Deploiement : 50 FT = 700 GB. 50 LoRA = 15 GB. Ratio 47x
- [ ] Init B=0 : garantit qu'au step 0, le modele LoRA = modele base. Si A et B etaient tous les deux random, le modele partirait deja biaise
