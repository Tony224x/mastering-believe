# Exercice difficile — J19 VLA introduction

## Objectif

Comparer **action tokenization** (RT-1/OpenVLA style) vs **action regression** (Octo/Diffusion Policy style) sur la **même tâche jouet**, et caractériser empiriquement les tradeoffs vus en théorie (§4 du cours).

## Consigne

Toujours sur le tiny-VLA du module (en mode multi-step `K=4` de l'exercice moyen, ou en mode single-step si tu veux rester simple) :

1. **Variante A — Tokenization** (déjà fait) : action heads en classification sur `N_BINS=16`, loss CE.
2. **Variante B — Regression MSE** : action head `nn.Linear(D_MODEL, N_ACTION_DIMS * K)` en sortie continue, loss MSE.
3. **Variante C — Regression "multi-mode" simulé** : pour stresser-test la multimodalité, ajoute une ambiguïté contrôlée — pour l'instruction `"go"` (mot ajouté au vocab), la cible est uniformément l'une des 4 directions avec probabilité 1/4 chacune (à chaque sample, on tire au sort). Re-entraîne A et B sur ce dataset avec ambiguïté.

Pour chacune des 3 variantes, mesure :

- **Accuracy** au sens "argmax direction recovered" (utilise un classifieur post-hoc qui retrouve la direction depuis le delta prédit).
- **Erreur quadratique** sur le delta prédit (continu).
- Sur la variante C, **diversité des prédictions** : combien des 4 directions le modèle couvre-t-il sur 100 inférences avec instruction `"go"` ?

## Criteres de reussite

- Variantes A et B atteignent une accuracy > 0.85 sur le dataset non-ambigu.
- Collapse de la régression mesuré : sur le dataset ambigu, la norme du delta prédit par B pour `"go"` est `< 0.3 ×` la norme moyenne des deltas cibles (il prédit ~ la moyenne (0,0)) — tu imprimes les deux normes.
- Multimodalité de la tokenization mesurée : sur 100 inférences `"go"` avec sampling softmax (température 1.0), la variante A couvre **au moins 3 des 4 directions**, là où B n'en couvre qu'une (toujours la même prédiction déterministe).
- Tu produis un tableau Markdown `variante × (accuracy, MSE, n_directions_couvertes)` pour les 3 variantes, suivi de 3-4 phrases qui relisent le tableau "Tokenization vs Regression" du §4 du cours à la lumière de tes chiffres.

## Indices

- Pour mesurer la multimodalité côté tokenization : remplace `argmax` par un **sampling** sur la distribution softmax (avec température) → tu verras les 4 modes apparaître.
- Pour la régression MSE : la collapse vers la moyenne est une conséquence directe de `argmin_a E[(a - y)^2] = E[y]`. Note bien que c'est ce qui motive l'usage de **diffusion** ou **flow matching** comme action heads chez Octo / π0 / Diffusion Policy (REFERENCES.md #17, #19).
- Tu peux limiter à `K=1` (single-step) pour cet exercice si la version multi-step est trop lourde.
