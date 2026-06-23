# Exercices Hard — Jour 6 : Transformer Architecture

---

## Exercice 7 : Bloc Transformer entrainable (forward + backward complet) en NumPy

### Objectif

Implementer un bloc Transformer entierement differentiable en NumPy (attention + FFN + LayerNorm + residuals, forward ET backward) et le valider par gradient check de bout en bout.

### Consigne

C'est l'exercice le plus complet de la semaine : un bloc Transformer pre-norm avec tous les gradients.

1. Reutiliser/implementer les briques avec leur backward :
   - `linear_forward/backward` (avec biais optionnel)
   - `layernorm_forward/backward` (Medium ex. 5)
   - `gelu_forward/backward`
   - `attention_forward/backward` (single-head suffit ; jour 5 hard ex. 7)

2. Assembler `block_forward(x, params)` (pre-norm) en cachant TOUTES les valeurs intermediaires :
   ```
   a = LayerNorm(x, ln1)
   attn_out = SelfAttention(a)          # Q=K=V=a @ projections
   x1 = x + attn_out                    # residual 1
   b = LayerNorm(x1, ln2)
   ffn_out = FFN(b)
   y = x1 + ffn_out                     # residual 2
   ```

3. Implementer `block_backward(dy, cache)` qui propage le gradient a travers les DEUX residuals (un residual ajoute son `dy` au chemin principal ET au chemin du sous-module). Renvoyer `dx` et les gradients de tous les parametres.

4. **Gradient check de bout en bout** : loss `0.5*||y||^2`, comparer chaque parametre (projections d'attention, W_O, FFN, les 2 LayerNorm) au gradient numerique. Erreur relative < 1e-4 pour tous.

5. **Mini-entrainement** : entrainer le bloc seul (+ une tete lineaire) sur une tache jouet (ex : copier la moyenne des tokens, ou classer une sequence) pour montrer que la loss descend — preuve que les gradients sont exploitables.

### Criteres de reussite

- [ ] Toutes les briques (linear, layernorm, gelu, attention) ont forward + backward corrects
- [ ] Le residual est correctement gere au backward (le gradient se duplique : chemin principal + sous-module)
- [ ] `block_backward` renvoie `dx` et les gradients de TOUS les parametres
- [ ] Le gradient check de bout en bout passe (erreur < 1e-4)
- [ ] Le mini-entrainement fait descendre la loss (les gradients sont corrects ET utilisables)

---

## Exercice 8 : Pre-norm vs Post-norm — etude de stabilite en profondeur

### Objectif

Demontrer empiriquement pourquoi le pre-norm a remplace le post-norm dans les Transformers profonds, en mesurant la norme des activations et des gradients selon la profondeur.

### Consigne

Construis deux variantes du bloc (forward uniquement suffit pour l'analyse de la propagation avant ; pour les gradients, reutilise le backward de l'ex. 7) :
```
Post-norm (original 2017) : x = LayerNorm(x + Sublayer(x))
Pre-norm  (GPT-2+)        : x = x + Sublayer(LayerNorm(x))
```

1. **Propagation avant** : empile N blocs (N ∈ {2, 8, 32, 64}) avec des poids initialises aleatoirement. Pour chaque N, mesure la norme moyenne de l'activation a la sortie du stack, pour les deux variantes. Le post-norm a-t-il tendance a amplifier/attenuer ?

2. **Propagation arriere (gradients)** : pour chaque variante et chaque N, injecte un gradient `dy = ones` a la sortie et propage-le jusqu'a l'entree (en reutilisant le backward). Mesure la norme du gradient a l'entree `||dx_0||`. Quelle variante preserve mieux le gradient en profondeur ?

3. **Tableau recapitulatif** : `N | ||activation|| post | ||activation|| pre | ||grad entree|| post | ||grad entree|| pre`.

4. **Analyse du chemin residuel** : explique pourquoi en pre-norm le chemin residuel est une somme directe `x_0 + sum(corrections)` (le gradient `d(sortie)/d(x_0)` contient un terme identite), alors qu'en post-norm chaque residual est re-normalise (le LayerNorm "casse" le chemin identite).

5. Question : les modeles tres profonds modernes (ex : 100+ couches) utilisent parfois des variantes (DeepNorm, ReZero, sandwich-norm). Quel probleme residuel du pre-norm cherchent-ils a corriger ?

### Criteres de reussite

- [ ] Les deux variantes (pre/post-norm) sont implementees correctement
- [ ] La norme d'activation est mesuree pour N ∈ {2, 8, 32, 64} dans les 2 variantes
- [ ] La norme du gradient a l'entree est mesuree (le pre-norm la preserve mieux en profondeur)
- [ ] Le tableau recapitulatif est correct et lisible
- [ ] L'analyse du chemin residuel est juste : pre-norm garde un chemin identite (gradient ~stable), post-norm re-normalise et attenue le signal en profondeur
- [ ] La reponse mentionne que le pre-norm peut souffrir d'une croissance de la variance du chemin residuel (d'ou DeepNorm/sandwich-norm pour les stacks tres profonds)
