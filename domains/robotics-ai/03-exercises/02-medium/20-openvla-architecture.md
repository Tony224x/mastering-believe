# Exercice J20 — niveau medium

## Objectif

Implementer LoRA from scratch et **prouver mathematiquement et numeriquement** la propriete d'identite a l'initialisation : un modele avec LoRA frais doit produire **exactement** les memes outputs que le modele de base.

Reference : `domains/robotics-ai/01-theory/20-openvla-architecture.md` section 4.2 + papier LoRA (Hu 2021, https://arxiv.org/abs/2106.09685).

## Consigne

1. **Implemente** une classe `LoRALinear(nn.Module)` qui wrap une `nn.Linear` existante :
   - Gele les params de la `Linear` de base (`requires_grad = False`).
   - Ajoute deux parametres trainables : `lora_A` de shape `(r, d_in)` initialisee Gaussian (`std = 1/r`) et `lora_B` de shape `(d_out, r)` initialisee a **zero**.
   - Le forward retourne `base(x) + (alpha/r) * x @ A.T @ B.T`.

2. **Ecris** une fonction `count_trainable_ratio(linear, r, alpha)` qui :
   - Wrap une `nn.Linear(d_in, d_out)` avec `LoRALinear`.
   - Retourne le ratio `params_trainables / params_totaux` exprime en pourcentage.
   - Verifie sur `nn.Linear(4096, 4096)` avec `r=32` : tu dois obtenir **environ 1.56%**.

3. **Probe adversariale** : ecris un test `test_lora_identity_at_init()` qui :
   - Construit `base = nn.Linear(64, 64)`, applique `lora = LoRALinear(base, r=8, alpha=16)`.
   - Genere un input aleatoire `x = torch.randn(4, 64)`.
   - Verifie que `torch.allclose(base(x), lora(x), atol=1e-6)`.
   - Si ca echoue, **explique pourquoi** dans un commentaire (typiquement : init de B != 0 ou bug de scaling).

4. **Bonus comprehension** : ajoute une methode `effective_weight()` qui renvoie la matrice equivalente `W_eff = W + (alpha/r) * B @ A`, et verifie que `lora(x)` est numeriquement equivalent a `F.linear(x, effective_weight())` (tolerance `1e-5`).

## Criteres de reussite

- [ ] `LoRALinear` ne contient **aucun** parametre trainable hors `lora_A` et `lora_B`.
- [ ] Le ratio sur `Linear(4096, 4096), r=32` est environ 1.56% (tolerance ±0.05%).
- [ ] `test_lora_identity_at_init()` passe.
- [ ] `effective_weight()` reconstruit bien la matrice equivalente.
- [ ] Tu commentes pourquoi le scaling vaut `alpha/r` et pas juste `1` (rep : pour decoupler la magnitude de l'update du choix de rang ; sinon changer `r` casse l'optimisation).
