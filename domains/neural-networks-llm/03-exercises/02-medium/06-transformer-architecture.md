# Exercices Medium — Jour 6 : Transformer Architecture

---

## Exercice 4 : LayerNorm from scratch — forward, backward, gradient check

### Objectif

Implementer LayerNorm complet (avec gamma/beta) y compris son backward — la normalisation est presente 2 fois par bloc, autant la connaitre exactement.

### Consigne

1. Implementer le forward :

```python
def layernorm_forward(x, gamma, beta, eps=1e-5):
    """x: (batch, d). Normalise sur le DERNIER axe.
    y = gamma * (x - mean) / sqrt(var + eps) + beta
    Retourne y et un cache pour le backward."""
```

2. Verifier sur un batch aleatoire (8, 16) : chaque ligne de `(y - beta) / gamma` a moyenne ~0 (|m| < 1e-9) et variance ~1 (|v - 1| < 1e-3 avec eps).

3. Implementer le backward `layernorm_backward(dy, cache)` retournant `dx, dgamma, dbeta`. La formule de `dx` (D = dimension normalisee, `xhat` = x normalise) :

```
dx = (gamma / sqrt(var + eps)) * (dy - mean(dy) - xhat * mean(dy * xhat))
```

   ou les moyennes sont prises sur l'axe normalise. `dgamma = sum(dy * xhat, axis=0)`, `dbeta = sum(dy, axis=0)`.

4. Gradient check par differences finies (eps=1e-6) sur `L = sum(y * G)` avec G fixe : verifier `dx`, `dgamma`, `dbeta` element par element, erreur relative < 1e-6.

5. Question piege (repondre en commentaire) : pourquoi `dx` n'est PAS simplement `dy * gamma / sqrt(var + eps)` ? (Indice : mean et var dependent de x.)

### Criteres de reussite

- [ ] Le forward normalise sur le bon axe (le dernier, par token — pas par batch)
- [ ] Les statistiques post-normalisation sont verifiees (moyenne 0, variance 1)
- [ ] Gradient check < 1e-6 sur dx, dgamma, dbeta (tous les elements)
- [ ] La reponse a la question piege est correcte : mean/var sont des fonctions de x, leurs gradients ajoutent les 2 termes correctifs
- [ ] Le code gere un batch de shape arbitraire (batch, T, d) en plus de (batch, d)

---

## Exercice 5 : Compteur de parametres — retrouver les 124M de GPT-2

### Objectif

Savoir compter les parametres d'un Transformer de tete et par code — competence d'entretien classique et indispensable pour estimer memoire/compute.

### Consigne

1. Ecrire `count_transformer_params(vocab_size, d_model, n_layers, n_heads, d_ff, max_pos, tied_embeddings=True)` qui detaille :
   - embeddings de tokens : `vocab_size * d_model`
   - embeddings de position : `max_pos * d_model`
   - par bloc : attention (4 matrices `d_model x d_model` + 4 bias), FFN (`d_model*d_ff + d_ff` + `d_ff*d_model + d_model`), 2 LayerNorm (`2 * 2 * d_model`)
   - LayerNorm final + lm_head (0 si tied avec l'embedding)

2. Verifier sur **GPT-2 small** : `vocab=50257, d_model=768, n_layers=12, n_heads=12, d_ff=3072, max_pos=1024`, embeddings ties. Le total doit tomber a 124M ± 2%.

3. Afficher la decomposition en % : embeddings vs attention vs FFN vs norms. Verifier que FFN ≈ 2x les parametres de l'attention (8d² vs 4d² par bloc).

4. Predire AVANT de calculer (ecrire la prediction en commentaire) : si on double `d_model` a `n_layers` constant, le nombre de parametres des blocs est multiplie par combien ? Verifier par le code (GPT-2 small vs un clone avec d_model=1536).

5. Calculer pour GPT-2 medium (`d_model=1024, n_layers=24, n_heads=16, d_ff=4096`) et comparer aux ~350M annonces.

### Criteres de reussite

- [ ] GPT-2 small : total dans [121.5M, 126.5M]
- [ ] La part embeddings est correcte (~31% pour GPT-2 small grace au tying)
- [ ] Le ratio FFN/attention par bloc vaut 2 (a ~1% pres, bias inclus)
- [ ] La prediction "x4 sur les blocs quand d_model double" est ecrite avant le calcul et verifiee
- [ ] GPT-2 medium tombe dans [340M, 360M]

---

## Exercice 6 : Pre-LN vs Post-LN — mesurer la stabilite

### Objectif

Montrer experimentalement pourquoi tous les LLMs modernes utilisent Pre-LN : la profondeur degrade le flot du gradient en Post-LN.

### Consigne

On simplifie le bloc a `FFN + residual` (sans attention) pour isoler l'effet de la position du LayerNorm :

- **Post-LN** : `x = LayerNorm(x + FFN(x))`
- **Pre-LN** : `x = x + FFN(LayerNorm(x))`

1. Implementer les deux variantes avec un FFN 2 couches (`d=32, d_ff=64`, activation ReLU, init `randn * 0.5` — volontairement un peu grande), et empiler `N=12` blocs (poids differents par bloc, seed fixe identique pour les deux variantes).

2. **Forward** : pour un input ~N(0,1) de shape (4, 32), tracer la norme moyenne de l'activation apres chaque bloc pour les deux variantes. Observer : Post-LN reste a norme constante (force par le LN final de chaque bloc), Pre-LN croit (le residual stream accumule).

3. **Gradient** : avec la loss `L = sum(output**2)`, estimer par differences finies (eps=1e-4) le gradient de L par rapport a **20 poids echantillonnes du PREMIER bloc** pour chaque variante. Comparer la norme moyenne des gradients du bloc 1 vs ceux du bloc 12 (memes 20 indices).

4. Construire le tableau : `variante | ||grad bloc 1|| | ||grad bloc 12|| | ratio bloc1/bloc12`. En Post-LN le ratio doit etre tres petit (le gradient s'attenue en remontant) ; en Pre-LN il doit rester d'un ordre de grandeur raisonnable (> 100x celui du Post-LN).

5. Conclure en commentaire : pourquoi Pre-LN permet d'entrainer sans warmup soigneux la ou Post-LN (Transformer original) en avait besoin.

### Criteres de reussite

- [ ] Les deux variantes sont implementees correctement (position du LN verifiee)
- [ ] Les normes d'activation par profondeur sont affichees et montrent le contraste attendu
- [ ] Le gradient est estime sur les MEMES indices de poids pour les deux variantes (comparaison equitable)
- [ ] Le ratio gradient-bloc-1 Pre-LN / Post-LN est > 100 (le vanishing du Post-LN est demontre numeriquement)
- [ ] La conclusion relie le resultat au chemin identite du residual stream (en Pre-LN le gradient a une autoroute sans LN sur le chemin)
