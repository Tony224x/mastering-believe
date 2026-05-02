# J9 — MDPs, Bellman, value/policy iteration

> Source unique : Sutton & Barto, *Reinforcement Learning: An Introduction*, 2e ed., MIT Press, 2018, ch. 3-4 [Sutton & Barto, 2018, ch. 3-4]. Lectures CS285 L4 (Levine, Berkeley 2023) pour la perspective robotique-RL.

## 1. Le problème, en concret

Imagine un robot dans une grille **4x4**. Le robot occupe une case `(i, j)`. À chaque pas de temps, il choisit une action parmi `{haut, bas, gauche, droite}` :

```
+---+---+---+---+
| S | . | . | . |
+---+---+---+---+
| . | X | . | . |
+---+---+---+---+
| . | . | . | . |
+---+---+---+---+
| . | . | . | G |
+---+---+---+---+
```

- `S` (start) : case (0, 0)
- `G` (goal) : case (3, 3) — récompense `+1`, fin d'épisode
- `X` (lava) : case (1, 1) — récompense `-1`, fin d'épisode
- toute autre case : récompense `0` à chaque pas (penalty `-0.04` parfois pour pousser à finir vite)
- les actions ne sont **pas déterministes** : 80% du temps l'action voulue est exécutée, 10% le robot dérape à 90 degrés à gauche, 10% à droite. Si le robot tape un mur, il reste sur place.

**Question** : quelle action choisir dans chaque case pour maximiser la récompense espérée totale ?

C'est l'archétype du problème de **prise de décision séquentielle sous incertitude**. La réponse est une **politique** `π : S → A` — une recette qui dit, dans chaque état, quelle action prendre.

Pour répondre, il nous faut :
1. Un cadre formel pour décrire ce monde (le MDP).
2. Un objectif quantifiable (la valeur).
3. Un algorithme qui calcule la politique optimale (value iteration, policy iteration).

## 2. MDP : définition formelle

Un **Markov Decision Process** est un tuple `(S, A, P, R, γ)` :

| Symbole | Nom | Description |
|---|---|---|
| `S` | États | Ensemble fini ou continu d'états du monde |
| `A` | Actions | Ensemble (souvent fini) d'actions disponibles |
| `P(s' \| s, a)` | Transitions | Probabilité d'arriver en `s'` en faisant `a` depuis `s` |
| `R(s, a, s')` | Récompense | Récompense reçue lors de la transition `(s, a) → s'` |
| `γ ∈ [0, 1[` | Discount | Pondère les récompenses futures (`0` = myope, proche de `1` = patient) |

**Propriété de Markov** : `P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, ...) = P(s_{t+1} | s_t, a_t)`. Le futur ne dépend que du présent, pas de l'historique. Si ce n'est pas vrai dans ton problème, c'est qu'il manque de l'info dans `s_t` — il faut enrichir l'état.

**Pourquoi `γ < 1` ?**
- mathématiquement : garantit que la somme infinie `Σ γ^t r_t` converge (si récompenses bornées)
- pratiquement : exprime une préférence pour le présent (1 € aujourd'hui > 1 € dans 100 pas)

Pour le GridWorld plus haut :
- `S = {(i, j) : 0 ≤ i, j < 4}` — 16 états, dont 2 terminaux (`G`, `X`)
- `A = {haut, bas, gauche, droite}` — 4 actions
- `P` : 80%/10%/10% pour exécution/dérape gauche/dérape droite ; identité si action contre un mur
- `R` : `+1` à l'entrée de `G`, `-1` à l'entrée de `X`, `-0.04` ailleurs (living penalty)
- `γ = 0.9` typiquement

## 3. Politique, retour, fonctions de valeur

### Politique
Une **politique** `π(a | s)` est la probabilité de choisir l'action `a` dans l'état `s`. Si déterministe : `π : S → A`.

### Retour (return)
Le **retour** depuis l'instant `t` est la somme actualisée des récompenses futures :

```
G_t = R_{t+1} + γ R_{t+2} + γ^2 R_{t+3} + ... = Σ_{k=0}^∞ γ^k R_{t+k+1}
```

C'est cette quantité qu'on cherche à maximiser **en espérance** (les transitions sont stochastiques).

### Value function `V_π(s)`
Espérance du retour si on part de `s` et qu'on suit `π` :

```
V_π(s) = E_π [ G_t | S_t = s ]
```

Réponse à : « combien je vaux si je suis dans l'état `s` et que je joue selon `π` ? »

### Action-value (Q-function) `Q_π(s, a)`
Espérance du retour si on part de `s`, on prend `a` (une seule fois), puis on suit `π` :

```
Q_π(s, a) = E_π [ G_t | S_t = s, A_t = a ]
```

Réponse à : « combien je vaux si je force la première action puis joue selon `π` ? »

Lien : `V_π(s) = Σ_a π(a | s) Q_π(s, a)`. Pour une politique déterministe, `V_π(s) = Q_π(s, π(s))`.

## 4. Équations de Bellman

Cœur du chapitre. Deux versions à bien distinguer.

### 4.1 Bellman *expected* (évaluation d'une politique fixée `π`)

```
V_π(s) = Σ_a π(a | s) Σ_{s'} P(s' | s, a) [ R(s, a, s') + γ V_π(s') ]
```

Lecture : « la valeur de `s` sous `π` = espérance, sur l'action choisie par `π` puis sur la transition, de la récompense immédiate plus la valeur actualisée du successeur ».

C'est un **système linéaire** de `|S|` équations à `|S|` inconnues. Solvable directement si `|S|` est petit (inverser une matrice `(I - γ P_π)`). Pour les grands `|S|`, on itère (cf. `policy_evaluation` plus bas).

### 4.2 Bellman *optimality* (politique optimale)

La politique optimale `π*` maximise `V` partout simultanément. Elle vérifie :

```
V*(s) = max_a Σ_{s'} P(s' | s, a) [ R(s, a, s') + γ V*(s') ]
```

```
Q*(s, a) = Σ_{s'} P(s' | s, a) [ R(s, a, s') + γ max_{a'} Q*(s', a') ]
```

L'opérateur `max_a` rend ces équations **non linéaires**. On ne les résout pas par inversion ; on les résout par **itération de point fixe**.

**Théorème (Banach)** : l'opérateur de Bellman optimal `T*` est une **contraction** de coefficient `γ` sous la norme infinie. Donc :
- une unique solution `V*` existe
- toute itération `V_{k+1} := T* V_k` converge géométriquement vers `V*` à vitesse `γ^k`

C'est ce qui rend value iteration fiable : on est garanti de converger.

### Politique optimale extraite de `V*`
```
π*(s) = argmax_a Σ_{s'} P(s' | s, a) [ R(s, a, s') + γ V*(s') ]
```

Si on connaît `Q*` directement : `π*(s) = argmax_a Q*(s, a)`. Q-fonction = représentation plus pratique pour la politique car pas besoin de modèle au moment de l'action.

## 5. Value Iteration (VI)

Algo le plus simple. On itère **directement** l'opérateur de Bellman optimal sur `V` :

```
Initialiser V_0(s) = 0 pour tout s
Répéter :
    Pour chaque s :
        V_{k+1}(s) = max_a Σ_{s'} P(s' | s, a) [ R(s, a, s') + γ V_k(s') ]
Jusqu'à || V_{k+1} - V_k ||_∞ < ε
Retourner π(s) = argmax_a Σ_{s'} P(s' | s, a) [ R(s, a, s') + γ V_k(s') ]
```

- Une seule passe = une seule application de `T*`
- Convergence garantie (Banach)
- Coût d'une itération : `O(|S|^2 |A|)` (pour chaque état, chaque action, somme sur les successeurs)

## 6. Policy Iteration (PI)

Alterne deux phases :

### Phase 1 — Évaluation de politique
On fixe `π` et on calcule `V_π` exactement (en itérant Bellman *expected*, ou en résolvant le système linéaire). Cette phase est elle-même un point fixe :
```
V_{k+1}(s) = Σ_a π(a | s) Σ_{s'} P(s' | s, a) [ R(s, a, s') + γ V_k(s') ]
```

### Phase 2 — Amélioration de politique
On rend `π` gloutonne par rapport à `V_π` :
```
π_new(s) = argmax_a Σ_{s'} P(s' | s, a) [ R(s, a, s') + γ V_π(s') ]
```

**Théorème de policy improvement** : si `π_new ≠ π`, alors `V_{π_new}(s) ≥ V_π(s)` pour tout `s`, avec inégalité stricte quelque part. Donc on progresse strictement à chaque itération.

On répète jusqu'à `π_new == π` (point fixe = optimum).

**PI converge en très peu d'itérations** (souvent < 10 même pour des centaines d'états), mais chaque itération fait un calcul d'évaluation potentiellement coûteux.

## 7. VI vs PI : quand utiliser quoi ?

| | Value Iteration | Policy Iteration |
|---|---|---|
| Itérations externes | Beaucoup (proportionnel à `1/(1-γ)`) | Très peu (~ log) |
| Coût par itération | Faible (`O(|S|^2 |A|)`) | Élevé (résolution du système d'évaluation) |
| Implémentation | Très simple | Deux boucles imbriquées |
| Quand préférer | `|S|` grand, `γ` faible | `|S|` modéré, `γ` proche de 1 |

**Modified Policy Iteration** : compromis. On fait `k` étapes d'évaluation (au lieu d'aller jusqu'à convergence) puis amélioration. `k = 1` redonne VI ; `k → ∞` redonne PI.

## 8. Convergence — intuition

Pour VI :
- `|| V_{k+1} - V* ||_∞ ≤ γ || V_k - V* ||_∞` (contraction)
- donc l'erreur est divisée par `γ` à chaque itération
- si `γ = 0.9`, après 50 itérations l'erreur initiale est multipliée par `0.9^50 ≈ 0.005`

Critère d'arrêt en pratique : on stoppe quand `|| V_{k+1} - V_k ||_∞ < ε`, ce qui borne `|| V_k - V* ||_∞ ≤ ε γ / (1 - γ)`.

## 9. Limites des MDPs tabulaires

- **Curse of dimensionality** : `|S|` explose avec la dimension (un robot 6-DOF discrétisé à 10 valeurs/joint = `10^6` états)
- **Modèle requis** : VI/PI supposent `P` et `R` connus. Si non → RL **model-free** (Q-learning, policy gradients — J10, J11)
- **États continus** : il faut approximer `V` ou `Q` (réseau de neurones — DQN, J10)
- **Observabilité partielle** : si l'agent ne voit pas l'état complet → POMDP (cadre étendu, hors scope ici)

Mais : MDPs tabulaires + Bellman = **les fondations** sur lesquelles repose tout le RL moderne. Tu ne comprends pas DQN sans comprendre Bellman optimality. Tu ne comprends pas l'actor-critic sans comprendre policy improvement.

## Points clés à retenir

> **MDP = `(S, A, P, R, γ)`** + propriété de Markov.
> **Bellman expected** = équation linéaire pour évaluer une politique fixée.
> **Bellman optimality** = équation non-linéaire (présence de `max`) pour la politique optimale.
> **VI** = itération directe de Bellman optimal sur `V`, simple, converge géométriquement.
> **PI** = alternance évaluation / amélioration, peu d'itérations externes.
> Tous deux convergent vers `V*` et `π*` ; choix pratique = compromis vitesse vs simplicité.

## Flash-cards (spaced repetition)

1. **Q.** Pourquoi a-t-on besoin du discount factor `γ` ?
   **R.** Pour garantir la convergence de la somme infinie de récompenses (si bornées) et pour exprimer la préférence pour le présent. Mathématiquement, `γ < 1` rend l'opérateur de Bellman contractant.

2. **Q.** Quelle est la différence entre `V_π` et `V*` ?
   **R.** `V_π` est la valeur sous une politique fixée `π` (équation de Bellman *expected*, linéaire). `V*` est la valeur optimale, max sur toutes les politiques (équation de Bellman *optimality*, non-linéaire à cause du `max`).

3. **Q.** Pourquoi value iteration converge-t-elle ?
   **R.** L'opérateur de Bellman optimal `T*` est une contraction de coefficient `γ` en norme infinie. Banach garantit existence/unicité du point fixe et convergence géométrique à taux `γ`.

4. **Q.** Combien d'itérations externes prend policy iteration sur un MDP tabulaire fini ?
   **R.** Très peu, souvent `< 10`, car `π` ne peut prendre qu'un nombre fini de valeurs et chaque itération améliore strictement (théorème de policy improvement) jusqu'au point fixe.

5. **Q.** Qu'est-ce qui justifie d'extraire la politique gloutonne de `V*` à la fin de VI ?
   **R.** L'équation de Bellman optimality dit que `π*(s) = argmax_a Σ_{s'} P(s'|s,a) [R + γ V*(s')]`. Une fois `V*` calculée, l'argmax sur `a` donne directement la politique optimale.

---

**Sources**
- Sutton & Barto, *Reinforcement Learning: An Introduction*, 2e ed., 2018, ch. 3 (Finite MDPs) et ch. 4 (Dynamic Programming) [Sutton & Barto, 2018, ch. 3-4].
- Levine, S., *CS285 — Deep Reinforcement Learning*, Berkeley, Fall 2023, Lecture 4 (« Introduction to Reinforcement Learning »).
