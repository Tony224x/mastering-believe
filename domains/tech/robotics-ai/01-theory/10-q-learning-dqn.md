# J10 — Q-learning, DQN

> Acquis fin de jour : implementer Q-learning tabulaire sur GridWorld, et un DQN qui resout CartPole-v1.
> Sources : `[Sutton & Barto, 2018, ch. 6]`, `[Mnih et al., 2015]`, `[CS285, L7-8]`, `[CleanRL, dqn.py]`.

---

## 1. Le probleme concret avant la theorie

Reprends le robot de J9. En J9, tu avais un MDP `(S, A, P, R, gamma)` **complet** : la matrice de transition `P(s' | s, a)` etait connue. Value Iteration applique alors directement l'equation de Bellman optimale :

```
Q*(s, a) = sum_{s'} P(s' | s, a) * [R(s, a, s') + gamma * max_a' Q*(s', a')]
```

**Probleme reel** : un robot qui apprend a marcher ne connait pas `P`. Il ne connait pas non plus `R` analytiquement. Il observe juste une suite `(s_t, a_t, r_{t+1}, s_{t+1})` en interagissant avec son environnement.

Question centrale du J10 :

> *Comment apprendre `Q*(s, a)` **sans jamais ecrire `P` ni `R`**, juste a partir de transitions observees ?*

Reponse : remplacer **l'esperance sur `P`** par un **echantillon** (Monte-Carlo ou bootstrap), et corriger l'estimation de `Q` a chaque pas. C'est l'idee du *temporal-difference learning* (TD).

---

## 2. TD-learning : le coeur du sujet

### 2.1 TD(0) — apprendre `V` sans modele

Prediction (policy `pi` fixe). On veut estimer `V^pi(s)`. A chaque transition `(s, a, r, s')` observee :

```
V(s) <- V(s) + alpha * [ r + gamma * V(s') - V(s) ]
                        \-----------------------/
                            cible TD = r + gamma * V(s')
                        \-----/  \-----------------/
                       reward       bootstrap (estimation du futur)
```

Trois ideas a retenir :
1. **Sans modele** : on n'a pas besoin de `P`, on echantillonne `s'` en interagissant.
2. **Bootstrap** : on utilise notre propre estimation `V(s')` pour mettre a jour `V(s)`. C'est ce qui distingue TD de Monte-Carlo (qui attend la fin de l'episode pour calculer le retour reel `G_t`).
3. **Online, incremental** : une mise a jour par transition, pas besoin d'attendre la fin de l'episode.

`alpha` est le pas d'apprentissage (typiquement 0.01 a 0.5 en tabulaire).

### 2.2 n-step TD : un pont entre TD(0) et Monte-Carlo

TD(0) bootstrappe immediatement (apres 1 step). Monte-Carlo attend la fin (pas de bootstrap, juste le retour reel `G_t`). n-step TD interpole :

```
G_{t:t+n} = r_{t+1} + gamma * r_{t+2} + ... + gamma^{n-1} * r_{t+n} + gamma^n * V(s_{t+n})
```

- `n = 1` : TD(0).
- `n -> infini` (tronque a la fin de l'episode) : Monte-Carlo.

Trade-off : grand `n` reduit le biais (moins de bootstrap) mais augmente la variance (plus de bruit echantillonnage). En pratique, `n = 3` ou `n = 5` est souvent un sweet spot.

> Voir `[Sutton & Barto, 2018, ch. 6.1-6.3 et ch. 7]` pour la derivation complete.

---

## 3. Q-learning : controle off-policy

### 3.1 Du `V` au `Q`

`V(s)` te dit *combien vaut un etat*, mais pas *quelle action prendre*. Pour le controle, on apprend directement `Q(s, a)`. La regle de mise a jour TD devient SARSA si on utilise l'action *reellement prise* `a'` au prochain pas :

```
SARSA :   Q(s, a) <- Q(s, a) + alpha * [ r + gamma * Q(s', a') - Q(s, a) ]
```

C'est **on-policy** : `a'` vient de la meme policy qui genere les donnees.

### 3.2 Q-learning (Watkins, 1989)

Q-learning remplace `Q(s', a')` par `max_a' Q(s', a')` :

```
Q-learning :   Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]
                                              \-----------------------------/
                                                  cible greedy = optimal step ahead
```

Pourquoi c'est important :
- **Off-policy** : la policy qui genere les donnees (souvent `epsilon`-greedy pour explorer) est differente de la policy qu'on apprend (la greedy `argmax_a Q(s, a)`).
- **Convergence** : sous tabulaire + visite infinie de `(s, a)` + decay correct de `alpha`, Q-learning converge vers `Q*`. C'est le resultat de Watkins & Dayan, 1992.

### 3.3 Exploration : `epsilon`-greedy

Pour visiter assez de paires `(s, a)`, on perturbe la policy :

```
a = random action          avec probabilite epsilon
a = argmax_a Q(s, a)       avec probabilite 1 - epsilon
```

Decay typique : `epsilon` part de 1.0, descend lineairement a 0.05 sur les premiers % de l'entrainement.

### 3.4 Mnemonique

> **TD = "evaluer maintenant en utilisant ta propre estimation du futur".**
> **Q-learning = "TD + on prend toujours le max sur les actions futures".**

---

## 4. Du tabulaire a la fonction d'approximation

Tabulaire = un scalaire `Q(s, a)` par paire `(s, a)`. Marche quand `|S| * |A|` tient en RAM (GridWorld 100 cases x 4 actions = 400 valeurs, trivial).

CartPole : etats continus `(x, x_dot, theta, theta_dot)` dans R^4. Impossible de tabuler. **On approxime `Q(s, a; theta)` avec un reseau de neurones** parametre par `theta` (les poids). C'est *Deep Q-Network* (DQN).

L'idee est evidente. Le piege ne l'est pas.

### 4.1 Pourquoi naive Q-learning + reseau echoue

Si tu fais simplement :

```
loss = ( Q_theta(s, a) - [ r + gamma * max_a' Q_theta(s', a') ] )^2
theta <- theta - lr * grad(loss)
```

Trois problemes catastrophiques :

1. **Echantillons fortement correles**. Les transitions consecutives `(s_t, a_t, r_{t+1}, s_{t+1}), (s_{t+1}, a_{t+1}, ...)` sont trop similaires. SGD assume i.i.d., donc l'optimisation diverge.
2. **Cible mouvante**. La cible `r + gamma * max_a' Q_theta(s', a')` depend de `theta` qu'on est en train de mettre a jour. C'est comme courir derriere son ombre : non-stationnaire, oscillations.
3. **Distribution shift**. La policy change a chaque update, donc la distribution des `(s, a)` collectes change aussi. Le reseau "oublie" les regions deja apprises.

### 4.2 Les deux astuces de Mnih et al. 2015

`[Mnih et al., 2015]` (Nature, "Human-level control through deep reinforcement learning") introduit deux mecanismes simples mais decisifs.

#### Astuce 1 : Replay Buffer

Au lieu de mettre a jour sur la transition fraiche, on stocke chaque `(s, a, r, s', done)` dans un buffer circulaire `D` de taille ~100k a 1M. A chaque step d'apprentissage :

```
mini-batch = random.sample(D, batch_size=32 ou 64)
loss = MSE sur ce mini-batch
```

Effets :
- Decorrele les echantillons (tirage aleatoire dans un buffer "vieux").
- Reutilise chaque transition plusieurs fois (sample efficiency).
- Lisse la distribution d'entrainement.

#### Astuce 2 : Target Network

On maintient une **deuxieme copie** des poids, `theta_target`, gelee la plupart du temps. La cible TD utilise `theta_target` :

```
target = r + gamma * max_a' Q_{theta_target}(s', a')
loss   = ( Q_theta(s, a) - target )^2
```

Periodiquement (toutes les `C` steps, typiquement 500-10000), on synchronise : `theta_target <- theta`. Variante "soft update" (Polyak) : `theta_target <- tau * theta + (1 - tau) * theta_target` avec `tau ~ 0.005`.

Effet : la cible est **stationnaire pendant `C` steps**. Le reseau apprend une cible fixe, puis on rafraichit. Beaucoup plus stable.

### 4.3 Pseudocode DQN complet

```
Initialiser Q_theta (reseau), Q_target = copie de Q_theta
Initialiser replay buffer D (capacite N)
epsilon = 1.0

Pour chaque episode :
    s = env.reset()
    Pour chaque step :
        a = epsilon-greedy(Q_theta, s)
        s', r, done = env.step(a)
        D.push( (s, a, r, s', done) )

        Si len(D) >= batch_size :
            batch = D.sample(batch_size)
            Pour chaque (s_i, a_i, r_i, s'_i, done_i) :
                y_i = r_i si done_i sinon r_i + gamma * max_a' Q_target(s'_i, a')
            loss = MSE( Q_theta(s_i, a_i), y_i )
            theta -= lr * grad(loss)

        Tous les C steps : Q_target <- Q_theta
        Decay epsilon
        s = s'
        Si done : break
```

C'est exactement la structure de `cleanrl/cleanrl/dqn.py` (`[CleanRL, dqn.py]`). 200 lignes de code lisibles ligne a ligne.

---

## 5. Variantes utiles

### 5.1 Double DQN (van Hasselt et al., 2015)

Le `max_a' Q_target(s', a')` dans la cible est biaise vers le haut (overestimation bias) parce que le `max` selectionne et evalue avec les memes valeurs bruitees.

Fix : decoupler selection et evaluation.

```
a*  = argmax_a' Q_theta(s', a')          # selection avec online net
y   = r + gamma * Q_target(s', a*)        # evaluation avec target net
```

Plus stable, meilleurs scores Atari. Modification d'une ligne dans le code.

### 5.2 Dueling DQN (Wang et al., 2016)

Architecture decomposee :

```
Q(s, a) = V(s) + ( A(s, a) - mean_a A(s, a) )
```

Une tete pour `V(s)` (valeur d'etat), une tete pour `A(s, a)` (avantage). Utile quand le choix d'action n'a pas grand impact sur certains etats (la valeur de l'etat domine). Aucun changement a la loss, juste l'architecture.

### 5.3 Autres extensions du "Rainbow"

Hessel et al., 2018, combinent : Double + Dueling + Prioritized Experience Replay + Multi-step + Distributional + Noisy Nets. Chaque composant apporte ~5-15% sur Atari, le combo est le SOTA Q-learning.

---

## 6. Limites : pourquoi DQN ne marche pas sur MuJoCo nativement

DQN suppose un **espace d'actions discret fini**. La cible TD calcule `max_a' Q(s', a')` qui necessite un argmax sur toutes les actions.

Sur CartPole (2 actions discretes : gauche / droite) ou Atari (~18 actions discretes) : OK.

Sur MuJoCo (HalfCheetah, Ant, Humanoid) : actions = vecteurs continus dans `R^d` (typiquement `d = 6` a `21`). Calculer `argmax_a` sur `R^d` est intractable.

Workarounds :
- **Discretiser** chaque dimension en `K` bins -> action space de taille `K^d`. Explose : `5^6 = 15625` actions, deja lourd.
- **Q-learning continu** : NAF (Gu et al., 2016) parametre `Q(s, a) = -1/2 (a - mu(s))^T P(s) (a - mu(s)) + V(s)`. Le `argmax` est analytique : `a* = mu(s)`. Marche, mais peu utilise en pratique.
- **Sortir du Q-learning** : algorithmes actor-critic continus (DDPG, TD3, SAC) qui apprennent une policy `mu(s)` explicite parametree par un reseau. C'est le sujet des J11-J12.

> Retenue : **DQN = discret. MuJoCo = continu. Pour MuJoCo, on passe a PPO/SAC/TD3.**

---

## 7. Cheatsheet

| Algo            | On/off-policy | Tabular | Function approx | Discret/Continu | Ressource cle              |
|-----------------|---------------|---------|------------------|------------------|----------------------------|
| TD(0)           | on            | oui     | possible         | les deux         | S&B ch. 6.1                |
| SARSA           | on            | oui     | possible         | discret          | S&B ch. 6.4                |
| Q-learning      | off           | oui     | possible         | discret          | S&B ch. 6.5                |
| DQN             | off           | non     | NN obligatoire   | discret          | Mnih 2015, CleanRL         |
| Double DQN      | off           | non     | NN               | discret          | van Hasselt 2015           |
| Dueling DQN     | off           | non     | NN               | discret          | Wang 2016                  |
| DDPG / TD3 / SAC| off           | non     | NN               | continu          | J11-J12                    |

---

## 8. Flash-cards (5 Q/R, spaced-rep)

1. **Q :** Pourquoi TD-learning n'a-t-il pas besoin de connaitre `P(s' | s, a)` ?
   **R :** Parce qu'il echantillonne `s'` directement en interagissant avec l'environnement, et utilise sa propre estimation `V(s')` pour bootstrapper la cible. L'esperance sur `P` est remplacee par un seul echantillon.

2. **Q :** Quelle est la difference exacte entre la regle de mise a jour SARSA et Q-learning ?
   **R :** SARSA utilise `Q(s', a')` ou `a'` est l'action **reellement prise** au pas suivant (on-policy). Q-learning utilise `max_a' Q(s', a')` (off-policy : la cible suppose qu'on prend l'action greedy, meme si on a explore).

3. **Q :** Quels sont les deux mecanismes ajoutes par Mnih et al. 2015 pour stabiliser Q-learning + reseau de neurones, et a quoi sert chacun ?
   **R :** (1) Replay buffer : decorrele les echantillons consecutifs et permet de reutiliser les transitions. (2) Target network : fournit une cible TD stationnaire pendant `C` steps, sinon le reseau "courrait derriere son ombre".

4. **Q :** Pourquoi le `max` dans la cible Q-learning cause-t-il un biais d'overestimation, et comment Double DQN le corrige ?
   **R :** Le `max` selectionne et evalue avec les memes valeurs bruitees, donc selectionne systematiquement les estimations sur-estimees. Double DQN decouple : selection avec le reseau online (`argmax`), evaluation avec le target network.

5. **Q :** Pourquoi DQN ne marche pas tel quel sur HalfCheetah (MuJoCo) ?
   **R :** Parce que les actions sont continues (`R^d`), et la cible TD requiert un `argmax_a` sur l'espace d'actions, qui est intractable en continu. Solutions : discretiser (explose), NAF, ou changer de famille (DDPG/TD3/SAC, sujet du J11-J12).

---

## 9. References

- `[Sutton & Barto, 2018, ch. 6 (TD prediction & control), ch. 7 (n-step TD)]` — la bible.
- `[Mnih et al., 2015]` — "Human-level control through deep reinforcement learning", Nature 518.
- `[CS285, Fall 2023, L7-L8 (value-based methods, deep RL)]` — derivations rigoureuses + intuitions failure modes.
- `[CleanRL, dqn.py]` — implementation single-file de reference (https://github.com/vwxyzjn/cleanrl).
- van Hasselt, Guez, Silver, 2015 — Double DQN (`arXiv:1509.06461`).
- Wang et al., 2016 — Dueling DQN (`arXiv:1511.06581`).
