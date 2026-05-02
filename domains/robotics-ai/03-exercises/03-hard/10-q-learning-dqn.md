# Exercice (hard) - J10 : Double DQN sur LunarLander-v3

## Objectif

Etendre l'implementation DQN du cours en **Double DQN** (van Hasselt et al., 2015), l'entrainer sur `LunarLander-v3` (action space discret : 4 actions), et **mesurer empiriquement** le biais d'overestimation : compares les Q-values predites par Vanilla DQN vs Double DQN sur les memes etats.

C'est un exercice integratif : tu touches au reseau, a la cible TD, au target network, au logging, et tu produis un mini-rapport quantitatif.

## Consigne

### Partie A — Implementation Double DQN

Pars du DQN du cours (`02-code/10-q-learning-dqn.py`). Modifie **uniquement** la cible TD pour decoupler selection et evaluation :

```
# Vanilla DQN :
target_max = target_net(next_states).max(dim=1).values
td_target  = rewards + gamma * target_max * (1 - dones)

# Double DQN :
next_actions = q_net(next_states).argmax(dim=1, keepdim=True)
target_max   = target_net(next_states).gather(1, next_actions).squeeze(1)
td_target    = rewards + gamma * target_max * (1 - dones)
```

Permet de basculer entre les deux modes via un parametre `double: bool`.

### Partie B — Entrainement sur LunarLander-v3

Configure :
- Hyperparams : `total_steps=200_000`, `buffer_capacity=100_000`, `batch_size=128`, `lr=2.5e-4`, `gamma=0.99`, `target_update_frequency=1000`, `epsilon_decay_steps=80_000`.
- Reseau : MLP `(8) -> 256 -> 256 -> 4`.
- Lance 3 seeds (0, 1, 2) pour chaque variante. Logge le retour moyen par episode et la **Q-value moyenne predite** sur un set fixe de 1000 etats (echantillon aleatoire du replay buffer apres 50k steps, conserve constant pour tout le reste de l'entrainement).

### Partie C — Analyse

Produis un script + un rapport `solution.md` (max 1 page) qui repond a :
1. Quel est le retour moyen final (moyenne sur 3 seeds, std incluse) pour Vanilla DQN vs Double DQN ?
2. Trace la Q-value moyenne predite au fil de l'entrainement pour les deux variantes. Y a-t-il une difference systematique (signature de l'overestimation) ?
3. Le retour reel (rollouts greedy) est-il plus proche de la Q-value pour Double DQN que pour Vanilla DQN ?

## Criteres de reussite

- Au moins une seed de Vanilla DQN ET de Double DQN atteignent un retour moyen >= **200** sur les 100 derniers episodes (LunarLander considere "resolu" a 200).
- Le rapport repond aux 3 questions avec des chiffres concrets et un graphique au minimum.
- Tu identifies au moins UN run ou Double DQN est visiblement plus stable / moins biaise (c'est statistiquement attendu, mais sur 3 seeds ce n'est pas garanti — discute le si tu n'observes pas de difference claire).

## Hints

- **Pas tres rapide**. 200k steps sur LunarLander prennent 30-90 minutes sur CPU. Si trop lent, tente sur Colab GPU ou reduis `total_steps` a 100k (LunarLander reste apprenable).
- L'overestimation est plus visible **tot dans l'entrainement**, quand Q est encore bruyant. Logge a haute frequence dans les 50k premiers steps.
- Si Vanilla DQN **diverge** (return s'effondre apres avoir bien appris), c'est une autre signature de l'overestimation : Double DQN ne devrait pas avoir ce comportement. Garde les courbes pour ton rapport.
- Pour LunarLander-v3 (Gymnasium 1.x), `env = gym.make("LunarLander-v3")` requiert `gymnasium[box2d]` (`pip install "gymnasium[box2d]"`).
