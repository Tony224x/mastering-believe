# Exercice (medium) - J10 : Replay Buffer + comparaison SARSA / Q-learning

## Objectif

(1) Coder un `ReplayBuffer` proprement teste, et (2) confronter empiriquement SARSA et Q-learning sur un GridWorld "cliff" — l'exemple canonique de `[Sutton & Barto, 2018, ch. 6.5]` qui exhibe la difference on-policy / off-policy.

## Consigne

### Partie A — ReplayBuffer

Code une classe `ReplayBuffer` avec :
- `__init__(self, capacity: int)` : taille max, comportement FIFO une fois pleine.
- `push(self, state, action, reward, next_state, done)` : ajoute une transition.
- `sample(self, batch_size: int)` : tirage **sans remise** dans le buffer, retourne 5 arrays numpy `(states, actions, rewards, next_states, dones)`.
- `__len__`.

Tests obligatoires (ecris-les) :
- Pousser 1000 transitions dans un buffer de capacite 100 -> `len(buffer) == 100`, et les 100 derniers states sont presents (pas les 900 premiers).
- `sample(32)` retourne des arrays de longueur 32 et ne plante pas.
- Avec un seed numpy fixe, deux runs produisent les memes batches (reproductibilite).

### Partie B — Cliff Walking : SARSA vs Q-learning

Implemente l'environnement "Cliff" du chapitre 6.5 de S&B :
- Grid 4 lignes x 12 colonnes.
- Start `(3, 0)`, Goal `(3, 11)`.
- Toute la ligne 3 entre les colonnes 1 et 10 est un "cliff" (reward `-100`, l'agent est teleporte au start, episode continue).
- Reward `-1` par step normal, `0` au goal qui termine l'episode.
- Actions : 4 directions, deterministe.

Implemente **SARSA** ET **Q-learning** tabulaires avec memes hyperparams (`alpha=0.5`, `gamma=1.0`, `epsilon=0.1` constant, 500 episodes).

Compare :
- Le retour moyen par episode lisse sur 50 episodes (matplotlib ou print numerique).
- La policy greedy finale extraite de chaque Q-table : trace ASCII du chemin que l'agent prendrait.

## Criteres de reussite

- Tous les tests du `ReplayBuffer` passent.
- Q-learning apprend la **trajectoire optimale** (le long du bord du cliff, retour total `-13`).
- SARSA apprend une **trajectoire safe** (s'eloigne du cliff, retour total moins bon en moyenne mais variance reduite, ~`-17`).
- Tu expliques en 3-5 lignes pourquoi cette difference est attendue (lien avec on-policy vs off-policy).

## Hints

- L'`epsilon` reste a `0.1` pendant tout l'entrainement (pas de decay), donc SARSA "voit" l'exploration et apprend une policy qui en tient compte.
- Q-learning, lui, apprend la policy greedy *optimale* meme si la policy de comportement explore : il bootstrap avec `max_a' Q`, pas avec l'action exploree.
- Si ta `Q-learning` n'apprend pas la trajectoire le long du cliff, verifie que la cible TD est bien `r + gamma * max_a' Q(s', a')` et non l'action effectivement prise.
