# J11 — Exercice facile : REINFORCE from scratch sur CartPole

## Objectif

Implementer **vanilla REINFORCE** (sans baseline, sans GAE, sans rien) sur `CartPole-v1` et observer son comportement, pour bien sentir d'ou viennent les ameliorations apportees par PPO.

## Consigne

1. Cree un script `reinforce_cartpole.py` qui :
   - Construit une policy `pi_theta(a | s)` : MLP `[obs_dim -> 64 -> 64 -> n_actions]` avec une softmax (utilise `torch.distributions.Categorical(logits=...)`).
   - Pour chaque episode :
     - Roll out une trajectoire complete (jusqu'a `terminated`/`truncated`).
     - Stocke les `(s_t, a_t, r_t)`.
   - Calcule les **returns-to-go** `R_t = sum_{k>=t} gamma^{k-t} r_k` avec `gamma = 0.99`.
   - Loss : `loss = - mean( log_prob(a_t | s_t) * R_t )`.
   - Update Adam (`lr = 1e-3`).

2. Affiche le retour moyen tous les 10 episodes pendant **500 episodes**.

3. Reponds en commentaire en haut du fichier :
   - Combien d'episodes faut-il (en moyenne sur 3 seeds) pour atteindre un retour moyen > 200 ?
   - Que se passe-t-il si tu utilises `R(tau)` (le retour total identique pour tous les `t` d'une meme trajectoire) au lieu du return-to-go ?

## Criteres de reussite

- Le script tourne en moins de 2 minutes sur CPU.
- Apres ~300-500 episodes, le retour moyen depasse 200 (variance importante attendue : c'est le point).
- Tu as observe et documente la difference `R(tau)` vs return-to-go (la version naive est plus bruitee).

## Indices

- N'utilise **pas** de baseline ni de GAE — c'est le but de cet exo de voir REINFORCE nu.
- Inverse l'ordre du tableau pour calculer les returns-to-go en O(T) :
  ```python
  returns = np.zeros_like(rewards)
  G = 0.0
  for t in reversed(range(len(rewards))):
      G = rewards[t] + gamma * G
      returns[t] = G
  ```
- `Categorical(logits=logits).log_prob(action_tensor)` donne directement le `log pi(a|s)`.
