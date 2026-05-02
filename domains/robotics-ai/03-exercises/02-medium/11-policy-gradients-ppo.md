# J11 — Exercice medium : ajouter une baseline et GAE a REINFORCE

## Objectif

Partir du REINFORCE de l'exo facile et **mesurer la reduction de variance** apportee par (a) une baseline value `V_phi(s)`, puis (b) GAE.

## Consigne

1. Pars de `reinforce_cartpole.py` (exo facile).

2. **Variante 1 : baseline value (A2C basique)**
   - Ajoute un reseau `V_phi` (MLP `[obs_dim -> 64 -> 64 -> 1]`).
   - Loss policy : `loss_pi = - mean( log_prob(a_t | s_t) * (R_t - V_phi(s_t).detach()) )`.
   - Loss value : `loss_v = mean( (V_phi(s_t) - R_t)^2 )`.
   - Tu peux entrainer les deux dans le meme `loss = loss_pi + 0.5 * loss_v` ou separement.

3. **Variante 2 : GAE**
   - Remplace `R_t - V(s_t)` par l'advantage GAE :
     ```
     delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
     A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
     ```
   - Avec `gamma = 0.99`, `lambda = 0.95`.
   - `loss_pi = - mean( log_prob(a_t|s_t) * A_t )`, `returns = A_t + V(s_t)` pour la value loss.

4. Compare les **3 versions** (REINFORCE nu, baseline, GAE) sur 5 seeds chacune :
   - Plot le retour moyen episode-par-episode.
   - Calcule la **variance des gradients** (norme du gradient batch-par-batch). Tu devrais voir : nu > baseline > GAE.

## Criteres de reussite

- 3 courbes plottees (`matplotlib`) sur le meme graphe avec ecart-type sur 5 seeds.
- GAE atteint le seuil de 475 (CartPole "solved") plus vite que la baseline simple, qui elle-meme bat REINFORCE nu.
- En commentaire : explique en 3 lignes pourquoi `lambda < 1` est un compromis biais/variance.

## Indices

- Tu peux faire des "rollouts batch" : collecter K episodes complets avant chaque update — c'est plus stable qu'un seul episode.
- Pour calculer GAE en pratique, fais-le **par episode** (pas a travers les boundaries d'episodes : on respecte la Markov property en mettant `next_value = 0` au terminal).
- Pour mesurer la variance du gradient : `total_norm = torch.nn.utils.clip_grad_norm_(...)` retourne la norme **avant clipping**. Stocke-la pour analyse.
