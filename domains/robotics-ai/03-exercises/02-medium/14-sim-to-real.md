# J14 - Exercice MEDIUM - Implementer un wrapper de domain randomization

## Objectif

Coder un wrapper Gymnasium reutilisable qui randomise des parametres physiques d'un environnement Pendulum-v1, et mesurer empiriquement l'impact d'une plage trop large vs trop etroite sur la convergence d'une policy.

## Consigne

1. Ecris une classe `DomainRandomizationWrapper(gym.Wrapper)` qui :
   - Prend en parametre un dict `param_ranges = {'mass': (lo, hi), 'length': (lo, hi), 'friction': (lo, hi), 'obs_noise_std': (lo, hi)}`.
   - A chaque `reset()`, tire un nouveau xi uniformement dans chaque plage et l'applique au pendule (via `env.unwrapped.m`, `env.unwrapped.l`, etc.).
   - Stocke le xi courant dans `info['phys_params']` au reset (utile pour debugger).
   - Ajoute un argument `seed` qui rend les tirages reproductibles.

2. Reutilise la policy heuristique `policy(obs, params)` du fichier `02-code/14-sim-to-real.py` et le random search.

3. Compare 3 strategies d'entrainement :
   - **A** (no randomization) : plages = nominal exact (singleton).
   - **B** (narrow randomization) : plages = ±10% autour du nominal.
   - **C** (wide randomization) : plages = ±50% autour du nominal.
   - **D** (too wide) : plages = ±200% (mass ∈ [0.05, 3.0], length ∈ [0.05, 3.0], etc.)

4. Pour chaque strategie, lance random search avec 80 trials et evalue les params trouves sur **le meme** env "reel" (mass=1.25, length=0.92, friction=0.05, bruit=0.02, latence=2 steps).

5. Trace ou tabule les 4 scores. Reponds :
   - Quelle strategie marche le mieux sur le reel ?
   - Pourquoi D peut etre **pire** que C, et pas seulement plus lent ?

## Criteres de reussite

- `DomainRandomizationWrapper` est generique : prend n'importe quel dict de plages, ne modifie pas l'env passe en parametre destructivement.
- `info['phys_params']` est bien rempli a chaque reset.
- Les 4 scores sont reportes avec ecart-type sur 20 episodes d'eval.
- L'analyse explique le **trade-off** : trop etroit → overfit a sim, trop large → sous-optimisation (la policy ne peut pas etre bonne partout, elle moyenne sur trop d'environnements differents).
- Bonus : trace l'evolution du `phys_params['mass']` au cours d'un rollout pour verifier que la randomization tire bien des valeurs differentes.
