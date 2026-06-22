# J14 - Exercice HARD - System Identification + Adaptive Domain Randomization

## Objectif

Combiner SysID, domain randomization, et un debut d'**adaptive** policy : la policy recoit en entree un encodage de l'historique recent pour inferer implicitement les parametres physiques courants. C'est la version simplifiee de l'idee de [RMA - Rapid Motor Adaptation, Kumar 2021] et de la policy adaptative d'OpenAI sur Rubik's Cube.

## Consigne

### Partie 1 - System Identification (SysID)

1. Implemente `system_identify(env, n_traj=5, traj_steps=200)` qui :
   - Genere `n_traj` trajectoires sur l'env "reel" avec une policy aleatoire (action uniforme dans `[-2, 2]`).
   - Estime `(mass, length, friction)` de l'env reel par minimisation de l'erreur quadratique entre observations reelles et observations simulees.
   - Utilise une **grid search** sur les 3 parametres (resolution ~10 valeurs par dim) ou `scipy.optimize.minimize`.

2. Verifie que les valeurs estimees sont proches des vraies valeurs cachees (mass=1.25, length=0.92, friction=0.05).

### Partie 2 - Domain randomization centree sur l'estimation SysID

3. Au lieu de randomiser sur des plages "aveugles" (ex: mass ∈ [0.5, 1.5]), randomise autour de l'estimation SysID : `mass ∈ [m_hat * 0.9, m_hat * 1.1]` etc.

4. Compare la performance de cette policy "SysID + narrow DR" vs la policy "wide DR sans SysID" du fichier `02-code/14-sim-to-real.py`. Quelle marche mieux et pourquoi ?

### Partie 3 - Adaptive policy (proxy)

5. Etends ta policy pour qu'elle prenne en entree, en plus de `obs`, un buffer des **N=20 dernieres** observations et actions. La nouvelle "policy" est une fonction parametree par 6 gains : 3 pour le comportement nominal, 3 pour des "ajustements" en fonction de la variance recente de `theta_dot` (proxy d'une physique avec haute friction si la variance est faible, basse friction sinon).

   Forme suggeree :
   ```
   var_dotmean = compute variance de theta_dot sur le buffer
   adjustment_factor = 1.0 + alpha * (var_dotmean - var_baseline)
   k_p_eff = k_p * adjustment_factor
   ...
   ```

6. Re-entraine cette policy adaptative avec random search sur les 6 gains, evaluee sur la distribution randomisee. Compare les 3 strategies sur le reel :
   - non-adaptive + sans DR
   - non-adaptive + DR
   - adaptive + DR

## Criteres de reussite

- SysID converge vers des valeurs ±10% des vraies valeurs cachees.
- Le narrow DR centre sur SysID est strictement meilleur que le wide DR aveugle (ou strictement pas pire, et alors il faut justifier pourquoi).
- La policy adaptative beat les deux baselines non-adaptives sur le reel.
- L'analyse explique l'idee centrale de RMA : le buffer permet a la policy de **detecter implicitement** la physique courante et d'adapter son comportement, sans avoir besoin de connaitre xi explicitement.
- Bonus : implementer l'**Automatic Domain Randomization** (ADR) : commencer avec une plage etroite, l'elargir si la policy reussit > seuil, la retrecir sinon. Justifier que c'est un curriculum sur l'env.
