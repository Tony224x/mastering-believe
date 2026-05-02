# J13 - Exercice HARD : GAIL minimal sur CartPole

## Objectif

Implementer une version minimaliste de **GAIL** (`[Ho & Ermon, 2016]`) : un discriminateur adversarial qui distingue (s, a) expert vs student, et une policy entrainee par policy gradient avec **reward = -log(1 - D(s, a))**.

C'est l'exercice le plus dur du jour : tu touches au RL on-policy + adversarial training. Lecture obligatoire avant de commencer : `[CS224R L2 - Finn]` section GAIL et la section 6 du cours theorique.

## Consigne

Sur `CartPole-v1` :

1. **Expert** : reutiliser l'expert heuristique du cours, generer 20 demos -> dataset `D_E = {(s, a)}_expert`.
2. **Discriminateur** `D_ψ(s, a) -> [0, 1]` :
   - MLP simple `(obs_dim + n_actions_onehot) -> 64 -> 64 -> 1` avec sigmoid.
   - Loss BCE : maximiser `log D(s_E, a_E) + log(1 - D(s_S, a_S))`.
3. **Policy** `π_θ(a | s)` : MLP categorical comme dans le cours.
4. **Boucle GAIL** (10-30 outer iterations) :
   - **a)** Roll-out de `π_θ` -> trajectoires `{(s_t, a_t)}_S`.
   - **b)** Update `D_ψ` sur un batch mixant demos expert + rollouts student (1-3 epochs).
   - **c)** Update `π_θ` par policy gradient simple (REINFORCE ou PPO simplifie) avec **reward** `r_t = -log(1 - D(s_t, a_t)) + ε`.
5. **Evaluer** la policy GAIL sur 20 episodes. Comparer a un BC entraine sur les memes 20 demos.

## Criteres de reussite

- GAIL converge a return ≥ 400 sur CartPole (idealement 500). Patience : la convergence adversariale est instable, prevoir 2-3 essais avec seeds differents.
- BC sur les 20 demos atteint deja un score eleve sur CartPole — tu **ne battras pas** forcement BC en valeur finale ; le but pedagogique est de **voir GAIL fonctionner**, pas de battre BC sur cet env trop facile.
- Le code separe clairement les 3 modules : `Discriminator`, `Policy`, `gail_step`.
- Tu logges la **loss du discriminateur** : si elle tend vers `-log(0.5) ≈ 0.69`, le discriminateur ne sait plus distinguer = victoire du generateur.

## Pieges courants

- **Reward shaping** : `-log(1 - D)` peut diverger quand `D -> 1`. Clipper a `[-10, 10]` ou utiliser `log D - log(1 - D)`.
- **Trop entrainer le discriminateur** : il devient parfait, gradient policy nul. Limiter a 1-3 epochs par iter.
- **Normaliser les rewards** sur l'episode (centrer/reduire) pour stabiliser le PG.
- **One-hot encoding** des actions discretes pour l'input du discriminateur.

## Pour aller plus loin (optionnel)

- Implementer **AIRL** (`[Fu et al., 2018]`) : variante GAIL ou le discriminateur a une structure speciale qui permet de **recuperer** une fonction de reward `R_φ(s, a)` reutilisable. Tester si elle generalise a une variante `CartPole` avec une masse modifiee.
- Comparer la performance GAIL vs BC quand on **reduit drastiquement** le nombre de demos (n=2). C'est la ou GAIL est cense briller — verifie sur ton environnement.
