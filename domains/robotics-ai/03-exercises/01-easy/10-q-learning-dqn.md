# Exercice (easy) - J10 : TD-error et `epsilon`-greedy

## Objectif

Verifier que tu sais derouler **a la main** une mise a jour Q-learning et coder une policy `epsilon`-greedy correcte. Aucun reseau de neurones, aucun `gym`. Juste numpy.

## Consigne

1. **TD-error a la main.**
   On a une Q-table 2x2 (2 etats `s0, s1`, 2 actions `a0, a1`) :

   ```
   Q[s0, a0] = 0.5    Q[s0, a1] = 0.3
   Q[s1, a0] = 0.0    Q[s1, a1] = 1.0
   ```

   On observe la transition `(s=s0, a=a0, r=0.1, s'=s1)`, episode pas termine. Avec `alpha=0.5`, `gamma=0.9` :
   - Calcule la cible TD `r + gamma * max_a' Q(s', a')`.
   - Calcule la TD-error.
   - Donne la nouvelle valeur de `Q[s0, a0]` apres mise a jour.

   Ecris un petit script Python qui calcule et affiche ces 3 valeurs.

2. **`epsilon`-greedy.**
   Code une fonction `epsilon_greedy(q_values: np.ndarray, epsilon: float, rng) -> int` qui :
   - Avec proba `epsilon` retourne une action aleatoire uniforme parmi `len(q_values)`.
   - Sinon retourne `argmax(q_values)`.
   - Doit utiliser `rng` (un `numpy.random.Generator` ou `random.Random`) pour la reproductibilite — pas de `random.random()` global.

   Verifie sur `q_values = np.array([0.1, 0.5, 0.2, 0.4])` :
   - Avec `epsilon=0.0` : doit toujours retourner `1`.
   - Avec `epsilon=1.0` : sur 10000 tirages, chaque action doit etre prise ~25% du temps (tolerance 5%).

## Criteres de reussite

- Les 3 valeurs de la question 1 sont calculees et affichees correctement.
- `epsilon_greedy` passe les deux checks de la question 2.
- Le code utilise un RNG seede (pas de `random` non-seede au global).

## Hint

`np.argmax` retourne le **premier** index en cas d'egalite — c'est ce qu'on veut. Si plusieurs actions ont la meme Q-value et que tu veux casser les egalites au hasard, c'est une optimisation supplementaire mais pas demandee ici.
