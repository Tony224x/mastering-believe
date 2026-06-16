# Exercices Medium — Jour 13 : Emergent abilities & reasoning

---

## Exercice 4 : Self-consistency — modele binomial exact + simulation

### Objectif

Modeliser le gain de la self-consistency (majority vote) avec la loi binomiale exacte, le valider par simulation Monte-Carlo, et determiner le nombre de samples necessaire pour atteindre une fiabilite cible.

### Consigne

On suppose un reasoner avec une probabilite `p` de donner la bonne reponse par sample, et des erreurs reparties uniformement sur `m` mauvaises reponses possibles (donc chaque mauvaise reponse a proba `(1-p)/m`).

1. **Probabilite exacte de la majorite** : implementer `p_majority(p, n, m)` qui calcule la probabilite que la bonne reponse soit STRICTEMENT majoritaire parmi `n` samples.
   - Cas simple (`m` grand → les mauvaises reponses ne s'accordent jamais) : la bonne gagne des qu'elle a au moins 1 vote de plus que n'importe quelle autre. Approximation usuelle : `P(>= ceil(n/2) corrects)` via la binomiale.
   - Calculer pour p=0.6, n ∈ {1, 3, 5, 11, 25}

2. **Simulation Monte-Carlo** : implementer la self-consistency reelle (tirer `n` samples avec la distribution complete, faire le vote majoritaire, gerer les egalites). Lancer 50000 essais et comparer avec la formule. L'ecart doit etre faible (< 0.02).

3. **Nombre de samples pour 95%** : trouver le plus petit `n` impair tel que `p_majority(0.6, n) >= 0.95`. Comparer pour p ∈ {0.55, 0.6, 0.7, 0.8}.

4. **Loi des rendements decroissants** : tracer `p_majority` vs `n` (pour p=0.6) et montrer que le gain par sample supplementaire diminue. Calculer le gain marginal de passer de 5→7 vs 21→23 samples.

5. **Cout** : si 1 sample coute `c`, exprimer le cout pour atteindre 95% en fonction de p. Pourquoi self-consistency vaut le coup pour les taches critiques (math, code) mais pas pour tout ?

### Criteres de reussite

- [ ] `p_majority` est correct et la valeur p=0.6, n=5 ≈ 0.68 (coherent avec l'easy)
- [ ] La simulation Monte-Carlo matche la formule (ecart < 0.02)
- [ ] Le nombre de samples pour 95% est trouve et augmente fortement quand p baisse
- [ ] La loi des rendements decroissants est demontree (gain marginal calcule)
- [ ] L'analyse cout/benefice est correcte

---

## Exercice 5 : CoT vs direct — un solveur "verbalise" qui reduit les erreurs

### Objectif

Construire un mini-solveur de problemes arithmetiques multi-etapes qui, en mode "direct", commet des erreurs de propagation, et montrer que le mode "step-by-step" (CoT) les corrige.

### Consigne

1. Reprendre le `MockLLM` du code (`02-code/13-emergent-abilities-reasoning.py`) avec ses modes `direct` et `cot`.

2. Implementer un modele d'erreur plus realiste pour le mode direct : au lieu d'une erreur aleatoire globale, modeliser une **erreur par etape** (proba `e` de se tromper a CHAQUE operation, l'erreur se propageant). En mode CoT, l'erreur par etape est bien plus faible (le modele "voit" le resultat intermediaire).
   - Direct : la probabilite d'etre correct sur un probleme a `k` etapes est `(1 - e_direct)^k` (les erreurs s'accumulent)
   - CoT : `(1 - e_cot)^k` avec `e_cot << e_direct`

3. **Accuracy vs longueur du probleme** : pour des problemes de 1, 3, 5, 8, 12 etapes, mesurer l'accuracy direct vs CoT (sur 2000 problemes par taille). Tracer les deux courbes.

4. **Le decrochage** : montrer que l'accuracy direct s'effondre exponentiellement avec le nombre d'etapes, alors que CoT decroit beaucoup plus lentement. A partir de combien d'etapes l'ecart devient-il enorme ?

5. **Self-consistency sur le CoT** : ajouter un vote majoritaire de 5 CoT par probleme. Montrer le gain supplementaire au-dela du CoT seul.

6. Analyser : pourquoi le CoT aide-t-il surtout sur les problemes LONGS (multi-etapes) et pas sur les questions a 1 etape ?

### Criteres de reussite

- [ ] Le modele d'erreur par etape `(1-e)^k` est implemente correctement
- [ ] L'accuracy direct decroit exponentiellement avec le nombre d'etapes
- [ ] CoT decroit beaucoup plus lentement (verifie empiriquement)
- [ ] Self-consistency sur CoT donne un gain supplementaire chiffre
- [ ] L'analyse "CoT aide sur les problemes longs" est correcte et justifiee par le modele

---

## Exercice 6 : In-context learning — un "modele jouet" qui apprend des exemples

### Objectif

Illustrer concretement l'in-context learning : montrer qu'un mecanisme tres simple (regression sur les exemples du prompt) peut "apprendre une fonction" a partir des demonstrations, sans aucune mise a jour de poids.

### Consigne

L'hypothese du cours : l'attention permet au modele d'implementer une forme de "meta-apprentissage" dans le forward pass. On l'illustre avec un modele jouet.

1. Definir une "tache" comme une fonction lineaire inconnue `y = w·x + b` (w, b tires aleatoirement par tache).

2. Construire un "prompt" = `k` paires (x_i, y_i) de demonstration + un x_query. Le modele doit predire y_query SANS connaitre w, b et SANS entrainement.

3. Implementer un **predicteur in-context** qui resout la regression lineaire SUR LES EXEMPLES DU PROMPT (moindres carres analytiques sur les k demos), puis applique le w_estime au x_query.
   - C'est l'analogue de ce que fait l'attention : "regarder" les exemples et inferer la regle.

4. **Few-shot scaling** : mesurer l'erreur de prediction en fonction de `k` (1, 2, 5, 10, 20 demos). Montrer que l'erreur baisse avec plus d'exemples (= more shots) — exactement le comportement zero/one/few-shot du cours.

5. **Robustesse au bruit** : ajouter du bruit sur les y_i des demos. Montrer comment l'erreur evolue et pourquoi il faut assez d'exemples pour "moyenner" le bruit.

6. Analyser : en quoi ce modele jouet capture l'idee de l'ICL ? Quelle est sa limite par rapport a un vrai LLM (qui peut faire de l'ICL sur des taches non lineaires, du raisonnement, etc.) ?

### Criteres de reussite

- [ ] La tache (fonction lineaire aleatoire) et le prompt few-shot sont construits correctement
- [ ] Le predicteur in-context (moindres carres sur les demos) fonctionne
- [ ] L'erreur baisse quand le nombre de demos augmente (courbe few-shot)
- [ ] L'effet du bruit est demontre et explique
- [ ] L'analyse relie le modele jouet a l'hypothese ICL et en pointe les limites
