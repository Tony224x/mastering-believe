# Exercices Hard — Jour 15 : Test-time compute & reasoning models

---

## Exercice 7 : Best-of-N avec reward model bruite vs self-consistency

### Objectif

Comparer rigoureusement deux strategies de test-time compute (best-of-N via reward model, self-consistency via majority vote) et caracteriser quand chacune gagne, en fonction de la qualite du reward model.

### Consigne

1. Construire un **generateur de solutions simule** (numpy) : pour un probleme, on echantillonne N solutions candidates, chacune ayant une qualite latente `q ~ Beta` (certaines bonnes, certaines mauvaises) et un flag `correct` derive de q (`correct = q > 0.5`).

2. Implementer **best-of-N avec reward model** : un RM attribue a chaque solution un score = `q + gauss(0, rm_noise)`. On retient la solution de score RM maximal. Le `rm_noise` modelise la qualite du RM (0 = parfait, grand = aleatoire).

3. Implementer **self-consistency** : chaque solution produit une "reponse finale" discrete ; les solutions correctes convergent vers la meme reponse, les fausses se dispersent (modeliser : reponse = vraie_reponse si correct, sinon une valeur aleatoire parmi M distracteurs). Majority vote.

4. Pour `N in [1, 4, 8, 16, 32]` et `rm_noise in [0.0, 0.2, 0.5, 1.0]`, mesurer sur 1000 problemes :
   - accuracy best-of-N ;
   - accuracy self-consistency ;
   - accuracy d'un oracle (qui choisit toujours une solution correcte si elle existe).

5. **Analyse** :
   - A `rm_noise = 0` (RM parfait), best-of-N doit approcher l'oracle. Verifier.
   - A `rm_noise` grand, best-of-N degenere vers du random pick. Self-consistency le bat-il ? A partir de quel `rm_noise` ?
   - Self-consistency ne marche que si les bonnes reponses **concordent**. Construire un cas (M distracteurs petit, beaucoup de reponses fausses qui concordent par hasard) ou self-consistency echoue alors que best-of-N reussit.

6. **Synthese decisionnelle** : produire un mini-tableau "quelle strategie selon (qualite du RM disponible, verifiabilite de la reponse, budget N)". Relier au cours (best-of-N exige un RM de qualite ; self-consistency est gratuit mais exige une reponse finale extractible et concordante).

### Criteres de reussite

- [ ] Les 3 strategies (best-of-N, self-consistency, oracle) sont implementees correctement
- [ ] A rm_noise=0, best-of-N ~ oracle ; degrade quand rm_noise augmente
- [ ] Le seuil de rm_noise ou self-consistency depasse best-of-N est identifie empiriquement
- [ ] Un contre-exemple ou self-consistency echoue (concordance des erreurs) est construit
- [ ] Le tableau de synthese est coherent avec la theorie
- [ ] Code numpy, seed, commente WHY ; resultats deterministes

---

## Exercice 8 : Pipeline RL reasoning complet (GRPO) sur une tache verifiable + reward hacking

### Objectif

Assembler un pipeline reasoning miniature de bout en bout : une "policy" parametree qui apprend par GRPO a resoudre une tache a reward verifiable, puis demontrer le reward hacking et une mitigation.

### Consigne

On simule un modele qui doit choisir une **strategie de resolution** parmi K (chaque strategie a une proba cachee de produire la bonne reponse) ET produire un format `<think>...</think><answer>...</answer>`.

1. **Environnement** :
   - K strategies, chacune avec `p_correct[k]` cachee (ex : [0.2, 0.4, 0.9, 0.6, 0.1]).
   - La policy = softmax sur K logits (quelle strategie privilegier).
   - reward = `1.0 * is_correct + format_bonus` ou `format_bonus = 0.1` si les balises sont presentes.

2. **Boucle GRPO** : groupes de G, advantage `(r - mean)/std`, update policy. Faire converger la policy vers la strategie 2 (p=0.9).

3. **Tracer** : proba de la meilleure strategie au fil des steps, reward moyen, et un proxy d'"allongement spontane" (ici : fraction de reponses bien formatees qui monte avec le training).

4. **Reward hacking** : introduire une **faille de verification**. Ajouter une strategie K+1 "triche" qui obtient toujours `format_bonus` ET passe une verification naive (`reward = 1` si la sortie *contient* la bonne reponse n'importe ou, y compris dans le `<think>`), sans reellement resoudre. Montrer que la policy converge vers la triche au lieu de la vraie strategie. C'est le reward hacking du cours.

5. **Mitigation** : remplacer la verification naive par une verification robuste (la reponse doit etre extraite **uniquement** du bloc `<answer>` et matcher exactement). Re-entrainer et montrer que la triche disparait.

6. **Analyse** :
   - Pourquoi le `format_bonus` seul peut detourner l'optimisation si la reward de correction est rare/bruitee ?
   - Citer 2 autres formes de reward hacking mentionnees dans le cours et une parade pour chacune.
   - Pourquoi distiller les traces d'un gros modele dans un petit marche-t-il mieux que d'entrainer le petit directement par RL ? (relier a l'exploration et a la base de connaissances).

### Criteres de reussite

- [ ] La boucle GRPO fait converger la policy vers la meilleure strategie verifiable
- [ ] Les courbes (proba meilleure strat, reward, formatage) montent de facon coherente
- [ ] La faille de verification provoque un reward hacking reproductible (la policy choisit la triche)
- [ ] La verification robuste (extraction du seul bloc `<answer>`, match exact) elimine la triche
- [ ] L'analyse cite 2 formes supplementaires de reward hacking + parades, et explique distillation > RL pour les petits modeles
- [ ] Code numpy, seed, modulaire, commente WHY
