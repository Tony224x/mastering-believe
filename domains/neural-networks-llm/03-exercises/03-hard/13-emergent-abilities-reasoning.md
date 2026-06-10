# Exercices Hard — Jour 13 : Emergent abilities & reasoning

---

## Exercice 7 : Tree-of-Thought vs greedy sur le jeu du compte est bon

### Objectif

Implementer une recherche Tree-of-Thought (beam search sur des etats de raisonnement partiels avec heuristique) et mesurer ce que le compute de recherche apporte par rapport a un raisonnement glouton.

### Consigne

**Le probleme** (Countdown simplifie) : etant donnes 4 nombres et une cible, atteindre EXACTEMENT la cible en combinant les nombres avec +, -, * (chaque nombre utilise au plus une fois ; chaque operation remplace deux nombres par leur resultat).

1. Generer 60 instances resolubles (seed fixe) : tirer 4 nombres dans [1, 12], appliquer 3 operations aleatoires pour fabriquer la cible (garantit la resolubilite), rejeter les cibles triviales (deja dans les nombres).

2. **Solveur glouton ("CoT a une seule chaine")** : a chaque etape, appliquer l'operation qui rapproche le plus un des nombres restants de la cible (`min |resultat - cible|`), sans retour arriere. Une seule trajectoire par instance.

3. **Solveur ToT (beam search)** : etat = multiset de nombres restants + trace des operations.
   - expansion : tous les couples x tous les operateurs valides
   - evaluation heuristique de chaque etat (le "value model") : `score = -min_n |n - cible|` avec bonus si la cible est atteignable par une derniere operation simple
   - garder les `w` meilleurs etats par profondeur (beam width w), profondeur max 3
   - succes si un etat contient exactement la cible

4. **Mesures** sur les 60 instances :
   - taux de succes : glouton vs ToT pour w ∈ {1, 3, 10}
   - verifier : ToT(w=1) ≈ glouton (meme logique, a l'heuristique pres) ; ToT(w=10) >= glouton + 20 points
   - compute : compter les etats EVALUES par instance (le proxy des "tokens depenses") et tracer succes vs compute moyen — la courbe de test-time scaling
   - verifier chaque solution trouvee en REJOUANT la trace d'operations (un assert : la trace produit bien la cible) — un solveur qui triche doit etre detecte

5. **Analyse d'echec** : exhiber une instance precise ou le glouton echoue et ToT reussit, afficher les deux traces, et expliquer en commentaire le piege (un coup localement sous-optimal etait necessaire — exactement ce que le beam preserve).

### Criteres de reussite

- [ ] Les 60 instances sont resolubles par construction et la generation est deterministe
- [ ] Toutes les solutions retournees passent la verification par rejeu (assert)
- [ ] ToT(w=10) >= glouton + 20 points de succes
- [ ] La courbe succes/etats-evalues est affichee (tableau) et montre le scaling du compute de recherche
- [ ] Un cas d'echec glouton est analyse avec les deux traces affichees
- [ ] Execution < 30 s

---

## Exercice 8 : Best-of-N avec verifier — la frontiere du test-time compute

### Objectif

Modeliser le scaling du test-time compute : echantillonner N solutions et laisser un verifier choisir — et quantifier comment la QUALITE du verifier borne ce que le compute peut acheter.

### Consigne

Modele : un generateur produit des solutions correctes avec probabilite p=0.3. Un verifier attribue un score `~N(mu_c, 1)` aux solutions correctes et `~N(0, 1)` aux incorrectes ; on choisit la solution au score max. `mu_c` mesure la qualite du verifier (separation d-prime).

1. Implementer `best_of_n(p, mu_c, N, n_trials, rng)` → accuracy Monte Carlo (>= 20 000 essais) : pour chaque essai, tirer la correctitude des N solutions, leurs scores, choisir le max, compter si la choisie est correcte. Gerer le cas "aucune solution correcte parmi les N" (echec force — la borne fondamentale).

2. **Verifications analytiques** :
   - verifier oracle (mu_c → infini, utiliser 50) : accuracy == `1 - (1-p)^N` a ± 1 pt (des qu'une solution correcte existe, l'oracle la trouve)
   - verifier aveugle (mu_c = 0) : accuracy == p a ± 1 pt POUR TOUT N (le compute ne sert a rien sans signal) — verifier pour N ∈ {1, 64}

3. **La surface compute x qualite** : tableau accuracy pour N ∈ {1, 2, 4, 8, 16, 32, 64} x mu_c ∈ {0, 0.5, 1, 2, 50} et verifier :
   - a mu_c fixe > 0, l'accuracy croit avec N mais PLAFONNE sous la borne oracle ; mesurer le plateau (accuracy(64) - accuracy(32) < 2 pts pour mu_c=1) — non : verifier plutot que le gain marginal decroit : gain(1→2) > gain(32→64) pour chaque mu_c > 0
   - le verifier moyen (mu_c=1) avec N=64 reste SOUS le verifier fort (mu_c=2) avec N=8 : la qualite du signal bat la quantite de compute — verifier numeriquement
4. **Comparaison avec majority voting** : implementer le vote majoritaire (les solutions incorrectes se dispersent sur 8 reponses distinctes uniformes, les correctes coincident) et comparer a best-of-N a meme N=16 pour p=0.3 : le vote n'a pas besoin de verifier mais exige que les erreurs ne coincident pas ; best-of-N(mu_c=2) doit le battre, best-of-N(mu_c=0.5) doit etre battu. Afficher les 3 valeurs.

5. Synthese en commentaire : relier aux modeles o1/R1 (le RL apprend a generer de MEILLEURES chaines = monter p ; le test-time compute multiplie les essais = monter N ; les process reward models = monter mu_c) et formuler la regle : le test-time compute n'achete de l'accuracy que proportionnellement au signal du verifier.

### Criteres de reussite

- [ ] Les deux cas limites analytiques (oracle, aveugle) sont verifies a ± 1 pt
- [ ] Le tableau N x mu_c est produit et les gains marginaux decroissants sont demontres
- [ ] Le resultat "qualite > quantite" (mu_c=2, N=8 bat mu_c=1, N=64) est verifie numeriquement
- [ ] La comparaison majority voting vs best-of-N est implementee et les 3 valeurs s'ordonnent comme attendu
- [ ] La synthese o1/R1 est correcte et relie p, N, mu_c aux leviers reels
- [ ] Execution < 60 s
