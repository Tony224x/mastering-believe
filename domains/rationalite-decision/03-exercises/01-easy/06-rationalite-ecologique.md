# Exercices — Module 06 : Rationalité Écologique

> **Difficulté** : progressif (easy → intermédiaire → hard)
> **Prérequis** : avoir lu `01-theory/06-rationalite-ecologique.md`
> **Format** : réflexion écrite + calcul simple. Pas de code.

---

## Exercice 1 — Reconnaissance et classements de villes (easy)

### Objectif

Appliquer l'heuristique de reconnaissance à un problème de classement et comprendre pourquoi la méconnaissance partielle peut être un avantage.

### Consigne

Vous êtes soumis aux paires de villes suivantes. Pour chaque paire, indiquez quelle ville vous reconnaissez, puis appliquez l'heuristique de reconnaissance pour prédire laquelle a la plus grande population. Enfin, répondez aux questions d'analyse.

**Paires :**

| Paire | Ville A | Ville B |
|---|---|---|
| 1 | Lyon | Châteauroux |
| 2 | Metz | Épinal |
| 3 | Toulouse | Albi |
| 4 | Bordeaux | Périgueux |
| 5 | Marseille | Salon-de-Provence |

**Travail à faire :**

1. Pour chaque paire, notez si vous reconnaissez A, B, les deux, ou aucune.
2. Si vous ne reconnaissez qu'une des deux, appliquez la règle de reconnaissance (A reconnue → A plus grande).
3. Si vous reconnaissez les deux, vous devez estimer — notez votre réponse intuitive.
4. Retrouvez ensuite les populations réelles (ordre de grandeur : Lyon ≈ 500 000 hab., Toulouse ≈ 480 000, Bordeaux ≈ 250 000, Marseille ≈ 870 000, Metz ≈ 115 000, Châteauroux ≈ 43 000, Épinal ≈ 30 000, Albi ≈ 50 000, Périgueux ≈ 30 000, Salon-de-Provence ≈ 43 000).

**Questions d'analyse :**

a. Pour les paires où vous n'avez reconnu qu'une ville, l'heuristique a-t-elle donné la bonne réponse ?
b. Pourquoi la notoriété d'une ville tend-elle à être corrélée à sa taille ? Quelles régularités de l'environnement médiatique expliquent cela ?
c. Imaginez un touriste étranger qui ne reconnaît aucune de ces villes. Peut-il utiliser l'heuristique de reconnaissance ? Quelle stratégie lui conseillez-vous ?

### Critères de réussite

- [ ] L'heuristique de reconnaissance est correctement appliquée pour les cas de reconnaissance partielle (ne reconnaître qu'une ville sur deux).
- [ ] L'analyse explique le mécanisme de corrélation entre notoriété et taille (médiatisation, économie, histoire).
- [ ] La limite de l'heuristique (cas où les deux villes sont reconnues, ou aucune) est identifiée.
- [ ] La réponse à (c) propose une stratégie alternative valide (ex. : choisir aléatoirement, chercher un autre indice).

---

## Exercice 2 — Take-The-Best sur un choix de club sportif (intermédiaire)

### Objectif

Appliquer Take-The-Best (TTB) à une décision multi-attributs et comprendre quand TTB peut battre une intégration complète de l'information.

### Consigne

Vous êtes directeur sportif d'une ligue régionale d'athlétisme. Deux clubs (Club A et Club B) candidatent pour accueillir les championnats régionaux. Vous disposez des indices suivants, **classés par validité décroissante** (du plus prédictif au moins prédictif pour la qualité d'organisation) :

| Rang | Indice | Club A | Club B |
|---|---|---|---|
| 1 | A déjà organisé un championnat régional dans les 5 ans | ✓ | ✗ |
| 2 | Dispose d'une piste homologuée IAAF | ✓ | ✓ |
| 3 | Budget infrastructure > 50 000 € | ✗ | ✓ |
| 4 | Nombre de bénévoles formés ≥ 30 | ✓ | ✓ |
| 5 | Présence d'un hôtel partenaire à moins de 5 km | ✗ | ✓ |

**Travail à faire :**

1. Appliquez les 3 étapes de TTB :
   - Étape 1 : Quel est l'indice le plus valide ?
   - Étape 2 : Discrimine-t-il les deux clubs ?
   - Étape 3 (si non) : Passer à l'indice suivant.
2. Quelle décision TTB recommande-t-elle ? À quelle étape s'est-elle arrêtée ?
3. Comparez avec une intégration complète : si chaque indice vaut 1 point pour le club qui l'a, calculez le score total de chaque club. Quelle est la décision de l'intégration complète ?
4. Les deux méthodes convergent-elles ou divergent-elles ici ? Que révèle ce cas sur les forces et limites de TTB ?

**Question bonus** : modifiez le tableau pour que TTB et l'intégration complète donnent des résultats opposés. Décrivez la situation modifiée.

### Critères de réussite

- [ ] Les 3 étapes de TTB sont appliquées correctement et en ordre (ne pas sauter d'étape).
- [ ] La décision TTB est justifiée par l'indice discriminant identifié (indice 1).
- [ ] Le score total de l'intégration complète est calculé correctement (Club A : 3, Club B : 4).
- [ ] L'analyse identifie que TTB s'arrête au premier indice discriminant, même si les autres indices favorisent l'autre option.
- [ ] La question bonus propose un tableau cohérent où les deux méthodes divergent (ex. : reclasser les indices pour que le premier discriminant ne soit plus le plus important globalement).

---

## Exercice 3 — Quand le *less-is-more* s'applique-t-il ? (hard)

### Objectif

Distinguer les conditions où une heuristique simple bat un modèle complexe (effet *less-is-more*) des conditions où c'est l'inverse, en raisonnant à partir de la structure de l'environnement.

### Consigne

Lisez les trois scénarios ci-dessous. Pour chacun, indiquez : (a) si l'effet *less-is-more* est susceptible de s'appliquer, (b) pourquoi, et (c) quelle stratégie décisionnelle recommanderiez-vous (heuristique simple ou modèle intégratif ?).

**Scénario A — Prédire le vainqueur d'un championnat de tennis amateur local**

Une association sportive doit établir un calendrier en plaçant les joueurs par niveau estimé. Elle dispose des données suivantes pour chaque joueur : résultats des 3 derniers tournois (victoires, défaites), classement officiel, âge, club d'origine, nombre d'années de pratique, taille, et poids. Les données sont fiables mais portent sur seulement 2 saisons.

**Scénario B — Prédire le classement de 50 villes dans un palmarès de qualité de vie**

Un magazine publie chaque année un classement de 50 villes sur 20 critères mesurés depuis 15 ans (logement, transports, emploi, culture, santé, environnement, etc.). Les données sont complètes, propres, et stables dans le temps. L'équipe a accès à un modèle de régression entraîné sur 10 ans de données historiques.

**Scénario C — Évaluer en temps réel la dangerosité d'une situation sur une piste cyclable**

Un cycliste doit décider en moins d'une seconde s'il freine ou accélère face à un croisement. Les informations disponibles sont : couleur du feu, présence ou absence d'un piéton visible, vitesse ressentie du véhicule venant de la droite.

**Pour chaque scénario :**
1. Identifiez si les données sont abondantes ou rares.
2. Identifiez si l'environnement est stable ou bruité.
3. Concluez sur l'applicabilité de *less-is-more* et la stratégie recommandée.

**Question de synthèse** : rédigez en 3-4 phrases un critère général permettant de savoir *a priori* si une heuristique simple ou un modèle intégratif sera plus performant.

### Critères de réussite

- [ ] Scénario A : *less-is-more* identifié comme applicable (données rares sur 2 saisons, bruit probable). Stratégie : heuristique simple (ex. : TTB sur classement officiel uniquement).
- [ ] Scénario B : *less-is-more* non applicable (données abondantes, propres, stables). Stratégie : modèle intégratif (régression sur 20 critères).
- [ ] Scénario C : heuristique rapide imposée par la contrainte de temps (décision en < 1 sec). La question de *less-is-more* ne se pose pas — l'heuristique est la seule option praticable.
- [ ] La question de synthèse formule correctement les conditions : données rares / bruit élevé → heuristique ; données abondantes / environnement stable → modèle intégratif ; contrainte de temps extrême → heuristique indépendamment des autres facteurs.
- [ ] La réponse distingue bien les trois dimensions : quantité de données, stabilité de l'environnement, contrainte de temps.
