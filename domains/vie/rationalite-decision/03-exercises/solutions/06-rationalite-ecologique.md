# Solutions — Module 06 : Rationalité Écologique

> Corrigé modèle. Vérifiez votre raisonnement, pas uniquement vos réponses finales.

---

## Solution — Exercice 1 : Reconnaissance et classements de villes

### Populations réelles (ordre de grandeur, commune seule)

| Paire | Ville A | Pop. A | Ville B | Pop. B | Grande ville |
|---|---|---|---|---|---|
| 1 | Lyon | ~500 000 | Châteauroux | ~43 000 | Lyon ✓ |
| 2 | Metz | ~115 000 | Épinal | ~30 000 | Metz ✓ |
| 3 | Toulouse | ~480 000 | Albi | ~50 000 | Toulouse ✓ |
| 4 | Bordeaux | ~250 000 | Périgueux | ~30 000 | Bordeaux ✓ |
| 5 | Marseille | ~870 000 | Salon-de-Provence | ~43 000 | Marseille ✓ |

Dans toutes les paires, la ville connue (A) est effectivement la plus peuplée. L'heuristique de reconnaissance donne 5/5 dans ce jeu de données.

### a. Pourquoi la reconnaissance fonctionne ici

Pour un francophone scolarisé, les grandes villes françaises apparaissent fréquemment dans les médias (actualité, sport, économie, culture), dans les cours de géographie et dans les conversations. Cette exposition médiatique est **corrélée positivement** à la taille : plus une ville est grande, plus elle génère d'événements, de commerce, de couverture presse. La reconnaissance n'est donc pas aléatoire — elle encode une régularité de l'environnement informationnel.

### b. Mécanisme de corrélation notoriété-taille

- **Médiatisation** : les événements sportifs, culturels, économiques se concentrent dans les grandes villes.
- **Fonctions administratives** : préfectures et métropoles sont nommées plus souvent dans les textes officiels.
- **Effets de réseau** : une grande ville attire des entreprises, des universités, des flux — qui génèrent à leur tour de la couverture.

Cette corrélation n'est pas universelle (ex. : des villes touristiques ou historiquement célèbres peuvent être reconnues sans être grandes), mais elle est suffisamment robuste pour rendre l'heuristique utile dans ce contexte.

### c. Stratégie pour le touriste étranger (ne reconnaît aucune ville)

L'heuristique de reconnaissance n'est pas applicable. Stratégies alternatives :
- **Chercher un autre indice** : présence d'un aéroport, d'une gare TGV, d'un club de football en Ligue 1 — indices corrélés à la taille.
- **Choisir aléatoirement** (50 % de réussite espérée) si aucun indice n'est disponible.
- **Consulter une source externe** (classement officiel INSEE) — mais cela sort du cadre de la décision rapide et frugale.

---

## Solution — Exercice 2 : Take-The-Best sur un choix de club sportif

### Étape 1 : Quel est l'indice le plus valide ?

L'indice de rang 1 est le plus valide : **« A déjà organisé un championnat régional dans les 5 ans »**.

### Étape 2 : Discrimine-t-il ?

Club A : ✓ | Club B : ✗ → **OUI, il discrimine.**

TTB s'arrête immédiatement. **Décision : Club A.**

### Étape 3 : Non applicable (arrêt à l'étape 2).

### Comparaison avec l'intégration complète

| Indice | Club A | Club B |
|---|---|---|
| A organisé un championnat | ✓ (1) | ✗ (0) |
| Piste homologuée IAAF | ✓ (1) | ✓ (1) |
| Budget infra > 50 000 € | ✗ (0) | ✓ (1) |
| Bénévoles formés ≥ 30 | ✓ (1) | ✓ (1) |
| Hôtel partenaire < 5 km | ✗ (0) | ✓ (1) |
| **Total** | **3** | **4** |

L'intégration complète (somme des indices) choisit **Club B** (score 4 vs 3).

### Les deux méthodes divergent ici

TTB choisit A (expérience organisationnelle déterminante), l'intégration complète choisit B (meilleures ressources globales). Ce cas illustre la différence fondamentale :
- TTB suppose que **l'indice le plus valide domine** et que les autres sont secondaires.
- L'intégration complète suppose que **chaque indice contribue de manière additive et pondérée**.

Lequel a raison ? Cela dépend de la structure réelle de l'environnement : si l'expérience organisationnelle est effectivement le facteur le plus déterminant pour la réussite d'un championnat, TTB est plus fiable. Si les ressources matérielles compensent le manque d'expérience, l'intégration complète est meilleure.

### Question bonus : faire diverger les méthodes dans l'autre sens

Exemple : reclasser les indices en mettant **budget > 50 000 €** en rang 1.

| Rang | Indice | Club A | Club B |
|---|---|---|---|
| 1 | Budget infra > 50 000 € | ✗ | ✓ |
| 2 | A organisé un championnat | ✓ | ✗ |
| ... | ... | ... | ... |

TTB choisit B (rang 1 discrimine). L'intégration complète choisit toujours B (score 4 vs 3). Dans ce cas, les deux convergent. Pour les faire diverger, il faut un cas où le premier indice discriminant favorise un club, mais la somme des autres indices favorise l'autre — par exemple, A gagne sur l'indice 1 mais perd sur 4 des 5 indices restants.

---

## Solution — Exercice 3 : Quand le *less-is-more* s'applique-t-il ?

### Scénario A — Championnat de tennis amateur local

**Analyse :**
- Données : rares (2 saisons seulement), 7 variables disponibles.
- Environnement : bruité (les performances en tournoi amateur varient selon la forme du jour, l'adversaire tiré au sort, la météo, etc.).
- Conclusion : conditions favorables à *less-is-more*.

**Recommandation** : heuristique simple. Utiliser TTB avec comme indice prioritaire le **classement officiel** (seul indice stable et calibré sur plusieurs tournois). Ignorer les variables secondaires (taille, poids) qui ajoutent du bruit sans améliorer la prédiction avec si peu de données.

Un modèle intégratif sur 7 variables et 2 saisons aurait des paramètres trop peu robustes — il apprendrait les particularités de ces 2 saisons plutôt que les régularités profondes du niveau des joueurs.

---

### Scénario B — Palmarès qualité de vie de 50 villes

**Analyse :**
- Données : abondantes (15 ans d'historique, 20 critères propres et stables).
- Environnement : stable (les critères de qualité de vie varient lentement).
- Conclusion : conditions défavorables à *less-is-more*.

**Recommandation** : modèle intégratif (régression entraînée sur 10 ans). Avec des données abondantes et propres, un modèle complexe peut estimer des poids fiables pour chaque critère. Une heuristique simple (ex. : TTB sur un seul critère) laisserait de l'information utile inutilisée et serait moins précise.

---

### Scénario C — Décision en temps réel sur piste cyclable

**Analyse :**
- Contrainte de temps : < 1 seconde.
- Données disponibles : 3 indices visuels immédiats.
- Conclusion : *less-is-more* ne s'applique pas comme concept statistique — la question ne se pose pas, car un modèle complexe est **physiquement impossible** à calculer dans ce délai.

**Recommandation** : heuristique rapide imposée par la contrainte de temps. La règle de priorité (feu rouge = freiner, peu importe le reste) est la seule stratégie praticable. Ce scénario illustre une autre raison d'utiliser des heuristiques simples : **la contrainte computationnelle** (temps, énergie cognitive), indépendamment de la quantité de données disponibles.

---

### Synthèse : critère général

Une heuristique simple sera plus performante qu'un modèle intégratif quand au moins l'une des trois conditions suivantes est remplie :

1. **Données rares ou bruitées** : le modèle complexe surajuste et généralise mal.
2. **Environnement non stationnaire** : les paramètres estimés hier ne tiennent plus demain ; la simplicité offre une robustesse aux changements.
3. **Contrainte de temps ou d'énergie cognitive extrême** : l'intégration complète est physiquement ou cognitivement impossible.

À l'inverse, quand les données sont abondantes, propres et que l'environnement est stable, un modèle intégratif reprend l'avantage — il peut estimer les poids de chaque indice avec précision et exploiter toute l'information disponible.

> **Formulation synthétique** : *less-is-more* n'est pas une philosophie générale mais un phénomène empirique conditionnel. La bonne question n'est pas « quelle méthode est la meilleure ? » mais « quelles sont les propriétés de cet environnement et de ces données ? » — c'est l'essence même de la rationalité écologique.
