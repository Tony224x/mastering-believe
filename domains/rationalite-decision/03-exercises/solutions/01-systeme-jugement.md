# Solutions — Module 1 : Le système du jugement

> Ces corrigés sont des réponses modèles. D'autres formulations sont acceptables si les concepts clés sont présents.

---

## Exercice 1 — Identifier Système 1 vs Système 2

| # | Situation | Système | Justification |
|---|-----------|---------|---------------|
| A | Joueur d'échecs expert repère un cavalier en prise | **S1** | Expertise consolidée : après des milliers de parties, la reconnaissance de patterns est automatisée. Un débutant ferait la même tâche en S2. |
| B | Comptable vérifie 47 lignes de chiffres d'un autre | **S2** | Tâche nouvelle, vérification minutieuse ligne par ligne : traitement délibératif et fatigant. |
| C | Freinage brusque sur balle surgissante | **S1** | Réflexe moteur automatisé : le stimulus visuel déclenche la réponse sans délibération consciente. |
| D | Ingénieur résout pour la 1ère fois un problème réseau | **S2** | Problème nouveau et complexe : recherche consciente, pas de routine disponible. |
| E | Reconnaître la voix d'un ami au téléphone | **S1** | Reconnaissance perceptuelle immédiate et inconsciente. |
| F | Remplir sa déclaration fiscale pour la 1ère fois | **S2** | Règles nombreuses et inconnues, lecture attentive, vérifications = effort délibératif continu. |
| G | Médecin urgentiste reconnaît une fracture connue | **S1 acquis** | Même logique que A : l'expertise a transféré la tâche de S2 vers S1. Un étudiant en médecine ferait la même tâche en S2. |
| H | Étudiant calcule 37 × 48 de tête | **S2** | Calcul non mémorisé, effort conscient pas à pas. |

**Point clé sur A et G** : la même tâche objective peut relever de S1 ou S2 *selon le niveau d'expertise*. L'entraînement répété convertit des tâches S2 en tâches S1 — c'est le mécanisme de l'automatisation. Un expert fait en S1 ce qu'un novice fait en S2.

---

## Exercice 2 — Déconstruire un biais à partir d'un problème chiffré

### 1. Réponse correcte

Les 10 billes sont remises dans l'urne après chaque tirage. Les tirages sont **indépendants** : la composition de l'urne n'a pas changé. La probabilité d'obtenir une bille rouge au 6ᵉ tirage est identique à celle de tout autre tirage, soit le **rapport entre le nombre de billes rouges et le total** (information non donnée dans l'énoncé — ce qui est le piège : la réponse ne peut pas être "plus élevée qu'avant" quel que soit ce rapport).

Si l'urne contient, par exemple, 5 billes rouges sur 10 : P(rouge) = 50 % à chaque tirage, quel que soit l'historique. L'historique ne modifie pas l'urne — donc ne modifie pas la probabilité.

### 2. Nom du biais

**Biais du joueur** (*gambler's fallacy*) : croyance erronée que des événements aléatoires indépendants passés influencent les événements futurs — par exemple, attendre un "rééquilibrage" qui n'a aucune base probabiliste.

### 3. Application de la définition formelle

| Critère | Application à cet exemple |
|---------|--------------------------|
| **Systématique** | L'erreur va toujours dans le même sens : les gens surestiment la probabilité après une "série" (attente de rééquilibrage). |
| **Prévisible** | On peut prédire que la majorité des sujets donnera une réponse > probabilité réelle après une longue série d'un même résultat. |
| **Écart normatif** | La norme est calculable : indépendance des événements → probabilité inchangée. L'écart par rapport à cette norme est mesurable. |

### 4. Rôle de S1

S1 est un détecteur de patterns : il cherche des régularités pour prédire l'avenir. Dans des séquences causales réelles (météo, comportement humain), cette tendance est utile. Appliquée à des événements aléatoires *sans mémoire*, elle produit une fausse attente de rééquilibrage. S2, s'il est activé, rappelle la règle d'indépendance et corrige l'intuition — mais sous pression de temps ou de fatigue, S1 l'emporte.

---

## Exercice 3 — Mindware gap : diagnostiquer et combler

### 1. Biais identifié

**Heuristique de disponibilité** (ou représentativité sur petit échantillon — les deux sont défendables) :

- *Disponibilité* : les 3 livraisons récentes réussies sont saillantes et facilement accessibles en mémoire. Lucas juge la fiabilité du fournisseur par la facilité avec laquelle des exemples positifs lui viennent à l'esprit, pas par les données historiques complètes.
- *Représentativité* : 3 observations semblent "représentatives" d'une tendance fiable alors que l'échantillon est trop petit pour estimer un taux à 40 % de retard.

### 2. Mindware manquant

Lucas manque de deux outils conceptuels :
1. **La notion de taux de base** : avant de juger sur l'expérience récente, consulter le taux de performance historique sur une période suffisante (ici, 2 ans de données existantes).
2. **La loi des grands nombres (version intuitive)** : 3 observations ne suffisent pas à estimer un taux de fiabilité — la variance est trop grande sur un petit échantillon.

### 3. Procédure en 3 étapes

**Étape 1 — Collecter les données historiques.**
Récupérer les données de livraison du fournisseur sur *au moins 12 mois* (idéalement 24 mois) : nombre de livraisons, taux de retard, ampleur des retards.

**Étape 2 — Comparer à un taux de référence.**
Définir un seuil acceptable (ex. : taux de retard ≤ 10 %) et comparer le taux historique du fournisseur à ce seuil, *indépendamment des 3 dernières livraisons*.

**Étape 3 — Décision sur règle, pas sur intuition.**
La règle est posée à l'avance : si le taux historique dépasse le seuil, le contrat n'est pas renouvelé sauf justification documentée d'une amélioration structurelle (changement de processus, audit tiers). L'expérience récente peut être un *signal faible* mais ne remplace pas la règle.

### 4. Distinction rationalité / intelligence (Stanovich)

La procédure ne demande pas à Lucas d'être *plus intelligent* (puissance algorithmique). Elle lui demande d'activer son **esprit réflexif** : s'arrêter, interroger son intuition, appliquer une règle formelle plutôt que se fier à S1.

Lucas était probablement parfaitement capable de calculer un taux de retard s'il l'avait cherché — son algorithme n'était pas en défaut. Ce qui lui manquait : la *disposition* à interroger son intuition et le *mindware* (concept de taux de base, règle de décision explicite). C'est précisément ce que Stanovich appelle la dysrationalia : raisonner mal non par manque d'intelligence, mais par manque d'esprit réflexif et d'outils conceptuels.
