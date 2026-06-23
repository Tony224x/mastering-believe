# Solutions Medium — Module 07 : Capstone — La Boîte à Outils du Jugement

> Ces exercices sont ouverts (portfolio). Les corrigés ci-dessous sont des **exemplaires-types** : un remplissage neutre et complet à imiter, pas une réponse unique.

---

## Exercice 1 — Journal de décisions structuré (exemple rempli)

**Phase 1 — Enregistrement (4 décisions neutres)**

| # | Décision | Option retenue | p(succès) % | 2 raisons clés | Indicateur de succès | Date de revue |
|---|----------|----------------|-------------|----------------|----------------------|---------------|
| 1 | Réparer ou remplacer le grille-pain | Réparer (résistance ~20 €) | 60 | Panne simple identifiée ; pièce dispo | Fonctionne encore dans 3 mois | 2026-09-16 |
| 2 | Quelle formation tableur acheter | Cours « niveau intermédiaire » (90 €) | 45 | Niveau adapté ; format court | Formation finie d'ici la revue | 2026-08-16 |
| 3 | Itinéraire domicile-travail | Passer par l'itinéraire B (vélo) | 70 | Moins de feux ; testé 2 fois | Trajet < 30 min en moyenne | 2026-07-16 |
| 4 | Acheter un casque audio maintenant ou attendre soldes | Attendre les soldes | 55 | Pas urgent ; remises probables | Économie ≥ 20 % à l'achat | 2026-07-01 |

**Phase 2 — Template rétrospectif (rempli à la date de revue)**

| # | Résultat (0/1) | Surprise ? | Cause : décision OU hasard ? | Leçon |
|---|----------------|-----------|------------------------------|-------|
| 1 | 1 | Non | Décision (diagnostic correct) | Bon processus, bon résultat. |
| 2 | 0 | Un peu | Décision (sous-estimé le temps dispo) | Mauvais choix de format, pas de malchance → ajuster `p` à la baisse pour les MOOC. |
| 3 | 1 | Non | Décision (test préalable payant) | Tester avant de trancher = bon réflexe. |
| 4 | 0 | Oui | Hasard (rupture de stock pendant les soldes) | **Bonne décision, mauvais résultat** : ne pas se flageller, le raisonnement tenait. |

> Point clé : la décision 4 illustre l'anti-*resulting* — on ne « rétrograde » pas le processus parce que l'issue a déçu pour une raison imprévisible.

---

## Exercice 2 — Score de Brier sur ses propres prédictions (exemple chiffré)

10 prédictions neutres déjà résolues (`o` = 1 si l'événement s'est produit) :

| # | Prédiction | p | o | (p − o)² |
|---|------------|------|---|----------|
| 1 | Colis livré sous 48h | 0,80 | 1 | 0,0400 |
| 2 | Pluie demain matin | 0,30 | 0 | 0,0900 |
| 3 | Tâche finie avant vendredi | 0,70 | 1 | 0,0900 |
| 4 | Réunion maintenue lundi | 0,90 | 1 | 0,0100 |
| 5 | Équipe suivie gagne le match | 0,55 | 0 | 0,3025 |
| 6 | Bus en retard > 5 min | 0,40 | 0 | 0,1600 |
| 7 | Rapport relu à temps | 0,65 | 1 | 0,1225 |
| 8 | Température > 25 °C samedi | 0,20 | 1 | 0,6400 |
| 9 | Finir le livre ce week-end | 0,75 | 1 | 0,0625 |
| 10 | Place de parking au 1er essai | 0,50 | 0 | 0,2500 |

**Calcul.**
Somme des `(p − o)²` = 0,0400 + 0,0900 + 0,0900 + 0,0100 + 0,3025 + 0,1600 + 0,1225 + 0,6400 + 0,0625 + 0,2500 = **1,7675**.

**Score de Brier = 1,7675 / 10 = 0,17675 ≈ 0,177.**

**Interprétation.** 0,177 < baseline 0,25 → on fait mieux qu'un « toujours 50 % ». Marge de progrès vers la cible < 0,20 ? Déjà atteinte, mais c'est fragile.

**Prédiction la plus coûteuse.** La #8 (0,64) domine la somme : prédire 20 % pour un événement qui s'est produit = **sous-confiance / mauvaise classe de référence** (en juin, > 25 °C un samedi local est loin d'être rare). Reclasser sur la fréquence saisonnière réelle (≈ 50-60 %) aurait fait chuter ce coût. La #5 (0,3025) confirme qu'attribuer des probabilités proches de 50 % à des événements qu'on finit par trancher coûte aussi cher.

---

## Exercice 3 — Pré-décision + pre-mortem (exemple rempli)

**Décision** : remplacer un vieil ordinateur portable lent par un modèle reconditionné (~450 €), ou ajouter de la RAM à l'actuel (~60 €).

**Étape 1 — Checklist courte**
1. Décision exacte : reconditionné 450 € **vs** upgrade RAM 60 € **vs** ne rien changer.
2. Biais examiné — *ancrage* : le prix « 450 € » sert d'ancre et fait paraître les 60 € « dérisoires » ; estimation indépendante → la lenteur vient-elle vraiment de la RAM ? (à vérifier avant). *Disponibilité* : un article récent vantait les reconditionnés (1 source ≠ taux de base de fiabilité).
3. Probabilité de succès (= machine fluide 12 mois) : upgrade RAM 55 % ; reconditionné 80 %. Classe de référence : pannes/obsolescence de portables de 5+ ans.
4. Scénario pessimiste : RAM ajoutée mais le disque/CPU reste le goulot → 60 € + temps perdus, toujours lent. Pas de scénario ruineux (montants modestes).
5. Information à vérifier : la lenteur est-elle bien liée à la RAM (moniteur d'activité) ou au disque/CPU ?
6. Décision provisoire : tester d'abord l'hypothèse RAM (diagnostic), revue à +1 semaine.

**Étape 2 — Pre-mortem** (« 6 mois plus tard, c'est un échec »)
1. La lenteur venait du disque dur, pas de la RAM → upgrade inutile.
2. Le reconditionné acheté tombe en panne hors garantie courte.
3. J'ai sur-investi (450 €) pour un usage qui n'en avait pas besoin.

**Étape 3 — Atténuations / signaux**
1. → Diagnostic logiciel **avant** d'acheter quoi que ce soit (moniteur d'usage RAM vs disque).
2. → N'acheter qu'un reconditionné avec **garantie ≥ 12 mois** ; vérifier la politique de retour.
3. → Fixer un budget plafond et un critère d'usage (« si la bureautique suffit, RAM d'abord »).

**Étape 4 — Décision finale (ajustée)**
Décision : **diagnostiquer d'abord** ; upgrade RAM si la RAM est le goulot, sinon viser un reconditionné sous garantie. `p(succès)` révisée à **70 %** (vs 55 % pour l'achat à l'aveugle) grâce au diagnostic préalable. Date de revue : +1 semaine pour le diagnostic, +6 mois pour l'issue.
