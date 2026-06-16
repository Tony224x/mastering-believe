# Exercices — Module 06 : Calibration & Vérification de l'Information

---

## Exercice 1 — Calculer et interpréter un score de Brier

### Objectif
Calculer le score de Brier sur un ensemble de prédictions et identifier les zones de sur- ou sous-confiance.

### Consigne

Un apprenant a enregistré les 8 prédictions suivantes dans son journal :

| # | Question | p prédit | Outcome (1=oui, 0=non) |
|---|----------|----------|------------------------|
| 1 | Il pleuvra demain | 0,80 | 1 |
| 2 | Mon colis arrivera à temps | 0,70 | 0 |
| 3 | L'équipe locale gagnera | 0,55 | 1 |
| 4 | Je terminerai ce rapport avant vendredi | 0,90 | 1 |
| 5 | Le film sera dans le top 10 du box-office | 0,40 | 0 |
| 6 | La réunion sera annulée | 0,20 | 0 |
| 7 | Je serai en retard ce matin | 0,30 | 1 |
| 8 | Le match se terminera aux prolongations | 0,15 | 0 |

**Questions** :
1. Calculez (p − o)² pour chacune des 8 prédictions.
2. Calculez le score de Brier global (moyenne des 8 valeurs).
3. Comparez ce score à la baseline (0,25). L'apprenant est-il meilleur que le hasard ?
4. Identifiez la prédiction la plus coûteuse (plus grand (p−o)²) et expliquez ce que cela révèle.

### Critères de réussite
- [ ] Les 8 valeurs (p−o)² sont calculées correctement.
- [ ] Le score de Brier global est calculé à ± 0,005.
- [ ] La comparaison à la baseline (0,25) est faite et interprétée.
- [ ] La prédiction la plus coûteuse est identifiée avec une interprétation (sur-confiance ou sous-confiance).

---

## Exercice 2 — Vérification d'une citation douteuse

### Objectif
Appliquer le protocole SIFT pour vérifier une citation présentée comme scientifique.

### Consigne

Un article de blog affirme : "Selon une étude de Harvard de 2022 publiée dans *Nature Medicine*, les personnes qui méditent 10 minutes par jour voient leur QI augmenter de 15 points en 3 mois."

Vous souhaitez vérifier cette affirmation avant de la partager.

**Questions** :
1. Appliquez SIFT : décrivez les 4 étapes concrètes que vous effectuez (avec les requêtes de recherche exactes que vous utiliseriez).
2. Quels sont les 3 signaux d'alerte spécifiques dans cette affirmation qui justifient une vérification approfondie ?
3. Si cette citation provenait d'un LLM, quel serait votre protocole de vérification spécifique (étapes exactes) ?

### Critères de réussite
- [ ] Les 4 étapes SIFT sont décrites avec des actions concrètes (pas des généralités).
- [ ] Au moins 3 signaux d'alerte sont identifiés (ex. : "Harvard" sans auteur nommé, effet de 15 points de QI est très grand, 3 mois est très court, "augmenter le QI" est contesté dans la littérature).
- [ ] Le protocole LLM inclut : recherche Google Scholar titre exact + vérification doi.org.

---

## Exercice 3 — Démarrer son journal de prévisions

### Objectif
Formuler 5 prédictions proprement et préparer le cadre de suivi.

### Consigne

Formulez 5 prédictions sur des événements neutres et observables dans les 7 à 30 prochains jours (météo, sport, transports, délais de tâches personnelles, etc.). Pour chaque prédiction :

**Format requis** :
```
Question : [formulée de façon binaire, avec date de résolution]
Probabilité : _____ %
Classe de référence utilisée : ___________
Date de résolution : ___________
```

**Règles** :
- La question doit être binaire (oui/non) et avoir une réponse non ambiguë.
- La date de résolution doit être dans les 7-30 jours.
- La probabilité doit être un chiffre (pas "probable" ou "très probable").
- Indiquer la classe de référence (ex. : "Il a plu 12 des 20 derniers lundis à cet endroit → taux de base 60 %").

**Après résolution** (à faire une fois les événements passés) :
- Calculez le score de Brier pour chaque prédiction résolue.
- Calculez votre score moyen.

### Critères de réussite
- [ ] 5 prédictions formulées avec la structure complète (question binaire + probabilité + classe de référence + date).
- [ ] Toutes les questions sont binaires et observables.
- [ ] La classe de référence est justifiée pour au moins 3 prédictions.
- [ ] (Si résolutions disponibles) : score de Brier calculé pour chaque prédiction résolue.
