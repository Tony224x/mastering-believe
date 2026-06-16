# Exercices — Module 06 : Calibration & Vérification de l'Information

## Exercice 1 — Calculer et interpréter un score de Brier

### Objectif
Calculer le score de Brier sur un ensemble de prédictions.

### Consigne
8 prédictions :

| # | Question | p prédit | Outcome |
|---|----------|----------|---------|
| 1 | Pluie demain | 0,80 | 1 |
| 2 | Colis à temps | 0,70 | 0 |
| 3 | Équipe gagne | 0,55 | 1 |
| 4 | Rapport fini avant vendredi | 0,90 | 1 |
| 5 | Film top 10 box-office | 0,40 | 0 |
| 6 | Réunion annulée | 0,20 | 0 |
| 7 | En retard ce matin | 0,30 | 1 |
| 8 | Match aux prolongations | 0,15 | 0 |

1. Calculez (p − o)² pour chacune.
2. Calculez le score de Brier global (moyenne des 8).
3. Comparez à la baseline (0,25). Meilleur que le hasard ?
4. Identifiez la prédiction la plus coûteuse et expliquez ce qu'elle révèle.

### Critères de réussite
- [ ] Les 8 valeurs (p−o)² sont correctes.
- [ ] Score de Brier calculé à ± 0,005.
- [ ] Comparaison à 0,25 faite et interprétée.
- [ ] Prédiction la plus coûteuse identifiée avec interprétation.

---

## Exercice 2 — Vérification d'une citation douteuse

### Objectif
Appliquer SIFT pour vérifier une citation présentée comme scientifique.

### Consigne
Un article de blog affirme : "Selon une étude de Harvard de 2022 publiée dans *Nature Medicine*, les personnes qui méditent 10 minutes par jour voient leur QI augmenter de 15 points en 3 mois."

1. Appliquez SIFT : décrivez les 4 étapes avec les requêtes exactes utilisées.
2. Quels sont les 3 signaux d'alerte dans cette affirmation ?
3. Si cette citation venait d'un LLM, quel serait votre protocole de vérification ?

### Critères de réussite
- [ ] Les 4 étapes SIFT décrites avec actions concrètes.
- [ ] Au moins 3 signaux d'alerte identifiés.
- [ ] Protocole LLM inclut : Google Scholar + doi.org.

---

## Exercice 3 — Démarrer son journal de prévisions

### Objectif
Formuler 5 prédictions proprement et préparer le cadre de suivi.

### Consigne
Formulez 5 prédictions sur des événements neutres et observables dans les 7-30 prochains jours.

**Format requis pour chaque prédiction** :

```
Question : [binaire, avec date de résolution]
Probabilité : _____ %
Classe de référence utilisée : ___________
Date de résolution : ___________
```

Après résolution : calculez le score de Brier pour chaque prédiction.

### Critères de réussite
- [ ] 5 prédictions avec structure complète.
- [ ] Toutes les questions sont binaires et observables.
- [ ] Classe de référence justifiée pour au moins 3 prédictions.
- [ ] (Si résolutions disponibles) : score de Brier calculé.
