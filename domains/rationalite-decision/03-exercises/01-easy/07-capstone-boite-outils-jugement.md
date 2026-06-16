# Exercices — Module 07 : Capstone — La Boîte à Outils du Jugement

---

## Exercice 1 — Assembler sa checklist de pré-décision personnalisée

### Objectif
Adapter la checklist générique du module 07 à une décision concrète et personnelle, et documenter le processus complet.

### Consigne

Choisissez une décision réelle et neutre que vous devez prendre dans les 2 prochaines semaines (ex. : choisir une formation, organiser un déplacement, décider d'un achat important, accepter ou refuser une tâche supplémentaire).

Remplissez la checklist complète pour cette décision :

```
DÉCISION : ___________
DATE : ___________

1. CLARIFIER
   - Décision exacte : ___________
   - Options réelles (lister toutes, pas seulement A vs statu quo) : ___________
   - Horizon de temps : ___________

2. BIAIS VÉRIFIÉS
   - Ancrage : y a-t-il un premier chiffre/idée qui m'influence ? ___________
     → Estimation indépendante avant : ___________
   - Disponibilité : quel exemple récent domine ma réflexion ? ___________
     → Taux de base historique : ___________
   - Cadrage : ma préférence changerait-elle en reformulant en gains/pertes ? ___________
   - Confirmation : argument le plus fort *contre* mon option favorite : ___________

3. PROBABILITÉS
   - Scénario principal : probabilité estimée _____ %
   - Classe de référence utilisée : ___________
   - Ajustements depuis le taux de base : ___________

4. CONSÉQUENCES
   - Scénario optimiste (prob : ___%) : ___________
   - Scénario central (prob : ___%) : ___________
   - Scénario pessimiste (prob : ___%) : ___________
   - Scénario ruineux à éviter absolument : ___________

5. VÉRIFICATION
   - Fait clé 1 : vérifié via ___________
   - Fait clé 2 : vérifié via ___________

6. DÉCISION ET SUIVI
   - Décision prise : ___________
   - Probabilité de succès : _____ %
   - Date de revue : ___________
```

### Critères de réussite
- [ ] La décision est réelle, neutre et concrète (pas hypothétique).
- [ ] Toutes les sections de la checklist sont remplies.
- [ ] Au moins 2 biais sont explicitement examinés (pas juste cochés).
- [ ] La probabilité est un chiffre avec une classe de référence justifiée.
- [ ] Une date de revue est fixée pour scorer le résultat.

---

## Exercice 2 — Construire son tableau de journal de prévisions (1 mois)

### Objectif
Mettre en place un journal de prévisions opérationnel sur un mois et calculer le score de Brier initial.

### Consigne

**Phase 1 (maintenant)** : formulez 10 prédictions sur des événements neutres et observables dans les 4 prochaines semaines.

Format de chaque entrée :
```
| Date | Question binaire | p (%) | Date résolution | Outcome | (p−o)² | Note |
```

Règles de formulation :
- Questions binaires avec réponse non ambiguë et date de résolution dans 7-28 jours.
- Couvrir des domaines variés : météo, sport, délais de tâches, décisions de tiers, etc.
- Indiquer la classe de référence utilisée pour chaque probabilité.

**Phase 2 (après résolution)** : une fois les événements passés, remplir les colonnes "Outcome" et "(p−o)²".

**Phase 3 (analyse)** :
1. Calculez votre score de Brier moyen sur les prédictions résolues.
2. Tracez mentalement (ou sur papier) votre courbe de calibration : pour les prédictions à 30-40 %, quelle proportion s'est réalisée ? Pour 60-70 % ? Pour 80-90 % ?
3. Identifiez une zone de sur-confiance (prédit 80 %, réalité 50 %) ou sous-confiance, si elle existe.

### Critères de réussite
- [ ] 10 prédictions formulées avec toutes les colonnes remplies (date, question, p, date résolution).
- [ ] Toutes les questions sont binaires et non ambiguës.
- [ ] Les classes de référence sont mentionnées pour au moins 7 prédictions.
- [ ] (Phase 2 si résolutions disponibles) : score de Brier calculé et comparé à 0,25.
- [ ] (Phase 3) : une zone de biais est identifiée ou le score est commenté.

---

## Exercice 3 — Appliquer le protocole complet sur une décision + vérification

### Objectif
Combiner les 3 outils (checklist, journal, protocole de vérification) sur un seul cas concret.

### Consigne

Choisissez une décision qui implique des informations externes (ex. : choisir un prestataire, évaluer une opportunité, comparer des options en ligne). Appliquez les 3 outils :

**Outil 1 — Checklist de pré-décision** (version courte, 6 questions) :
1. Quelle est la décision exacte ?
2. Quel biais ai-je vérifié en premier ? Résultat ?
3. Quelle est ma probabilité estimée pour l'option favorite ? Classe de référence ?
4. Quel est le scénario ruineux à éviter ?
5. Y a-t-il des informations à vérifier ?
6. Décision prise + date de revue ?

**Outil 2 — Journal** : enregistrer une prédiction liée à cette décision (ex. : "cette option sera satisfaisante dans 3 mois : 70 %").

**Outil 3 — Vérification** : identifier une information clé utilisée dans la décision et appliquer le protocole complet (SIFT ou vérification de citation). Documenter les étapes réalisées et le résultat.

**Livrable** : un document de 1 page (ou équivalent) résumant les 3 outils appliqués.

### Critères de réussite
- [ ] Les 3 outils sont tous utilisés sur le même cas.
- [ ] La checklist identifie au moins un biais examiné.
- [ ] La prédiction du journal est binaire avec une probabilité et une date de résolution.
- [ ] La vérification documente les étapes concrètes réalisées et leur résultat (information confirmée, infirmée ou impossible à vérifier).
- [ ] Le document de synthèse est lisible par quelqu'un d'autre.
