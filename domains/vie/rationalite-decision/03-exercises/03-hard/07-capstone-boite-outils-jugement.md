# Exercices Hard — Module 07 : Capstone — La Boîte à Outils du Jugement

> **Niveau** : Hard | **Temps estimé** : ~60 min (projet étendu, suivi sur 30 jours)

---

## Exercice 1 — Protocole personnel de calibration sur 30 jours

### Objectif
Ne pas seulement *lister* des prédictions, mais **concevoir un protocole** de calibration reproductible : classes de référence, cadence de log, revue mensuelle avec score de Brier **et** courbe de calibration par tranches, puis une **règle de recalibration**. C'est le cœur du pilier « Jugement » mis en routine.

### Consigne
Produisez un document de protocole en **4 parties** :

**Partie A — Conception.**
- Choisissez **3 à 4 classes de référence** neutres et récurrentes dans votre quotidien (ex. : « météo locale à 24h », « livraisons annoncées sous 48h », « tâches planifiées finies à la date prévue », « résultats d'une équipe sportive suivie »). Pour chaque classe, notez un **taux de base** estimé (et sa source ou votre raisonnement).
- Définissez la **cadence de log** (combien de prédictions / jour ou / semaine, quand) et une règle anti-triche : probabilités **fixées avant** résolution, jamais éditées après.

**Partie B — Collecte (≥ 20 prédictions sur 30 jours).**
- Tableau : `| Date | Classe | Question binaire | p (%) | Date résolution | o (0/1) | (p−o)² |`.

**Partie C — Revue mensuelle.**
1. **Score de Brier global** = moyenne des `(p−o)²`.
2. **Courbe de calibration par tranches** : regroupez vos prédictions par buckets de confiance (ex. 50-60, 60-70, 70-80, 80-90, 90-100). Pour chaque bucket non vide : *confiance moyenne déclarée* vs *fréquence réelle observée* (= moyenne des `o` du bucket).
3. Identifiez la **zone de sur-confiance** (déclaré > observé) et/ou de **sous-confiance** (déclaré < observé).

**Partie D — Règle de recalibration.**
- Écrivez **une règle explicite et chiffrée** à appliquer le mois suivant (ex. : « dans le bucket 80-90, je décote mes probabilités de 15 points tant que l'écart déclaré−observé dépasse 10 points »). La règle doit être *testable* à la prochaine revue.

### Critères de réussite
- [ ] 3-4 classes de référence neutres, chacune avec un taux de base justifié.
- [ ] Cadence de log définie + règle « probabilité fixée avant résolution ».
- [ ] ≥ 20 prédictions binaires neutres collectées avec toutes les colonnes.
- [ ] Score de Brier global calculé (arithmétique juste).
- [ ] Courbe de calibration par buckets : déclaré vs observé pour chaque bucket non vide.
- [ ] Zone de sur-/sous-confiance identifiée avec chiffres.
- [ ] Règle de recalibration explicite, chiffrée et testable.

---

## Exercice 2 — Cas intégré : checklist + journal + SIFT sur une décision info-dépendante

### Objectif
Démontrer la maîtrise **intégrée** des 6 modules sur **une seule décision neutre** dont l'issue dépend d'une information externe à vérifier. Livrable : une synthèse 1 page mobilisant les trois outils en chaîne.

### Consigne
Choisissez **une décision neutre qui dépend d'une affirmation factuelle externe** (ex. : acheter un appareil parce qu'« il consomme 40 % de moins », choisir un trajet parce qu'« une nouvelle voie réduit le temps de 15 min », souscrire un service parce qu'« 9 utilisateurs sur 10 le recommandent »). L'affirmation choisie doit être **factuelle et apolitique**.

Déroulez les trois outils **en chaîne** :

**Outil 1 — Checklist de pré-décision** (les 6 sections) avec **au moins 2 biais réellement examinés** (ancrage, disponibilité, cadrage ou confirmation — pas seulement cochés), une probabilité de succès chiffrée avec classe de référence, et un scénario pessimiste.

**Outil 2 — Journal de prévisions** : enregistrez **une prédiction binaire** liée à la décision (probabilité + date de résolution + indicateur observable).

**Outil 3 — Protocole de vérification SIFT** sur l'affirmation externe douteuse :
- **S**top (ne pas relayer ; noter la source initiale),
- **I**nvestiguer la source,
- **F**ind better coverage (lecture latérale : que disent des sources indépendantes ?),
- **T**race jusqu'à la source originale.
- Si l'affirmation provient d'un texte/chiffre potentiellement généré par une IA : ajoutez une **vérification anti-hallucination** (la citation existe-t-elle vraiment ? le chiffre est-il sourçable ?).
- Concluez par un **verdict calibré** (« étayé / partiellement étayé / non vérifiable »), pas binaire.

**Étape finale — Synthèse 1 page** : reliez les trois. *Comment le verdict SIFT modifie-t-il la probabilité de succès du journal et la décision finale ?* C'est le test de cohérence : la vérification doit **rétroagir** sur la probabilité.

### Critères de réussite
- [ ] Une seule décision neutre, dépendant d'une affirmation factuelle apolitique.
- [ ] Checklist complète avec ≥ 2 biais réellement examinés + probabilité chiffrée + classe de référence + scénario pessimiste.
- [ ] Journal : prédiction binaire avec probabilité, date de résolution et indicateur observable.
- [ ] SIFT : les 4 étapes documentées + (si pertinent) contrôle anti-hallucination, finissant sur un verdict calibré (non binaire).
- [ ] Synthèse : le verdict SIFT rétroagit explicitement sur la probabilité de succès et la décision finale.
- [ ] Le document mobilise des notions d'au moins 4 des 6 modules précédents (probabilités/Bayes, biais, décision sous incertitude, calibration/Brier, vérification SIFT).
