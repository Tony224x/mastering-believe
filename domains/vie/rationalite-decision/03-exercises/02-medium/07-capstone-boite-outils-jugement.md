# Exercices Medium — Module 07 : Capstone — La Boîte à Outils du Jugement

> **Niveau** : Medium | **Temps estimé** : ~40 min

---

## Exercice 1 — Journal de décisions structuré (3-5 décisions réelles)

### Objectif
Étendre le journal de prévisions du niveau easy vers un **journal de décisions** : non plus juste « va-t-il pleuvoir ? », mais « la décision que je prends maintenant va-t-elle se révéler bonne ? ». L'enjeu est de rendre vos décisions *traçables* et *évaluables a posteriori*.

### Consigne
**Phase 1 — Enregistrer.** Choisissez **3 à 5 décisions réelles et neutres** que vous prenez en ce moment (exemples : acheter telle formation plutôt qu'une autre, réparer ou remplacer un appareil, choisir un itinéraire domicile-travail, planifier une tâche avant telle date, acheter tel équipement). Pour chacune, remplissez une ligne :

```
| # | Décision | Option retenue | p(succès) % | 2 raisons clés | Indicateur de succès | Date de revue |
```

Règles :
- `p(succès)` = probabilité chiffrée que, à la date de revue, vous jugiez la décision « bonne » selon **votre indicateur défini à l'avance** (pas reconstruit après coup).
- L'indicateur de succès doit être **observable** (« l'appareil fonctionne encore » / « la formation est finie » / « le trajet prend < 35 min »), pas vague (« je suis content »).
- Chaque décision a une **date de revue fixe**.

**Phase 2 — Revoir (template rétrospectif).** À la date de revue, remplissez pour chaque décision :

```
| # | Résultat (0/1) | Surprise ? | Cause du résultat (décision OU hasard ?) | Leçon |
```

Le point clé de la colonne « décision OU hasard ? » : une **bonne décision peut donner un mauvais résultat** (et inversement). On évalue d'abord le *processus*, pas seulement l'*issue* (anti-*resulting*).

### Critères de réussite
- [ ] 3 à 5 décisions réelles, neutres et concrètes.
- [ ] Chaque décision a une `p(succès)` chiffrée fixée *avant* l'issue.
- [ ] Chaque décision a un indicateur de succès observable défini à l'avance.
- [ ] Chaque décision a une date de revue fixe.
- [ ] Le template rétrospectif distingue explicitement « bonne décision » et « bon résultat ».

---

## Exercice 2 — Calculer son score de Brier sur ses propres prédictions

### Objectif
Passer du concept au calcul : prendre **un lot de vos prédictions déjà résolues**, calculer le score de Brier, et l'interpréter par rapport à la baseline 0,25.

### Consigne
Rassemblez **8 à 10 prédictions binaires neutres déjà résolues** (issues de votre journal easy, ou reconstituées honnêtement : météo, livraisons, délais, résultats sportifs, est-ce que telle tâche a été finie à temps…).

Pour chacune, notez :
- `p` = la probabilité que vous aviez attribuée (en %, convertie en 0-1).
- `o` = l'issue réelle (1 si l'événement s'est produit, 0 sinon).
- `(p − o)²`.

Puis :
1. Calculez le **score de Brier** = moyenne des `(p − o)²`.
2. Comparez à la **baseline 0,25** (= prédire systématiquement 50 %). En dessous = mieux qu'une pièce de monnaie ; au-dessus = à recalibrer.
3. Repérez la (ou les) prédiction(s) qui contribue(nt) le plus à la somme : que vous apprennent-elles (sur-confiance ? mauvaise classe de référence ?) ?

> Rappel formule : le Brier va de 0 (parfait) à 1 (pire). Une prédiction à 90 % qui se réalise coûte `(0,9 − 1)² = 0,01` ; la même qui rate coûte `(0,9 − 0)² = 0,81`. La confiance se paie cher quand on se trompe.

### Critères de réussite
- [ ] 8 à 10 prédictions binaires neutres, déjà résolues.
- [ ] Chaque `(p − o)²` est calculé correctement.
- [ ] Le score de Brier (moyenne) est calculé et l'arithmétique est juste.
- [ ] Comparaison explicite à la baseline 0,25.
- [ ] Au moins une prédiction « coûteuse » est analysée (cause identifiée).

---

## Exercice 3 — Pré-décision + pre-mortem

### Objectif
Combiner la checklist de pré-décision avec un **pre-mortem** : se projeter dans un futur où la décision a échoué, et en déduire des mesures d'atténuation *avant* d'agir. C'est une difficulté désirable contre l'excès d'optimisme.

### Consigne
Prenez **une décision neutre** à venir (achat, réparation vs remplacement, planning, choix de prestataire neutre…). Déroulez :

**Étape 1 — Checklist courte (6 questions)** :
1. Décision exacte et options réelles ?
2. Un biais examiné (pas juste coché) + résultat de l'examen ?
3. Probabilité de succès + classe de référence ?
4. Conséquence pessimiste / scénario ruineux ?
5. Information clé à vérifier ?
6. Décision provisoire + date de revue ?

**Étape 2 — Pre-mortem** : « Nous sommes 6 mois plus tard et cette décision a clairement échoué. » Écrivez **3 causes plausibles** de cet échec imaginaire.

**Étape 3 — Atténuations** : pour **chaque** cause, une mesure concrète prise *dès maintenant* (ou un signal d'alerte à surveiller).

**Étape 4 — Décision finale** : ajustée (ou maintenue) à la lumière du pre-mortem, avec la `p(succès)` éventuellement révisée.

### Critères de réussite
- [ ] Décision neutre, réelle, avec options.
- [ ] Checklist : au moins 1 biais réellement examiné + probabilité chiffrée avec classe de référence.
- [ ] Pre-mortem : 3 causes d'échec plausibles et distinctes.
- [ ] Chaque cause a une mesure d'atténuation ou un signal d'alerte concret.
- [ ] La décision finale indique si elle a été ajustée et avec quelle `p(succès)`.
