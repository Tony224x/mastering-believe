# Solutions — Module 05 : Heuristiques & Biais

> Corrigé détaillé. À consulter *après* avoir tenté les exercices.

---

## Exercice 1 — Identifier et nommer les biais

### Situation A — Ancrage numérique

**Biais** : Ancrage numérique.

**Pourquoi** : L'écran à 1 800 € a servi d'ancre. L'écran à 1 100 € est perçu comme bon marché *par rapport à l'ancre*, non par rapport à sa valeur réelle ou à des comparatifs indépendants. Si la première vitrine avait affiché un écran à 600 €, la perception de l'achat à 1 100 € aurait été radicalement différente.

---

### Situation B — Disponibilité

**Biais** : Heuristique de disponibilité.

**Pourquoi** : Les pannes récentes, couvertes intensément par les médias, sont saillantes et viennent facilement à l'esprit. La saillance est confondue avec la fréquence réelle du risque. Les statistiques de sécurité ferroviaire (accidents par km parcouru) ne sont pas spontanément accessibles et sont donc ignorées.

---

### Situation C — Biais de confirmation

**Biais** : Biais de confirmation.

**Pourquoi** : Le chef de projet cherche à valider une décision déjà prise. Il filtre les informations de manière asymétrique : les témoignages favorables sont lus, les défavorables survolés. Il ne se demande pas « qu'est-ce qui devrait être vrai si ce logiciel était mauvais ? »

---

### Situation D — Cadrage (*Framing*)

**Biais** : Effet de cadrage.

**Pourquoi** : « 80 % à temps » et « 20 % en retard » sont mathématiquement identiques. La formulation A encode un résultat positif (cadrage gain), la formulation B encode un résultat négatif (cadrage perte). L'aversion à la perte rend B subjectivement moins attractif malgré une réalité identique.

---

## Exercice 2 — Tâche de Wason

### Partie A — Solution

**Cartes à retourner : △ et 3.**

| Carte | Retourner ? | Raison |
|-------|------------|--------|
| **△** | Oui | La règle concerne les triangles : il faut vérifier que l'autre côté est pair. |
| **□** | Non | La règle ne dit rien sur ce qui se trouve derrière un carré. Aucune violation possible. |
| **6** | Non | La règle dit qu'un triangle *implique* un pair, pas l'inverse. Un pair peut avoir n'importe quel symbole. |
| **3** | **Oui** | 3 est impair. Si l'autre côté est un triangle, la règle est violée. C'est la carte la plus souvent oubliée — pourtant la seule capable de *réfuter* la règle. |

**Lien avec le biais de confirmation** : la tentation naturelle est de retourner △ et 6 — pour *confirmer* (trouver un triangle avec un pair). Retourner 3 est l'acte de *réfutation* que le biais de confirmation rend contre-intuitif.

---

### Partie B — Contrôle qualité

**Fiches à examiner : Fiche 1 et Fiche 4.**

| Fiche | Examiner ? | Raison |
|-------|-----------|--------|
| **Fiche 1** (prioritaire) | Oui | Règle : prioritaire → ≥ 3 contrôles. Il faut vérifier si les contrôles sont au nombre requis. |
| **Fiche 2** (non marquée) | Non | La règle ne contraint pas les pièces non prioritaires. |
| **Fiche 3** (5 contrôles) | Non | 5 ≥ 3. Même si la pièce était prioritaire, la règle est satisfaite. |
| **Fiche 4** (1 contrôle) | **Oui** | 1 < 3. Si cette pièce est marquée prioritaire, la règle est violée. C'est l'équivalent de la carte 3 : seule cette fiche peut révéler une violation. |

---

## Exercice 3 — Analyse et débiaisage d'une décision

### Étape 1 — Ancrage numérique

**Biais** : Ancrage numérique. Le chiffre « 3 semaines » a été introduit avant toute estimation indépendante de l'équipe.

**Action corrective** : Demander à chaque membre de l'équipe de noter son estimation sur papier *avant* que le chef de projet communique quelque information de référence. Agréger ces estimations, puis seulement les confronter aux données externes.

---

### Étape 2 — Disponibilité

**Biais** : Heuristique de disponibilité. Les deux migrations récentes et bien mémorisées dominent le raisonnement ; les trois migrations difficiles, moins fraîches en mémoire, ne sont pas citées.

**Action corrective** : Constituer un registre écrit des projets similaires passés (qu'ils aient bien ou mal tourné) et le consulter *systématiquement* avant chaque estimation. Demander explicitement : « Quelles migrations *difficiles* ou *en retard* connaissons-nous ? »

---

### Étape 3 — Cadrage

**Biais** : Effet de cadrage. L'option « gagner 2 semaines » semble plus attractive que « risquer d'en perdre 2 », alors que les deux décrivent la même réalité (±2 semaines par rapport au planning).

**Action corrective** : Reformuler systématiquement toute option dans le cadrage opposé avant de voter. Ici : « Cette option peut signifier 2 semaines de retard supplémentaires — est-ce acceptable ? » Si le choix change selon la formulation, c'est un signal d'alerte.

---

### Étape 4 — Négligence du taux de base

**Biais** : Négligence du taux de base (*base rate neglect*). L'équipe n'a pas cherché à savoir quelle est la durée médiane de ce type de migration dans le secteur, ni quel pourcentage de projets similaires dérapent.

**Action corrective** : Avant de valider une estimation, demander : « Quelle est la durée médiane observée pour des migrations de ce type ? Quel pourcentage dépassent le délai initial ? » Ancrer l'estimation sur ces données sectorielles, puis l'ajuster selon les spécificités du projet (méthode de référence de classe, *reference class forecasting* — Kahneman & Lovallo, 2003).

---

### Protocole de débiaisage pour les réunions d'estimation

1. **Estimation à l'aveugle d'abord** : avant toute information externe (benchmarks, durée d'un projet similaire), chaque participant note son estimation de façon indépendante. Le facilitateur collecte les chiffres sans les révéler immédiatement.

2. **Consultation des données de base** : le responsable du projet apporte les statistiques sectorielles pertinentes (durée médiane, taux de dépassement) et les partage *après* la collecte des estimations individuelles.

3. **Inventaire des cas défavorables** : le facilitateur pose systématiquement la question « Quels projets similaires se sont mal passés ? Pourquoi ? » avant de consolider l'estimation.

4. **Double cadrage obligatoire** : toute option décisionnelle est présentée dans les deux formulations — gain *et* perte. Si les préférences divergent selon le cadrage, la décision est suspendue jusqu'à clarification.

5. **Hypothèse réfutante** : en fin de réunion, un participant désigné défend la position contraire à la conclusion émergente (« Qu'est-ce qui pourrait tout faire échouer ? ») — pré-mortem allégé.
