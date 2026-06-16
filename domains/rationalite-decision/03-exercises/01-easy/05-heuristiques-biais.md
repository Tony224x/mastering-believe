# Exercices — Module 05 : Heuristiques & Biais

> **Prérequis** : avoir lu `01-theory/05-heuristiques-biais.md`
> **Format** : 3 exercices gradués (easy → medium → hard). Pas de code.
> **Solutions** : `03-exercises/solutions/05-heuristiques-biais.md`

---

## Exercice 1 — Identifier et nommer les biais *(easy)*

### Objectif
Reconnaître le biais à l'œuvre dans des situations concrètes neutres.

### Consigne

Lisez chacune des quatre situations ci-dessous. Pour chacune :
1. **Nommez le biais principal** en jeu (ancrage, disponibilité, cadrage ou biais de confirmation).
2. **Expliquez en une phrase** pourquoi ce biais s'applique.

**Situation A**
Un acheteur entre dans un magasin de matériel informatique. Le premier écran qu'il voit coûte 1 800 €. Il finit par acheter un écran à 1 100 €, qu'il perçoit comme « une bonne affaire ».

**Situation B**
Après plusieurs semaines de reportages sur des pannes de trains spectaculaires, un voyageur choisit l'avion pour son prochain déplacement, convaincu que le train est « trop risqué » — malgré les statistiques de sécurité.

**Situation C**
Un chef de projet cherche des arguments pour justifier l'adoption d'un nouveau logiciel qu'il a choisi. Il lit en priorité les témoignages positifs et survole les avis négatifs.

**Situation D**
Un directeur logistique doit choisir entre deux prestataires de livraison.
- Prestataire A : 80 % des livraisons arrivent à temps.
- Prestataire B : 20 % des livraisons sont en retard.

Il préfère A, bien que les deux propositions soient identiques.

### Critères de réussite

- [ ] Les quatre biais sont correctement nommés.
- [ ] Chaque explication identifie le mécanisme (chiffre arbitraire / saillance vs fréquence / recherche de confirmation / formulation gain vs perte).
- [ ] Aucune confusion entre les quatre biais.

---

## Exercice 2 — Appliquer la tâche de Wason *(medium)*

### Objectif
Mettre en pratique la détection du biais de confirmation sur une tâche logique abstraite, puis transférer le raisonnement à un contexte réel de contrôle qualité.

### Consigne

**Partie A — Tâche abstraite**

On vous montre 4 cartes. Chaque carte a un symbole d'un côté et un nombre de l'autre.

```
[ △ ]   [ □ ]   [ 6 ]   [ 3 ]
```

**Règle à vérifier** : *« Si une carte a un triangle (△) d'un côté, alors elle a un nombre pair de l'autre. »*

1. Quelles cartes devez-vous retourner pour tester cette règle ?
2. Justifiez le choix de chaque carte (pourquoi retourner / pourquoi ne pas retourner).

**Partie B — Transfert en contexte de contrôle qualité**

Un responsable qualité vérifie une règle de production : *« Si une pièce est marquée "prioritaire", elle doit avoir subi au moins 3 contrôles. »*

Il dispose de 4 fiches de pièces :

| Fiche 1 | Fiche 2 | Fiche 3 | Fiche 4 |
|---------|---------|---------|---------|
| Marquée « prioritaire » | Non marquée | 5 contrôles effectués | 1 contrôle effectué |

Quelles fiches doit-il examiner pour détecter une violation de la règle ? Justifiez.

### Critères de réussite

- [ ] Partie A : les cartes △ et 3 sont identifiées (et seulement celles-là).
- [ ] Justification correcte pour chaque carte, incluant pourquoi la carte 6 est inutile.
- [ ] Partie B : Fiche 1 et Fiche 4 identifiées.
- [ ] Explication du lien entre biais de confirmation et tendance à ne retourner que les cas « validants ».

---

## Exercice 3 — Analyse et débiaisage d'une décision *(hard)*

### Objectif
Appliquer les contre-mesures de plusieurs biais à une situation de décision réelle et neutre.

### Consigne

**Contexte**
Une petite équipe de développement doit estimer le temps nécessaire pour migrer une base de données vers un nouveau système. La réunion d'estimation se déroule ainsi :

1. Le chef de projet ouvre la réunion en disant : « J'ai entendu qu'une migration similaire avait pris **3 semaines** dans une autre entreprise. »
2. L'équipe cite spontanément deux migrations récentes qui s'étaient bien passées en 4 semaines — sans mentionner les trois migrations difficiles dont ils avaient entendu parler l'an dernier.
3. Le chef de projet propose : « On peut cadrer ça comme : soit on gagne 2 semaines sur le planning actuel, soit on risque de perdre 2 semaines. » Tout le monde choisit l'option "gain".
4. Avant de conclure, personne ne demande quelle est la durée *médiane* de ce type de migration d'après les données sectorielles.

**Travail demandé**
1. Identifiez **quel biais** est actif à chacune des étapes 1, 2, 3 et 4.
2. Pour chaque biais identifié, proposez **une action concrète** que l'équipe aurait pu faire pour corriger son jugement.
3. Rédigez en 5-6 phrases un **protocole de débiaisage** que cette équipe pourrait appliquer systématiquement à toute future réunion d'estimation.

### Critères de réussite

- [ ] Les quatre biais sont correctement attribués aux quatre étapes.
- [ ] Les actions correctives sont concrètes et directement applicables (pas de vœux pieux).
- [ ] Le protocole intègre au moins : génération d'estimation avant exposition à un ancre, recherche de cas défavorables, reformulation gain/perte, consultation de données de base (base rates).
- [ ] Le protocole est rédigé en termes opérationnels (qui fait quoi, quand).
