# Solutions — Module 02 : Arguments & sophismes

> Corrigé modèle. Les formulations alternatives restent valides si elles respectent les critères de réussite.

---

## Exercice 1 — Repérer un sophisme

### Argument A
> « Les données montrent que les équipes utilisant des tableaux Kanban ont des délais plus courts. Donc, utiliser un tableau Kanban réduit les délais. »

**Sophisme : Corrélation → causalité (fausse cause / *post hoc*)**

La donnée montre une association statistique, pas un lien de cause à effet. Les équipes qui adoptent Kanban sont peut-être aussi celles qui ont déjà de bonnes pratiques de gestion du flux — c'est un confondant probable. Pour conclure à la causalité, il faudrait une expérience contrôlée (ou au minimum, contrôler les facteurs alternatifs : taille d'équipe, type de projet, ancienneté des membres...).

---

### Argument B
> « Notre responsable logistique a dit d'augmenter le stock tampon. Mais il ne connaît pas notre nouveau logiciel — on ne peut pas l'écouter. »

**Sophisme : *Ad hominem***

L'argument attaque la personne (ignorance supposée d'un outil) plutôt que la recommandation elle-même. La méconnaissance du logiciel est peut-être un élément à prendre en compte pour évaluer *la qualité de l'information* du responsable, mais elle ne réfute pas la proposition d'augmenter le stock tampon. Il faut évaluer la recommandation sur ses mérites : est-elle justifiée par les données de rupture de stock ? Est-elle cohérente avec les contraintes de coût ?

---

### Argument C
> « Si on autorise les livreurs à choisir leur itinéraire, bientôt ils choisiront aussi leurs horaires, puis leur charge de travail, et finalement l'entreprise perdra tout contrôle. »

**Sophisme : Pente glissante**

La chaîne de conséquences est présentée comme inévitable, sans justification des liens causaux entre chaque étape. Pourquoi l'autonomie sur l'itinéraire entraînerait-elle nécessairement l'autonomie sur les horaires ? Rien ne le prouve. Le sophisme est identifié quand les transitions entre étapes ne sont pas étayées par des données ou mécanismes explicites.

*Note : si des données empiriques montraient que dans des contextes similaires, chaque étape a effectivement suivi, l'argument cesserait d'être fallacieux — ce serait alors une prédiction empiriquement fondée.*

---

### Argument D
> « Ce projet, soit on le lance avec un budget de 500 000 €, soit on l'abandonne. »

**Sophisme : Faux dilemme**

L'espace des options est arbitrairement réduit à deux. Des alternatives intermédiaires sont plausibles : lancement en version réduite (budget partiel), report à un trimestre plus favorable, phase pilote, co-financement, etc. Le faux dilemme est souvent utilisé pour forcer une décision binaire sous pression.

---

### Argument E
> « Un champion du monde d'échecs recommande cette application de mémorisation — c'est la meilleure méthode pour apprendre le vocabulaire d'une langue. »

**Sophisme : Appel à l'autorité hors domaine**

Le champion est expert en échecs (mémorisation de positions tactiques), pas en acquisition de vocabulaire en langue étrangère. Son autorité est réelle mais hors du domaine concerné. Pour que l'appel à l'autorité soit légitime, l'expert devrait être compétent précisément sur la question en jeu (ex. : un chercheur en psychologie de l'apprentissage des langues, ou des études contrôlées sur cette application).

---

## Exercice 2 — Disséquer un argument

**Extrait analysé :**
> « Les entrepôts qui ont mis en place un tri automatique ont réduit leurs erreurs de picking de 30 % en moyenne. Notre entrepôt a récemment connu une hausse des erreurs. Par conséquent, nous devrions mettre en place un tri automatique. »

---

### Étape 1 — Décomposition

**Prémisse 1 :** Les entrepôts ayant adopté un système de tri automatique ont réduit leurs erreurs de picking de 30 % en moyenne.

**Prémisse 2 :** Notre entrepôt a récemment connu une hausse des erreurs de picking.

**Conclusion :** Nous devrions mettre en place un système de tri automatique.

---

### Étape 2 — Validité

L'argument est **invalide** tel qu'il est formulé. Même si P1 et P2 sont vraies, la conclusion ne suit pas nécessairement. P1 dit que le tri automatique *peut* réduire les erreurs dans d'autres entrepôts ; P2 dit que notre entrepôt *a un problème*. Mais rien dans ces deux prémisses n'établit que :
- Le tri automatique est la meilleure solution pour *notre* problème spécifique.
- Notre hausse d'erreurs a la même cause que celles des entrepôts de la statistique.
- Il n'existe pas d'alternative moins coûteuse ou plus adaptée.

Il manque une prémisse reliant les deux (voir Étape 4).

---

### Étape 3 — Questions critiques sur les prémisses

**Sur P1 :**
- Ces entrepôts sont-ils comparables au nôtre en taille, volume et type de marchandises ?
- La réduction de 30 % est-elle une moyenne robuste, ou tirée vers le haut par quelques cas atypiques ? Sur quelle période ?
- Les erreurs de picking de ces entrepôts avaient-elles la même origine que les nôtres ?

**Sur P2 :**
- Quelle est la cause de la hausse de nos erreurs (formation insuffisante, nouveaux produits, surcharge, erreur de système informatique) ? Si la cause est connue, le tri automatique est peut-être inutile.
- La hausse est-elle statistiquement significative ou dans la variabilité normale ?

---

### Étape 4 — Prémisse implicite

> **Prémisse implicite :** Si un système réduit les erreurs de picking dans d'autres entrepôts similaires, alors c'est la solution appropriée (ou au moins préférable aux alternatives) pour résoudre notre hausse d'erreurs.

Cette prémisse est le chaînon manquant. Elle est contestable : il faudrait comparer le tri automatique aux alternatives (formation, réorganisation physique, meilleur système de scan) avant de conclure.

---

## Exercice 3 — Reconstruction charitable

**Argument original :**
> « La réunion hebdomadaire du lundi, ça ne sert à rien. Ça fait deux fois qu'on la fait et on n'a toujours pas résolu le problème des retards en zone B. »

---

### Étape 1 — Faiblesses de l'argument original

1. **Prémisse mal quantifiée :** « deux fois » est un échantillon très restreint pour évaluer l'efficacité d'un dispositif. Il faudrait définir un horizon temporel raisonnable.
2. **Conclusion excessive :** « ça ne sert à rien » est une généralisation totale à partir d'un seul critère (résolution du problème en zone B). La réunion pourrait avoir d'autres utilités (coordination, information, suivi d'autres sujets).
3. **Attribution causale implicite non justifiée :** l'argument suppose que si la réunion *avait* servi à quelque chose, le problème serait résolu — mais la résolution de ce problème dépend peut-être de facteurs extérieurs à la réunion (ressources, temps, compétences).

---

### Étape 2 — Reconstruction charitable

```
Prémisse 1 : Un dispositif de coordination est utile s'il contribue à résoudre
             les problèmes opérationnels qu'il est censé traiter, dans un délai raisonnable.

Prémisse 2 : La réunion du lundi a pour objectif déclaré de traiter le problème
             des retards en zone B (entre autres sujets).

Prémisse 3 : Après deux itérations de ce dispositif, le problème des retards
             en zone B n'a pas progressé de façon mesurable.

Conclusion : Il y a des raisons de douter que la réunion du lundi, dans son
             format actuel, soit efficace pour résoudre ce problème spécifique —
             et son format ou son ordre du jour devrait être réévalué.
```

---

### Étape 3 — Évaluation sous forme reconstruite

Sous cette forme, l'argument est **plus valide** (la conclusion suit des prémisses) et **partiellement défendable** selon les faits. La conclusion reconstruite est aussi plus utile : elle ne dit pas « supprimer la réunion », mais « réévaluer son format ».

**Ce qu'il faudrait pour trancher :**
- Quel était l'objectif précis des deux premières réunions ? A-t-on défini des actions concrètes avec responsable et délai ?
- Le problème des retards en zone B a-t-il une cause identifiée qui dépasse les attributions de l'équipe présente en réunion ?
- Y a-t-il eu des progrès sur d'autres sujets traités en réunion — auquel cas la réunion n'est pas inutile, mais inadaptée à *ce* problème spécifique ?

**Morale pédagogique :** la reconstruction charitable transforme une critique émotionnelle vague (« ça ne sert à rien ») en diagnostic actionnable — c'est précisément là où l'analyse d'argument devient utile en pratique.
