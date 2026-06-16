# Solutions — Module 12 : Pensée systémique

> Corrigé modèle. Chaque réponse est chiffrée et justifiée.

---

## Exercice 1 — Identifier stocks, flux et boucles

### Tableau complété

| Système | Stock | Flux entrant | Flux sortant | Type de boucle |
|---------|-------|-------------|-------------|----------------|
| **Baignoire** | Volume d'eau (litres) | Débit du robinet (L/min) | Débit d'évacuation ou fermeture du robinet | **E** (équilibrante) : l'écart entre niveau réel et niveau souhaité pilote la fermeture du robinet. |
| **Entrepôt** | Palettes en stock (unités) | Livraisons reçues (palettes/jour) | Expéditions clients (palettes/jour) | **E** (équilibrante) : le stock bas déclenche une commande qui vise à ramener le stock au-dessus du seuil. |
| **Rumeur** | Nombre de personnes informées | Nouveaux partages (personnes/heure) | (sortie négligeable dans le court terme) | **R** (renforçante) : plus de personnes informées → plus de relais → encore plus de personnes informées. |

### Réponses aux questions

**Q1 — Fermeture trop lente et délai :**
Si vous fermez le robinet trop lentement, l'eau continue d'arriver pendant que vous agissez. Le stock (volume) monte encore après votre intervention, car il y a un **délai entre la décision et son effet**. Résultat : sur-remplissage ou débordement. C'est exactement l'effet de délai décrit dans le module : l'action correctrice prend du temps à se propager au stock.

**Q2 — Boucle R vs boucle E :**
La rumeur utilise une **boucle renforçante** : le stock (personnes informées) *alimente* lui-même le flux entrant (nouveaux partages), sans objectif ni mécanisme correcteur. C'est l'opposé du thermostat (boucle E) qui compare le stock à une *cible* et réduit son action corrective à mesure que l'écart diminue. La rumeur n'a pas de cible : elle croît tant qu'il y a des personnes non atteintes (saturation naturelle).

**Q3 — Stock de sécurité minimum :**
Délai de livraison = 7 jours × 100 unités/jour = **700 unités** de stock de sécurité minimum pour ne subir aucune rupture pendant le réapprovisionnement (en supposant une demande constante). En pratique, on ajoute une marge pour la variabilité (coefficient de sécurité).

---

## Exercice 2 — Analyser un système avec délai

### Tableau Scénario A (commandes normales)

| Jour | Stock début | Commande passée ? | Livraison reçue ? | Stock fin |
|------|-------------|-------------------|-------------------|-----------|
| J0 | 80 | **Oui** (80 < 100) | Non | 60 |
| J1 | 60 | Non | Non | 40 |
| J2 | 40 | Non | Non | 20 |
| J3 | 20 | Non | Non | 0 |
| J4 | 0 | Non | **Oui (+120)** | 100 |
| J5 | 100 | Non | Non | 80 |
| J6 | 80 | **Oui** (80 < 100) | Non | 60 |
| J7 | 60 | Non | Non | 40 |
| J8 | 40 | Non | Non | 20 |
| J9 | 20 | Non | Non | 0 |

*Note : rupture de J3 à J3 soir (stock = 0 mais livraison reçue à J4 matin). Le cycle se répète.*

### Scénario B — Panique : deux commandes à J0

Le gestionnaire passe 2 commandes à J0 (240 unités au total). Les deux livraisons arrivent à J4 (+240 unités). Le stock en J4 = 0 (stock résiduel après consommation) + 240 = **240 unités**. Avec une consommation de 20/jour, ce sur-stock ne sera résorbé qu'en J16. Entre J4 et J7, le stock est très élevé (entre 240 et 160 unités) alors que le seuil de réalerte est 100 — le gestionnaire ne commandera pas mais a immobilisé des ressources et de l'espace inutilement. C'est le **sur-ajustement** classique induit par le délai.

### Règle synthèse

> **Ne passer qu'une seule commande à la fois et attendre la livraison avant d'en relancer une** — ou, formulé en termes de délai : *agir doucement et proportionnellement à l'écart observé, en attendant que chaque action ait produit son effet avant d'en initier une nouvelle*.

---

## Exercice 3 — Points de levier

### Tableau complété

| Intervention | Niveau Meadows | Impact | Facilité |
|---|---|---|---|
| 1. Réduire traitement de 8→7 min | **Paramètre** (chiffre) | Faible (+12 % de débit, insuffisant pour ×1,8 pic) | Facile |
| 2. Stock-tampon de 2 agents réserve | **Taille du stock-tampon** | Moyen (absorbe un pic modéré, insuffisant pour pic fort) | Moyen |
| 3. Délai RH 48h → 4h | **Gain de la boucle de rétroaction** (réponse plus rapide à l'écart) | Fort (permet de réagir avant que la file explose) | Difficile (processus RH) |
| 4. Alerte automatique à 50 commandes | **Structure de boucle** (nouvelle boucle E de surveillance) | Fort (détecte le pic tôt, déclenche l'action avant la saturation) | Moyen |
| 5. Tarification incitative vendredi → mercredi | **Structure des flux** (redistribue le flux entrant) | Fort (supprime la source du problème) | Difficile (résistance clients) |

### Réponses aux questions

**Q1 — Levier le plus puissant :**
L'intervention 4 (alerte automatique = nouvelle boucle de rétroaction) ou l'intervention 5 (redistribution du flux) sont les plus puissantes au sens de Meadows, car elles agissent sur la **structure** du système plutôt que sur un paramètre ; l'une crée une nouvelle boucle E, l'autre élimine le pic à la source — soit les deux niveaux les plus hauts de la hiérarchie.

**Q2 — Plus facile mais moins impactant :**
L'intervention 1 (réduire le temps de traitement de 8 à 7 min) est la plus facile à mesurer et à implémenter (formation, optimisation de poste), mais c'est un simple **changement de paramètre** : elle n'affecte pas la structure des boucles et ne résout pas un pic ×1,8 (gain de 12 % vs besoin de +80 %).

**Q3 — Effet de second ordre de l'intervention 5 (tarification incitative) :**
Un effet de second ordre plausible : les clients qui commandent habituellement le vendredi pour être livrés le lundi résistent à l'incitation et regroupent *deux semaines* de commandes le mercredi pour compenser — créant un nouveau pic le mercredi. Alternativement, certains clients migrent vers un concurrent sans surcharge tarifaire, réduisant le volume total mais aussi le chiffre d'affaires.
