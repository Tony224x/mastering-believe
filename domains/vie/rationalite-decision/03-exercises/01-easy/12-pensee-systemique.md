# Exercices — Module 12 : Pensée systémique

> **Niveau** : easy → medium → hard | **Prérequis** : théorie module 12
>
> Les trois exercices sont **100 % neutres** : thermostat, baignoire, stock logistique.

---

## Exercice 1 — Identifier stocks, flux et boucles (easy)

### Objectif

Reconnaître les éléments de base d'un système simple : stock, flux entrant, flux sortant, et type de boucle de rétroaction.

### Consigne

Pour chacun des trois systèmes décrits ci-dessous, complétez le tableau :

| Système | Stock | Flux entrant | Flux sortant | Type de boucle |
|---------|-------|-------------|-------------|----------------|
| **Baignoire** : vous ouvrez le robinet, l'eau monte ; quand l'eau atteint le niveau souhaité vous fermez le robinet. | ? | ? | ? | ? |
| **Entrepôt** : des palettes arrivent chaque jour, des palettes partent vers les clients ; quand le stock est bas, on commande davantage. | ? | ? | ? | ? |
| **Rumeur** : plus une rumeur circule, plus de gens la partagent, ce qui la fait encore davantage circuler. | ? | ? | ? | ? |

**Questions écrites (répondez en 2-3 phrases chacune) :**

1. Dans la baignoire, si vous fermez le robinet *trop lentement*, que se passe-t-il ? Quel phénomène du module 12 explique cela ?
2. Pour la rumeur : cette boucle est-elle renforçante ou équilibrante ? En quoi est-ce différent de la boucle thermostat ?
3. Dans l'entrepôt, si la commande prend 7 jours à arriver et que la demande est de 100 unités/jour, quel stock de sécurité minimum faut-il pour ne pas tomber en rupture pendant le réapprovisionnement ?

### Critères de réussite

- [ ] Les trois lignes du tableau sont correctement remplies (stock = quantité, flux = taux, type = R ou E).
- [ ] La réponse à Q1 cite explicitement la notion de **délai**.
- [ ] La réponse à Q2 distingue clairement boucle R (amplification) vs boucle E (régulation vers objectif).
- [ ] La réponse à Q3 aboutit à un chiffre justifié (700 unités ou raisonnement équivalent).

---

## Exercice 2 — Analyser un système avec délai (medium)

### Objectif

Modéliser mentalement (ou à la main) l'évolution d'un stock face à un délai, et identifier l'erreur d'ajustement classique.

### Consigne

Un gestionnaire de stock logistique suit ce tableau de bord à 09h00 chaque matin :

- **Stock actuel** : 80 unités  
- **Seuil d'alerte** : 100 unités  
- **Consommation** : 20 unités/jour  
- **Délai de livraison fournisseur** : 4 jours  
- **Quantité commandée à chaque réapprovisionnement** : 120 unités  

**Scénario A — Comportement normal**

Complétez le tableau sur 10 jours (en supposant qu'une commande est passée dès que le stock passe sous le seuil) :

| Jour | Stock début de journée | Commande passée ? | Livraison reçue ? | Stock fin de journée |
|------|------------------------|-------------------|-------------------|----------------------|
| J0 | 80 | ? | Non | ? |
| J1 | ? | ? | ? | ? |
| … | … | … | … | … |
| J9 | ? | ? | ? | ? |

**Scénario B — Panique du gestionnaire**

Le gestionnaire panique : voyant le stock baisser, il passe *deux* commandes à J0 (au lieu d'une seule). Décrivez en 3 phrases ce qui arrive au niveau du stock entre J4 et J7.

**Question synthèse** : quelle règle simple (une phrase) le gestionnaire devrait-il s'imposer pour éviter la sur-commande induite par les délais ?

### Critères de réussite

- [ ] Le tableau Scénario A est complété avec des valeurs cohérentes (rupture ou non selon les calculs).
- [ ] Le Scénario B décrit clairement un sur-stock à J4–J7 (les deux livraisons arrivent alors que le pic de demande est passé).
- [ ] La règle synthèse cite explicitement soit la notion de **délai**, soit le principe d'**action douce/progressive face aux délais**.

---

## Exercice 3 — Identifier les points de levier (hard)

### Objectif

Appliquer la hiérarchie des points de levier de Meadows à un système logistique fictif, et justifier quel levier est le plus puissant.

### Consigne

Voici la description d'un système de traitement de commandes :

> *Un entrepôt reçoit en moyenne 200 commandes/jour. Le temps de traitement moyen est de 8 min/commande. Le délai entre la détection d'un pic de commandes et l'ajout d'un agent supplémentaire est de 48h (processus RH). Résultat : lors des pics hebdomadaires (×1,8 la demande normale le vendredi), la file d'attente explose et les délais de livraison passent de 1 jour à 3–4 jours.*

**Travail demandé :**

Pour chacune des 5 interventions proposées ci-dessous, (a) classez-la selon la hiérarchie Meadows (paramètre / taille du stock-tampon / structure des flux / gain de boucle / structure de boucle) et (b) estimez qualitativement son impact (faible / moyen / fort) et sa facilité de mise en œuvre (facile / difficile).

| Intervention | Niveau Meadows | Impact | Facilité |
|---|---|---|---|
| 1. Réduire le temps de traitement de 8 min à 7 min (optimisation de process). | ? | ? | ? |
| 2. Constituer un stock-tampon de 2 agents en réserve, disponibles sous 2h. | ? | ? | ? |
| 3. Raccourcir le délai de déclenchement RH de 48h à 4h (nouveau protocole). | ? | ? | ? |
| 4. Ajouter un signal d'alerte automatique quand la file dépasse 50 commandes (nouvelle boucle de feedback). | ? | ? | ? |
| 5. Introduire une tarification incitative pour décaler les commandes du vendredi vers le mercredi (redistribution du flux entrant). | ? | ? | ? |

**Questions écrites :**

1. Quelle intervention est selon vous la plus puissante au sens de Meadows ? Justifiez en une phrase.
2. Quelle intervention est la plus facile à mettre en œuvre mais la moins impactante ? Pourquoi ?
3. Décrivez un **effet de second ordre négatif** possible de l'intervention 5 (tarification incitative).

### Critères de réussite

- [ ] Les 5 interventions sont classées dans le bon niveau Meadows (±1 niveau acceptable avec justification).
- [ ] Les réponses écrites montrent que le candidat distingue impact et facilité d'implémentation.
- [ ] L'effet de second ordre de Q3 est concret et plausible (ex. : résistance clients, report sur un autre pic, etc.).
- [ ] La réponse à Q1 cible une intervention qui touche les boucles (niveaux 4 ou 5), pas les paramètres.
