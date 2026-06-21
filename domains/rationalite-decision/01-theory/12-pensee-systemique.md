# Module 12 — Pensée systémique

> **Temps estimé** : 45 min | **Prérequis** : Modules 01–11
>
> **Objectif** : Comprendre comment les systèmes produisent des comportements surprenants — et identifier où agir pour un effet maximal.

---

## 1. Pourquoi les systèmes nous surprennent

Une baignoire se remplit trop vite. Vous fermez le robinet. L'eau continue de monter quelques secondes. Vous fermez encore plus. La baignoire déborde. Résultat : vous avez **réagi à un retard** que vous n'aviez pas vu.

C'est le cœur de la pensée systémique : **nos intuitions linéaires échouent face aux dynamiques non linéaires, aux délais et aux boucles**.

> **À retenir** : un système est un ensemble d'éléments reliés par des flux et des règles, qui produit un comportement *émergent* — souvent différent de ce qu'on attendrait en regardant les éléments un par un.

---

## 2. Stocks et flux — les briques de base

### 2.1 Le stock

Un **stock** est une *quantité accumulée* à un instant t : eau dans une baignoire, marchandises dans un entrepôt, commandes en attente dans une file.

- Il change **lentement** (inertie).
- Il peut être mesuré *maintenant*, sans regarder le passé.
- Il crée de la **mémoire** dans le système.

### 2.2 Les flux

Un **flux** est un *taux de changement* du stock : litres/minute entrant ou sortant, colis/heure traités, commandes/jour reçues.

```
Stock(t+1) = Stock(t) + Flux_entrant(t) − Flux_sortant(t)
```

**Exemple concret — entrepôt logistique :**

| Élément | Valeur |
|---|---|
| Stock = palettes en attente | 200 palettes |
| Flux entrant = livraisons reçues | 50 palettes/jour |
| Flux sortant = expéditions | 40 palettes/jour |
| Stock demain | 200 + 50 − 40 = **210 palettes** |

> **À retenir** : on ne contrôle jamais directement un stock — on contrôle des *flux*. Et les stocks absorbent les chocs : c'est pourquoi ils existent.

---

## 3. Boucles de rétroaction (feedback loops)

La plupart des systèmes réels ne fonctionnent pas en ligne droite : les stocks *influencent* les flux qui les alimentent. C'est une **boucle de rétroaction**.

### 3.1 Boucle renforçante (R) — amplification

La boucle renforce le changement en cours. Si le stock monte, il monte encore plus vite ; si il descend, il descend encore plus vite.

**Exemple — file d'attente à saturation :**
Plus la file s'allonge → plus les agents sont surchargés → plus le temps de traitement augmente → plus la file s'allonge. Cercle vicieux ↑.

**Exemple symétrique (cercle vertueux) :**
Un opérateur logistique optimise ses routes → délais réduits → clients satisfaits → plus de commandes → ressources pour optimiser davantage.

```
Boucle R :  Stock ↑ → Flux entrant ↑ → Stock ↑↑  (ou Stock ↓ → Flux sortant ↑ → Stock ↓↓)
```

### 3.2 Boucle équilibrante (E) — régulation

La boucle *résiste* au changement : elle pousse le stock vers un objectif.

**Exemple — thermostat :**

1. Température ambiante (stock) < consigne (objectif).
2. Écart détecté → chauffage activé (flux entrant de chaleur).
3. Température remonte → écart diminue → chauffage se réduit.
4. La température *converge* vers la consigne.

```
Boucle E :  Stock < Objectif → Flux correcteur ↑ → Stock ↑ → Écart ↓ → Flux correcteur ↓
```

> **À retenir** : les boucles R amplifient (croissance ou effondrement) ; les boucles E stabilisent (régulation vers un but). Tout système complexe contient les deux, en tension.

---

## 4. Délais — la source des oscillations

Les délais s'intercalent entre cause et effet. Résultat : on agit trop fort, trop tard.

**Exemple — baignoire avec délai de perception :**

- Vous ouvrez le robinet à fond.
- L'eau met 5 secondes à arriver (délai de canalisation).
- Vous voyez le niveau monter seulement après 3 secondes (délai de perception).
- Vous fermez le robinet — mais 8 secondes de flux sont encore en route.

**Exemple — stock logistique avec délai de réapprovisionnement :**

- Stock de pièces tombe sous le seuil d'alerte lundi.
- Commande passée mardi.
- Livraison mercredi en 8 jours → pièces arrivent *le mercredi suivant*.
- Entre-temps, production à l'arrêt ou sur-commande pour anticiper.

> **À retenir** : plus le délai est long, plus les oscillations sont dangereuses. La règle d'or : **agir doucement face aux délais** — une correction forte provoque un sur-ajustement.

---

## 5. Effets de second ordre

Un effet de premier ordre est l'impact *direct* d'une action. Les effets de second (et troisième) ordre sont les conséquences *des conséquences*.

**Exemple — remise commerciale logistique :**

| Ordre | Effet |
|---|---|
| 1er | La remise de 20 % déclenche un pic de commandes (+40 %). |
| 2e | Le stock s'épuise ; délais de livraison s'allongent. |
| 3e | Clients mécontents des retards → certains annulent ou ne renouvellent pas. |
| 4e | La remise a *dégradé* la fidélisation client net. |

**Exemple — file d'attente :**

- Ajouter un agent supplémentaire (1er ordre : file réduite).
- Les clients perçoivent un service plus rapide → demande augmente (2e ordre).
- File revient à son niveau initial — mais avec plus de coûts fixes (3e ordre).

> **À retenir** : avant toute décision, demandez *"et ensuite ?"* au moins deux fois de suite.

---

## 6. Points de levier — où agir efficacement

Donella Meadows a identifié une **hiérarchie des points de levier** : certains endroits dans un système permettent des changements d'impact bien supérieur pour une intervention donnée.

Les 5 niveaux clés (du moins au plus puissant dans son cadre) :

| Niveau (Meadows) | Exemple concret |
|---|---|
| **Chiffres** (paramètres) | Changer le seuil d'alerte de réapprovisionnement de 50 à 80 unités. |
| **Taille des stocks-tampons** | Doubler le stock de sécurité pour absorber les variations. |
| **Structure des flux** | Raccourcir le délai de livraison fournisseur de 8 à 2 jours. |
| **Gains des boucles de rétroaction** | Augmenter la sensibilité du thermostat (réagit plus vite à l'écart). |
| **Structure des boucles** | Ajouter une boucle de contrôle qualité *avant* expédition (nouvelle boucle E). |

> **À retenir** : les paramètres (chiffres) sont les leviers *les plus faciles à changer* mais *les moins puissants*. Les leviers puissants touchent les boucles et leurs gains — mais ils sont politiquement et organisationnellement plus difficiles à modifier.

*Note — le latticework de Munger* : Munger plaidait pour combiner la pensée systémique avec d'autres modèles mentaux (probabilités, économie, biologie…). Cette combinaison — un « treillis de modèles » — est le thème du capstone J14.

---

## 7. Méthode : lire un système en 4 étapes

1. **Identifier les stocks** : qu'est-ce qui s'accumule ? (palettes, clients, énergie thermique, commandes)
2. **Tracer les flux** : qu'est-ce qui fait monter ou descendre ce stock ?
3. **Repérer les boucles** : y a-t-il une rétroaction ? Elle renforce ou équilibre ?
4. **Chercher les délais** : entre quelle cause et quel effet y a-t-il un délai, et de quelle durée ?

---

## Flash-cards

**Q1 : Quelle est la différence entre un stock et un flux ?**
> R : Un stock est une quantité accumulée (mesurable à un instant t) ; un flux est un taux de changement dans le temps (entrée ou sortie). On contrôle les stocks *via* les flux, jamais directement.

**Q2 : Qu'est-ce qu'une boucle renforçante ? Donnez un exemple neutre.**
> R : Une boucle où le changement du stock amplifie lui-même la variation — cercle vertueux ou vicieux. Exemple : file d'attente qui s'allonge → temps de traitement augmente → file s'allonge encore davantage.

**Q3 : Qu'est-ce qu'une boucle équilibrante ? Donnez un exemple neutre.**
> R : Une boucle qui ramène le stock vers un objectif. Exemple : thermostat — l'écart entre température réelle et consigne active le chauffage jusqu'à ce que l'écart soit nul.

**Q4 : Pourquoi les délais provoquent-ils des oscillations ?**
> R : Parce que l'action correctrice est décidée sur une information déjà dépassée ; on sur-corrige, puis on sur-corrige dans l'autre sens. Plus le délai est long, plus les oscillations sont amples.

**Q5 : Parmi les points de levier de Meadows, lesquels sont les plus puissants mais les plus difficiles à activer ?**
> R : Ceux qui modifient la *structure des boucles de rétroaction* (ajouter ou supprimer une boucle, changer son gain). Les simples paramètres numériques sont faciles à toucher mais faiblement impactants.

---

## Points clés à retenir

- **Stock + flux** = la grammaire de tout système dynamique.
- **Boucle R** → amplification ; **Boucle E** → régulation vers objectif.
- **Les délais** transforment une correction en oscillation si on agit trop fort.
- **Les effets de second ordre** annulent souvent les gains apparents de premier ordre.
- **Agir sur les boucles** (structure, gain) est plus puissant qu'agir sur les paramètres.

---

## Pour aller plus loin

- Meadows, D. H. (2008). *Thinking in Systems: A Primer*. Chelsea Green Publishing. ISBN 9781603580557. https://en.wikipedia.org/wiki/Thinking_In_Systems:_A_Primer
- Sterman, J. D. (2000). *Business Dynamics: Systems Thinking and Modeling for a Complex World*. McGraw-Hill. (manuel avancé avec Vensim)
- Simulation interactive : [ncase.me/loopy](https://ncase.me/loopy/) — outil visuel pour dessiner et tester des boucles de rétroaction.
