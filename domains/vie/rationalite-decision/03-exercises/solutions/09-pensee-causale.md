# Solutions — Module 09 : Pensée causale

> Corrigé modèle. D'autres formulations correctes existent — l'essentiel est d'identifier le mécanisme causal juste.

---

## Solution Exercice 1 — Identifier les confondants

**Corrélation A : parapluies ↔ bottes en caoutchouc**

- **Confondant Z** : les précipitations locales (la pluie / le mauvais temps).
- **Mécanisme** : la pluie pousse les clients à acheter des parapluies *et* des bottes en caoutchouc. Les deux produits répondent au même besoin de protection contre l'eau — ils ne se causent pas mutuellement. Un magasin implanté dans une ville très pluvieuse vendra beaucoup des deux.

**Corrélation B : volume de trafic de véhicules ↔ nombre de pannes**

- **Confondant Z** : l'itinéraire lui-même (distance totale parcourue / ancienneté des véhicules affectés à cet itinéraire).
- **Mécanisme** : les itinéraires les plus chargés cumulent plus de kilomètres, ce qui use davantage les véhicules *et* les fait passer plus souvent sur cet itinéraire. Ce n'est pas le volume qui cause les pannes directement : c'est l'usure liée au kilométrage total qui augmente à la fois l'utilisation et la probabilité de panne.

> **Note** : une autre lecture valide — Z = ancienneté du parc affecté à ces lignes. Les deux confondants sont défendables.

**Corrélation C : arrosage ↔ croissance foliaire**

- **Confondant Z** : la zone de la serre (exposition lumineuse / qualité du sol dans ce secteur).
- **Mécanisme** : l'agriculteur a peut-être configuré l'arrosage automatique plus intensément dans les zones où les plantes poussent déjà mieux (car il veut maximiser le rendement), ou bien ces zones ont un sol plus riche qui favorise à la fois une meilleure rétention d'eau et une meilleure croissance. La corrélation ne prouve pas que plus d'eau cause plus de croissance — il faut un essai contrôlé (même sol, même exposition, dosage aléatoire de l'arrosage).

---

## Solution Exercice 2 — Contrefactuel et groupe contrôle

**Question 1 — Le contrefactuel idéal**

Le contrefactuel idéal est : *les mêmes entrepôts de la région Nord, sur la même période de trois mois, mais sans le nouvel algorithme*. Autrement dit, on voudrait observer simultanément la même réalité avec et sans le traitement — ce qui est physiquement impossible pour un même entrepôt. On cherche donc un groupe contrôle qui *approxime* ce contrefactuel.

**Question 2 — Comparaison Nord/Sud : peut-on conclure à +5 % d'effet ?**

Non, pas sans précautions. Deux confondants plausibles :

1. **Différence de conditions de marché** : la région Nord et la région Sud peuvent avoir des profils de livraison différents (densité urbaine, types de marchandises, longueur des tournées). Si les entrepôts du Nord avaient déjà des tournées plus facilement optimisables, ils auraient pu baisser leurs coûts même sans le nouvel algorithme.

2. **Effet saisonnier différentiel** : si la comparaison couvre un trimestre où le Sud connaît plus d'événements perturbateurs (météo, jours fériés locaux, pics de congés), la baisse moindre du Sud peut refléter un contexte moins favorable, pas l'absence de l'algorithme.

En résumé : les groupes Nord et Sud ne sont pas comparables *a priori* — les différences observées peuvent venir de ces confondants, pas uniquement de l'algorithme.

**Question 3 — Dispositif pour mieux isoler l'effet**

Randomiser l'assignation de l'algorithme **au niveau de l'entrepôt** : tirer au sort, parmi tous les entrepôts du réseau, lesquels reçoivent le nouvel algorithme et lesquels gardent l'ancien. Mesurer ensuite l'évolution du coût par livraison sur la même période dans les deux groupes. Ce dispositif est réalisable sans essai long : quelques semaines de déploiement partiel suffisent pour une première estimation.

---

## Solution Exercice 3 — Concevoir un A/B test

**1. Unité d'assignation**

Les **visiteurs individuels** (identifiés par cookie ou identifiant de session). Chaque visiteur est assigné une seule fois au groupe contrôle ou au groupe traitement dès sa première visite sur la page produit concernée. Ce choix évite qu'un même visiteur voie les deux versions (contamination) et garantit l'indépendance des observations.

**2. Groupes**

- **Groupe contrôle (50 %)** : page produit sans indicateur de stock restant.
- **Groupe traitement (50 %)** : même page avec le bandeau "Plus que 3 en stock !" affiché quand le stock est ≤ 3 unités.
- Une seule différence entre les groupes : la présence de cet indicateur. Tout le reste (mise en page, prix, photos, texte) est identique.

**3. Mesure**

- **Variable Y** : taux de conversion = (nombre d'achats / nombre de visiteurs uniques exposés) × 100.
- **Durée** : au moins 2 à 4 semaines pour couvrir les variations hebdomadaires (les comportements du lundi diffèrent du week-end) et atteindre une puissance statistique suffisante (calculée en fonction du taux de base ≈ 3,5 % et de la différence minimale détectable souhaitée).

**4. Confondants neutralisés par la randomisation**

1. **Profil des acheteurs** : les visiteurs les plus susceptibles d'acheter (anciens clients fidèles, visiteurs ayant déjà une intention d'achat) sont répartis équitablement entre les deux groupes par le tirage au sort.
2. **Moment de la visite** : les pics de trafic (soldes, week-ends) touchent les deux groupes de manière équivalente, car l'assignation est faite aléatoirement au moment de chaque visite, quelle que soit la période.

**Conclusion causale : 3,5 % → 4,1 %**

Oui, on peut conclure à un **effet causal** sous deux conditions :
1. La taille d'échantillon est suffisante pour que la différence soit statistiquement significative (p < 0,05) et pratiquement significative (la différence de +0,6 point justifie le coût de développement).
2. Il n'y a pas eu de **contamination** : un visiteur assigné au contrôle n'a pas pu voir la version traitement (et vice versa) — ce qui est assuré si l'assignation par cookie est persistante.

La randomisation garantit que la différence observée entre 3,5 % et 4,1 % est causalement attribuable à l'indicateur de stock, et non à des différences pré-existantes entre les groupes.
