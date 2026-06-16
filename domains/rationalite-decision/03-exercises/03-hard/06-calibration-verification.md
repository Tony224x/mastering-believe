# Exercices Hard — Module 06 : Calibration & Vérification de l'Information

> **Niveau** : Hard | **Temps estimé** : ~50 min

---

## Exercice 1 — Diagnostiquer puis recalibrer un journal de prévisions

### Objectif

À partir du relevé d'un prévisionniste réparti par tranches de confiance : calculer le score de Brier, **décomposer** la miscalibration (quelles tranches, dans quelle direction), prescrire une **règle de recalibration concrète**, puis estimer son effet **chiffré** sur le Brier.

### Consigne

Un prévisionniste a résolu 16 prédictions sur des événements neutres (livraisons, trajets, deadlines, météo, sport). On les regroupe par probabilité annoncée :

| Tranche (p) | n | Outcomes (o) | Réalisés |
|-------------|---|--------------|----------|
| 0,60 | 5 | 1, 1, 0, 0, 1 | 3 / 5 |
| 0,75 | 4 | 1, 1, 1, 0 | 3 / 4 |
| 0,85 | 4 | 1, 0, 0, 1 | 2 / 4 |
| 0,95 | 3 | 1, 0, 0 | 1 / 3 |

1. Calculez la somme des (p − o)² par tranche, puis le **score de Brier global** (sur les 16). Comparez à 0,25.
2. **Décomposez la calibration** : pour chaque tranche, comparez la probabilité annoncée à la **fréquence réelle**. Quelles tranches sont calibrées ? Lesquelles montrent de la **surconfiance** ? Dans quelle direction et de combien (écart `fréquence − p`) ?
3. **Prescrivez une règle de recalibration** concrète : par exemple "ramener les prédictions à 0,85 vers 0,65" et "ramener les 0,95 vers 0,70", en laissant les tranches calibrées intactes. Justifiez chaque cible par la fréquence réelle observée.
4. **Estimez l'effet** : recalculez le Brier en réappliquant votre règle aux mêmes issues. De combien baisse-t-il ? Expliquez pourquoi la recalibration ne touche que les tranches surconfiantes.

### Critères de réussite

- [ ] Sommes par tranche correctes : 0,60 → **1,20** ; 0,75 → **0,75** ; 0,85 → **1,49** ; 0,95 → **1,8075**.
- [ ] Somme totale = 5,2475 ; **Brier global = 0,3280** (± 0,005) → **au-dessus** de la baseline 0,25.
- [ ] Décomposition : tranche 0,60 calibrée (freq 0,60, écart 0) ; tranche 0,75 calibrée (freq 0,75, écart 0).
- [ ] Tranche 0,85 **surconfiante** : freq réelle 0,50, écart = **−0,35**.
- [ ] Tranche 0,95 **très surconfiante** : freq réelle ≈ 0,333, écart ≈ **−0,617**.
- [ ] Règle prescrite cohérente avec les fréquences observées (rapprocher 0,85 → ~0,65 et 0,95 → ~0,70 ; ne pas toucher 0,60 et 0,75).
- [ ] Brier recalibré (0,85→0,65 ; 0,95→0,70) = **0,2569** (± 0,005), soit une baisse d'environ **0,071** ; il repasse sous/au niveau de la baseline.
- [ ] Explication : seules les tranches surconfiantes changent ; rapprocher p de la fréquence réelle réduit mécaniquement la moyenne des (p−o)² sans dégrader les tranches déjà calibrées.

---

## Exercice 2 — Protocole de vérification d'une réponse d'IA (citation + image) et conclusion calibrée

### Objectif

Synthétiser **calibration + vérification** : face à une réponse d'IA confiante accompagnée d'une **citation à l'allure fabriquée** et d'une **image repartagée**, écrire le protocole de vérification complet (lecture latérale, Google Scholar, doi.org, reverse image search), énumérer les signaux d'hallucination, et formuler une **conclusion probabiliste calibrée**.

### Consigne

Vous demandez à un assistant IA si une plante d'intérieur courante "dépollue l'air d'un bureau". Il répond, très sûr de lui :

> "Oui. Selon **Okonkwo & Lindqvist (2021)**, *« Indoor phytoremediation efficiency in office environments »*, publié dans le *International Journal of Atmospheric Bioremediation*, **DOI 10.1099/ijab.2021.88421**, une seule plante réduit les particules fines de **37 %** en 24 heures."

Il ajoute une **photo** "d'un bureau test de l'étude" (image qui circule par ailleurs sur les réseaux).

Rédigez le **protocole de vérification complet** :

1. **Citation — lecture latérale** : quelles requêtes pour vérifier l'existence de la revue *International Journal of Atmospheric Bioremediation* et des auteurs ? (requêtes exactes)
2. **Citation — Google Scholar** : comment chercher le **titre exact entre guillemets** ? Que conclure si Scholar ne renvoie rien ?
3. **Citation — DOI** : comment tester le DOI ? (URL exacte à ouvrir) Que signifie un DOI qui ne résout pas ?
4. **Image — reverse image search** : quelle démarche pour savoir si la photo appartient vraiment à l'étude ou si elle est hors contexte ?
5. **Signaux d'hallucination** : énumérez au moins 4 indices que la citation est probablement inventée.
6. **Conclusion calibrée** : formulez votre verdict en termes probabilistes (pas un "vrai/faux" binaire) et dites comment vous **pondérez** l'affirmation initiale.

### Critères de réussite

- [ ] **Lecture latérale** : requêtes du type `"International Journal of Atmospheric Bioremediation" indexé Scopus OR DOAJ` et `Okonkwo Lindqvist phytoremediation auteurs` (sortir de la page, vérifier ailleurs).
- [ ] **Google Scholar** : recherche du **titre exact entre guillemets** `"Indoor phytoremediation efficiency in office environments"` ; conclusion correcte = "0 résultat ⇒ forte présomption de fabrication".
- [ ] **DOI** : ouvrir `https://doi.org/10.1099/ijab.2021.88421` ; un DOI qui ne résout pas (erreur/aucune cible) est un signal majeur d'invention.
- [ ] **Reverse image search** : démarche décrite (Google Images / TinEye / Lens) pour retrouver l'origine et la date de l'image et détecter un usage **hors contexte**.
- [ ] **≥ 4 signaux d'hallucination** parmi : revue au nom générique invérifiable, DOI qui ne résout pas, titre absent de Scholar, chiffre précis et flatteur (37 %), aucune trace des auteurs, image non rattachable à l'étude, assurance excessive du modèle.
- [ ] **Conclusion calibrée** explicite : "**probablement halluciné — à pondérer en conséquence**" (probabilité de fiabilité basse), et non "faux à 100 %" ; on suspend l'affirmation tant que la source primaire n'est pas retrouvée.
- [ ] La synthèse relie les deux moitiés du module : la **vérification** ajuste la **calibration** de notre croyance (on baisse la probabilité accordée à la réponse de l'IA au lieu de l'accepter ou de la rejeter en bloc).
