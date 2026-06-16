# Exercices Medium — Module 04 : Heuristiques & Biais Cognitifs

> **Niveau** : Medium | **Temps estimé** : ~35 min

---

## Exercice 1 — Quantifier et neutraliser l'ancrage

### Objectif

Mesurer chiffre en main l'écart introduit par une ancre numérique, puis concevoir un protocole de débiaisage concret de type *consider-the-opposite*.

### Consigne

Vous devez estimer la durée d'un projet de migration logicielle. Avant tout calcul, un collègue lâche : « le dernier projet de ce genre a pris **8 semaines** » (c'est votre ancre).

Vos propres données objectives, indépendantes de cette remarque :

- 5 lots de travail, durée estimée par lot : 3, 4, 2, 5 et 3 semaines.
- Les lots s'exécutent **en série** (l'un après l'autre).
- Vos 3 derniers projets comparables ont dépassé leur estimation initiale de **+30 % en moyenne** (marge d'imprévu à ajouter).

Travail demandé :

1. Calculez votre estimation **indépendante** : somme des lots, puis application de la marge +30 %. Montrez le calcul.
2. Comparez à l'ancre de 8 semaines : quel est l'écart en semaines et en pourcentage ? Dans quelle direction l'ancre tire-t-elle votre jugement ?
3. Rédigez un **protocole de débiaisage** en 3 étapes, dont au moins une étape *consider-the-opposite* (« et si l'ancre était fausse, quels arguments iraient dans le sens contraire ? »).

### Critères de réussite

- [ ] Somme des lots correcte : 3 + 4 + 2 + 5 + 3 = **17 semaines**.
- [ ] Estimation indépendante avec marge correcte : 17 × 1,30 = **22,1 semaines** (≈ 22 semaines).
- [ ] Écart identifié : l'ancre (8) est **~14 semaines en dessous** de l'estimation indépendante (soit ~64 % plus basse) ; l'ancre tire l'estimation **vers le bas** (sous-estimation).
- [ ] Le protocole de débiaisage contient une étape « estimer en aveugle AVANT toute valeur de référence ».
- [ ] Le protocole contient une étape *consider-the-opposite* explicite (chercher activement les arguments qui contredisent l'ancre).
- [ ] L'apprenant note que l'ancrage est un effet **robuste/répliqué** (ce n'est pas un raccourci anecdotique).

---

## Exercice 2 — Nommer le biais et appliquer la contre-mesure

### Objectif

Diagnostiquer, sur plusieurs vignettes courtes et neutres, lequel des 5 biais robustes est en jeu, puis prescrire la contre-mesure exacte du tableau récapitulatif du cours.

### Consigne

Pour chacune des 5 vignettes : (a) nommez le biais robuste à l'œuvre parmi les 5 du cours (ancrage, disponibilité, cadrage, négligence du taux de base, biais de confirmation) ; (b) donnez la contre-mesure rapide associée. Justifiez en une phrase.

**Vignette A** — On vous demande d'estimer le nombre de billes dans un bocal. L'animateur dit d'abord « il y en a peut-être 500 ». La plupart des gens annoncent ensuite un nombre proche de 500, alors qu'un comptage par couches donnerait ~1 200.

**Vignette B** — Un test de contrôle qualité détecte un défaut réel dans 90 % des cas. Sur la ligne, seulement 2 % des pièces sont réellement défectueuses. Le test signale un défaut sur une pièce ; un opérateur conclut « il y a 90 % de chances qu'elle soit défectueuse ».

**Vignette C** — Après avoir vu trois reportages sur des pannes d'un modèle de machine à laver, un acheteur déclare ce modèle « peu fiable », sans consulter les taux de retour agrégés du fabricant.

**Vignette D** — Un chef d'équipe est convaincu que l'outil X est le plus rapide. Il parcourt les benchmarks et ne retient que ceux qui donnent X gagnant, sans regarder les tests où X est plus lent.

**Vignette E** — Une mutuelle propose un forfait présenté comme « 95 % des dossiers remboursés ». Un concurrent propose le même forfait présenté comme « 1 dossier sur 20 refusé ». Les clients préfèrent nettement la première formulation.

### Critères de réussite

- [ ] Vignette A = **ancrage** → contre-mesure : estimer en aveugle puis chercher des références (ici, compter par couches sans se laisser tirer par le « 500 »).
- [ ] Vignette B = **négligence du taux de base** → contre-mesure : calculer P(base) explicitement ; la vraie probabilité est très inférieure à 90 % (le taux de base de 2 % écrase le résultat).
- [ ] Vignette C = **disponibilité** → contre-mesure : chercher les statistiques agrégées (taux de retour réels).
- [ ] Vignette D = **biais de confirmation** → contre-mesure : chercher activement l'argument réfutant le plus fort (les benchmarks où X perd).
- [ ] Vignette E = **cadrage** → contre-mesure : reformuler dans les deux sens (gain/perte) ; « 95 % remboursés » = « 5 % refusés » = « 1 sur 20 », c'est identique.
- [ ] Chaque diagnostic est justifié en une phrase reliée au mécanisme du biais.

---

## Exercice 3 — Construire les deux cadrages d'une même option

### Objectif

Prendre un seul jeu d'options neutre, l'écrire en cadrage gain puis en cadrage perte, prouver par le calcul que les espérances sont identiques, et prédire le renversement de préférence.

### Consigne

Une usine doit gérer un lot de **600 composants** menacés par un défaut de fabrication. Deux plans d'intervention, de **même espérance**, sont possibles :

- **Plan sûr** : on sauve **200 composants** à coup sûr.
- **Plan risqué** : **1/3** de chance de sauver les **600**, **2/3** de chance d'en sauver **0**.

Travail demandé :

1. Calculez l'espérance du nombre de composants sauvés pour le plan sûr et pour le plan risqué. Montrez qu'elles sont égales.
2. Réécrivez **exactement le même jeu d'options** en **cadrage perte** (parler des composants perdus, pas sauvés). Donnez le plan sûr et le plan risqué version « perte ».
3. Indiquez quel plan la majorité des gens choisit en cadrage gain, et lequel en cadrage perte. Expliquez en 2-3 phrases le renversement de préférence en mobilisant l'aversion à la perte.
4. Concluez sur la contre-mesure du cours.

### Critères de réussite

- [ ] Espérance plan sûr = **200 composants sauvés** ; espérance plan risqué = (1/3 × 600) + (2/3 × 0) = **200** ; les deux sont égales.
- [ ] Cadrage perte correct : plan sûr = « **400 composants seront perdus** à coup sûr » ; plan risqué = « **1/3** de chance qu'**aucun** ne soit perdu, **2/3** de chance que les **600** soient perdus ».
- [ ] Le calcul montre que gain et perte décrivent la **même réalité** (200 sauvés ⇔ 400 perdus ; etc.).
- [ ] Prédiction correcte : en cadrage **gain** la majorité choisit le **plan sûr** ; en cadrage **perte** la majorité bascule vers le **plan risqué**.
- [ ] L'explication mobilise l'**aversion à la perte** : face à une perte certaine, on devient preneur de risque.
- [ ] Contre-mesure citée : reformuler systématiquement la décision dans les deux cadrages avant de trancher.
