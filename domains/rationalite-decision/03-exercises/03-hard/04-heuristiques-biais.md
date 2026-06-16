# Exercices Hard — Module 04 : Heuristiques & Biais Cognitifs

> **Niveau** : Hard | **Temps estimé** : ~50 min

---

## Exercice 1 — Audit d'un mémo d'estimation truffé de biais

### Objectif

Auditer un document de décision neutre, localiser **plusieurs biais robustes distincts** cohabitant dans un même texte, justifier chaque diagnostic, puis réécrire le mémo en version débiaisée avec une checklist opérationnelle.

### Consigne

Lisez le mémo ci-dessous (contexte neutre : choix d'un fournisseur de pièces pour une chaîne de production). Le but n'est pas de juger l'auteur mais d'isoler chaque défaillance de raisonnement.

> **MÉMO — Choix du fournisseur de roulements (rédigé par l'acheteur)**
>
> 1. Le commercial du fournisseur **Alpha** a ouvert la discussion en annonçant un prix de **42 €/pièce**. Après négociation j'ai obtenu **39 €/pièce**, ce qui me paraît une excellente affaire. Je n'ai pas calculé de prix cible de mon côté avant l'entretien.
> 2. La semaine dernière, une livraison du fournisseur **Beta** est arrivée avec **un carton endommagé**. Du coup je considère Beta comme « peu fiable » et je l'écarte, sans regarder son historique global de livraisons.
> 3. J'ai comparé les fiches techniques. Comme j'étais déjà convaincu qu'Alpha était le meilleur, je n'ai relevé que les specs où Alpha gagne ; je n'ai pas cherché les essais où Alpha est moins bon.
> 4. Le contrôle qualité d'Alpha détecte une pièce défectueuse dans **95 %** des cas. Or seules **1 %** des pièces produites sont réellement défectueuses. Le contrôle a signalé une pièce : j'en conclus qu'il y a **95 %** de chances qu'elle soit vraiment défectueuse, donc le process d'Alpha est risqué.
> 5. **Conclusion** : on signe avec Alpha à 39 €, c'est clairement le meilleur choix.

**Questions :**

1. Pour chacun des points 1 à 4, **nommez le biais robuste** en jeu (parmi les 5 du cours) et **justifiez** en une phrase.
2. Pour le point 4, **calculez** la vraie probabilité que la pièce signalée soit réellement défectueuse, en supposant un taux de faux positifs de **5 %**. La conclusion « 95 % » est-elle défendable ?
3. **Réécrivez** le mémo en version débiaisée : pour chaque point, indiquez ce que l'auteur aurait dû faire (contre-mesure du cours).
4. Produisez une **checklist anti-biais** de 5 items réutilisable pour toute future décision d'achat.

### Critères de réussite

- [ ] Point 1 = **ancrage** (le prix initial de 42 € sert d'ancre ; « 39 € = bonne affaire » est jugé par rapport à l'ancre, pas à un prix cible indépendant).
- [ ] Point 2 = **disponibilité** (un incident récent et saillant — un carton — surpondère le jugement par rapport à l'historique global).
- [ ] Point 3 = **biais de confirmation** (collecte sélective des specs favorables à la conviction initiale).
- [ ] Point 4 = **négligence du taux de base** (la prévalence de 1 % est ignorée au profit de la sensibilité de 95 %).
- [ ] Calcul du point 4 correct : P(défectueuse | signalée) = (0,01 × 0,95) / [(0,01 × 0,95) + (0,99 × 0,05)] = 0,0095 / (0,0095 + 0,0495) ≈ **16,1 %**, pas 95 %. La conclusion est **indéfendable** : le taux de base écrase le résultat.
- [ ] Mémo réécrit avec une contre-mesure correcte par point : (1) fixer un prix cible AVANT la négociation ; (2) consulter l'historique agrégé de livraisons de Beta ; (3) chercher activement les specs où Alpha perd ; (4) raisonner en fréquences / poser P(base) avant de conclure.
- [ ] Checklist de 5 items, chacun mappé à un biais et formulé comme une action concrète.
- [ ] L'apprenant note que les 4 biais audités sont tous des effets **robustes/répliqués** (l'audit ne repose sur aucun effet fragile).

---

## Exercice 2 — Rationalité écologique + tri robuste vs fragile

### Objectif

Argumenter, sur une tâche de prédiction neutre, pauvre en données et bruitée, quand une heuristique simple bat un modèle complexe (less-is-more), puis transférer la prudence épistémique en distinguant un effet **robuste** d'un effet de *priming social* **fragile** dans une liste d'effets « qui sonnent célèbre ».

### Consigne

**Partie A — Take the Best vs modèle complexe.**

On veut prédire, pour des paires de petites villes inconnues du décideur, laquelle a la plus grande population. On dispose de très peu d'observations (8 paires d'entraînement) et de plusieurs indices bruités (présence d'une gare, d'une université, d'un aéroport, d'une équipe sportive connue...).

1. Décrivez comment fonctionne l'heuristique **« Take the Best »** sur cette tâche (recherche d'un indice à la fois, dans l'ordre de validité, arrêt au premier indice qui discrimine).
2. Expliquez pourquoi, **dans cet environnement précis** (peu de données, beaucoup de bruit), Take the Best — ou l'**heuristique de reconnaissance** — peut **égaler ou battre** une régression multiple. Reliez votre réponse à l'**overfitting** et à l'effet **less-is-more**.
3. Indiquez **quand** cet avantage disparaît (quand un modèle complexe redevient supérieur) : décrivez les conditions d'environnement (beaucoup de données, faible bruit, indices stables).
4. Concluez en reliant explicitement les deux cadres : celui de Kahneman/Tversky (« heuristiques = biais ») et celui de Gigerenzer (« heuristiques = outils adaptatifs »). Montrez qu'ils sont **complémentaires** et non opposés.

**Partie B — Robuste ou fragile ?**

Pour chacun des 4 effets ci-dessous, dites s'il s'agit d'un effet **robuste/répliqué** ou d'un effet de **priming social fragile** (qui a largement échoué à la réplication après 2011), et justifiez en une phrase.

- (i) **Ancrage numérique** : un nombre présenté en premier tire les estimations vers lui.
- (ii) **Amorçage « vieillesse »** : lire des mots liés à la vieillesse ferait marcher les gens plus lentement en sortant de la pièce.
- (iii) **Effet de cadrage** : la même option décrite en gain vs en perte renverse les préférences (problème des 600).
- (iv) **Effet « café chaud »** : tenir une boisson chaude rendrait le jugement social plus « chaleureux » envers autrui.

### Critères de réussite

- [ ] Partie A.1 : Take the Best décrit correctement (un indice à la fois, par ordre de validité, **arrêt dès qu'un indice discrimine**, on ignore le reste).
- [ ] Partie A.2 : l'avantage est relié à l'**overfitting** (un modèle complexe ajuste le bruit de 8 observations) et au **less-is-more** (ignorer des indices = meilleure généralisation ici) ; mention possible de l'heuristique de **reconnaissance**.
- [ ] Partie A.3 : conditions où le modèle complexe redevient meilleur correctement listées (**beaucoup de données, faible bruit, indices nombreux et stables**).
- [ ] Partie A.4 : les deux cadres sont présentés comme **complémentaires** — l'efficacité d'une heuristique dépend de l'appariement règle/environnement (*rationalité écologique*), Kahneman décrit quand elle trompe, Gigerenzer quand elle aide.
- [ ] Partie B : (i) **robuste**, (ii) **fragile (priming social)**, (iii) **robuste**, (iv) **fragile (priming social)** — classement entièrement correct.
- [ ] Chaque classement de la partie B est justifié, et l'apprenant rappelle que Kahneman lui-même a reconnu la fragilité du priming social après la crise de réplication de 2011.
- [ ] L'apprenant ne présente **jamais** (ii) et (iv) comme des résultats établis.
