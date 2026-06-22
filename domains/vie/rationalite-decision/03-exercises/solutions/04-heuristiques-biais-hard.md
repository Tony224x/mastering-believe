# Solutions Hard — Module 04 : Heuristiques & Biais Cognitifs

---

## Exercice 1 — Audit d'un mémo d'estimation truffé de biais

### 1. Diagnostic des biais (points 1 à 4)

| Point | Biais robuste | Justification (1 phrase) |
|---|---|---|
| 1 — prix initial 42 €, « 39 € = bonne affaire » | **Ancrage** | Le 42 € sert d'ancre ; 39 € est jugé bon par rapport à elle, sans prix cible indépendant |
| 2 — un carton abîmé → Beta « peu fiable » | **Disponibilité** | Un incident récent et saillant surpondère le jugement face à l'historique global |
| 3 — ne retient que les specs favorables à Alpha | **Biais de confirmation** | Collecte sélective des preuves qui valident la conviction de départ |
| 4 — test 95 %, 1 % défectueux → « 95 % » | **Négligence du taux de base** | La prévalence de 1 % est ignorée au profit de la sensibilité de 95 % |

### 2. Calcul du point 4 (Bayes)

```
Données : prévalence (taux de base) = 1 % ; sensibilité = 95 % ; faux positifs = 5 %.

P(défectueuse | signalée)
  = (0,01 × 0,95) / [(0,01 × 0,95) + (0,99 × 0,05)]
  = 0,0095 / (0,0095 + 0,0495)
  = 0,0095 / 0,0590
  ≈ 0,161  =  16,1 %
```

**La conclusion « 95 % » est indéfendable.** La vraie probabilité est ~**16 %**, pas 95 %. Comme les pièces réellement défectueuses sont rares (1 %), les faux positifs (5 % de 99 % de pièces saines) dominent les vrais positifs. L'auteur a confondu P(signalée | défectueuse) = 95 % avec P(défectueuse | signalée) ≈ 16 %.

### 3. Mémo réécrit (version débiaisée)

1. **Prix (anti-ancrage)** : calculer un **prix cible** à partir du coût matière + marge acceptable **avant** d'entrer en négociation ; juger l'offre par rapport à ce prix cible, pas par rapport au premier chiffre annoncé.
2. **Beta (anti-disponibilité)** : consulter l'**historique agrégé** de livraisons de Beta (taux d'incidents sur 12 mois) ; un carton abîmé isolé ne dit rien sans le dénominateur.
3. **Specs (anti-confirmation)** : lister explicitement les essais où **Alpha perd** ; pré-définir des critères de comparaison **avant** de regarder les fiches.
4. **Contrôle qualité (anti-taux de base)** : raisonner en fréquences naturelles / poser P(base) ; conclure que ~16 % des pièces signalées sont réellement défectueuses, donc le process d'Alpha n'est **pas** « risqué » sur cette seule base.
5. **Conclusion révisée** : la décision Alpha vs Beta doit être ré-instruite avec un prix cible, l'historique de Beta, une comparaison équilibrée des specs et une lecture correcte du contrôle qualité — pas signée sur les quatre biais ci-dessus.

### 4. Checklist anti-biais (réutilisable)

- [ ] **Ancrage** — Ai-je fixé mon prix cible / mon estimation **avant** d'entendre le chiffre de l'autre partie ?
- [ ] **Disponibilité** — Mon jugement repose-t-il sur un incident saillant ou sur des **statistiques agrégées** ?
- [ ] **Confirmation** — Ai-je cherché activement la preuve qui **contredit** mon hypothèse de départ ?
- [ ] **Taux de base** — Ai-je intégré la **fréquence de base** avant d'interpréter un résultat individuel/un test ?
- [ ] **Cadrage** — Ai-je reformulé la décision en gain **et** en perte pour vérifier que ma préférence ne dépend pas de la formulation ?

**Note** : les 4 biais audités (ancrage, disponibilité, confirmation, taux de base) sont des effets **robustes et répliqués**. L'audit ne s'appuie sur aucun effet fragile de priming social — c'est ce qui le rend solide.

---

## Exercice 2 — Rationalité écologique + tri robuste vs fragile

### Partie A — Take the Best vs modèle complexe

**A.1 — Fonctionnement de Take the Best**

On classe les indices par **validité** (probabilité qu'un indice donne la bonne réponse quand il discrimine). Puis, pour comparer deux villes :

```
1. Regarder le 1er indice (le plus valide). 
   - S'il discrimine (l'un a, l'autre non) -> décider et S'ARRÊTER. On ignore tout le reste.
   - S'il ne discrimine pas -> passer à l'indice suivant.
2. Répéter jusqu'au premier indice qui tranche.
```

C'est une heuristique **« one-reason decision making »** : une seule raison suffit, on ne combine pas les indices. L'**heuristique de reconnaissance** en est un cas limite : si on reconnaît une ville et pas l'autre, on parie sur celle qu'on reconnaît.

**A.2 — Pourquoi elle peut battre la régression ici**

Avec seulement 8 paires d'entraînement et des indices bruités :

- Une **régression multiple** estime un poids par indice ; avec si peu de données, ces poids s'ajustent au **bruit** de l'échantillon (**overfitting**). En généralisation, elle se trompe.
- Take the Best **ignore** la plupart de l'information et n'estime quasiment aucun paramètre → elle ne peut pas overfitter le bruit. Elle capture le signal robuste (l'indice le plus valide) et jette le reste.
- C'est l'effet **less-is-more** : dans un environnement pauvre en données et bruité, **moins** d'information traitée conduit à de **meilleures** prédictions. Gigerenzer & Todd (1999) montrent que Take the Best égale ou bat la régression multiple sur la tâche des populations de villes.

**A.3 — Quand l'avantage disparaît**

Le modèle complexe **redevient supérieur** quand l'environnement le permet :

```
- Beaucoup de données d'entraînement (les poids sont estimés sans overfitter).
- Faible bruit (les indices sont fiables, le signal est net).
- Indices nombreux, stables et faiblement redondants (la combinaison apporte de l'information).
```

Autrement dit : less-is-more n'est **pas** une loi universelle ; c'est un avantage **conditionnel à l'environnement** (rationalité écologique).

**A.4 — Les deux cadres sont complémentaires**

```
Kahneman / Tversky : les heuristiques produisent des biais systématiques (ancrage, cadrage...).
                     -> décrit QUAND une heuristique trompe.
Gigerenzer         : les heuristiques frugales sont des outils adaptatifs.
                     -> décrit QUAND une heuristique aide (et bat un modèle complexe).
```

Le pont entre les deux est la **rationalité écologique** : une heuristique n'est ni bonne ni mauvaise en soi ; son efficacité dépend de l'**appariement entre la règle et la structure de l'environnement**. Dans un environnement où la règle « colle » (peu de données, bruit), elle gagne ; dans un environnement où elle ne colle pas, elle produit un biais. Les deux programmes décrivent les deux faces du même objet — ils ne s'opposent pas.

### Partie B — Robuste ou fragile ?

| Effet | Verdict | Justification |
|---|---|---|
| (i) Ancrage numérique | **Robuste / répliqué** | Effet maintes fois reproduit (roue de loterie : 20 points d'écart pour un nombre arbitraire) ; au programme des 5 biais solides |
| (ii) Amorçage « vieillesse » (marcher plus lentement) | **Fragile — priming social** | Effet emblématique du priming social qui a **largement échoué à la réplication** après 2011 |
| (iii) Effet de cadrage (problème des 600) | **Robuste / répliqué** | Renversement de préférence stable et reproduit (Tversky & Kahneman, 1981) |
| (iv) Effet « café chaud » (boisson chaude → jugement chaleureux) | **Fragile — priming social** | Effet de priming social non répliqué de façon fiable ; à ne pas présenter comme établi |

**Prudence épistémique (à expliciter)** : Kahneman lui-même a reconnu la fragilité des effets de priming social après la **crise de réplication de 2011** (cf. Open Science Collaboration, 2015). La règle pour l'apprenant : les biais (i) et (iii) peuvent être enseignés et utilisés comme des faits ; les effets (ii) et (iv) doivent être présentés comme **incertains/non répliqués**, jamais comme des résultats acquis. Savoir distinguer robuste de fragile **est** une compétence de rationalité.
