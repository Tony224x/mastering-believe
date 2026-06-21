# Module 14 — Capstone : assembler son gabarit de plan financier

> **Temps estime** : 90-120 min (capstone, travail actif) | **Prerequis** : Modules 01-13
>
> **Objectif** : Assembler un **gabarit parametrable** ou VOUS entrez VOS chiffres — budget et taux d'epargne, fonds d'urgence cible, plan de dette, allocation simulee, projection composee 20-30 ans (en **reel** vs **nominal**), et cible FIRE optionnelle. Le livrable est un outil de simulation, **pas un plan prescriptif**.

> ⚠️ **Disclaimer** : ce capstone est **purement educatif**. C'est un **gabarit de simulation aux hypotheses ajustables**, **pas une recommandation** d'investissement, fiscale ou juridique. Les projections reposent sur des hypotheses qui **ne garantissent pas** les resultats futurs. Le programme **ne vous dit pas** quoi acheter ni quelle part allouer a quelle classe d'actifs : il calcule les consequences **des chiffres que vous saisissez**. Tout investissement comporte un risque de perte en capital. Consultez un professionnel agree pour toute decision reelle.

---

## 1. Ce que ce capstone est — et n'est pas

| Ce capstone **EST** | Ce capstone **N'EST PAS** |
|---|---|
| Un gabarit ou vous saisissez vos chiffres | Un plan tout fait a copier |
| Un calculateur de consequences ("si... alors...") | Un conseiller qui prescrit une allocation |
| Un outil pour visualiser le reel vs le nominal | Une promesse de rendement |
| Reutilisable chaque annee a la revision | Fige une fois pour toutes |

L'idee maitresse : un plan ecrit change le comportement (les travaux de Gollwitzer sur les *implementation intentions* montrent qu'ecrire **quoi, quand, comment** augmente fortement la probabilite d'agir). Mais un plan reste **le votre** : le gabarit ne decide rien a votre place, il rend visibles les arbitrages.

---

## 2. Les 6 blocs du gabarit

Le simulateur (`02-code/14-capstone-plan-financier.py`) est organise en blocs. **Toutes les valeurs d'entree sont des variables en tete de fichier** : vous les modifiez, vous relancez, vous lisez les consequences.

### Bloc 1 — Budget et taux d'epargne
Vos revenus nets, vos depenses (essentielles / non essentielles), vos remboursements de dette. Le gabarit en deduit votre **epargne mensuelle** et votre **taux d'epargne** (Module 02). Pas de jugement de valeur impose : le programme affiche le chiffre, c'est vous qui l'interpretez.

### Bloc 2 — Fonds d'urgence cible
Vous choisissez la cible en **mois de depenses essentielles** (3, 6... selon votre situation — Module 03). Le gabarit calcule le montant cible et l'ecart avec votre reserve actuelle. Rappel Module 01 : la cible se pense **en mois de depenses** (qui montent avec l'inflation), pas en montant fige.

### Bloc 3 — Plan de dette
Vous saisissez vos dettes (montant, taux). Le gabarit identifie celles a **taux eleve** (a traiter en priorite — logique avalanche, Module 04) et rappelle l'arbitrage rembourser/investir selon le taux. Il **ne tranche pas** a votre place : il met le taux en face de l'hypothese de rendement.

### Bloc 4 — Allocation simulee
Vous saisissez VOS proportions entre classes d'actifs et un TER cible. Le gabarit en deduit un **rendement net hypothetique** pondere (Modules 05-06). **Aucune allocation n'est suggeree** : les pourcentages sont vos entrees, pas une sortie du programme.

### Bloc 5 — Projection composee 20-30 ans : reel ET nominal
Le coeur pedagogique. Le gabarit projette votre capital a plusieurs horizons en affichant **deux colonnes** :
- **Nominal** : le solde affiche en euros futurs.
- **Reel** : le pouvoir d'achat en euros d'aujourd'hui (rendement reel ≈ nominal − inflation, Module 01).

C'est decisif : sur 30 ans, l'ecart entre nominal et reel est enorme. Raisonner en nominal **surestime** systematiquement ce que le capital permettra reellement d'acheter.

### Bloc 6 — Cible FIRE (optionnelle)
Si vous l'activez, le gabarit calcule une cible de capital selon la regle des 4 % (25 × depenses annuelles) et estime un horizon (Module 12). **Avec ses limites affichees** : la regle des 4 % suppose un horizon ~30 ans et un retrait indexe sur l'inflation ; elle ne couvre pas le risque de sequence de rendements. C'est un **repere**, pas une promesse.

---

## 3. Utiliser le gabarit (workflow)

1. Ouvrez `02-code/14-capstone-plan-financier.py`.
2. En **tete de fichier**, modifiez le bloc de variables `# === VOS CHIFFRES ===` avec vos donnees.
3. Lancez :
   ```
   python domains/finance-personnelle/02-code/14-capstone-plan-financier.py
   ```
4. Lisez la synthese. **Le disclaimer educatif est imprime dans la sortie elle-meme.**
5. Faites varier UNE hypothese a la fois (ex. taux d'epargne, hypothese de rendement, inflation) et observez l'effet. C'est de la **deliberate practice** : vous testez votre comprehension des leviers.

> Le programme tourne aussi sans rien modifier : il utilise alors un profil fictif de demonstration (clairement etiquete) pour que vous voyiez le format de sortie avant d'entrer vos chiffres.

---

## 4. Criteres de reussite du capstone

Votre gabarit est complet et exploitable si :
- [ ] Vous avez saisi vos propres chiffres (budget, dettes, reserve, proportions d'allocation).
- [ ] Votre taux d'epargne est calcule et vous savez l'interpreter (sans note de valeur imposee par le programme).
- [ ] Votre cible de fonds d'urgence est exprimee **en mois de depenses**.
- [ ] Votre plan de dette distingue clairement les dettes a taux eleve.
- [ ] Votre projection affiche le capital en **nominal ET en reel** a 20 et 30 ans.
- [ ] L'hypothese de rendement et l'hypothese d'inflation sont **explicites** (vous pouvez les defendre).
- [ ] (Optionnel) La cible FIRE est calculee avec ses limites comprises.
- [ ] Vous pouvez relire le tout en 1-2 pages a la revision annuelle.

---

## 5. La revision annuelle

Le gabarit n'est pas un one-shot. Une fois par an (ou apres un evenement majeur : changement de revenu, naissance, achat immobilier) :
1. Mettez a jour vos chiffres reels et relancez.
2. Comparez reel vs plan (taux d'epargne effectif, capital effectif).
3. Reajustez vos hypotheses (rendement, inflation) si la realite a derive.
4. Le plan evolue avec votre vie — ce qui compte, c'est de le maintenir vivant.

---

## 6. Erreurs classiques (et la reponse du gabarit)

| Erreur | Reponse |
|---|---|
| Investir avant le fonds d'urgence | Bloc 2 avant Bloc 4 : la sequence est dans l'ordre des blocs |
| Ignorer l'inflation dans les projections | Bloc 5 affiche **toujours** la colonne reelle a cote du nominal |
| Confondre allocation suggeree et allocation saisie | Le programme **n'allocue rien** : vos % sont des entrees |
| Prendre la cible FIRE pour une garantie | Bloc 6 imprime ses limites (horizon, sequence de rendements) |
| Plan trop complexe | Simplicite = durabilite : peu de lignes, ajustables |

---

## 7. Flash-cards

**Q1 — En quoi ce capstone est-il un "gabarit" et pas un "plan prescriptif" ?**
> R : Il calcule les **consequences des chiffres que VOUS saisissez** ; il ne recommande aucune allocation ni aucun produit. Les pourcentages d'allocation sont des **entrees**, jamais des sorties.

**Q2 — Pourquoi le Bloc 5 affiche-t-il le capital en nominal ET en reel ?**
> R : Sur 20-30 ans, l'inflation erode fortement le pouvoir d'achat. Le **reel** (≈ nominal − inflation) mesure ce que le capital permettra reellement d'acheter ; raisonner en nominal seul **surestime** le resultat.

**Q3 — Comment est calculee la cible FIRE optionnelle, et avec quelles limites ?**
> R : 25 × depenses annuelles (regle des 4 %). Limites affichees : horizon suppose ~30 ans, retrait indexe sur l'inflation, **risque de sequence de rendements** non couvert. C'est un repere, pas une promesse.

**Q4 — Quelle est la bonne facon de "pratiquer" avec le gabarit ?**
> R : Faire varier **une hypothese a la fois** (taux d'epargne, rendement, inflation) et observer l'effet — pour comprendre quels leviers comptent le plus (deliberate practice).

**Q5 — Pourquoi exprimer le fonds d'urgence en mois de depenses plutot qu'en montant fige ?**
> R : Parce que les depenses montent avec l'inflation : une cible "en mois" reste pertinente dans le temps, un montant fige perd sa valeur reelle (Module 01).

---

## Points cles a retenir

1. Le capstone est un **gabarit parametrable** : vous entrez vos chiffres, il calcule les consequences — **aucune recommandation**.
2. Six blocs : budget/taux d'epargne, fonds d'urgence (en mois), plan de dette, allocation **saisie**, projection **reelle vs nominale**, cible FIRE optionnelle.
3. La projection en **reel** est le coeur pedagogique : le nominal seul trompe sur le long terme.
4. Le disclaimer educatif est **imprime dans la sortie** du programme.
5. Le gabarit se **revise chaque annee** ; un plan simple qu'on maintient bat un plan parfait qu'on abandonne.

---

## Pour aller plus loin

- **Compound Interest Calculator** — U.S. SEC (Investor.gov) : https://www.investor.gov/financial-tools-calculators/calculators/compound-interest-calculator
- **The Little Book of Common Sense Investing** — J. C. Bogle (Wiley, 2017) : https://www.wiley.com/en-us/9781119404507
- **The Psychology of Money** — M. Housel (Harriman House, 2020) : https://harriman-house.com/authors/morgan-housel/the-psychology-of-money/9780857197689/
- **AMF — Espace epargnants** (en francais) : https://www.amf-france.org/fr/espace-epargnants
- **La finance pour tous** — IEFP : https://www.lafinancepourtous.com/

> **Disclaimer** : ce module et son gabarit sont educatifs. Ils ne constituent pas un conseil financier, fiscal ou juridique personnalise et ne recommandent aucune allocation ni aucun produit. Les hypotheses sont ajustables ; les performances passees ne prejugent pas des performances futures ; tout investissement comporte un risque de perte en capital.
