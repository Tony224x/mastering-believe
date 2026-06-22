# Exercices Hard — Module 01 : Le systeme d'exploitation du jugement

> **Niveau** : Hard | **Temps estime** : ~45 min

---

## Exercice 1 : Diagnostic multi-erreurs d'un analyste brillant

### Objectif

Diagnostiquer, dans une etude de cas neutre d'estimation de projet, plusieurs erreurs de rationalite distinctes commises par une seule personne tres intelligente ; classer chaque erreur dans le bon composant de Stanovich (autonome / algorithmique / reflexif) et prescrire le mindware correctif precis pour chacune.

### Consigne

Lisez le cas, puis traitez les questions. Le but n'est pas de juger la personne mais d'isoler la nature de chaque defaillance.

**Le cas** — Karim est ingenieur, major de promo, redoutable en calcul et en algorithmie (esprit algorithmique tres puissant). On lui confie l'estimation du delai de livraison d'un nouveau module logiciel pour une chaine de production. Voici sa demarche, etape par etape :

1. **Etape 1** : On lui glisse d'abord "le dernier module similaire a pris 6 semaines". Il fixe ensuite son estimation a 6,5 semaines, alors meme que le nouveau module est notablement plus complexe et que toutes ses sous-taches chiffrees, additionnees, donnent ~11 semaines.
2. **Etape 2** : Il estime la duree en imaginant le scenario ou "tout se passe bien" et en additionnant les temps optimaux de chaque sous-tache, sans reserver de marge pour les imprevus — alors que ses 4 derniers projets ont tous depasse leur estimation de 30 a 50 %.
3. **Etape 3** : On lui presente un probleme de controle qualite : "Le test automatique du module detecte un vrai bug dans 90 % des cas (sensibilite). Sur l'ensemble des executions, seules 2 % contiennent un vrai bug. Le test vient de signaler un bug : quelle est la probabilite qu'il y ait vraiment un bug ?" Karim repond "90 %". Il n'a jamais appris la notion de taux de base ni le raisonnement bayesien.
4. **Etape 4** : Convaincu depuis le debut que la libraire X est la meilleure, il parcourt la documentation et ne retient que les benchmarks qui confirment ce choix, sans chercher les cas ou X est plus lente. Il sait pourtant, en theorie, qu'il "faudrait chercher la donnee qui contredit".
5. **Etape 5** : Sous pression de temps, il calcule a la volee le piege "machines/widgets" (5 machines, 5 widgets, 5 minutes -> combien pour 100 machines, 100 widgets ?) et lance "100 minutes" sans verifier. Repris, il corrige seul en 5 secondes.

**Questions** :

1. Pour chacune des 5 etapes, nommez le biais ou l'erreur de rationalite a l'oeuvre.
2. Pour chacune des 5 etapes, indiquez le composant de Stanovich principalement en cause (esprit **autonome** S1, esprit **algorithmique** S2-brut, ou esprit **reflexif**), et dites s'il s'agit d'un **mindware gap** (outil absent) ou d'un **echec de l'esprit reflexif** (outil present non active).
3. Pour chacune des 5 etapes, prescrivez le **mindware correctif** concret (une methode, une regle, un outil) qui aurait evite l'erreur.
4. Question de synthese : le QI eleve de Karim a-t-il protege contre ces erreurs ? Reliez votre reponse au concept de **dysrationalia**.

### Criteres de reussite

- [ ] Etape 1 = **ancrage** (anchoring) ; composant = **reflexif** (l'ancre 6 semaines n'a pas ete corrigee malgre ses propres chiffres a 11) ; correctif = estimer AVANT d'entendre toute valeur de reference, ou ajuster a partir de ses propres donnees et non de l'ancre
- [ ] Etape 2 = **planning fallacy / exces d'optimisme** ; composant = **reflexif** ; correctif = vue "outside view" / reference class forecasting (utiliser le depassement moyen historique de 30-50 % comme multiplicateur)
- [ ] Etape 3 = **base rate neglect** ; composant = **mindware gap** (notion de taux de base et de Bayes jamais apprise) ; correctif = apprendre le theoreme de Bayes / raisonner en frequences naturelles ; la bonne reponse intuitive a corriger est tres inferieure a 90 % (le taux de base de 2 % ecrase le resultat)
- [ ] Etape 4 = **biais de confirmation** ; composant = **reflexif** (la regle "chercher la preuve contraire" existe mais n'est pas activee) ; correctif = chercher activement la preuve infirmante / pre-engager des criteres de comparaison avant de regarder les benchmarks
- [ ] Etape 5 = **defaut d'activation du S2** sur un piege de reflexion cognitive ; composant = **reflexif** (il corrige seul une fois ralenti -> le mindware est present) ; correctif = regle "ralentir et verifier sur tout resultat qui vient trop vite"
- [ ] Distinction nette : SEULE l'etape 3 est un mindware gap ; les etapes 1, 2, 4, 5 sont des echecs de l'esprit reflexif (outils disponibles, non actives)
- [ ] Question 4 : reponse = non, le QI (esprit algorithmique) ne protege pas ; c'est exactement la **dysrationalia** (raisonner mal malgre une intelligence elevee, par manque d'esprit reflexif et/ou de mindware)
- [ ] Pour l'etape 3, l'apprenant signale que la valeur exacte se calcule au module 03 (Bayes) mais que la direction est claire : tres en-dessous de 90 % a cause du taux de base de 2 %

---

## Exercice 2 : Concevoir un item qui dissocie intelligence et rationalite

### Objectif

Transferer la comprehension theorique en production : concevoir de toutes pieces un court item de "test de rationalite" neutre qui dissocie l'intelligence (puissance algorithmique) de la rationalite (esprit reflexif), justifier pourquoi il les dissocie, et predire la direction de l'erreur typique.

### Consigne

Concevez **votre propre** item de test, original (ne reprenez pas tel quel batte-et-balle, machines/widgets ou nenuphars). Il doit etre **100 % neutre** (des, urnes, meteo, gestion de projet, sport, jeux de plateau, fabrication/qualite, temps de trajet, etc. — jamais de sujet politique, religieux ou clivant). Votre livrable comporte 6 sections :

1. **Enonce de l'item** : un probleme court qui declenche une reponse intuitive (S1) seduisante mais fausse.
2. **Reponse-leurre (S1)** : la reponse intuitive que la plupart des gens donneront.
3. **Reponse correcte + calcul** : la solution exacte, demontree etape par etape, verifiable.
4. **Pourquoi l'item dissocie intelligence et rationalite** : expliquez pourquoi une personne a fort QI peut quand meme tomber dans le leurre (l'item ne demande pas de la puissance de calcul, il demande d'inhiber/verifier une intuition).
5. **Prediction de la direction de l'erreur** : dans quel sens l'erreur typique va-t-elle (sur-estimation, sous-estimation, mauvaise categorie...), et pourquoi cette direction est systematique et previsible (rappel de la definition d'un biais).
6. **Test de validite anti-piege** : verifiez que le calcul est correct (recalculez-le par une seconde voie) et que l'item n'est pas un simple casse-tete difficile (un casse-tete dur teste l'esprit algorithmique ; un bon item de rationalite reste calculable facilement UNE FOIS qu'on a inhibe l'intuition).

### Criteres de reussite

- [ ] L'item est **original** (distinct des exemples du cours) et **100 % neutre**
- [ ] Il existe une reponse-leurre S1 clairement identifiee, et elle est differente de la reponse correcte
- [ ] La reponse correcte est demontree etape par etape ET re-verifiee par une seconde methode (section 6)
- [ ] La justification montre que l'item teste l'**esprit reflexif** (inhibition/verification d'une intuition), pas la puissance algorithmique : une fois l'intuition mise de cote, le calcul est facile
- [ ] La direction de l'erreur est predite et reliee a la definition d'un biais (systematique + previsible)
- [ ] Controle de validite : l'item n'est PAS un casse-tete difficile (qui testerait le QI) — il est facile a calculer mais piege l'intuition rapide
- [ ] Bonus de qualite : l'apprenant explique comment il "calibrerait" l'item (le tester sur quelques personnes pour verifier que le leurre fonctionne reellement)
