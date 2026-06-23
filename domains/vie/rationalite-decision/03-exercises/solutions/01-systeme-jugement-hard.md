# Solutions Hard — Module 01 : Le systeme d'exploitation du jugement

---

## Exercice 1 — Diagnostic multi-erreurs d'un analyste brillant

Rappel des reperes utilises :
- Composants de Stanovich : **autonome** (S1), **algorithmique** (S2 brut, ~QI), **reflexif** (interroger/corriger).
- **Mindware gap** = outil conceptuel absent. **Echec reflexif** = outil present, non active.

### Tableau de diagnostic (questions 1 a 3)

| Etape | Erreur (Q1) | Composant Stanovich + nature (Q2) | Mindware correctif (Q3) |
|---|---|---|---|
| 1 | **Ancrage** (anchoring) | **Reflexif**, echec d'activation : l'ancre "6 semaines" n'a pas ete corrigee malgre ses propres sous-taches a ~11 | Estimer AVANT d'entendre toute valeur de reference ; ajuster a partir de ses propres donnees, pas de l'ancre |
| 2 | **Planning fallacy / exces d'optimisme** | **Reflexif**, echec d'activation : il connait l'historique mais ne l'utilise pas | "Outside view" / reference class forecasting : appliquer le depassement moyen historique (30-50 %) comme multiplicateur |
| 3 | **Base rate neglect** | **Mindware gap** : notion de taux de base / Bayes jamais apprise | Apprendre Bayes ; raisonner en frequences naturelles ; integrer le taux de base de 2 % |
| 4 | **Biais de confirmation** | **Reflexif**, echec d'activation : il sait qu'il "faudrait chercher la preuve contraire" mais ne le fait pas | Chercher activement la preuve infirmante ; pre-engager les criteres de comparaison avant de regarder les benchmarks |
| 5 | **Defaut d'activation du S2** sur un piege de reflexion | **Reflexif**, echec d'activation : il corrige seul une fois ralenti -> mindware present | Regle "ralentir et verifier tout resultat qui vient trop vite" |

### Detail de l'etape 3 (le seul mindware gap)

L'enonce : sensibilite 90 %, taux de base 2 %, le test signale un bug. La reponse "90 %" confond P(signal | bug) avec P(bug | signal). C'est du **base rate neglect** pur : Karim ignore que seules 2 % des executions contiennent un vrai bug.

```
Intuition (fausse) : P(bug | signal) = sensibilite = 90 %.
Realite : sur 2 % de vrais bugs seulement, meme un bon test produit beaucoup de
fausses alertes en valeur absolue, parce que 98 % des cas sont sains.
=> La vraie probabilite est TRES en-dessous de 90 %.
(Le calcul exact via Bayes est fait au module 03 ; ici, l'essentiel est la DIRECTION :
le taux de base de 2 % ecrase le resultat. La reponse correcte n'est pas 90 %.)
```
C'est un **mindware gap** et non un echec reflexif : Karim n'a jamais appris la notion de taux de base. Meme en ralentissant, il ne saurait pas la mobiliser. C'est la difference cruciale avec les etapes 1, 2, 4, 5, ou l'outil existe mais reste inutilise.

### Distinction transversale

```
Mindware gap (outil ABSENT)          : etape 3 uniquement.
Echec de l'esprit reflexif (outil    : etapes 1, 2, 4, 5.
  PRESENT mais non active)
```
Indice diagnostique : si la personne corrige seule des qu'on la ralentit ou l'invite a verifier (etapes 1, 2, 4, 5), l'outil etait la -> echec reflexif. Si elle reste bloquee meme en prenant son temps (etape 3), l'outil manque -> mindware gap.

### Question 4 — Synthese : QI et dysrationalia

Non, le QI eleve de Karim ne l'a pas protege. Son **esprit algorithmique** est excellent (major de promo, redoutable en calcul), mais 4 de ses 5 erreurs viennent d'un **esprit reflexif** non active, et la cinquieme d'un **mindware gap** — deux dimensions que le QI ne mesure pas.

C'est exactement la **dysrationalia** au sens de Stanovich : la capacite a raisonner mal malgre une intelligence elevee, par manque d'esprit reflexif et/ou de mindware, et non par manque de puissance algorithmique. La lecon pratique : on ameliore le jugement en **installant du mindware** (etape 3) et en **cultivant l'habitude d'activer l'esprit reflexif** — ralentir, verifier, chercher le contre-exemple, estimer avant de s'ancrer (etapes 1, 2, 4, 5). L'intelligence brute n'y suffit pas.

---

## Exercice 2 — Concevoir un item qui dissocie intelligence et rationalite

Cet exercice est ouvert : il n'y a pas une seule bonne reponse. Voici un **exemple de corrige** qui respecte tous les criteres, suivi de la grille de validation a appliquer a la production de l'apprenant.

### Exemple d'item modele : "le tournoi a elimination"

**1. Enonce de l'item**
> Un tournoi de jeu de plateau a elimination directe (chaque match elimine un joueur) reunit 64 participants. Combien de matchs faut-il jouer au total pour designer le vainqueur ?

**2. Reponse-leurre (S1)**
La plupart des gens lancent un calcul de tours : 32 + 16 + 8 + 4 + 2 + 1, et beaucoup repondent vite et faux (ou abandonnent), ou repondent "64" par appariement de surface avec le nombre de joueurs. Le leurre dominant est de croire qu'il faut additionner les rounds laborieusement, voire de repondre "environ 64".

**3. Reponse correcte + calcul**
```
Insight : chaque match elimine exactement 1 joueur.
Pour qu'il ne reste qu'1 vainqueur sur 64, il faut eliminer 63 joueurs.
=> il faut donc exactement 63 matchs.
Reponse correcte : 63 matchs.
```

**4. Pourquoi l'item dissocie intelligence et rationalite**
L'item ne demande aucune puissance de calcul : une fois qu'on adopte le bon cadrage ("1 match = 1 elimine"), la soustraction 64 - 1 = 63 est triviale. Ce qui pose probleme, c'est d'**inhiber le reflexe** de sommer les tours (ou de repondre "64"). Une personne a fort QI peut tres bien sommer 32+16+8+4+2+1 correctement et tomber juste — mais elle peut aussi se precipiter sur le leurre si elle n'active pas l'esprit reflexif. L'item teste donc le **cadrage et la verification** (esprit reflexif), pas la capacite de calcul (esprit algorithmique).

**5. Prediction de la direction de l'erreur**
Direction typique : reponse erronee orientee vers le **nombre de joueurs (64)** ou un effort de sommation inutile. L'erreur est **systematique et previsible** (definition d'un biais) : le S1 s'accroche au chiffre saillant de l'enonce (64) et au schema "additionner les tours", au lieu de reformuler le probleme par l'invariant "1 match = 1 elimine".

**6. Test de validite anti-piege (re-verification par une 2e voie)**
```
2e methode (sommation des tours, pour controler) :
  64 -> 32 matchs -> reste 32
  32 -> 16 matchs -> reste 16
  16 -> 8  matchs -> reste 8
  8  -> 4  matchs -> reste 4
  4  -> 2  matchs -> reste 2
  2  -> 1  match  -> reste 1 (vainqueur)
  Total = 32 + 16 + 8 + 4 + 2 + 1 = 63 matchs.  OK, coherent avec 64 - 1.
```
L'item n'est **pas** un casse-tete difficile : le calcul final (64 - 1) est enfantin une fois l'intuition mise de cote. Il pieges la rapidite, pas la puissance. C'est precisement le profil d'un bon item de rationalite.

### Grille de validation (a appliquer a la production de l'apprenant)

| Critere | Verifie ? |
|---|---|
| Item original (pas batte/widgets/nenuphars) et 100 % neutre | a verifier |
| Reponse-leurre S1 clairement identifiee, differente de la bonne reponse | a verifier |
| Reponse correcte demontree pas a pas ET re-verifiee par une 2e methode | a verifier |
| L'item teste l'esprit reflexif (inhibition/verification), pas le QI (calcul facile une fois l'intuition ecartee) | a verifier |
| Direction de l'erreur predite + reliee a "systematique + previsible" | a verifier |
| Controle : ce n'est PAS un casse-tete dur (sinon il testerait l'esprit algorithmique) | a verifier |
| Bonus : plan de calibration (tester le leurre sur quelques personnes) | bonus |

**Note sur la calibration (bonus)** : un item de rationalite n'a de valeur que si le leurre fonctionne reellement. Le tester sur un petit echantillon permet de verifier qu'une majorite tombe d'abord dans le leurre — sinon l'item ne dissocie rien. C'est l'exigence empirique qui distingue un vrai item de test d'une simple devinette, dans la lignee de l'honnetete sur la preuve rappelee par la crise de replication (module 01, section 6).
