# Exercices Hard — Module 02 : Probabilites utiles

> **Niveau** : Hard | **Temps estime** : ~50 min

---

## Exercice 1 : Stratification du risque — qui faut-il tester ?

### Objectif

Synthetiser sensibilite/specificite/taux de base sur **deux sous-populations de prevalences differentes**, comparer la VPP par strate, mesurer la VPP que l'on obtient si l'on **ignore la stratification** (depistage de masse indifferencie), puis en deduire une politique de depistage.

### Consigne

Un test unique (memes caracteristiques partout) est envisage pour un depistage :

- **Sensibilite** : 95 %
- **Specificite** : 96 %

La population se divise en deux groupes que l'on peut identifier **avant** de tester :

- **Groupe a faible risque** : taux de base = **0,2 %** (0,002)
- **Groupe a haut risque** : taux de base = **8 %** (0,08)

On dispose de **10 000 personnes dans chaque groupe** (20 000 au total).

Travail demande :

1. Pour le **groupe a faible risque** : tableau (VP, FP) et VPP.
2. Pour le **groupe a haut risque** : tableau (VP, FP) et VPP.
3. **Depistage indifferencie** : si l'on teste tout le monde et qu'on melange les deux groupes sans distinguer le risque, quelle est la VPP globale ? (Sommez VP et FP des deux strates.)
4. **Decision** : comparez les trois VPP. Quelle politique recommandez-vous ? Quel est le cout d'un depistage de masse pour le groupe a faible risque, exprime en faux positifs par vrai positif ?

*Conseil : verifiez chaque strate independamment avec `domains/vie/rationalite-decision/02-code/02-probabilites-utiles.py` (une execution par prevalence).*

### Criteres de reussite

- [ ] Groupe faible risque : malades = 20, sains = 9 980 ; VP = 19, FP ≈ 399 ; VPP ≈ 19 / 418 ≈ **4,5 %**
- [ ] Groupe haut risque : malades = 800, sains = 9 200 ; VP = 760, FP = 368 ; VPP = 760 / 1 128 ≈ **67,4 %**
- [ ] Depistage indifferencie : VP total = 779, FP total ≈ 767 ; VPP globale ≈ 779 / 1 546 ≈ **50,4 %** (prevalence melangee = 4,1 %)
- [ ] La comparaison est faite : la VPP varie de ~4,5 % a ~67 % avec le **meme test**, uniquement par le taux de base
- [ ] La politique est argumentee : tester en priorite le groupe a haut risque ; dans le groupe a faible risque, on obtient environ **21 faux positifs pour 1 vrai positif** (399 / 19), ce qui rend le depistage de masse couteux et peu informatif
- [ ] La conclusion methodologique est nommee : agreger des sous-populations de risques tres differents masque l'information ; la VPP globale (50 %) ne represente fidelement aucun des deux groupes

---

## Exercice 2 : Transfert — detecteur de fraude sur des transactions

### Objectif

Transposer le raisonnement "faux positifs / taux de base" hors du contexte medical, sur un detecteur d'anomalies, derouler le tableau, calculer VPP et VPN, puis raisonner qualitativement sur l'arbitrage cout des faux positifs / cout des faux negatifs.

### Consigne

Une plateforme de paiement deploie un **detecteur de fraude** qui leve une alerte sur les transactions suspectes :

- **Sensibilite** : 92 % (il detecte 92 % des transactions reellement frauduleuses)
- **Specificite** : 99 % (il laisse passer correctement 99 % des transactions legitimes)
- **Taux de base de la fraude** : **0,3 %** des transactions (0,003)

On analyse un volume de **1 000 000 de transactions**.

Travail demande :

1. Derivez le tableau complet en effectifs : transactions frauduleuses vs legitimes, puis VP, FP, VN, FN.
2. Calculez la **VPP** : quand le detecteur leve une alerte, quelle est la probabilite que la transaction soit reellement frauduleuse ?
3. Calculez la **VPN** : quand il ne dit rien, quelle est la probabilite que la transaction soit reellement legitime ?
4. **Decision** : chaque alerte declenche un blocage (transaction gelee, client recontacte). Discutez qualitativement l'arbitrage entre le cout des **faux positifs** (9 970 transactions legitimes bloquees) et celui des **faux negatifs** (240 fraudes ratees). Quel levier (sensibilite ? specificite ? taux de base via filtrage prealable ?) ameliorerait le plus la VPP ?

*Conseil : la structure du calcul est identique a un test medical — `02-code/02-probabilites-utiles.py` accepte n'importe quelle paire sensibilite/specificite/prevalence.*

### Criteres de reussite

- [ ] Effectifs de base : frauduleuses = 3 000, legitimes = 997 000
- [ ] Tableau : VP = 2 760, FN = 240, VN = 987 030, FP = 9 970
- [ ] VPP = 2 760 / (2 760 + 9 970) = 2 760 / 12 730 ≈ **21,7 %** : ~78 % des alertes sont de fausses alertes
- [ ] VPN = 987 030 / (987 030 + 240) ≈ **99,98 %** : une transaction non signalee est quasi certainement legitime
- [ ] L'arbitrage est analyse : un faux positif gene un client legitime (friction, perte de confiance) ; un faux negatif laisse passer une fraude (perte directe) — le bon seuil depend du cout relatif des deux erreurs
- [ ] Le levier dominant est identifie : a 0,3 % de prevalence, c'est la **specificite** (et/ou un pre-filtrage qui releve le taux de base avant le detecteur) qui pese le plus sur la VPP ; gagner en sensibilite change surtout les faux negatifs, pas la VPP
