# Exercices Easy — Principes fondamentaux

---

## Exercice 1 : Estimation QPS d'un service de notifications

### Objectif
Pratiquer le calcul back-of-the-envelope pour le QPS et la bande passante.

### Consigne
Un service de notifications push a les caracteristiques suivantes :
- 5 millions de DAU (Daily Active Users)
- Chaque utilisateur recoit en moyenne 8 notifications par jour
- Chaque notification fait 500 octets (payload JSON)
- Le pic de trafic est 4x la moyenne (matin 8h-9h)

Calcule :
1. Le QPS moyen
2. Le QPS pic
3. La bande passante sortante en pic (en Mbps)
4. Le stockage necessaire pour 30 jours de retention

Montre tes calculs etape par etape. Arrondis aux ordres de grandeur.

### Criteres de reussite
- [ ] QPS moyen calcule correctement (~460 req/s)
- [ ] QPS pic calcule avec le bon facteur (~1850 req/s)
- [ ] Bande passante en Mbps (pas en MBps — attention bits vs bytes)
- [ ] Stockage en Go avec retention appliquee

---

## Exercice 2 : CP ou AP — Choisis ton camp

### Objectif
Savoir identifier rapidement si un systeme necessite CP ou AP.

### Consigne
Pour chacun des systemes suivants, indique s'il faut privilegier **CP** ou **AP**, et justifie en une phrase :

1. Un systeme de vote en ligne pour une election nationale
2. Un compteur de "vues" sur des videos YouTube
3. Un service de gestion d'inventaire e-commerce (stock restant)
4. Un feed d'actualites sur un reseau social
5. Un systeme de transfert d'argent entre comptes bancaires
6. Un cache DNS

### Criteres de reussite
- [ ] 6/6 choix corrects
- [ ] Chaque justification mentionne la consequence d'une mauvaise consistance OU d'une indisponibilite
- [ ] Au moins un cas ou tu mentionnes que la reponse depend du contexte exact

---

## Exercice 3 : Les nines en pratique

### Objectif
Comprendre l'impact concret des SLAs sur les operations.

### Consigne
Ton service web a un SLA de 99.95% d'uptime.

1. Calcule le downtime autorise par mois et par jour
2. Tu as une fenetre de maintenance planifiee de 15 minutes chaque dimanche. Est-ce compatible avec ton SLA ? Montre le calcul.
3. Un incident a cause 2 heures de downtime ce mois-ci. Combien de temps de downtime "supplementaire" te reste-t-il pour le mois ?
4. Ton systeme depend de 2 services externes, chacun avec un SLA de 99.99%. Quel est le SLA combine maximum theorique de ton systeme ?

### Criteres de reussite
- [ ] Downtime/mois et /jour calcules correctement
- [ ] Analyse correcte de la compatibilite avec la maintenance
- [ ] Calcul du budget restant apres l'incident
- [ ] SLA combine calcule par multiplication des probabilites
