# Exercices — Maitriser la dette et le credit (Module 03)

> **Niveau** : Debutant | **Temps estime** : 45-60 min
>
> **Matiere premiere** : Theorie du Module 03
>
> **Disclaimer** : exercices educatifs avec situations fictives. Les taux sont illustratifs. Ne pas extrapoler en conseil personnalise.

---

## Exercice 1 — Calculer le cout reel d'un credit et comparer des offres

### Objectif
Utiliser le TAEG pour comparer des offres de credit et calculer le cout total reel d'un emprunt.

### Consigne

Paul souhaite financer un ordinateur professionnel de **3 500 €** et compare trois offres :

| Offre | Montant | Duree | Taux nominal | Frais de dossier | Assurance | TAEG communique |
|-------|---------|-------|-------------|-----------------|-----------|-----------------|
| Banque A | 3 500 € | 24 mois | 5,9 % | 0 € | 0 € | 5,9 % |
| Organisme B | 3 500 € | 24 mois | 4,5 % | 150 € | 0 € | 6,8 % |
| Banque en ligne C | 3 500 € | 24 mois | 6,2 % | 0 € | 45 €/an | 7,1 % |

Questions :
1. **Quel est l'indicateur de comparaison valable** entre ces trois offres ? Pourquoi le taux nominal seul est-il insuffisant ?
2. Classez les trois offres du moins cher au plus cher selon le seul indicateur pertinent.
3. Pour l'offre A (TAEG 5,9 %, 24 mois), estimez la **mensualite approximative** en utilisant la formule simplifiee : `M ≈ (P × (r/12)) / (1 - (1 + r/12)^(-n))` avec r = 0.059 et n = 24. Arrondissez au centime.
4. Calculez le **total rembourse** (mensualite × 24) et le **cout total du credit** (total - 3 500 €) pour l'offre A.
5. Paul peut alternativement utiliser ses economies (3 500 € sur un livret a 3 %/an). Quel est le **manque a gagner en interets** sur 24 mois s'il paye comptant plutot qu'avec l'offre A ? Comparez au cout du credit A : payer comptant est-il financierement plus avantageux ici ?

### Criteres de reussite

- [ ] Le TAEG est identifie comme indicateur de comparaison et son avantage vs taux nominal explique
- [ ] Le classement des 3 offres est correct (A < B < C selon le TAEG)
- [ ] La mensualite de l'offre A est calculee correctement (tolerance ±1 €)
- [ ] Le cout total du credit A est calcule et exprime en euros
- [ ] Le manque a gagner sur le livret est calcule et compare au cout du credit (conclusion logique)

---

## Exercice 2 — Choisir une strategie de remboursement de dette

### Objectif
Comparer les methodes Avalanche et Boule de Neige sur un cas concret et justifier un choix.

### Consigne

Amelie a 3 dettes simultanees :

| Dette | Solde | Taux annuel | Mensualite minimum |
|-------|-------|-------------|-------------------|
| Carte credit renouvelable | 1 800 € | 19,5 % | 55 €/mois |
| Credit auto | 8 500 € | 4,2 % | 180 €/mois |
| Pret personnel | 2 200 € | 7,8 % | 65 €/mois |

Elle dispose de **400 €/mois** au total pour rembourser ses dettes (les minimums cumulent a 300 €, il lui reste 100 € de surplus).

**Partie A — Methode Avalanche :**
1. Quelle dette cible-t-on en premier avec la methode Avalanche ? Pourquoi ?
2. Comment repartit-elle ses 400 €/mois au debut (minimums + surplus) ?
3. Estimez en combien de mois la premiere dette sera eliminee (calcul simplifie : solde / paiement mensuel alloue a cette dette, en ignorant les interets pour l'estimation).

**Partie B — Methode Boule de Neige :**
1. Quelle dette cible-t-on en premier avec la methode Boule de Neige ? Pourquoi ?
2. Estimez en combien de mois la premiere dette sera eliminee.
3. Quelle est la motivation psychologique de cette methode ?

**Partie C — Decision :**
1. Laquelle recommandez-vous a Amelie ? Justifiez en considerant a la fois le critere mathematique ET le profil psychologique (supposez qu'Amelie a tente de rembourser ses dettes plusieurs fois sans succes par manque de motivation).
2. Calculez le cout annuel en interets de la carte renouvelable seule (1 800 € a 19,5 %) — quelle urgence cela cree-t-il ?

### Criteres de reussite

- [ ] La cible Avalanche est correctement identifiee (carte renouvelable : taux le plus eleve)
- [ ] La cible Boule de Neige est correctement identifiee (carte renouvelable aussi ici — coincidence ; noter que si les soldes differaient, la reponse changerait)
- [ ] Les estimations de duree sont calculees avec le bon paiement alloue
- [ ] La recommandation est justifiee par le profil d'Amelie (echecs precedents = priorite a la motivation)
- [ ] Le cout annuel de la carte (environ 351 €/an en interets) est calcule

---

## Exercice 3 — Evaluer si s'endetter vaut le coup

### Objectif
Appliquer le concept de cout d'opportunite pour decider si un credit a la consommation est pertinent.

### Consigne

Julien, 32 ans, veut acheter un velo electrique a **2 000 €**. Il a deux options :

**Option 1 — Payer comptant** : utiliser ses 2 000 € d'epargne (actuellement sur un livret a 2,5 %/an).

**Option 2 — Credit a la consommation** : financement propose a TAEG 9,9 % sur 18 mois.

Questions :
1. Calculez la **mensualite approximative** du credit (formule : `M ≈ (P × (r/12)) / (1 - (1 + r/12)^(-n))` avec r = 0.099, n = 18).
2. Calculez le **cout total du credit** (mensualite × 18 - 2 000 €).
3. Si Julien garde ses 2 000 € sur le livret a 2,5 % pendant 18 mois plutot que de payer comptant, combien gagne-t-il en interets ?
4. **Comparaison finale** : combien lui coute reellement le credit en net (cout du credit - interets gagnes sur le livret) ?
5. Julien argue que "pendant les 18 mois de credit, mes 2 000 € continuent de fructifier en bourse a 7 % (hypothese illustrative)". Recalculez le gain potentiel en bourse sur 18 mois (7 %/an pro rata) et comparez au cout du credit. Cela change-t-il la recommandation ? Identifiez la nuance importante (risque vs rendement certain).
6. **Decision** : dans quels cas payer comptant est-il clairement preferable ? Dans quels cas le credit peut-il se justifier ?

### Criteres de reussite

- [ ] La mensualite est calculee correctement (tolerance ±1 €)
- [ ] Le cout total du credit est calcule (montant en euros)
- [ ] Les interets du livret sur 18 mois sont calcules
- [ ] La comparaison nette (cout credit - interets livret) est effectuee
- [ ] La question 5 distingue le rendement hypothetique (bourse, risque) du rendement certain (livret), et nuance la recommandation
- [ ] La question 6 formule une regle de decision generaliste (taux du credit > rendement sans risque disponible = payer comptant ; sinon, le credit peut valoir le coup si les fonds sont places de facon sure ou productive)
