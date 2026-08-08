# Module 07 — Formules, tableaux croises & modeles

> **Temps estime** : 45–60 min | **Prerequis** : Module 06
>
> **Objectif** : Construire un mini-modele (recettes / depenses / solde) avec formules expliquees, pret pour le projet tresorerie de J9.

---

## 1. Scene concrete : le modele "serviette de table"

Sur une serviette : Recettes − Depenses = Solde. En Excel, le meme modele devient :

- une feuille **Transactions** ;
- une feuille **Resume** avec totaux et % ;
- plus tard : 3 scenarios (base / optimiste / pessimiste).

Few (2012) insiste : un tableau clair bat un graphique joli mais trompeur. Commence par des **nombres lisibles**. [Few, 2012]

> **Key takeaway :** Un bon modele Excel est d'abord **lisible par un humain** (toi dans 2 semaines), pas "impressionnant".

---

## 2. Formules a savoir demander (FR)

| Besoin | Famille | Exemple d'intention pour l'IA |
|--------|---------|-------------------------------|
| Total simple | `SOMME` | Somme de D2:D100 |
| Total conditionnel | `SOMME.SI` / `SOMME.SI.ENS` | Total ou Type="sortie" |
| Compter | `NB.SI` | Nombre de transactions "Loyer" |
| Recherche | `RECHERCHEX` / `RECHERCHEV` | Trouver le budget d'une categorie |
| % | Division + format % | Depense cat / total depenses |

Demande toujours : *formule + explication + cas de test*. [Microsoft formulas overview]

---

## 3. Tableaux structures

Dans Excel, convertir la plage en **Tableau** (Insertion > Tableau) aide les references (`[@Montant]`). Prompt :

```
Explique comment ecrire une colonne calculee "Sens" qui affiche +Montant si Type=entree et -Montant si sortie, dans un Tableau Excel nomme Transactions.
```

---

## 4. Mini-modele du jour (a construire)

**Feuille Transactions** (10 lignes fictives minimum)  
**Feuille Resume** :

- Total entrees  
- Total sorties  
- Solde  
- Top categorie de depense (tu peux la calculer a la main si la formule est trop avancee — l'IA t'aide a progresser)

Prompt d'assemblage :

```
Voici mon schema : [..]. Propose la structure Resume (cellules B2:B5) et les formules exactes FR. Liste 3 erreurs que je risque en copiant.
```

---

## 5. Lien vers J9

J9 = ce modele + scenarios + resume 5 lignes pour un **Budget PME Demo** 100 % fictif.

---

## Spaced repetition

**Q1.** Quelle formule pour totaliser sous condition ?
**R1.** `SOMME.SI` ou `SOMME.SI.ENS` (FR).

**Q2.** Pourquoi separer Transactions et Resume ?
**R2.** Donnees brutes vs indicateurs ; plus facile a auditer et etendre.

**Q3.** Que demander en plus de la formule ?
**R3.** Explication + test numerique attendu.

**Q4.** Quel principe Few pour la clarte ?
**R4.** Prioriser la lisibilite des nombres / eviter la decoration inutile. [Few, 2012]

**Q5.** A quoi sert un Tableau Excel structure ?
**R5.** References stables et colonnes calculees plus robustes.
