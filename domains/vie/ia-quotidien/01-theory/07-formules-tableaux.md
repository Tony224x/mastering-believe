# Module 07 — Formules, tableaux croisés & modèles

> **Temps estimé** : 45–60 min | **Prérequis** : Module 06
>
> **Objectif** : Construire un mini-modèle (recettes / dépenses / solde) avec formules expliquées, prêt pour le projet trésorerie de J9.

---

![Schéma du module : Chaque formule se découpe en morceaux compréhensibles.](../assets/07-formule-expliquee.png)

> **En une phrase :** Chaque formule se découpe en morceaux compréhensibles.


## 1. Scène concrète : le modèle « serviette de table »

Sur une serviette : Recettes − Dépenses = Solde. En Excel, le même modèle devient :

- une feuille **Transactions** ;
- une feuille **Résumé** avec totaux et % ;
- plus tard : 3 scénarios (base / optimiste / pessimiste).

Few (2012) insiste : un tableau clair bat un graphique joli mais trompeur. Commence par des **nombres lisibles**. [Few, 2012]

> **À retenir :** Un bon modèle Excel est d'abord **lisible par un humain** (toi dans 2 semaines), pas « impressionnant ».

---

## 2. Formules à savoir demander (FR)

| Besoin | Famille | Exemple d'intention pour l'IA |
|--------|---------|-------------------------------|
| Total simple | `SOMME` | Somme de D2:D100 |
| Total conditionnel | `SOMME.SI` / `SOMME.SI.ENS` | Total où Type="sortie" |
| Compter | `NB.SI` | Nombre de transactions « Loyer » |
| Recherche | `RECHERCHEX` / `RECHERCHEV` | Trouver le budget d'une catégorie |
| % | Division + format % | Dépense cat / total dépenses |

Demande toujours : *formule + explication + cas de test*. [Microsoft formulas overview]

---

## 3. Tableaux structurés

Dans Excel, convertir la plage en **Tableau** (Insertion > Tableau) aide les références (`[@Montant]`). Prompt :

```
Explique comment écrire une colonne calculée "Sens" qui affiche +Montant si Type=entrée et -Montant si sortie, dans un Tableau Excel nommé Transactions.
```

---

## 4. Mini-modèle du jour (à construire)

**Feuille Transactions** (10 lignes fictives minimum)  
**Feuille Résumé** :

- Total entrées  
- Total sorties  
- Solde  
- Top catégorie de dépense (tu peux la calculer à la main si la formule est trop avancée — l'IA t'aide à progresser)

Prompt d'assemblage :

```
Voici mon schéma : [..]. Propose la structure Résumé (cellules B2:B5) et les formules exactes FR. Liste 3 erreurs que je risque en copiant.
```

---

## 5. Lien vers J9

J9 = ce modèle + scénarios + résumé 5 lignes pour un **Budget PME Demo** 100 % fictif.

---

## Spaced repetition

**Q1.** Quelle formule pour totaliser sous condition ?
**R1.** `SOMME.SI` ou `SOMME.SI.ENS` (FR).

**Q2.** Pourquoi séparer Transactions et Résumé ?
**R2.** Données brutes vs indicateurs ; plus facile à auditer et étendre.

**Q3.** Que demander en plus de la formule ?
**R3.** Explication + test numérique attendu.

**Q4.** Quel principe Few pour la clarté ?
**R4.** Prioriser la lisibilité des nombres / éviter la décoration inutile. [Few, 2012]

**Q5.** À quoi sert un Tableau Excel structuré ?
**R5.** Références stables et colonnes calculées qui tiennent quand tu ajoutes des lignes.

<!-- NAV:START -->
---

← [Module 06 — Excel + IA : les bases](./06-excel-bases-ia.md) · [Index des chapitres](../README.md#index-des-chapitres-cliquable) · [Mission easy](../03-exercises/01-easy/07-formules-tableaux.md) · [Progression](../PROGRESS.md) · [Module 08 — Nettoyer, analyser, visualiser](./08-nettoyer-analyser.md) →

<!-- NAV:END -->
