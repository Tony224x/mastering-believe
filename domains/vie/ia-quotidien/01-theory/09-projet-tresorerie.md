# Module 09 — Projet secondaire : trésorerie / budget PME (fictif)

> **Temps estimé** : 60 min | **Prérequis** : Modules 06–08
>
> **Objectif** : Livrer un classeur **Budget PME Demo** réutilisable (transactions, résumé, 3 scénarios), 100 % fictif, documenté.

---

![Schéma du module : Quatre onglets, zéro donnée réelle d'employeur.](../assets/09-classeur-onglets.svg)

> **En une phrase :** Quatre onglets, zéro donnée réelle d'employeur.


```mermaid
flowchart TB
  subgraph Classeur["Budget-PME-Demo.xlsx"]
    T[Transactions]
    R[Resume]
    S[Scenarios]
    M[Readme]
  end
  T --> R
  R --> S
  T -.-> M
```

> **Visuel mental** : 4 onglets, zéro donnée réelle d'employeur.


## 1. Scène concrète : le livrable « portfolio Excel »

À la fin de l'heure, tu as un fichier que tu peux montrer (école, entretien, ou à toi-même) :

**`Budget-PME-Demo.xlsx`**

- Onglet `Transactions` (20–40 lignes inventées)
- Onglet `Résumé` (totaux, solde, % de dépenses)
- Onglet `Scénarios` (base / optimiste / pessimiste)
- Onglet `Readme` (5 lignes : hypothèse, outils, ce que l'IA a fait, ce que tu as vérifié)

Ce n'est **pas** le budget de ton employeur. [CNIL IA] [NIST AI RMF]

> **À retenir :** Un projet fictif bien fait enseigne le geste pro **sans** exposer de données sensibles.

---

## 2. Cahier des charges (acceptance)

- [ ] Au moins 20 transactions fictives cohérentes (dates sur 1–2 mois)
- [ ] Catégories claires (loyer, salaires, marketing, fournitures, ventes…)
- [ ] Formules de totaux (pas de totaux tapés à la main)
- [ ] Solde = entrées − sorties
- [ ] 3 scénarios documentés (ex. +10 % ventes / −10 % ventes)
- [ ] Aucune donnée réelle identifiable
- [ ] Notes « généré avec aide ChatGPT » + liste des formules clés

---

## 3. Session guidée avec ChatGPT (45 min)

**Bloc A (10 min)** — Générer le jeu de données
```
Crée un jeu CSV fictif de 25 transactions pour une PME de café mobile en ville (janv. 2026).
Colonnes: date, libelle, categorie, montant, type(entree/sortie).
Montants réalistes mais inventés. Pas de vrais noms de personnes.
```

**Bloc B (15 min)** — Formules Résumé
(reprendre prompts J6–J7) [Microsoft formulas overview]

**Bloc C (10 min)** — Scénarios
```
Explique comment modéliser 3 scénarios en gardant les transactions fixes et en appliquant des % sur un onglet Scenarios. Formules FR.
```

**Bloc D (10 min)** — Readme humain
Tu écris 5 lignes **sans** IA, puis tu demandes une relecture de clarté seulement.

---

## 4. Pièges

- Scénario « optimiste » avec chiffres magiques non reliés aux formules
- Catégories en double (`Marketing` / `marketing`)
- Coller un vrai extrait bancaire « pour gagner du temps » → **interdit**

---

## Spaced repetition

**Q1.** Pourquoi imposer le fictif ici ?
**R1.** Confidentialité + éthique + apprentissage transférable. [CNIL IA]

**Q2.** Quels onglets minimum dans le livrable ?
**R2.** Transactions, Résumé, Scénarios, Readme (ou équivalent).

**Q3.** Que doit être calculé par formule ?
**R3.** Totaux et solde (pas saisis manuellement).

**Q4.** Que mettre dans le Readme ?
**R4.** Hypothèses, rôle de l'IA, vérifications faites.

**Q5.** Lien avec le capstone PPT ?
**R5.** Les insights du budget peuvent illustrer 1–2 slides du pitch formation (toujours fictif).
