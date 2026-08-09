# Module 09 — Projet secondaire : trésorerie / budget PME (fictif)

> **Temps estimé** : 60 min | **Prérequis** : Modules 06–08
>
> **Objectif** : Livrer un classeur **Budget PME Demo** réutilisable (transactions, résumé, 3 scénarios), 100 % fictif, documente.

---

![Schéma du module : Quatre onglets, zero donnée réelle d employeur.](../assets/09-classeur-onglets.svg)

> **En une phrase :** Quatre onglets, zero donnée réelle d employeur.


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

> **Visuel mental** : 4 onglets, zero donnée réelle d'employeur.


## 1. Scène concrete : le livrable "portfolio Excel"

À la fin de l'heure, tu as un fichier que tu peux montrer (école, entretien, ou a toi-même) :

**`Budget-PME-Demo.xlsx`**

- Onglet `Transactions` (20–40 lignes inventées)
- Onglet `Résumé` (totaux, solde, % depenses)
- Onglet `Scénarios` (base / optimiste / pessimiste)
- Onglet `Readme` (5 lignes : hypothese, outils, ce que l'IA a fait, ce que tu as verifie)

Ce n'est **pas** le budget de ton employeur. [CNIL IA] [NIST AI RMF]

> **À retenir :** Un projet fictif bien fait enseigne le geste pro **sans** exposer de données sensibles.

---

## 2. Cahier des chargés (acceptance)

- [ ] Au moins 20 transactions fictives cohérentes (dates sur 1–2 mois)
- [ ] Catégories claires (loyer, salaires, marketing, fournitures, ventes…)
- [ ] Formules de totaux (pas de totaux tapes à la main)
- [ ] Solde = entrées − sorties
- [ ] 3 scénarios documentes (ex. +10 % ventes / −10 % ventes)
- [ ] Aucune donnée réelle identifiable
- [ ] Notes "genere avec aide ChatGPT" + liste des formules cles

---

## 3. Session guidée avec ChatGPT (45 min)

**Bloc A (10 min)** — Generer le jeu de données 
```
Cree un jeu CSV fictif de 25 transactions pour une PME de cafe mobile en ville (janv. 2026).
Colonnes: date, libelle, categorie, montant, type(entree/sortie).
Montants realistes mais inventes. Pas de vrais noms de personnes.
```

**Bloc B (15 min)** — Formules Résumé 
(reprendre prompts J6–J7) [Microsoft formulas overview]

**Bloc C (10 min)** — Scénarios 
```
Explique comment modeliser 3 scenarios en gardant les transactions fixes et en appliquant des % sur un onglet Scenarios. Formules FR.
```

**Bloc D (10 min)** — Readme humain 
Tu ecris 5 lignes **sans** IA, puis tu demandes une relecture de clarté seulement.

---

## 4. Pièges

- Scénario "optimiste" avec chiffres magiques non relies aux formules 
- Catégories en double (`Marketing` / `marketing`) 
- Coller un vrai extrait bancaire "pour gagner du temps" → **interdit**

---

## Spaced repetition

**Q1.** Pourquoi imposer le fictif ici ?
**R1.** Confidentialite + éthique + apprentissage transferable. [CNIL IA]

**Q2.** Quels onglets minimum dans le livrable ?
**R2.** Transactions, Résumé, Scénarios, Readme (ou equivalent).

**Q3.** Que doit être calcule par formule ?
**R3.** Totaux et solde (pas saisis manuellement).

**Q4.** Que mettre dans le Readme ?
**R4.** Hypotheses, rôle de l'IA, vérifications faites.

**Q5.** Lien avec le capstone PPT ?
**R5.** Les insights du budget peuvent illustrer 1–2 slides du pitch formation (toujours fictif).
