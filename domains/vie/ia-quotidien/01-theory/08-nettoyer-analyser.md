# Module 08 — Nettoyer, analyser, visualiser

> **Temps estime** : 45 min | **Prerequis** : Modules 06–07
>
> **Objectif** : Passer d'une liste "sale" fictive a un tableau propre + un graphique sobre, avec l'aide de ChatGPT pour la methode (pas pour inventer des chiffres).

---

## 1. Scene concrete : l'export "crade"

Tu recois (fictif) une liste :

```
12/01/2026; loyer ; 1200 ; sortie
2026-01-15; Cafe equipe; 45,5; Sortie
...
```

Problemes : dates melangees, espaces, majuscules inconsistantes, virgules/points.

**Role de l'IA :** te donner une **checklist de nettoyage** et des formules (`SUPPRESPACE`, `MAJUSCULE`, `DATEVALUE` selon locale) — pas inventer des lignes manquantes.

> **Key takeaway :** Garbage in, garbage out. Nettoyer **avant** d'analyser.

---

## 2. Checklist de nettoyage (a coller dans ChatGPT)

```
Voici 15 lignes echantillon (fictives).
1) Liste les problemes de qualite.
2) Propose un ordre de nettoyage en 6 etapes dans Excel (sans Power Query d'abord).
3) Donne les formules FR utiles pour espaces et casse.
Ne complete pas les montants manquants : signale-les.
```

---

## 3. Analyser sans se noyer

Questions utiles (Few : commencer par la question metier) [Few, 2012] :

1. Combien sort / entre ce mois ?
2. Quelle categorie domine les sorties ?
3. Y a-t-il des doublons de libelles ?

---

## 4. Visualiser avec sobriete

| A faire | A eviter |
|---------|----------|
| 1 graphique = 1 message | 3D, arcs-en-ciel |
| Barres pour comparer categories | Camembert a 12 parts |
| Titre qui dit la conclusion | Titre "Graphique 1" |

Prompt :

```
J'ai categories en A et totaux en B. Quel type de graphique recommander et pourquoi (max 5 phrases) ? Style sobre type Stephen Few.
```

---

## 5. Livrable du jour

- Tableau propre (fictif) 
- 1 graphique 
- 3 puces d'insight **ecrites par toi** (l'IA peut proposer, tu valides)

---

## Spaced repetition

**Q1.** Pourquoi ne pas laisser l'IA "completer" les trous de montants ?
**R1.** Risque d'invention ; les trous doivent rester visibles.

**Q2.** Donne 2 symptomes de donnees sales.
**R2.** Dates multi-formats ; categories avec casses differentes.

**Q3.** Quel graphique pour comparer 5 categories de depenses ?
**R3.** Barres (plutot qu'un camembert surcharge).

**Q4.** Que doit exprimer le titre du graphique ?
**R4.** Le message / la conclusion, pas un numero.

**Q5.** Reference design de donnees citee ?
**R5.** Stephen Few, Show Me the Numbers. [Few, 2012]
