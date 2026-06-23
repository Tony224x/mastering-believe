# Exercices — Independance financiere et retrait soutenable (Module 12)

> **Niveau** : Debutant | **Temps estime** : 45-60 min
>
> **Matiere premiere** : Theorie du Module 12 + simulateur `02-code/12-independance-financiere.py`
>
> **Disclaimer** : exercices educatifs. Les hypotheses (rendement, taux de retrait) sont illustratives et ne garantissent rien. Tout investissement comporte un risque de perte en capital.

---

## Exercice 1 — Capital cible et retrait selon la regle des 4 %

### Objectif
Maitriser les deux calculs de base : capital cible a partir des depenses, et retrait soutenable a partir d'un capital.

### Consigne

1. Julie estime ses depenses annuelles "en independance" a **28 000 €/an**. Calculez son **capital cible** selon la regle des 4 % (rappel : cible = depenses / 0,04 = depenses × 25).
2. Marc a accumule **600 000 €** investis. Calculez le **retrait annuel** et le **retrait mensuel** que la regle des 4 % autorise la premiere annee.
3. Verifiez vos deux resultats avec le script `02-code/12-independance-financiere.py` (fonctions `capital_cible()` et `montant_retrait()`).
4. En une phrase : la regle des 4 % suppose que le retrait est ajuste chaque annee. Ajuste de quoi, et pourquoi ?

### Criteres de reussite

- [ ] Le capital cible de Julie est correct (28 000 × 25)
- [ ] Le retrait annuel ET mensuel de Marc sont corrects (600 000 × 0,04, puis /12)
- [ ] Le script confirme les deux resultats (sortie visible)
- [ ] La reponse 4 mentionne l'ajustement **sur l'inflation** (maintien du pouvoir d'achat)

---

## Exercice 2 — L'effet (non lineaire) du taux d'epargne

### Objectif
Constater que le taux d'epargne agit sur deux leviers a la fois et raccourcit l'horizon plus vite qu'une regle de trois.

### Consigne

A l'aide du simulateur (`annees_jusqu_independance(taux_epargne, rendement_reel=0.05, taux_retrait=0.04)`), remplissez le tableau pour un depart de zero :

| Taux d'epargne | Annees jusqu'a l'independance |
|---|---|
| 20 % | ? |
| 40 % | ? |
| 60 % | ? |

Puis repondez :
1. Passer de 20 % a 40 % d'epargne (×2) divise-t-il l'horizon par exactement 2 ? Par plus ? Par moins ?
2. Expliquez en deux phrases le **double effet** du taux d'epargne qui produit ce resultat (pensez : ce qu'on accumule **et** la taille de la cible).
3. Pourquoi, dans ce modele, le **niveau du salaire** ne change-t-il PAS le nombre d'annees (a taux d'epargne donne) ? (indice : voir la note "pourquoi le revenu s'annule" dans le script)

### Criteres de reussite

- [ ] Les trois durees sont relevees depuis le script (tolerance ±0,5 an)
- [ ] La reponse 1 constate que l'horizon est divise par **plus** que 2
- [ ] La reponse 2 cite les deux effets : accumulation plus rapide ET cible plus basse
- [ ] La reponse 3 explique que le revenu apparait au numerateur (cible) et au denominateur (versement) et se simplifie

---

## Exercice 3 — Critiquer la regle des 4 % (ses limites)

### Objectif
Savoir accompagner le chiffre "4 %" de ses limites, sans le presenter comme une garantie.

### Consigne

Pour chacune des situations suivantes, indiquez **quelle limite** de la regle des 4 % est en jeu (sequence des rendements / horizon trop long / donnees US uniquement / fiscalite et frais / absence de flexibilite) et expliquez en une ou deux phrases.

1. Sofia vise l'independance a **42 ans** et compte vivre jusqu'a **92 ans** (50 ans de retrait). Elle applique 4 % sans ajustement.
2. Hugo part en retrait juste avant un **krach severe** dans ses deux premieres annees ; son voisin, parti deux ans plus tot, a traverse le meme krach en milieu de parcours. Meme rendement moyen sur la periode, issues tres differentes.
3. Lina a calcule sa cible "depenses × 25" mais a oublie que ses retraits seront **imposes** et que son portefeuille a des **frais de gestion**.

Puis, en conclusion (2-3 phrases) : citez **un** mecanisme concret qui ameliore la robustesse d'un plan de retrait face a ces limites.

### Criteres de reussite

- [ ] Situation 1 → "horizon trop long" (calibre pour ~30 ans ; viser plutot 3-3,5 %)
- [ ] Situation 2 → "sequence des rendements" (un krach en debut de retrait est bien plus destructeur, a moyenne egale)
- [ ] Situation 3 → "fiscalite et frais non inclus" (reduisent le taux de retrait soutenable reel)
- [ ] La conclusion propose un mecanisme valable (flexibilite des depenses, taux de retrait plus prudent 3-3,5 %, revenu complementaire, reserve de liquidites pour ne pas vendre en bas)
