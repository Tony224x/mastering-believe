# Exercices — Interets composes et valeur du temps (Module 01)

> **Niveau** : Debutant | **Temps estime** : 45-60 min
>
> **Matiere premiere** : Theorie du Module 01 + calculateur `02-code/01-interets-composes.py`
>
> **Disclaimer** : exercices educatifs. Les taux utilises sont illustratifs. Aucun resultat ne constitue une promesse de rendement.

---

## Exercice 1 — Calculer la croissance d'un capital initial

### Objectif
Appliquer la formule des interets composes a un capital initial unique (sans versements reguliers) et comprendre intuitivement l'effet de la duree.

### Consigne

Vous placez **5 000 €** sur un support donnant **5 % de rendement annuel** (capitalisation annuelle). Calculez **a la main** (ou avec la formule, pas encore le script) les montants suivants :

1. Capital apres **5 ans**
2. Capital apres **20 ans**
3. Capital apres **40 ans**

Ensuite :

4. Verifiez vos calculs avec le script `02-code/01-interets-composes.py` en modifiant les parametres de `demo_calcul_de_base()` (ou en appellant `capital_final()` directement depuis un shell Python interactif).
5. Repondez en une phrase : pourquoi le gain entre 20 et 40 ans est-il bien superieur au gain entre 0 et 20 ans, meme si la duree est la meme (20 ans) ?

**Rappel formule** : `A = P × (1 + r)^t` (avec `r` = 0.05, `n` = 1, pas de versements).

### Criteres de reussite

- [ ] Les trois montants sont calcules correctement (tolerance : ±5 € d'arrondi acceptable)
- [ ] Le script Python confirme les resultats (sortie visible)
- [ ] La reponse a la question 5 mentionne explicitement le mot "exponentiel" ou "capital plus grand qui genere plus d'interets"

---

## Exercice 2 — Comparer deux strategies d'epargne

### Objectif
Comprendre que le montant verse compte moins que le moment ou l'on commence.

### Consigne

Comparez les deux strategies suivantes sur une periode allant jusqu'a **65 ans**, avec un taux annuel de **6 %** (illustratif) :

**Strategie A (epargner tot, peu) :**
- Versement de **150 €/mois** de 22 a 32 ans (10 ans)
- Plus aucun versement ensuite — on laisse fructifier jusqu'a 65 ans

**Strategie B (epargner tard, beaucoup) :**
- Versement de **300 €/mois** de 42 a 65 ans (23 ans)

Calculez pour chaque strategie :
1. Le total des sommes versees
2. Le capital a 65 ans (utilisez le script ou la formule)
3. Le rapport capital final / sommes versees

Puis repondez :
- Quelle strategie produit le plus grand capital a 65 ans ?
- Quelle strategie a le meilleur rapport "rendement sur effort verse" ?
- Quel est le "cout de l'attente" de 20 ans (difference de capital entre les deux strategies) ?

### Criteres de reussite

- [ ] Les totaux verses sont calcules exactement (A = 150 × 120 mois ; B = 300 × 276 mois)
- [ ] Les capitaux a 65 ans sont calcules correctement (avec le script ou la formule, tolerance ±200 €)
- [ ] Le rapport capital/verse est compare numeriquement pour les deux strategies
- [ ] La reponse identifie correctement quelle strategie gagne et explique pourquoi

---

## Exercice 3 — La regle des 72 et le cout de l'inaction

### Objectif
Utiliser la regle des 72 pour des estimations rapides et comprendre le cout de remettre a plus tard.

### Consigne

**Partie A — Regle des 72 :**

Pour chacun des taux suivants, estimez la duree de doublement du capital avec la regle des 72, puis calculez la valeur exacte (formule `ln(2) / ln(1+r)`) :
- 3 % (livret conservateur)
- 7 % (portefeuille diversifie, illustratif)
- 12 % (hypothese optimiste, non garantie)

Creez un petit tableau (taux / estimation regle 72 / calcul exact / ecart en %).

**Partie B — Le cout de l'inaction :**

Marie a 30 ans et hesite a commencer a epargner 250 €/mois. Elle se dit "je commencerai dans 5 ans". Calculez :
1. Capital a 65 ans si elle commence a **30 ans** (35 ans de versements, 7 %)
2. Capital a 65 ans si elle commence a **35 ans** (30 ans de versements, 7 %)
3. Le manque a gagner (en euros) lie a ces 5 ans d'attente
4. En combien d'annees supplementaires de versements aurait-elle du compenser ce manque si elle commence a 35 ans ? (estimation, pas de calcul exact requis)

### Criteres de reussite

- [ ] Le tableau de la Partie A est complet avec les trois taux
- [ ] L'ecart regle-des-72 vs exact est calcule en pourcentage pour chaque taux
- [ ] Les deux capitaux de la Partie B sont calcules avec le script ou la formule (tolerance ±500 €)
- [ ] Le manque a gagner est chiffre precisement
- [ ] La reponse a la question 4 donne une estimation raisonnee (pas forcement exacte, mais logiquement argumentee)

---

## Exercice 4 — Du nominal au reel : l'effet de l'inflation

### Objectif
Distinguer rendement nominal et rendement reel, et mesurer l'erosion du pouvoir d'achat — pour comprendre pourquoi fonds d'urgence et regle des 4 % se pensent en reel.

### Consigne

**Partie A — Taux reel :**

Pour chacun des couples (rendement nominal / inflation) suivants, calculez le rendement reel avec l'approximation `nominal − inflation`, puis avec la formule exacte `(1 + nominal) / (1 + inflation) − 1` :
1. Nominal 2 % / inflation 3 % (livret en periode d'inflation)
2. Nominal 7 % / inflation 3 % (portefeuille diversifie, illustratif)

Pour chaque cas, indiquez si le pouvoir d'achat **augmente ou diminue**.

**Partie B — Erosion d'un cash dormant :**

Vous laissez **15 000 €** sur un compte non remunere (0 %). L'inflation est de **3 %/an**. Calculez le **pouvoir d'achat reel** de cette somme (en euros d'aujourd'hui) apres 10 ans puis 25 ans, avec `montant / (1 + inflation)^annees`.

**Partie C — Raisonnement :**

3. Verifiez vos resultats avec `demo_rendement_reel()` du script `02-code/01-interets-composes.py`.
4. En une phrase : pourquoi vaut-il mieux dimensionner un fonds d'urgence en **mois de depenses** plutot qu'en montant fixe en euros ?

### Criteres de reussite

- [ ] Les taux reels sont calcules par les deux methodes (approximation et exact) pour les deux couples
- [ ] La direction du pouvoir d'achat (hausse/baisse) est correcte pour chaque cas (cas 1 = baisse, cas 2 = hausse)
- [ ] Les deux pouvoirs d'achat de la Partie B sont calcules (tolerance ±50 €)
- [ ] La reponse a la question 4 relie explicitement le fonds d'urgence a la hausse des prix (les depenses augmentent avec l'inflation)
