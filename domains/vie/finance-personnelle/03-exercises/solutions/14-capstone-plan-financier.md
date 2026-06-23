# Solutions — Module 14 : Capstone, gabarit de plan financier

> ⚠️ Contenu educatif. Les chiffres ci-dessous decoulent des hypotheses du gabarit (rendement et inflation ajustables, NON garantis). Ce ne sont pas des recommandations. Vos valeurs exactes peuvent legerement varier selon l'arrondi.

> Profil fictif de l'exercice 1 : revenus 2 600 €, dep. essentielles 1 500 €, non essentielles 400 €, dette 150 €/mois ; fonds d'urgence 800 € / cible 6 mois ; versement investi 350 €/mois ; capital deja investi 2 000 €. Allocation et hypotheses par defaut du fichier (60/20/20 ; rdt bruts 7/8/3 % ; TER 0,20 % ; inflation 2 %).

---

## Exercice 1 — Personnaliser le gabarit

**1. Taux d'epargne** : epargne = 2 600 − (1 500 + 400 + 150) = **550 €/mois**, soit **~21,2 %** des revenus nets.

**2. Fonds d'urgence** : cible = 1 500 € × 6 = **9 000 €** ; reserve actuelle 800 € → **manque 8 200 €**.

**3. Projection a 30 ans** (rendement net hypothetique 6,20 %/an, inflation 2 %) :
- Capital **nominal** : ~**365 800 €**
- Capital **reel** (pouvoir d'achat d'aujourd'hui) : ~**201 900 €**
- Le reel est inferieur au nominal d'environ **45 %**. C'est tout l'enjeu du bloc 5 : sur 30 ans, l'inflation absorbe pres de la moitie du chiffre nominal.

**4. Allocation** : **non, le programme ne suggere aucune allocation.** Le bloc 4 affiche les proportions **que vous avez saisies** (60/20/20 ici) et en deduit un rendement net hypothetique. Les % sont des **entrees**, pas une sortie. Si vous changez les proportions, le rendement net change — mais le programme ne dit jamais "mettez 60 % en actions".

---

## Exercice 2 — Tester un levier a la fois

Capital **reel** a 30 ans selon la variation (base = ~201 900 €) :

| Variation | Changement | Capital reel a 30 ans | Sens / ampleur |
|---|---|---|---|
| **Base** | — | ~201 900 € | reference |
| **A** | versement 350 → 500 € | ~**285 600 €** | **monte** fortement (+~42 %) |
| **B** | inflation 2 % → 4 % | ~**112 800 €** | **baisse** fortement (−~44 %) |
| **C** | rdt brut −1 pt/poche (net 6,2 % → 5,2 %) | ~**168 300 €** | **baisse** (−~17 %) |

**1.** A monte, B baisse, C baisse — sens conformes a l'intuition.

**2.** Dans CE cas, **l'inflation (B)** et **le versement (A)** ont les effets les plus marquants sur le reel, devant la variation de rendement (C). Lecon : l'inflation n'est pas un detail cosmetique — elle pese autant qu'un changement majeur d'effort d'epargne. Et le levier que vous controlez le mieux (votre versement) a un impact direct et puissant. (L'ordre exact des leviers depend des amplitudes choisies ; l'important est de l'avoir mesure, pas suppose.)

**3.** Changer **une seule variable a la fois** permet d'**attribuer l'effet a sa cause**. Si on bouge tout en meme temps, on ne sait plus quel levier a produit quel resultat — c'est la base de toute experimentation propre (deliberate practice).

---

## Exercice 3 — Lire la cible FIRE et ses limites

**1. Cible de capital** : depenses annuelles (hors dette) = (1 500 + 400) × 12 = 22 800 € ; cible = 22 800 / 0,04 = **570 000 €**. Le facteur vient de la regle des 4 % : depenses / 0,04 = **25 × depenses annuelles** (1/0,04 = 25).

**2. Horizon estime** : ~**37 ans** avec ces hypotheses (350 €/mois, 2 000 € de depart, rendement net 6,2 %).

**3. Les deux limites imprimees par le programme** :
- La regle des 4 % suppose un **horizon ~30 ans** et un **retrait indexe sur l'inflation**.
- Elle **ne couvre pas le risque de sequence de rendements** (l'ordre dans lequel arrivent les bonnes/mauvaises annees, surtout en debut de retrait).

**4. Repere, pas garantie** : la cible FIRE repose sur des hypotheses moyennes (rendement, inflation) calibrees historiquement sur un horizon donne. La realite peut diverger — notamment via le risque de sequence et un horizon plus long que 30 ans (Module 12). C'est donc un **point de reperage** pour situer un ordre de grandeur, jamais une promesse que le capital tiendra. Un educateur responsable affiche toujours ces limites a cote du chiffre.
