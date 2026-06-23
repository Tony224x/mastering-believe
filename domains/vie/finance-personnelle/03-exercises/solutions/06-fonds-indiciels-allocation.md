# Solutions — Fonds indiciels et allocation (Module 06)

> **Disclaimer** : solutions educatives. Aucun produit, emetteur ou ticker n'est recommande. Chiffres illustratifs ; tout investissement comporte un risque de perte en capital.

---

## Exercice 1 — Chiffrer l'impact compose des frais

Valeurs relevees de la **DEMO 1** (10 000 € initial, 200 €/mois, 7 % brut, 30 ans) :

| TER | Capital net final |
|---|---|
| 0,1 % | ~303 642 € |
| 0,5 % | ~279 549 € |
| 1,0 % | ~252 338 € |

1. Voir le tableau ci-dessus.

2. Le cas **1,0 %** est en retard de **~51 304 €** sur le cas 0,1 % (303 642 − 252 338), soit **~16,9 %** de capital final en moins (51 304 / 303 642).

3. Versements annuels = 200 €/mois × 12 = **2 400 €/an**. Surcout des frais ≈ 51 304 €. Soit 51 304 / 2 400 ≈ **~21 ans de versements** "envoles" en frais — alors que les versements totaux ne couvrent que 30 ans. Autrement dit, environ 0,9 point de frais en plus engloutit l'equivalent de plus de deux tiers de l'effort d'epargne.

4. On ne controle pas le rendement futur des marches (il est incertain et hors de notre pouvoir), mais on peut **comparer et choisir le TER** avant d'investir : c'est une variable connue a l'avance et directement actionnable. D'ou "le levier le plus directement sous votre controle".

---

## Exercice 2 — Construire et ajuster une allocation "3 fonds"

1. Allocation de base **40 / 40 / 20** (rendements esperes 7 % / 7 % / 3 %, TER 0,15 %) :
   - Rendement brut pondere = 0,40 × 7 % + 0,40 × 7 % + 0,20 × 3 % = **6,20 %**
   - Rendement espere **net** = 6,20 % − 0,15 % = **~6,05 %/an** (conforme a la sortie du script).

2. Allocation **prudente 30 / 20 / 50** : brut pondere = 0,30 × 7 % + 0,20 × 7 % + 0,50 × 3 % = 2,10 + 1,40 + 1,50 = **5,00 %** ; net ~4,85 %. Le rendement espere **baisse**, car on remplace des actions (7 %) par des obligations (3 %) : moins de moteur de croissance, mais moins de volatilite (cf. Module 05).

3. Allocation **dynamique 50 / 40 / 10** : brut pondere = 0,50 × 7 % + 0,40 × 7 % + 0,10 × 3 % = 3,50 + 2,80 + 0,30 = **6,60 %** ; net ~6,45 %. Le rendement espere **monte**, en contrepartie d'une volatilite plus elevee.

4. Avec **50 / 40 / 20** (somme = 110 %), l'instruction `assert abs(total - 1.0) < 1e-9` echoue et le script s'arrete avec une `AssertionError` ("Les poids doivent sommer a 100 %"). Ce garde-fou est utile car une allocation qui ne somme pas a 100 % n'a pas de sens (on investirait plus ou moins que le capital reel) : il attrape une erreur de saisie avant tout calcul errone.

5. L'allocation **prudente (30 / 20 / 50)** convient mieux a un **horizon de 3 ans** (peu de temps pour absorber une baisse → priorite a la stabilite). L'allocation **dynamique (50 / 40 / 10)** convient mieux a un **horizon de 30 ans** (le temps permet d'encaisser la volatilite des actions). C'est le couple actif + horizon du Module 05.

> Rappel : ces poids sont des illustrations. Aucune allocation n'est prescrite ; chacun ajuste selon sa situation.

---

## Exercice 3 — Lire SPIVA honnetement

1. **Argument arithmetique (Sharpe)** : pris dans leur ensemble, les gerants actifs detiennent collectivement le marche. Leur rendement moyen *avant frais* est donc, par construction, egal au rendement du marche. Comme la gestion active coute plus cher (frais, rotation), *apres frais* leur rendement moyen est **necessairement inferieur** a celui de l'indice. Ce n'est pas un jugement de valeur, c'est de l'arithmetique.

2. **Nuance methodologique** : le chiffre ~90 %+ est **equipondere** — il compte les **fonds** (chaque fonds compte pour 1, gros ou petit). Si l'on pondere plutot par les **encours** (l'argent reellement investi), la sous-performance apparait **moins prononcee** (une partie des capitaux se concentre dans des fonds qui s'en sortent mieux). Des travaux academiques (Cremers et al. notamment) rappellent que le resultat depend de la methode de mesure. Point cle : l'ecart **se reduit mais ne s'inverse pas** — en moyenne et net de frais, la gestion active reste derriere sur le long terme.

3. Exemple de reformulation : *"Les preuves suggerent qu'il est difficile, pour la grande majorite des epargnants, de battre durablement un indice large net de frais."* (et non "Vous devez faire de l'indiciel").

4. Correction de l'ami : *"SPIVA montre que la plupart des fonds actifs sous-performent net de frais sur le long terme, mais le chiffre depend de la methode (fonds vs encours) et certains gerants surperforment ; ce n'est pas 'inutile', c'est 'difficile a faire mieux en moyenne, surtout apres frais'."* On reste factuel : ni militance pro-passif, ni rejet de la nuance.
