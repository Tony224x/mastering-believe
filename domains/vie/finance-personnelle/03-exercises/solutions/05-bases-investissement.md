# Solutions — Bases de l'investissement (Module 05)

> **Disclaimer** : solutions educatives. Les chiffres de simulation dependent de la graine aleatoire fixee (`random.seed(42)`) ; tant qu'elle n'est pas modifiee, la sortie est reproductible. Aucun rendement n'est garanti.

---

## Exercice 1 — Classer le couple risque/rendement

1. **Classement par risque croissant** : W (vol 1 %) < X (vol 6 %) < Y (vol 18 %). Le rendement espere suit exactement le meme ordre : W (2 %) < X (4 %) < Y (7 %). C'est l'illustration de la **prime de risque** : on n'obtient un rendement espere plus eleve qu'en acceptant plus d'incertitude.

2. **Classes d'actifs plausibles** :
   - **W** (2 % / vol 1 %) → **liquidites** (compte d'epargne) : rendement faible, quasi pas de variation.
   - **X** (4 % / vol 6 %) → **obligations** : rendement modere, volatilite faible a moyenne.
   - **Y** (7 % / vol 18 %) → **actions** : rendement espere plus eleve, forte volatilite.

3. **Placement Z (9 % espere, 0 % de volatilite)** : c'est un **signal d'alarme**. Un rendement espere eleve **sans aucun risque** contredit le couple risque/rendement (il n'existe pas de prime de risque "gratuite"). Un rendement eleve promis comme garanti est l'un des marqueurs classiques d'arnaque (voir Module 08).

---

## Exercice 2 — Voir le risque avec la simulation

1. Valeurs relevees de la **DEMO 1** (graine 42, 10 000 € initial, 6 % espere, 20 ans) :

| Volatilite | Mediane | p10 | p90 |
|---|---|---|---|
| 3 % | ~31 925 € | ~27 043 € | ~37 585 € |
| 10 % | ~29 452 € | ~16 699 € | ~51 349 € |
| 20 % | ~22 145 € | ~7 127 € | ~66 434 € |

2. **Amplitude (p90 − p10)** :
   - 3 % : ~37 585 − 27 043 = **~10 542 €**
   - 10 % : ~51 349 − 16 699 = **~34 650 €**
   - 20 % : ~66 434 − 7 127 = **~59 307 €**
   L'amplitude **explose** quand la volatilite augmente : c'est la definition meme du risque (dispersion des resultats possibles).

3. La mediane du profil 20 % (~22 145 €) est **plus basse** que celle du profil 3 % (~31 925 €), alors que le rendement espere est identique (6 %). Raison : avec des rendements tres disperses, la **composition** penalise la trajectoire (une annee −20 % puis +20 % donne 0,8 × 1,2 = 0,96, soit une perte nette). Plus la volatilite est forte, plus cet effet ronge la croissance mediane.

4. Un epargnant qui a besoin de son argent dans 1 an doit fuir le profil 20 % car, sur un horizon aussi court, il risque de devoir **vendre en bas de fourchette** (le p10 tombe a ~7 127 €) sans avoir le temps que le marche se redresse : volatilite elevee + horizon court = risque de perte realisee.

---

## Exercice 3 — Le "repas gratuit" de la diversification

Valeurs relevees de la **DEMO 2** (graine 42, 7 % espere, vol 25 % par actif, 20 ans) :

| Config | Mediane | p10 | p90 |
|---|---|---|---|
| 1 seul actif | ~22 042 € | ~4 798 € | ~86 941 € |
| 5 actifs, faible correlation | ~31 542 € | ~13 381 € | ~70 396 € |
| 30 actifs, faible correlation | ~34 036 € | ~17 843 € | ~64 998 € |
| 30 actifs, forte correlation (0,8) | ~25 436 € | ~6 751 € | ~83 886 € |

1. De 1 actif a 30 actifs peu correles : la fourchette se **resserre nettement**. Le p10 passe de ~4 798 € a ~17 843 €, soit une amelioration d'environ **+13 000 €** du scenario defavorable ; le p90 baisse en parallele (de ~86 941 € a ~64 998 €). On echange un peu de "loto" a la hausse contre beaucoup moins de risque a la baisse.

2. Le rendement espere de chaque actif est maintenu a 7 % partout : c'est **le point cle**. Cela isole l'effet de la diversification comme une **reduction de risque pure** — on ne paie pas la baisse du risque par une baisse du rendement espere. C'est le "repas gratuit" de Markowitz.

3. Avec une correlation de 0,8, les actifs montent et descendent presque ensemble : leurs chocs **propres** (diversifiables) ne se compensent plus, car le **choc commun** domine. La diversification aide donc beaucoup moins. Ce qui reste est le **risque de marche** (systematique), **non diversifiable**.

4. Non, la diversification **ne protege pas contre toutes les baisses**. Elle reduit le risque specifique (une entreprise/un secteur), mais une baisse generale du marche touche presque tous les actifs en meme temps (cas correle). La diversification reduit le risque, elle ne l'elimine jamais totalement.
