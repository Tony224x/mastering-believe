# Module 06 — Fonds indiciels et allocation : la mise en oeuvre

> **Temps estime** : 50 min | **Prerequis** : Module 05 (risque/rendement, classes d'actifs, diversification)
>
> **Objectif** : Passer du *principe* de diversification (Module 05) a la *mise en oeuvre* : comprendre ce qu'est un fonds indiciel / ETF, construire une allocation simple en "3 fonds", mesurer l'impact compose des frais, et lire le debat actif vs passif a travers la donnee.

> ⚠️ **Disclaimer** : ce module est **purement educatif** et ne constitue pas un conseil financier, fiscal ou en investissement personnalise. Toute decision doit tenir compte de votre situation propre ; consultez un conseiller agree. Les performances passees ne prejugent pas des performances futures. Tout investissement comporte un **risque de perte en capital**. Aucun produit, emetteur ou ticker n'est recommande ici.

---

## 1. Exemple d'abord : meme capital, frais differents

Marie et Paul investissent chacun 10 000 € aujourd'hui et ajoutent 200 €/mois pendant 30 ans. Rendement **brut** hypothetique identique : 7 %/an. La seule difference, ce sont les **frais annuels**.

- **Marie** : un fonds aux frais eleves — **1,80 %/an**
- **Paul** : un fonds a bas frais — **0,20 %/an**

| | Capital final estime |
|---|---|
| Marie (1,80 % de frais) | ≈ 220 000 € |
| Paul (0,20 % de frais) | ≈ 311 000 € |

Une difference de **~91 000 €**, pour 1,6 point de frais par an. Ce n'est ni le talent ni la chance qui les separe : ce sont **les frais composes sur 30 ans**.

> **A retenir** : sur un long horizon, les frais ne sont pas un detail. Ils se composent exactement comme les rendements — mais en votre defaveur.

---

## 2. Qu'est-ce qu'un fonds indiciel (ETF) ?

Un **fonds indiciel** cherche simplement a **repliquer** un indice de marche (par exemple un indice des grandes entreprises mondiales) plutot qu'a le "battre". Quand il est cote en Bourse et s'achete/se vend comme une action, on parle d'**ETF** (*Exchange Traded Fund*).

**Ce que ca apporte concretement :**
- **Diversification immediate** : un seul ETF "actions monde" peut contenir des milliers d'entreprises (le principe de Markowitz du Module 05, applique en un achat).
- **Frais tres bas** : repliquer un indice ne demande pas une equipe de gerants qui selectionne des titres, d'ou des frais reduits.
- **Transparence** : on sait ce que le fonds detient (l'indice est public).

**Ce qu'il ne fait pas :**
- Il ne protege pas d'une baisse generale du marche : il replique l'indice **a la baisse comme a la hausse**. La diversification reduit le risque specifique, pas le risque de marche (rappel du Module 05).

> Pour rester neutre et durable, ce cours parle toujours de categories generiques — par exemple **"un ETF actions monde a bas frais"** — jamais d'un produit, d'un emetteur ou d'un ticker precis.

---

## 3. Lire le debat actif vs passif par la donnee

Le coeur du sujet : la **gestion active** (un gerant tente de battre le marche en choisissant des titres) fait-elle mieux que la **gestion passive** (repliquer l'indice) ?

### 3.1 L'argument arithmetique

Avant meme les donnees, une logique simple (formalisee par William Sharpe) : pris dans leur ensemble, **les gerants actifs detiennent collectivement le marche**, donc obtiennent *avant frais* le rendement du marche. *Apres frais*, en moyenne, ils font necessairement **moins bien** que l'indice. Ce n'est pas un jugement de valeur, c'est de l'arithmetique.

### 3.2 La donnee : le rapport SPIVA

Le rapport **SPIVA** (*S&P Indices Versus Active*), publie par S&P Dow Jones Indices, mesure regulierement combien de fonds actifs sous-performent leur indice de reference.

Ordre de grandeur souvent observe sur le long terme (horizon ~20 ans, actions US) : **environ 90 % et plus des fonds actifs sous-performent leur indice**.

### 3.3 GARDE-FOU : la nuance methodologique (a ne JAMAIS omettre)

Ce chiffre demande une lecture honnete :

- Le ~92 % est calcule **en comptant les fonds** (chaque fonds compte pour 1, qu'il soit gros ou minuscule) : c'est une mesure **equiponderee**.
- Si l'on pondere plutot **par les encours** (l'argent reellement investi), la sous-performance est **un peu moins prononcee** : une partie des capitaux se concentre dans des fonds qui s'en sortent mieux. Des travaux academiques (par ex. Cremers et al. sur la gestion active et l'*active share*) rappellent que la mesure depend fortement de la methodologie.
- **L'ecart se reduit, mais ne s'inverse pas** : meme pondere par les encours, l'ensemble de la gestion active reste, en moyenne et net de frais, derriere l'indice sur le long terme.

> **A retenir** : **les preuves suggerent** qu'il est difficile, pour la grande majorite des epargnants, de battre durablement un indice large net de frais. Ce n'est pas "vous *devez* faire de l'indiciel" — c'est "voici ce que la donnee montre, avec ses nuances de mesure".

---

## 4. L'allocation "3 fonds" : simple et robuste

Inspiree de la philosophie de **John Bogle** (fondateur de la gestion indicielle grand public ; *The Little Book of Common Sense Investing*) et popularisee par la communaute **Bogleheads**, l'allocation "3 fonds" applique la diversification du Module 05 avec un minimum d'ingredients :

| Bloc | Role |
|---|---|
| ETF **actions domestiques** (a bas frais) | Croissance, exposition a son economie locale |
| ETF **actions internationales** (a bas frais) | Diversification geographique (les economies ne montent pas ensemble) |
| ETF **obligations** (a bas frais) | Amortisseur, reduction de la volatilite globale |

**Exemples d'allocation selon l'horizon (illustrations, pas des prescriptions) :**
- Horizon long (~30 ans) : davantage d'actions (ex. 80 % actions / 20 % obligations)
- Horizon plus court (~10 ans) : davantage d'obligations (ex. 50 % / 50 %)

> Ces proportions sont des **illustrations pedagogiques**. La bonne allocation depend de **votre** horizon, de votre tolerance au risque, de votre situation et de vos objectifs (cf. Module 05, section horizon). Le cours ne vous dit pas combien mettre en actions.

---

## 5. Les frais : l'ennemi compose silencieux

Les frais d'un fonds (souvent appeles **TER**, *Total Expense Ratio*, ou "frais courants") s'appliquent **chaque annee**, sur **l'ensemble du capital accumule**. Leur effet est donc **compose**, exactement comme les rendements — c'est ce que Bogle appelle la *tyrannie des couts composes*, par opposition a la *magie des rendements composes*.

Ordre de grandeur de l'impact sur un capital qui atteindrait ~200 000 € brut sur 30 ans :

| TER annuel | Impact cumule approximatif |
|---|---|
| 0,10 % | quelques milliers d'euros |
| 0,50 % | quelques dizaines de milliers |
| 1,00 % et plus | plusieurs dizaines de milliers |

C'est l'un des rares leviers **entierement sous votre controle** : vous ne choisissez pas les rendements futurs, mais vous pouvez comparer les frais. Le script `02-code/06-fonds-indiciels-allocation.py` chiffre precisement l'ecart entre 0,1 % et 1 % sur 30 ans.

> **A retenir** : cherchez toujours le **TER / frais courants** d'un fonds. Sur un long horizon, un point de frais en plus peut couter autant que plusieurs annees de versements.

*(Note : les frais sont ancres ici, au Module 06 ; ils ne seront pas redemontres au Module 07 sur la fiscalite, qui traite un frottement different.)*

---

## 6. Ce que ce module ne couvre pas

- Les **enveloppes fiscales** et le frottement fiscal → Module 07 (principes generaux, sans loi nationale).
- Le **stock-picking / day-trading** : activites a haut risque, hors du "20 % qui compte" pour la majorite.
- La **crypto** et les **robo-advisors** : traites de facon neutre au Module 13.
- L'**immobilier** (acheter vs louer) : Module 10.

---

## Flash-cards (spaced repetition)

**Q1 — Qu'est-ce qu'un fonds indiciel (ETF) et que cherche-t-il a faire ?**
> R : Un fonds qui replique un indice de marche au lieu de tenter de le battre. Il offre une diversification immediate et des frais bas. Cote en Bourse, il s'appelle un ETF.

**Q2 — Quel est l'argument arithmetique contre la gestion active ?**
> R : Collectivement, les gerants actifs detiennent le marche : avant frais ils obtiennent le rendement du marche, donc apres frais ils font en moyenne moins bien que l'indice. C'est un resultat mathematique, pas un jugement.

**Q3 — Que dit SPIVA, et quelle est sa nuance methodologique cruciale ?**
> R : Environ 90 %+ des fonds actifs sous-performent leur indice sur ~20 ans. Nuance : ce chiffre est equipondere (chaque fonds = 1) ; pondere par les encours, la sous-performance se reduit, mais ne s'inverse pas.

**Q4 — Que sont le TER et la "tyrannie des couts composes" ?**
> R : Le TER (frais courants) est le cout annuel total du fonds en % du capital. Comme il s'applique chaque annee sur tout le capital, son effet se compose et peut couter des dizaines de milliers d'euros sur 30 ans.

**Q5 — Quels sont les trois blocs de l'allocation "3 fonds" ?**
> R : Actions domestiques + actions internationales + obligations, chacun via un fonds indiciel a bas frais. Les proportions s'ajustent selon l'horizon et la tolerance au risque.

---

## Points cles a retenir

1. **Les frais se composent** comme les rendements, mais contre vous : c'est un levier sous votre controle (le TER).
2. **Un fonds indiciel / ETF** applique la diversification de Markowitz en un achat, a bas cout et de facon transparente.
3. **Les preuves suggerent** (arithmetique + SPIVA) qu'il est difficile de battre durablement un indice large net de frais — **avec** la nuance fonds vs encours.
4. **L'allocation "3 fonds"** (actions domestiques + internationales + obligations) est une mise en oeuvre simple et robuste de la diversification.
5. Les proportions restent **vos** choix, selon horizon et tolerance au risque — le cours n'en prescrit aucune.

---

## Pour aller plus loin

- **The Little Book of Common Sense Investing** — John C. Bogle, John Wiley & Sons, ed. mise a jour 2017 (ISBN 9781119404507) : https://www.wiley.com/en-us/9781119404507
- **A Random Walk Down Wall Street** — Burton G. Malkiel, W. W. Norton (ed. 50e anniversaire) : https://wwnorton.com/books/A-Random-Walk-Down-Wall-Street/
- **Three-fund portfolio** — Bogleheads Wiki (application pratique minimaliste) : https://www.bogleheads.org/wiki/Three-fund_portfolio
- **SPIVA U.S. Year-End** — S&P Dow Jones Indices (donnees actif vs passif) : https://www.spglobal.com/spdji/en/research-insights/spiva/
- **AMF — Espace epargnants** (guides "frais" en francais) : https://www.amf-france.org/fr/espace-epargnants
- **Simulation du domaine** — `02-code/06-fonds-indiciels-allocation.py` (stdlib Python)

> **Disclaimer** : contenu educatif, pas un conseil financier personnalise. Aucun produit ni emetteur n'est recommande. Les performances passees ne prejugent pas des performances futures ; tout investissement comporte un risque de perte en capital.
