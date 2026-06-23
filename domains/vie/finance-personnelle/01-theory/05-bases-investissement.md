# Module 05 — Les bases de l'investissement : risque, rendement, diversification

> **Temps estime** : 45 min | **Prerequis** : Modules 01 a 04 (interets composes, budget, fonds d'urgence, dette)
>
> **Objectif** : Comprendre le couple risque/rendement, les grandes classes d'actifs et le *principe* de diversification — la fondation conceptuelle avant de passer a la mise en oeuvre concrete (allocation et fonds indiciels) au Module 06.

> ⚠️ **Disclaimer** : ce module est **purement educatif** et ne constitue pas un conseil financier, fiscal ou en investissement personnalise. Tout investissement comporte un **risque de perte en capital**. Les performances passees ne prejugent pas des performances futures. Les chiffres utilises sont illustratifs.

---

## 1. Exemple d'abord : deux placements, deux profils

Imaginez deux placements sur lesquels vous deposez 10 000 € pendant 1 an.

- **Placement A** : un compte d'epargne securise. Quoi qu'il arrive, vous retrouvez 10 200 € (rendement certain de 2 %).
- **Placement B** : un panier d'actions. Selon les annees passees, il pourrait finir a 13 000 € (+30 %)... ou a 8 000 € (−20 %). En **moyenne** sur le long terme, ce type de panier a historiquement rapporte davantage que A — mais aucune annee n'est garantie.

Le placement B a un **rendement espere plus eleve**, mais aussi une **incertitude plus grande**. C'est tout le sujet de ce module : on ne peut pas parler de rendement sans parler du risque qui l'accompagne.

> **A retenir** : il n'existe pas de placement a la fois tres rentable, totalement sur et parfaitement liquide. Choisir, c'est arbitrer entre ces trois dimensions.

---

## 2. Le couple risque / rendement

En finance, **risque** et **rendement espere** vont ensemble. Personne n'accepterait de prendre plus de risque sans esperer, en contrepartie, un rendement moyen plus eleve. C'est ce qu'on appelle la **prime de risque**.

| Concept | Definition simple |
|---|---|
| **Rendement** | Le gain (ou la perte) d'un placement, en % du capital, sur une periode |
| **Rendement espere** | La moyenne des rendements possibles, ponderee par leur probabilite |
| **Risque** | L'incertitude autour de ce rendement : a quel point les resultats peuvent s'ecarter de la moyenne |

Point crucial : un **rendement espere eleve n'est pas un rendement garanti**. Si quelqu'un vous promet un rendement eleve *sans* risque, c'est un signal d'alarme (voir Module 08 sur les arnaques).

> **A retenir** : plus le rendement espere est eleve, plus l'incertitude (le risque) l'est aussi. Le couple est indissociable.

---

## 3. Mesurer le risque : la volatilite

La facon la plus courante de chiffrer le risque d'un actif est la **volatilite** : elle mesure l'ampleur des variations du prix autour de sa moyenne. Concretement, on la calcule comme **l'ecart-type** des rendements.

- Un actif **peu volatil** (ex. obligations d'Etat de qualite) varie peu : ses rendements annuels restent proches les uns des autres.
- Un actif **tres volatil** (ex. actions, et plus encore certaines classes speculatives) peut gagner 30 % une annee et en perdre 25 % la suivante.

Une intuition utile : la volatilite ne dit pas seulement "ca peut monter beaucoup", elle dit surtout **"ca peut descendre beaucoup, et a un mauvais moment"**. C'est pourquoi le risque se gere en fonction de votre **horizon** (section 6).

> Le script `02-code/05-bases-investissement.py` simule des portefeuilles de differentes volatilites et montre concretement comment la dispersion des resultats s'elargit quand la volatilite augmente.

---

## 4. Les grandes classes d'actifs

Un **actif** est quelque chose que l'on detient dans l'espoir d'un revenu ou d'une plus-value. Le regulateur americain (SEC / Investor.gov) regroupe l'essentiel en quelques **classes d'actifs**, chacune avec son profil risque/rendement/liquidite typique.

| Classe d'actifs | Ce que c'est | Rendement espere | Risque / volatilite | Liquidite |
|---|---|---|---|---|
| **Liquidites** (cash, comptes d'epargne) | Argent disponible immediatement | Faible | Tres faible (mais erode par l'inflation) | Tres elevee |
| **Obligations** | Prets a des Etats ou entreprises, remboursables avec interets | Modere | Faible a moyen | Moyenne a elevee |
| **Actions** | Parts de propriete d'entreprises | Plus eleve (historiquement) | Eleve | Elevee (cotees) |
| **Immobilier** | Biens physiques (residence, locatif) ou fonds immobiliers | Modere a eleve | Moyen ; **peu liquide** | Faible (vente longue) |

Quelques nuances importantes :

- **Liquidites** : utiles pour le fonds d'urgence (Module 03) et le court terme, mais sur le long terme l'inflation grignote leur pouvoir d'achat (taux reel souvent negatif).
- **Obligations** : jouent un role d'**amortisseur** dans un portefeuille ; elles montent rarement autant que les actions, mais baissent generalement moins.
- **Actions** : moteur de croissance a long terme, au prix d'une forte volatilite a court terme.
- **Immobilier** : sa **faible liquidite** (on ne vend pas un appartement en un clic) et ses couts de transaction eleves en font une classe a part. On l'etudiera specifiquement plus tard.

> **A retenir** : chaque classe d'actifs a un profil different. Aucune n'est "la meilleure" dans l'absolu ; tout depend de votre horizon, de vos objectifs et de votre tolerance au risque.

---

## 5. Le principe de diversification (Markowitz, 1952)

Voici l'idee centrale du module — et l'une des plus importantes de toute la finance personnelle.

En 1952, l'economiste **Harry Markowitz** publie *Portfolio Selection* (*The Journal of Finance*), un article fondateur (prix Nobel 1990). Sa demonstration, vulgarisee :

> En combinant des actifs dont les performances ne montent et ne descendent **pas exactement en meme temps** (on dit qu'ils sont **peu correles**), on peut **reduire le risque global du portefeuille sans necessairement reduire le rendement espere**.

### 5.1 L'intuition

Imaginez deux activites : un vendeur de glaces et un vendeur de parapluies. Pris seuls, chacun a des revenus tres irreguliers (tres dependants de la meteo). Mais si vous **possedez les deux**, quand l'un baisse, l'autre monte : votre revenu total est bien plus stable, pour un niveau moyen identique.

C'est exactement ce que fait la diversification avec des actifs financiers : on ne supprime pas le risque, on **lisse les a-coups** en evitant de tout miser sur une seule source.

### 5.2 Le "panier d'oeufs"

La SEC resume le principe par l'image populaire : **"ne mettez pas tous vos oeufs dans le meme panier"** (*Beginners' Guide to Asset Allocation, Diversification, and Rebalancing*). La diversification opere a deux niveaux :

- **Entre classes d'actifs** : repartir entre actions, obligations, liquidites, etc.
- **Au sein d'une classe** : detenir des centaines d'entreprises plutot qu'une seule (si une fait faillite, elle ne pese qu'une fraction infime du total).

### 5.3 Ce que la diversification fait... et ne fait pas

- **Elle reduit** le *risque specifique* (celui propre a une entreprise ou un secteur).
- **Elle ne supprime pas** le *risque de marche* (celui qui touche presque tous les actifs en meme temps, ex. une crise globale). Aucun portefeuille n'est immunise contre une baisse generale.

> **A retenir** : la diversification est le seul "repas gratuit" reconnu en finance — elle peut reduire le risque sans sacrifier le rendement espere. Mais elle reduit le risque, elle ne l'elimine jamais totalement.

---

## 6. L'horizon de temps : la cle qui relie tout

Votre **horizon** (dans combien de temps aurez-vous besoin de cet argent ?) determine le niveau de risque que vous pouvez raisonnablement accepter.

- **Court terme (0-3 ans)** : besoin d'argent bientot → privilegier la securite et la liquidite (liquidites, obligations courtes). Mettre en actions un capital dont on aura besoin dans 1 an, c'est risquer de devoir vendre en pleine baisse.
- **Long terme (10 ans et plus)** : le temps permet d'**absorber** la volatilite des actions. Les baisses, meme severes, ont historiquement ete suivies de reprises sur des horizons longs (sans garantie pour autant).

C'est la rencontre entre la **volatilite** (section 3) et le **temps** : plus l'horizon est long, plus on peut tolerer la volatilite, car on n'est pas force de vendre au mauvais moment.

> **A retenir** : ce n'est pas l'actif seul qui est "risque" ou "sur", c'est le **couple actif + horizon**. Un actif volatil sur 1 an n'a pas le meme profil de risque sur 25 ans.

---

## 7. Ou l'on s'arrete (et la suite)

Ce module pose **les principes** : risque/rendement, classes d'actifs, diversification, horizon. Il s'arrete volontairement **au principe** de diversification.

La question naturelle qui suit — *"concretement, comment je repartis mon argent et avec quels supports ?"* — est le sujet du **Module 06** (fonds indiciels, portefeuille type, impact des frais). On passe alors du *pourquoi* au *comment*.

---

## Flash-cards (spaced repetition)

**Q1 — Pourquoi ne peut-on pas separer rendement espere et risque ?**
> R : Parce qu'un rendement espere plus eleve est la contrepartie (la "prime") d'une incertitude plus grande. Personne ne prendrait plus de risque sans esperer, en moyenne, un meilleur rendement. Un rendement eleve promis *sans* risque est un signal d'alarme.

**Q2 — Qu'est-ce que la volatilite mesure ?**
> R : L'ampleur des variations du prix d'un actif autour de sa moyenne (mesuree par l'ecart-type des rendements). Plus elle est elevee, plus les resultats annuels peuvent s'ecarter — a la hausse comme a la baisse.

**Q3 — Classez liquidites, obligations et actions du moins au plus volatil.**
> R : Liquidites (tres faible) < obligations (faible a moyen) < actions (eleve). Le rendement espere suit generalement le meme ordre croissant.

**Q4 — En une phrase, qu'a montre Markowitz en 1952 ?**
> R : Qu'en combinant des actifs peu correles, on peut reduire le risque global d'un portefeuille sans necessairement reduire son rendement espere.

**Q5 — La diversification supprime-t-elle tout le risque ?**
> R : Non. Elle reduit le risque *specifique* (propre a une entreprise ou un secteur), mais pas le risque *de marche* (une crise globale touche presque tous les actifs en meme temps).

---

## Points cles a retenir

1. **Risque et rendement espere sont indissociables** : la prime de risque est le prix de l'incertitude.
2. **La volatilite (ecart-type) chiffre le risque** : elle dit a quel point les resultats peuvent s'ecarter de la moyenne.
3. **Quatre grandes classes d'actifs** — liquidites, obligations, actions, immobilier — avec des profils risque/rendement/liquidite distincts.
4. **La diversification (Markowitz)** reduit le risque sans sacrifier le rendement espere : c'est le seul "repas gratuit" de la finance.
5. **L'horizon decide du risque acceptable** : c'est le couple actif + duree qui compte, pas l'actif seul.
6. Ces chiffres sont illustratifs ; aucun rendement n'est garanti.

---

## Pour aller plus loin

- **Beginners' Guide to Asset Allocation, Diversification, and Rebalancing** — U.S. SEC / Investor.gov : https://www.investor.gov/additional-resources/general-resources/publications-research/info-sheets/beginners-guide-asset
- **Asset Allocation** (Getting Started) — U.S. SEC / Investor.gov : https://www.investor.gov/introduction-investing/getting-started/asset-allocation
- **Portfolio Selection** — Harry Markowitz, *The Journal of Finance*, vol. 7, n°1, 1952 (DOI : 10.1111/j.1540-6261.1952.tb01525.x) : https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.1952.tb01525.x
- **A Random Walk Down Wall Street** — Burton G. Malkiel, W. W. Norton (ed. 50e anniversaire) : https://wwnorton.com/books/A-Random-Walk-Down-Wall-Street/
- **Simulation du domaine** — `02-code/05-bases-investissement.py` (stdlib Python, jouable en local)

> **Disclaimer** : contenu educatif, pas un conseil financier personnalise. Les exemples et taux sont illustratifs ; tout investissement comporte un risque de perte en capital, et les performances passees ne prejugent pas des performances futures.
