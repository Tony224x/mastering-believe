# Module 01 — Le moteur : interets composes et valeur du temps

> **Temps estime** : 45 min | **Prerequis** : aucun
>
> **Objectif** : Comprendre pourquoi le temps est votre actif le plus precieux et comment les interets composes transforment une epargne modeste en capital significatif.

---

## 1. Le point de depart : un calcul qui change tout

Imaginez deux personnes, Alice et Bob :

- **Alice** epargne 200 € par mois de **25 a 35 ans** (10 ans, 24 000 € verses), puis **arrete de verser** mais laisse son capital investi jusqu.a 65 ans.
- **Bob** commence a **35 ans** et epargne 200 € par mois jusqu'a **65 ans** (30 ans d'effort).
- Tous deux obtiennent un rendement annuel moyen de **7 %** (typique d'un portefeuille diversifie sur 30 ans, avant inflation — chiffre illustratif, pas une garantie).

Resultats a 65 ans :

| | Epargne totale versee | Capital final |
|---|---|---|
| **Alice** (10 ans, 25-35 ans) | 24 000 € | ~**281 000 €** |
| **Bob** (30 ans, 35-65 ans) | 72 000 € | ~**244 000 €** |

Alice a verse **3 fois moins** que Bob (24 000 € contre 72 000 €), et finit pourtant avec **plus de capital** (~281 000 € contre ~244 000 €). Pourquoi ? Parce que ses premiers versements ont eu **40 ans pour composer** : le temps a davantage compte que le montant total verse.

> **A retenir** : le temps en bourse est votre actif le plus precieux. Commencer tot bat presque toujours verser plus tard.

---

## 2. Le mecanisme : "l'interet sur l'interet"

### 2.1 Interet simple vs interet compose

**Interet simple** : chaque annee, vous gagnez les interets uniquement sur votre capital initial.
- Exemple : 1 000 € a 5 % = +50 € chaque annee. Au bout de 10 ans : 1 500 €.

**Interet compose** : les interets s'ajoutent au capital, et generent eux-memes des interets.
- Exemple : 1 000 € a 5 %, capitalisation annuelle. Annee 1 : +50 € → 1 050 €. Annee 2 : +52,5 € → 1 102,5 €. Au bout de 10 ans : **1 629 €** (au lieu de 1 500 €).

La difference semble modeste sur 10 ans, mais elle devient colossale sur 30-40 ans. C'est la croissance **exponentielle**.

### 2.2 La formule

Pour un capital initial investi sans versements reguliers :

```
A = P × (1 + r/n)^(n × t)
```

- **A** = montant final
- **P** = capital initial (principal)
- **r** = taux d'interet annuel (en decimal ; ex. 5 % = 0,05)
- **n** = nombre de capitalisations par an (annuelle = 1, mensuelle = 12)
- **t** = duree en annees

Pour des versements reguliers (epargne mensuelle **M**) :

```
A = P × (1 + r/n)^(n×t) + M × [((1 + r/n)^(n×t) - 1) / (r/n)]
```

> Le calculateur `02-code/01-interets-composes.py` implemente ces deux formules et vous permet de jouer avec les parametres.

---

## 3. Les trois variables qui comptent

### 3.1 Le temps (la plus importante)

Exemple : 1 000 € a 7 % par an, capitalisation annuelle.

| Duree | Montant final |
|-------|--------------|
| 10 ans | 1 967 € |
| 20 ans | 3 870 € |
| 30 ans | 7 612 € |
| 40 ans | 14 974 € |

En 40 ans, votre capital est multiplie par **presque 15**, sans rien faire de plus.

### 3.2 Le taux (important mais moins controlable)

Exemple : 10 000 € pendant 30 ans.

| Taux annuel | Montant final |
|-------------|--------------|
| 2 % (livret) | 18 113 € |
| 4 % | 32 434 € |
| 6 % | 57 435 € |
| 8 % | 100 627 € |

La difference entre 4 % et 8 % n'est "que" du simple au double en taux, mais triple le capital final sur 30 ans.

> **Attention** : des taux eleves s'accompagnent en general de risques plus eleves. Il n'existe pas de rendement garanti de 8 %. Les performances passees ne prejugent pas des performances futures.

### 3.3 Le montant investi regulierement

La regularite bat le gros coup unique pour la plupart des epargnants. Verser 200 € par mois pendant 30 ans a 7 % produit ~**244 000 €** (pour 72 000 € verses).

---

## 4. La frequence de capitalisation

Plus les interets sont capitalises frequemment, plus la croissance est rapide.

Exemple : 10 000 € a 6 % sur 10 ans.

| Frequence | Montant final |
|-----------|--------------|
| Annuelle | 17 908 € |
| Trimestrielle | 18 061 € |
| Mensuelle | 18 194 € |
| Continue | 18 221 € |

La difference est reelle mais modeste. Ne pas optimiser la frequence au detriment du plus important : **commencer tot**.

---

## 5. Le cout de l'attente

Chaque annee d'attente a un prix. Voici combien 200 €/mois a 7 % produit selon l'age de depart, jusqu'a 65 ans :

| Age de depart | Duree | Capital a 65 ans |
|--------------|-------|-----------------|
| 25 ans | 40 ans | ~527 000 € |
| 30 ans | 35 ans | ~370 000 € |
| 35 ans | 30 ans | ~257 000 € |
| 40 ans | 25 ans | ~175 000 € |
| 45 ans | 20 ans | ~116 000 € |

Attendre de 25 a 35 ans coute ici pres de **270 000 €** de capital final, meme si vous versez exactement la meme somme par la suite.

> **A retenir** : "le meilleur moment pour planter un arbre etait il y a 20 ans. Le deuxieme meilleur moment, c'est maintenant." — Proverbe chinois adapte.

---

## 6. Inflation & pouvoir d'achat : le rendement reel

Tous les chiffres precedents sont **nominaux** : ils comptent des euros, pas ce que ces euros permettent d'acheter. Or l'inflation erode silencieusement le pouvoir d'achat. Un capital qui "grossit" de 5 % pendant que les prix montent de 3 % ne vous enrichit que de ~2 % en termes reels.

### 6.1 Du nominal au reel

Le **taux reel** mesure le gain de pouvoir d'achat, une fois l'inflation retiree.

```
taux reel = (1 + nominal) / (1 + inflation) - 1
          ≈ nominal - inflation   (approximation valable quand les taux sont faibles)
```

Exemple : rendement nominal 5 %, inflation 3 %.
- Approximation : 5 − 3 = **~2 %**
- Exact : (1,05 / 1,03) − 1 = **1,94 %**

L'approximation "nominal − inflation" suffit dans la vie courante ; l'ecart avec le calcul exact reste minime tant que les taux sont modestes.

### 6.2 Pourquoi c'est decisif sur le long terme

L'inflation est elle-meme un interet compose… qui joue **contre** vous sur la valeur de votre argent.

A 3 % d'inflation par an, le pouvoir d'achat de 10 000 € devient :

| Duree | Pouvoir d'achat reel de 10 000 € |
|-------|----------------------------------|
| 10 ans | ~7 441 € |
| 20 ans | ~5 537 € |
| 30 ans | ~4 120 € |

En 30 ans, l'argent "dormant" perd ~60 % de son pouvoir d'achat. C'est pourquoi laisser toute son epargne sur un compte non remunere (ou a un taux inferieur a l'inflation) revient a perdre de l'argent **en termes reels**, meme si le solde nominal ne bouge pas.

> **A retenir** : ce qui compte pour votre niveau de vie futur, c'est le rendement **reel** (nominal − inflation), pas le rendement affiche. Un livret a 2 % quand l'inflation est a 4 % vous appauvrit de ~2 %/an.

### 6.3 Deux regles a penser en reel

- **Le fonds d'urgence (3-6 mois de depenses)** : son role est d'etre liquide et sur, pas de rapporter. Mais comme il est detenu en cash sur la duree, l'inflation le grignote. La bonne reponse n'est pas de l'investir en actifs risques (ce serait trahir sa fonction de filet), mais de le **dimensionner en mois de depenses** — qui montent avec l'inflation — plutot qu'en montant fixe fige, et de le placer sur un support remunere proche de l'inflation (livret reglemente, compte d'epargne). On reverra le fonds d'urgence en detail au Module 03.

- **La regle des 4 % (retrait soutenable)** : ce repere de la phase de retraite suppose un retrait **indexe sur l'inflation** chaque annee, et repose donc sur un rendement **reel** positif du portefeuille. Si on raisonnait en nominal, on surestimerait largement ce qu'un capital peut financer sur 30 ans. Detail et limites au Module 12 — ici, retenez seulement : la regle des 4 % se pense en pouvoir d'achat, pas en euros nominaux.

> Le calculateur `02-code/01-interets-composes.py` calcule desormais le rendement **reel** (ajuste de l'inflation) en plus du nominal.

---

## 7. La regle des 72 (estimation rapide)

Pour estimer combien de temps il faut pour doubler votre capital : divisez 72 par le taux annuel.

```
Annees pour doubler ≈ 72 / taux_en_pourcentage
```

Exemples :
- A 4 % : 72 / 4 = **18 ans** pour doubler
- A 6 % : 72 / 6 = **12 ans** pour doubler
- A 9 % : 72 / 9 = **8 ans** pour doubler

---

## 8. Flash-cards

**Q1 : Quelle est la difference entre interet simple et interet compose ?**
> R : L'interet simple calcule les interets uniquement sur le capital initial. L'interet compose calcule les interets sur le capital **plus** les interets precedemment accumules — le capital croit donc exponentiellement.

**Q2 : Dans la formule A = P(1 + r/n)^(nt), que represente "n" ?**
> R : Le nombre de capitalisations par an (n=1 : annuelle ; n=12 : mensuelle ; n=365 : journaliere).

**Q3 : Alice investit 1 000 € a 7 % pendant 30 ans. Combien obtient-elle approximativement ?**
> R : Environ 7 600 € (facteur ~7,6). Utiliser la regle des 72 : 72/7 ≈ 10 ans pour doubler → x2 a 10 ans, x4 a 20 ans, x8 a 30 ans ≈ 8 000 € (la regle est une approximation).

**Q4 : Pourquoi commencer tot est-il plus important que verser des montants eleves plus tard ?**
> R : Les interets des premieres annees generent eux-memes des interets pendant des decennies. Le capital initial a le plus de temps pour croitre de facon exponentielle — l'effet est non lineaire.

**Q5 : Que dit la regle des 72 sur un investissement a 6 % ?**
> R : 72 / 6 = 12 ans pour doubler le capital. En 24 ans, le capital est multiplie par 4 ; en 36 ans, par 8.

**Q6 : Vous gagnez 5 % nominal alors que l'inflation est de 3 %. Quel est votre rendement reel ?**
> R : Environ 2 % (approximation nominal − inflation ; exact : 1,05/1,03 − 1 ≈ 1,94 %). C'est ce rendement reel qui mesure votre gain de pouvoir d'achat — le seul qui compte pour votre niveau de vie futur.

---

## Points cles a retenir

1. **L'interet compose est une croissance exponentielle** — les gains generent des gains. Plus la duree est longue, plus l'effet est spectaculaire.
2. **Le temps est le parametre le plus puissant** — devant le montant investi et le taux.
3. **Commencer tot bat verser plus tard** — meme de petits montants places tot peuvent surpasser de gros versements tardifs.
4. **La regle des 72** permet d'estimer rapidement la duree de doublement : 72 / taux (%).
5. **Chaque annee d'attente a un cout** — calculable, souvent superieur a ce qu'on imagine.
6. **Le rendement qui compte est le rendement reel** (nominal − inflation) — l'inflation est un interet compose qui joue contre votre pouvoir d'achat. Fonds d'urgence et regle des 4 % se pensent en reel.
7. Les exemples ici sont illustratifs. Les rendements reels varient et ne sont pas garantis.

---

## Pour aller plus loin

- **Compound Interest Calculator** — U.S. SEC (Investor.gov) : https://www.investor.gov/financial-tools-calculators/calculators/compound-interest-calculator
  *(Outil officiel pour experimenter avec vos propres chiffres)*
- **Compound Interest — Investing 101** — Investor.gov : https://www.investor.gov/introduction-investing/investing-basics/glossary/compound-interest
- **Calculateur du domaine** — `02-code/01-interets-composes.py` (stdlib Python, jouable en local)
- **Finance Theory I (15.401)** — MIT OCW (valeur-temps de l'argent) : https://ocw.mit.edu/courses/15-401-finance-theory-i-fall-2008/
- **Consumer Price Index (CPI) — inflation et pouvoir d'achat** — U.S. Bureau of Labor Statistics : https://www.bls.gov/cpi/
  *(Mesure officielle de l'inflation ; le CPI inflation calculator illustre l'erosion du pouvoir d'achat dans le temps)*

> **Disclaimer** : ce module est educatif. Il ne constitue pas un conseil financier personnalise. Les taux utilises dans les exemples sont illustratifs ; les performances passees ne prejudgent pas des performances futures.
