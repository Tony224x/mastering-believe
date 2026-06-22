# Solutions — Module 04 : Investir simplement et sur le long terme (niveau avancé)

> ⚠️ Corrigés modèles. Les rendements sont **hypothétiques et illustratifs**. Le débat actif vs passif est traité **par la donnée** (SPIVA, Sharpe), sans jugement. **Pas un conseil financier.** Tout investissement comporte un risque de perte en capital.

---

## Solution Exercice 1 — Quelle surperformance pour "valoir" ses frais ?

Données : 10 000 € + 200 €/mois, 25 ans, capitalisation mensuelle.

### Question 1 : fonds passif (net 6,85 %)

`capital_final_mensuel(10000, 200, 0.0685, 25)` ≈ **205 581 €**.

### Question 2 : fonds actif selon α

Rendement net actif = 7 % + α − 1,80 %.

| α brut | Rendement net | Capital final à 25 ans |
|---|---|---|
| 0 % | 5,20 % | ≈ **156 049 €** |
| +1 % | 6,20 % | ≈ **184 241 €** |
| +2 % | 7,20 % | ≈ **218 186 €** |

Avec α = 0 (le fonds actif réplique simplement l'indice avant frais), il finit ~50 000 € **derrière** le passif — uniquement à cause des frais (1,80 % vs 0,15 %). Même avec +1 % de surperformance brute chaque année, il reste **en dessous** du passif.

### Question 3 : seuil d'α pour égaler le passif

```
net actif = net passif
7 % + α − 1,80 % = 6,85 %
α = 6,85 % − 5,20 % = 1,65 %
```

Le fonds actif doit livrer **+1,65 % de surperformance brute, chaque année, pendant 25 ans**, juste pour **égaler** le fonds passif net de frais. Pour le *battre*, il faut encore plus. C'est un obstacle considérable et permanent.

### Question 4 : lien avec l'arithmétique de Sharpe

Sharpe (1991) démontre qu'**avant frais, l'ensemble des gérants actifs obtient exactement le rendement du marché** (puisqu'ils *sont*, collectivement, le marché). En moyenne, l'α brut agrégé est donc nul. Après frais (élevés pour l'actif), la moyenne des gérants fait **moins bien** que l'indice. Livrer durablement +1,65 % d'α net positif suppose de faire partie d'une minorité qui prend cet α aux autres — par construction, tout le monde ne peut pas y arriver. Ce n'est pas un jugement sur la compétence : c'est un résultat **arithmétique**.

### Question 5 : lien avec SPIVA (factuel)

SPIVA Year-End 2024 mesure que **~92 % des fonds actions US domestiques sous-performent leur indice sur 20 ans**. Conséquence factuelle : la probabilité de **sélectionner à l'avance** (ex-ante) le fonds actif rare qui livrera l'α requis est faible, et la persistance de la surperformance est statistiquement rare. On pose la donnée : ce n'est pas que la gestion active soit "mauvaise" ou "malhonnête" — c'est que, après frais et en moyenne, les chiffres penchent fortement vers le passif à bas coût sur le long terme. (Nuance honnête du module : pondéré par les encours, l'écart se réduit un peu mais ne s'inverse pas.)

---

## Solution Exercice 2 — Politique d'investissement écrite (Sofiane, 250 000 €)

### Partie A : sensibilité aux frais (250 000 €, 7 % brut, 25 ans, lump sum)

| TER | Rendement net | Capital final à 25 ans |
|---|---|---|
| 0,10 % | 6,90 % | ≈ **1 325 509 €** |
| 0,50 % | 6,50 % | ≈ **1 206 925 €** |
| 1,00 % | 6,00 % | ≈ **1 072 968 €** |
| 2,00 % | 5,00 % | ≈ **846 589 €** |

Perte due aux frais entre 0,10 % et 2,00 % : 1 325 509 − 846 589 = **~478 920 €**, soit **~36 %** du capital final de référence. Sur un gros capital et 25 ans, 1,9 point de frais détruit près de **la moitié d'un million d'euros**. La décision de frais, prise une seule fois, est l'une des plus lourdes du plan.

### Partie B : allocation et mise en œuvre

Allocation "3 fonds" cohérente (horizon 25 ans, pas de besoin de liquidité) : par exemple **65 % actions monde développé / 15 % actions émergentes / 20 % obligations**. TER cible global : **< 0,20 %** (privilégier des fonds indiciels larges à très bas coût). Mise en œuvre : l'investissement immédiat (lump sum) a en moyenne surperformé l'étalement historiquement ; mais étaler sur quelques mois est psychologiquement acceptable face à un gros capital reçu d'un coup (le détail DCA relève du Module 05). Choisir selon sa tolérance, en gardant le cap une fois la règle fixée.

### Partie C : politique d'investissement en 5 règles (exemple modèle)

1. **Allocation cible** : 65 % actions dév. / 15 % émergents / 20 % obligations, via fonds indiciels TER < 0,20 %. Aucun stock-picking, aucun produit non compris.
2. **Rééquilibrage** : 1 fois/an ; rééquilibrer uniquement si une classe s'écarte de **> 5 points** de la cible (en priorité via les flux entrants).
3. **Conduite en cas de krach > 20 %** : ne rien vendre ; maintenir l'allocation et le calendrier. Relire cette politique avant toute décision.
4. **Consultation du portefeuille** : trimestrielle au maximum ; notifications de cours désactivées.
5. **Sources** : régulateurs (AMF), SPIVA, wikis Bogleheads pour les décisions structurelles ; éviter médias financiers sensationnalistes et influenceurs sans transparence.

### Partie D : stress test (−30 % l'année suivante)

(a) **Ce qu'il faut faire** : ne pas vendre, suivre la politique écrite — une baisse n'est une perte réelle que si on vend. (b) **Ce que l'horizon de 25 ans change** : il donne le temps d'absorber et de récupérer une baisse ; historiquement, les marchés diversifiés se sont repris sur des horizons longs (sans garantie que le futur réplique le passé). (c) **Limite honnête** : la diversification réduit le risque mais **ne l'élimine pas** ; aucun rendement n'est garanti ; un marché peut rester déprimé longtemps. La politique écrite ne supprime pas le risque — elle empêche que l'émotion ne transforme une baisse temporaire en perte permanente.

---

## Résumé des enseignements clés (hard)

1. Pour seulement **égaler** un fonds passif, un fonds actif doit livrer ~+1,65 %/an de surperformance brute **chaque année** (effet des frais) — obstacle permanent.
2. L'**arithmétique de Sharpe** + **SPIVA** expliquent, par la donnée, pourquoi battre l'indice net de frais sur 20 ans est statistiquement rare — sans jugement de valeur.
3. Sur **gros capital et long horizon**, 1,9 point de frais peut détruire ~36 % du capital final.
4. Une **politique d'investissement écrite** (allocation, rééquilibrage, conduite en krach, consultation, sources) protège du comportement émotionnel.
5. Posture honnête : la diversification **réduit** le risque, ne l'**élimine pas** ; aucun rendement n'est garanti.
