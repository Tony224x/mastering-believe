# Solutions — Module 06 : Indépendance financière (niveau avancé)

> Corrigés modèles. Tolérance ± 5 %. Les rendements et taux de retrait sont **illustratifs** ; la règle des 4 % est un **point de départ historique, pas une garantie**. **Pas un conseil financier.** Aucun rendement n'est garanti.

---

## Solution Exercice 1 — Plan FIRE long horizon stress-testé (Réda, 45 ans, horizon 50 ans)

### Question 1 : capital cible selon le taux de retrait (dépenses 30 000 €/an)

| Taux | Capital cible | Multiple |
|---|---|---|
| 4 % | 750 000 € | 25× |
| 3,5 % | ~857 143 € | ~28,6× |
| 3 % | 1 000 000 € | ~33,3× |
| 2,5 % | 1 200 000 € | 40× |

Pour un horizon de **50 ans**, le module et la littérature (Bogleheads, chercheurs) recommandent **3 % à 3,5 %**. Le **4 % standard est inadapté** car il a été calibré par Bengen pour un horizon de **30 ans** : sur 50 ans, le risque d'épuisement du capital est nettement plus élevé (plus de cycles de marché traversés, plus d'exposition au risque de séquence). Réda devrait viser ~857 000 € à 1 000 000 € plutôt que 750 000 €.

### Question 2 : horizon d'accumulation (50 000 € + 1 500 €/mois, 5 % réel)

- Cible **4 %** (750 000 €) : atteinte en ≈ **21 ans**.
- Cible **3 %** (1 000 000 €) : atteinte en ≈ **25 ans**.

La prudence (viser 3 % au lieu de 4 %) impose **~4 années d'accumulation supplémentaires** — un coût modéré pour réduire fortement le risque d'épuisement sur 50 ans.

### Question 3 : risque de séquence des rendements

Pour une décumulation de 50 ans, ce risque est **central** : un krach dans les **premières** années de retrait force à vendre des actifs dépréciés pour vivre, amputant durablement le capital qui aurait dû se rétablir et financer les 40+ années suivantes. Deux adaptations concrètes :
1. **Réserve de liquidités** (2-3 ans de dépenses, ~60-90 k€) hors actifs volatils, pour ne pas vendre en pleine baisse au début.
2. **Taux de retrait prudent + flexibilité** : viser 3-3,5 %, et réduire les dépenses (ou reprendre une activité légère) lors des mauvaises années. On peut aussi **réduire la volatilité de l'allocation** en début de décumulation.

### Question 4 : longévité

Vivre jusqu'à 95 ans (50 ans de retrait) renforce le besoin d'un **taux prudent** : plus l'horizon est long, plus le capital doit durer et plus la marge de sécurité doit être grande. Le module présente la **flexibilité** comme complément essentiel à un taux fixe : un plan qui peut **ajuster les dépenses** ou intégrer un revenu d'appoint résiste mieux qu'un retrait rigide inflation-ajusté. La règle des 4 % suppose un retrait fixe sans flexibilité — irréaliste sur 50 ans.

### Question 5 : disclaimers honnêtes (exemple)

« Ces projections reposent sur un rendement réel **historique et illustratif** (5 %) qui ne garantit pas l'avenir ; les performances passées ne préjugent pas des performances futures. La règle des 4 % est issue de **données américaines du 20e siècle** et d'un **horizon de 30 ans** — elle n'est pas une garantie, surtout sur 50 ans. Les simulations **n'incluent ni fiscalité ni frais**, qui réduisent le taux de retrait soutenable réel. Ce plan est un cadre de réflexion, pas une promesse ; toute décision réelle suppose de consulter un conseiller agréé. »

---

## Solution Exercice 2 — Variantes FIRE chiffrées (Coast / Barista)

Hypothèses : 5 % réel, retraite à 65 ans, cible à 65 ans = 750 000 €.

### Partie A — Coast FIRE

`capital_aujourd'hui = 750 000 / (1,05)^n`

1. Capital "Coast FIRE" nécessaire **aujourd'hui** :
   - À **35 ans** (n = 30) : 750 000 / (1,05)^30 ≈ **~135 968 €**
   - À **40 ans** (n = 25) : 750 000 / (1,05)^25 ≈ **~221 477 €**
   (Commencer plus tôt réduit fortement le capital "Coast" nécessaire — effet du temps, Module 01.)
2. Une fois le Coast FIRE atteint, la croissance composée seule mène au capital de retraite à 65 ans **sans nouveaux versements**. La personne peut donc **réduire fortement, voire arrêter, son taux d'épargne** et "laisser croître" — par exemple travailler moins, changer de métier, ou consacrer son revenu à autre chose. C'est exactement la définition du module.

### Partie B — Barista FIRE (Lena)

Dépenses 2 400 €/mois, dont 900 € couverts par l'activité à mi-temps -> portefeuille doit générer **1 500 €/mois = 18 000 €/an**.

3. Capital cible Barista :
   - À **4 %** : 18 000 / 0,04 = **450 000 €**
   - À **3,5 %** : 18 000 / 0,035 ≈ **~514 286 €**
4. Horizon depuis 0 € + 600 €/mois à 5 % réel pour atteindre **450 000 €** : ≈ **29 ans**.
   Comparaison avec un **FIRE total** (couvrir 2 400 €/mois = 28 800 €/an -> 28 800 / 0,04 = 720 000 €) : le Barista réduit l'objectif de **720 000 € à 450 000 €**, soit **−270 000 € (−37,5 %)**. Le revenu d'appoint allège massivement le capital nécessaire.

### Partie C — Arbitrage : risques propres au Barista

1. **Dépendance au revenu d'appoint** : si l'activité à mi-temps disparaît (santé, marché du travail), les 900 €/mois manquent et le portefeuille (calibré sur 1 500 €) ne suffit plus. *Adaptation* : garder une marge (viser un capital qui couvre un peu plus que 1 500 €, ou conserver une réserve), et maintenir une employabilité.
2. **Séquence des rendements au moment du passage à mi-temps** : un krach juste quand Lena commence à retirer force à vendre bas. *Adaptation* : réserve de liquidités de 12-24 mois de dépenses résiduelles (1 500 × 18 = 27 000 €) hors portefeuille investi, et taux de retrait prudent (3,5 %).

---

## Résumé des enseignements clés (hard)

1. Sur un **horizon de 50 ans**, le 4 % standard est inadapté ; viser 3-3,5 % (capital cible ~857 k€ à 1 M€) — surcoût ~4 ans d'accumulation.
2. Le **risque de séquence** est central en décumulation longue : réserve de liquidités + flexibilité des dépenses + taux prudent.
3. **Coast FIRE** : un capital atteint tôt croît seul jusqu'à la retraite — commencer tôt réduit drastiquement le capital nécessaire (effet du temps).
4. **Barista FIRE** : un revenu d'appoint réduit fortement le capital cible (ici −37,5 %), mais ajoute une dépendance à ce revenu.
5. Disclaimers obligatoires : règle calibrée US/30 ans, fiscalité et frais non inclus, **pas de garantie**.
