# Exercices — Module 04 : Investir simplement et sur le long terme (niveau intermédiaire)

> **Niveau** : Intermédiaire | **Temps estimé** : 60-75 min
>
> **Matière première** : Théorie du Module 04 + simulateur `02-code/04-investir-long-terme.py`
>
> **Disclaimer** : exercices éducatifs. Les rendements sont **hypothétiques et illustratifs** — les marchés réels ne garantissent aucun rendement. Ce contenu est éducatif et **ne constitue pas un conseil financier personnalisé**.

---

## Exercice 1 — Rééquilibrer un portefeuille qui a dérivé

### Objectif
Comprendre concrètement pourquoi et comment rééquilibrer une allocation après que les marchés l'ont déformée, et voir comment le rééquilibrage "achète bas, vend haut" mécaniquement.

### Consigne

Hakim a un portefeuille de **100 000 €** alloué **70 % actions / 30 % obligations** (donc 70 000 € / 30 000 €). Sur un an, les actions montent de **+25 %** et les obligations de **+2 %**.

1. Calculez la valeur de chaque poche après un an, puis la valeur totale du portefeuille.
2. Calculez la **nouvelle allocation** en % (actions / obligations). De combien a-t-elle dérivé par rapport à la cible 70/30 ?
3. Pour revenir à 70/30, combien faut-il **vendre d'actions** (ou rediriger) ? Montrez le calcul.
4. Le module dit que le rééquilibrage permet d'"acheter des actions en solde pendant les baisses". Ici le marché a **monté** : qu'est-ce que le rééquilibrage fait alors mécaniquement (sur la classe qui a le plus monté) ? En quoi est-ce une discipline anti-biais (lien Module 05) ?
5. À quelle fréquence le module recommande-t-il de rééquilibrer, et selon quel seuil de dérive ?

### Critères de réussite
- [ ] Valeurs des deux poches et total après un an calculés correctement
- [ ] Nouvelle allocation en % calculée et dérive identifiée
- [ ] Montant à vendre/rediriger pour revenir à 70/30 calculé
- [ ] Le mécanisme "vendre ce qui a monté / acheter ce qui a baissé" est expliqué et relié à la discipline anti-biais
- [ ] Fréquence (≈ 1 fois/an) et seuil (≈ 5 points) cités d'après le module

---

## Exercice 2 — Pourquoi diversifier : tester l'intuition de Markowitz

### Objectif
Manipuler concrètement le principe de diversification (Markowitz) sur un mini-cas, pour saisir que combiner des actifs peu corrélés réduit le risque sans détruire le rendement espéré.

### Consigne

On considère deux actifs hypothétiques sur 3 scénarios d'année (illustratifs, non prédictifs) :

| Scénario | Actif Actions | Actif Obligations |
|---|---|---|
| Bon | +20 % | +1 % |
| Moyen | +8 % | +3 % |
| Mauvais | −25 % | +4 % |

1. Calculez le rendement **moyen** de chaque actif seul (moyenne des 3 scénarios).
2. Construisez un portefeuille **70 % actions / 30 % obligations**. Calculez son rendement dans chacun des 3 scénarios, puis son rendement moyen.
3. Comparez l'**amplitude** (écart entre le meilleur et le pire scénario) de l'actif Actions seul vs du portefeuille mixte. Le portefeuille mixte réduit-il l'amplitude (proxy du risque) ?
4. Dans le scénario "Mauvais" (krach), comment les obligations jouent-elles leur rôle d'"amortisseur" décrit au module ? Chiffrez la différence de perte entre 100 % actions et le portefeuille mixte.
5. En une phrase, reformulez le principe de Markowitz à partir de vos calculs.

### Critères de réussite
- [ ] Rendements moyens des deux actifs seuls calculés
- [ ] Rendement du portefeuille 70/30 calculé pour les 3 scénarios + moyenne
- [ ] Amplitude (max − min) comparée entre actions seules et portefeuille mixte
- [ ] Rôle d'amortisseur des obligations en cas de krach chiffré
- [ ] Principe de Markowitz reformulé à partir des résultats

---

## Exercice 3 — Construire une allocation "3 fonds" et comparer trois profils de risque

### Objectif
Appliquer l'allocation "3 fonds" à un profil, puis comparer la projection à 30 ans de trois allocations (dynamique / équilibrée / prudente) pour visualiser l'arbitrage rendement/risque.

### Consigne

**Profil** : Inès, 35 ans, horizon 30 ans, capital investi 10 000 €, capacité d'épargne 350 €/mois.

Hypothèses illustratives (rendements bruts) : actions monde développé **7 %**, actions émergentes **8 %**, obligations **3 %**. TER **0,20 %**.

**Partie A** : Proposez une allocation "3 fonds" **équilibrée** pour Inès et justifiez en 2-3 phrases.

**Partie B** : Calculez le **rendement net pondéré** puis la projection à 30 ans (capital + 350 €/mois) pour trois allocations :
- **Dynamique** : 100 % actions monde développé
- **Équilibrée** : 60 % dev / 20 % émergents / 20 % obligations
- **Prudente** : 30 % dev / 10 % émergents / 60 % obligations

*Vous pouvez utiliser `02-code/04-investir-long-terme.py` (`capital_final_mensuel` et `demo_allocation_3_fonds`).*

**Partie C** : Comparez les trois capitaux finaux. L'écart entre dynamique et prudente justifie-t-il, à lui seul, de tout mettre en actions ? Quelle dimension absente du calcul (volatilité, comportement, horizon) faut-il intégrer avant de trancher ?

### Critères de réussite
- [ ] Allocation équilibrée proposée (total 100 %) et justifiée par l'âge/horizon
- [ ] Rendement net pondéré calculé pour les 3 allocations
- [ ] Projection à 30 ans calculée pour les 3 allocations
- [ ] Comparaison critique : l'écart de rendement ne suffit pas à décider sans intégrer le risque/comportement
