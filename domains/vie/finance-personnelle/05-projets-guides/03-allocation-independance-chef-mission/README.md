# Projet guide 03 — Allocation & indépendance financière : le tableau de bord d'un chef de mission

> **Disclaimer.** Projet **purement éducatif**. Rien ici n'est un conseil
> financier, fiscal ou en investissement personnalisé. Les rendements sont
> illustratifs et non garantis ; tout investissement comporte un risque de
> perte en capital. La « règle des 4 % » est une heuristique historique
> américaine (étude Trinity), **pas** une garantie. Contexte métier :
> [`shared/logistics-context.md`](../../../../../shared/logistics-context.md).

**Niveau** : difficile (intégrateur) · **Modules mobilisés** : 04 (investir long
terme), 05 (frais), 06 (indépendance financière), 01 (intérêts composés).

## 1. Contexte metier

Karim est **chef de mission** chez LogiSim : il pilote les déploiements FleetSim
chez les clients, gagne bien sa vie, et veut enfin **structurer son patrimoine**
au lieu de laisser dormir son argent. Il a quatre questions de fond :

1. Quelle **allocation** (actions / obligations) pour quel horizon ?
2. Combien lui coûtent vraiment les **frais** d'un fonds géré activement vs un fonds indiciel ?
3. À quel **capital** et dans **combien d'années** pourrait-il viser l'indépendance financière ?
4. Quel **risque** le guette précisément au moment où il commencera à vivre de son capital ?

Ce projet est le plus intégrateur du domaine : il rassemble intérêts composés,
allocation, frais et retrait soutenable dans **un seul tableau de bord**.

## 2. Objectif technique

Construire un **tableau de bord patrimonial** en quatre volets :
1. **allocation** suggérée par horizon (+ rendement attendu illustratif) ;
2. **actif vs passif** net de frais sur 30 ans (cadrage SPIVA) ;
3. **numéro FI** (capital cible) et **délai** pour l'atteindre selon le taux de retrait ;
4. **risque de séquence** : même rendement moyen, ordre des années différent.

## 3. Consigne

```python
def allocation_par_horizon(horizon_ans: int) -> Allocation: ...
def capital_apres_frais(versement_annuel, rendement_brut, frais, annees) -> float: ...
def numero_fi(depenses_annuelles, taux_retrait=0.04) -> float: ...
def annees_jusqu_a_fi(capital_actuel, epargne_annuelle, cible, rendement_net=0.05) -> int | None: ...
def simuler_retraite(capital_initial, retrait_annuel, rendements: list[float]) -> float: ...
```

Contraintes :
- **stdlib uniquement** (`dataclasses`, `statistics`), déterministe.
- `annees_jusqu_a_fi` retourne **`None`** si la cible n'est pas atteignable (garde-fou 100 ans).
- `simuler_retraite` retourne **0.0** si le capital est épuisé (pas de valeur négative).
- Le risque de séquence doit utiliser **les mêmes rendements** dans deux ordres (un seul `reversed`), pour que la moyenne soit *identique*.

## 4. Etapes guidees

1. **Allocation** — règle d'horizon simple et **assumée comme heuristique** : 90 % d'actions au-delà de 20 ans, puis on réduit. Le vrai déterminant n'est pas l'âge mais l'horizon **et** la tolérance au risque (capacité à ne pas vendre en baisse).
2. **Actif vs passif** — même `capital_apres_frais` appelé avec deux niveaux de frais (0,2 % ETF vs 1,8 % fonds actif), **même rendement brut**. Le cadrage SPIVA justifie l'hypothèse « même brut » : en moyenne, l'actif ne bat pas l'indice avant frais, encore moins après.
3. **Numéro FI** — `dépenses / taux_retrait`. À 4 %, c'est 25× les dépenses annuelles. Fais varier le taux (3,5 % → ~28,6× ; 5 % → 20×) et observe l'effet sur le capital cible **et** le délai.
4. **Délai jusqu'à FI** — simulation annuelle : `capital = capital·(1+r) + épargne`, arrêt quand `capital ≥ cible`. Garde-fou à 100 ans → `None`.
5. **Risque de séquence** — simule un retrait en **début** d'année puis applique le rendement. Compare une séquence avec le krach **à la fin** vs **au début** (l'inverse). Même moyenne, destins opposés.

## 5. Criteres de reussite

- [ ] `python solution/tableau_bord.py` tourne sans dépendance externe, déterministe
- [ ] L'allocation **diminue** la part actions quand l'horizon raccourcit
- [ ] Le fonds passif (0,2 %) **bat nettement** le fonds actif (1,8 %) à rendement brut égal — l'écart est chiffré et attribué aux frais
- [ ] Le numéro FI **augmente** quand le taux de retrait **baisse** (plus prudent = plus gros capital = plus d'années)
- [ ] `annees_jusqu_a_fi(...)` renvoie `None` si la cible est inatteignable
- [ ] Le risque de séquence montre un **écart significatif** entre krach-début et krach-fin, **à moyenne de rendement identique**

> **Piège à comprendre** : pendant la phase d'**accumulation**, l'ordre des
> rendements est *neutre* (seul le produit compte). Pendant la phase de
> **retrait**, l'ordre devient critique : subir un krach **juste après** avoir
> commencé à puiser dans le capital verrouille des pertes et épuise le
> portefeuille bien plus vite. Dans la solution, deux séquences de moyenne
> identique (4,6 %) finissent à **808 131 €** (krach en fin) contre **634 121 €**
> (krach au début) — **174 010 €** d'écart pour la *même* performance moyenne.

## 6. Corrige

Voir [`solution/tableau_bord.py`](./solution/tableau_bord.py) (commenté) et
[`solution/analyse.md`](./solution/analyse.md) pour la lecture détaillée, les
hypothèses et les limites (notamment le statut **non garanti** de la règle des 4 %).

## 7. Pour aller plus loin

- **Monte-Carlo léger** — au lieu de deux séquences, tire 1 000 ordres aléatoires (avec `random.shuffle`, seed fixée) et mesure le **taux d'échec** (portefeuille épuisé avant l'horizon).
- **Glide path** — fais décroître la part actions à l'approche de la retraite et mesure l'effet sur le risque de séquence.
- **Inflation** — indexe le retrait sur l'inflation (retrait *réel* constant) ; le risque de séquence s'aggrave.
- **Taux de retrait flexible** — réduis le retrait de 10 % les années de baisse et observe combien cela prolonge le capital.
