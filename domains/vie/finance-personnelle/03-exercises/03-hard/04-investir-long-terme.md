# Exercices — Module 04 : Investir simplement et sur le long terme (niveau avancé)

> **Niveau** : Avancé | **Temps estimé** : 75-90 min
>
> **Matière première** : Théorie du Module 04 (Sharpe, Fama, SPIVA, Markowitz) + simulateur `02-code/04-investir-long-terme.py`
>
> **Disclaimer** : exercices éducatifs. Les rendements sont **hypothétiques et illustratifs**. Le débat actif vs passif est traité **par la donnée** (SPIVA, arithmétique de Sharpe), sans jugement de valeur. Ce contenu est éducatif et **ne constitue pas un conseil financier**. Tout investissement comporte un risque de perte en capital.

---

## Exercice 1 — Quelle surperformance un fonds actif doit-il livrer pour "valoir" ses frais ?

### Objectif
Quantifier rigoureusement, par la donnée, l'obstacle que les frais imposent à un fonds actif — et relier ce calcul à l'arithmétique de Sharpe et aux données SPIVA.

### Consigne

On compare deux fonds pour un investissement de **10 000 € + 200 €/mois sur 25 ans** :
- **Fonds passif (indiciel)** : réplique l'indice, rendement brut **7 %**, TER **0,15 %**.
- **Fonds actif** : rendement brut = 7 % + α (sa "surperformance" brute éventuelle), TER **1,80 %**.

1. Calculez le capital final du fonds **passif** (rendement net 6,85 %) sur 25 ans (`capital_final_mensuel`).
2. Calculez le capital final du fonds **actif** pour α = **0 %**, **+1 %**, **+2 %** (rendements nets : 5,20 %, 6,20 %, 7,20 %). Présentez un tableau.
3. **Trouvez le seuil** : quel α brut le fonds actif doit-il livrer **chaque année pendant 25 ans** pour simplement **égaler** le fonds passif net de frais ? (Indice : net actif = net passif ⇒ 7 % + α − 1,80 % = 6,85 %.)
4. Reliez ce seuil à l'**arithmétique de Sharpe** : pourquoi est-il, en moyenne et par construction, très difficile pour l'ensemble des gérants actifs de livrer durablement cet α ?
5. Reliez aux données **SPIVA Year-End 2024** (≈ 92 % des fonds actions US sous-performent sur 20 ans) : qu'est-ce que cela dit sur la probabilité de choisir *à l'avance* le fonds actif qui livrera l'α requis ? Restez factuel (poser la donnée, pas de jugement).

### Critères de réussite
- [ ] Capital du fonds passif calculé (net 6,85 %, 25 ans)
- [ ] Tableau du fonds actif pour α = 0 / +1 / +2 % calculé
- [ ] Seuil d'α (≈ +1,65 %/an, chaque année) correctement déduit et interprété
- [ ] Lien avec l'arithmétique de Sharpe (avant frais = marché ; après frais, moyenne perdante) explicité
- [ ] Lien avec SPIVA formulé factuellement (difficulté de sélection ex-ante), sans jugement de valeur

---

## Exercice 2 — Concevoir une politique d'investissement écrite (Investment Policy Statement)

### Objectif
Synthétiser le module en une **politique d'investissement** personnelle, robuste face aux marchés et aux émotions, avec une analyse de sensibilité aux frais sur un patrimoine conséquent.

### Consigne

**Profil** : Sofiane, 40 ans, vient d'hériter et dispose de **250 000 €** à investir (capital déjà placé par ailleurs : nul), horizon 25 ans, pas de besoin de liquidité sur ces fonds, fonds d'urgence complet par ailleurs.

**Partie A — Sensibilité aux frais sur gros capital.**
Hypothèse de rendement brut **7 %**, capitalisation, 25 ans, **sans** versements (lump sum de 250 000 €). Calculez le capital final pour des TER de **0,10 %**, **0,50 %**, **1,00 %**, **2,00 %**. Puis chiffrez la **perte due aux frais** entre 0,10 % et 2,00 % (en € et en % du capital final de référence).

**Partie B — Allocation et mise en œuvre.**
Proposez une allocation "3 fonds" cohérente avec son horizon, un TER cible global, et une règle de mise en œuvre (investissement immédiat vs étalement — restez nuancé, le détail DCA relève du Module 05).

**Partie C — Politique écrite (5 règles).**
Rédigez une **politique d'investissement** en 5 règles maximum, couvrant : (1) l'allocation cible, (2) le seuil et la fréquence de rééquilibrage, (3) la règle de conduite en cas de krach > 20 %, (4) la fréquence de consultation du portefeuille, (5) les sources d'information retenues / évitées. Chaque règle doit être **opérationnelle** (pas un vœu vague).

**Partie D — Stress test.**
Le marché chute de 30 % l'année qui suit l'investissement. Que dit la théorie du module sur (a) ce qu'il faut faire, (b) ce que l'horizon de 25 ans change, et (c) la limite honnête de l'analyse (diversification réduit le risque mais ne l'élimine pas, aucun rendement garanti) ?

### Critères de réussite
- [ ] Tableau de sensibilité TER (0,10 / 0,50 / 1,00 / 2,00 %) calculé sur 250 000 € à 25 ans
- [ ] Perte due aux frais (0,10 % vs 2,00 %) chiffrée en € et en %
- [ ] Allocation "3 fonds" + TER cible + règle de mise en œuvre proposés
- [ ] Politique d'investissement en 5 règles, chacune opérationnelle
- [ ] Stress test traité avec honnêteté sur la preuve (horizon aide, mais aucun rendement garanti)
