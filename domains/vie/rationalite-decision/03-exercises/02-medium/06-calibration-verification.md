# Exercices Medium — Module 06 : Calibration & Vérification de l'Information

> **Niveau** : Medium | **Temps estimé** : ~35 min

---

## Exercice 1 — Score de Brier sur 12 prédictions + analyse de calibration

### Objectif

Calculer un score de Brier sur un ensemble plus large, le comparer à la baseline (0,25), puis **diagnostiquer la calibration** en regroupant les prédictions par tranches de confiance (over/under-confidence).

### Consigne

Vous tenez un journal de prévisions sur des événements neutres et observables. Après résolution, voici les 12 prédictions :

| # | Question | p prédit | Outcome |
|---|----------|----------|---------|
| 1 | Trajet domicile-bureau < 30 min | 0,55 | 1 |
| 2 | Réunion d'équipe écourtée | 0,55 | 0 |
| 3 | Équipe A bat équipe B (derby) | 0,60 | 1 |
| 4 | Température > 25 °C cet après-midi | 0,50 | 0 |
| 5 | Colis livré dans le créneau annoncé | 0,70 | 1 |
| 6 | Sprint terminé avant la deadline | 0,70 | 1 |
| 7 | Bus de 8h12 à l'heure | 0,70 | 0 |
| 8 | Mise à jour logicielle livrée ce vendredi | 0,70 | 1 |
| 9 | Pluie demain (> 50 % de couverture) | 0,90 | 1 |
| 10 | Livraison fournisseur à l'heure | 0,90 | 0 |
| 11 | Place de parking libre au sous-sol | 0,90 | 0 |
| 12 | Cafétéria ouverte à midi | 0,90 | 1 |

1. Calculez (p − o)² pour chacune des 12 prédictions, puis le **score de Brier global**.
2. Comparez à la baseline 0,25. Le résultat est-il **meilleur** que de toujours dire 0,50 ?
3. Regroupez les prédictions en **3 tranches** : 50-60 %, 70 %, 90 %. Pour chaque tranche, calculez la **probabilité moyenne annoncée** et la **fréquence réelle de réalisation** (part de `o = 1`).
4. Diagnostiquez : quelle(s) tranche(s) sont bien calibrées ? Laquelle révèle de la **surconfiance** (over-confidence) ou de la **sous-confiance** ? Quelle tranche tire le Brier vers le haut ?

### Critères de réussite

- [ ] Les 12 valeurs (p−o)² sont correctes (la plus coûteuse vaut 0,81, deux fois : #10 et #11).
- [ ] Somme = 3,315 ; **score de Brier = 0,2762** (± 0,005).
- [ ] Comparaison à 0,25 faite : ici le Brier est **légèrement au-dessus** de la baseline (donc moins bon que 0,50 partout) — c'est attendu vu le diagnostic.
- [ ] Tranche 50-60 % : p_moyen = 0,55 ; fréquence réelle = 2/4 = **0,50** → quasi calibrée.
- [ ] Tranche 70 % : p_moyen = 0,70 ; fréquence réelle = 3/4 = **0,75** → bien calibrée.
- [ ] Tranche 90 % : p_moyen = 0,90 ; fréquence réelle = 2/4 = **0,50** → **surconfiance** marquée, c'est cette tranche qui plombe le Brier.
- [ ] Conclusion nommée : deux tranches correctes, mais la surconfiance sur les "90 %" suffit à faire passer le Brier au-dessus de la baseline.

---

## Exercice 2 — Comparer deux prévisionnistes sur les mêmes événements

### Objectif

Comparer les scores de Brier de deux prévisionnistes sur **le même** lot d'événements et décider lequel est le mieux calibré — en distinguant informellement **calibration** et **résolution** (le fait d'oser s'éloigner de 0,50 quand c'est justifié).

### Consigne

Alex et Bo prédisent les **mêmes 10 événements neutres**. Voici leurs probabilités et les issues observées :

| # | Événement | p Alex | p Bo | Outcome |
|---|-----------|--------|------|---------|
| 1 | Colis n°1 livré aujourd'hui | 0,80 | 0,55 | 1 |
| 2 | Réunion reportée | 0,30 | 0,45 | 0 |
| 3 | Équipe locale gagne | 0,70 | 0,55 | 1 |
| 4 | Build CI passe du 1er coup | 0,90 | 0,60 | 1 |
| 5 | Pluie à midi | 0,20 | 0,45 | 0 |
| 6 | Vol au départ à l'heure | 0,60 | 0,55 | 1 |
| 7 | Tâche finie avant 17h | 0,40 | 0,50 | 0 |
| 8 | Match aux prolongations | 0,10 | 0,45 | 0 |
| 9 | Colis n°2 livré demain | 0,75 | 0,55 | 1 |
| 10 | Mise à jour déployée ce soir | 0,85 | 0,60 | 1 |

1. Calculez le **score de Brier d'Alex** et celui de **Bo** (chacun sur ses 10 prédictions).
2. Comptez, pour chacun, le nombre de **bons appels directionnels** (p > 0,5 quand o = 1, ou p < 0,5 quand o = 0).
3. Décidez : qui est **le mieux calibré** ? (Indice : plus bas = meilleur.)
4. Interprétez : les deux ont-ils le même nombre de bons appels ? Si oui, pourquoi leurs Brier diffèrent-ils autant ? Parlez de **résolution** (oser sortir de 0,50) vs **calibration**.

### Critères de réussite

- [ ] Score de Brier d'Alex = **0,0685** (somme des (p−o)² = 0,685) (± 0,005).
- [ ] Score de Brier de Bo = **0,1987** (somme = 1,9875) (± 0,005).
- [ ] Bons appels directionnels : Alex 10/10 **et** Bo 10/10 (même précision sur le sens).
- [ ] Conclusion : **Alex** est meilleur (Brier plus bas), bien que les deux aient les mêmes appels.
- [ ] L'écart est attribué à la **résolution** : Bo se colle à 0,50 (timide → gros (p−o)² même quand il a raison), Alex ose des probabilités tranchées et justifiées → récompensé par le Brier.
- [ ] La nuance est posée : un Brier bas combine **calibration** (les X % se réalisent X % du temps) ET **résolution** (s'écarter de 0,50 à bon escient).

---

## Exercice 3 — Appliquer SIFT à une affirmation virale douteuse

### Objectif

Dérouler la méthode **SIFT** (Stop, Investigate the source, Find better coverage, Trace to original) sur une affirmation neutre douteuse, avec des **requêtes exactes**, et lister au moins 3 signaux d'alerte.

### Consigne

Une publication très partagée affirme :

> "Une étude de 2023 menée par l'« Global Productivity Institute » et publiée dans le *Journal of Applied Cognitive Performance* montre que travailler debout 2 heures par jour augmente la productivité de **42 %**."

1. **Stop** : décrivez ce que vous faites *avant* de partager (état émotionnel, réflexe de pause).
2. **Investigate the source** : pratiquez la **lecture latérale** — quelles requêtes tapez-vous pour vérifier qui est le « Global Productivity Institute » et si la revue existe ? (Donnez les requêtes exactes.)
3. **Find better coverage** : quelles requêtes pour voir si des sources fiables ont relayé (ou démenti) ce chiffre ?
4. **Trace to original** : comment remontez-vous à l'étude réelle (titre, auteurs, DOI) ? Quelles requêtes sur Google Scholar et doi.org ?
5. Listez **au moins 3 signaux d'alerte** dans l'affirmation.

### Critères de réussite

- [ ] **Stop** explicite : ne pas partager avant vérification ; repérer le déclencheur émotionnel/"trop beau".
- [ ] **Investigate** : lecture latérale décrite, avec au moins une requête du type `"Global Productivity Institute" institut crédibilité` et `"Journal of Applied Cognitive Performance" indexé`.
- [ ] **Find** : requête de couverture du type `étude productivité travail debout 42 %` pour chercher recoupements/démentis.
- [ ] **Trace** : recherche du **titre exact entre guillemets** sur Google Scholar + vérification du **DOI sur doi.org** ; constat si l'étude est introuvable.
- [ ] **≥ 3 signaux d'alerte** parmi : chiffre rond et spectaculaire (42 %), institut au nom générique/invérifiable, revue jamais entendue/non indexée, absence de lien vers la source primaire, pas d'auteurs nommés, formulation "une étude montre" sans méthodologie.
- [ ] Conclusion calibrée : tant que la source primaire n'est pas retrouvée, traiter le chiffre comme **non vérifié** (probabilité de fiabilité basse), pas comme "faux" avec certitude.
