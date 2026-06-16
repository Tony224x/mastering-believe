# Solutions Medium — Module 06 : Calibration & Vérification de l'Information

*Les scores de Brier peuvent être vérifiés avec le script :*
`python domains/rationalite-decision/02-code/06-calibration-verification.py`

*Rappel du score de Brier : `B = (1/N) Σ (p − o)²`. 0 = parfait, 0,25 = baseline (toujours 0,50), 1 = pire ; plus bas = mieux ; objectif < 0,20.*

---

## Exercice 1 — Score de Brier sur 12 prédictions + analyse de calibration

**Étape 1 : (p − o)² pour chaque prédiction**

| # | p | o | (p − o)² |
|---|------|---|----------|
| 1 | 0,55 | 1 | 0,2025 |
| 2 | 0,55 | 0 | 0,3025 |
| 3 | 0,60 | 1 | 0,1600 |
| 4 | 0,50 | 0 | 0,2500 |
| 5 | 0,70 | 1 | 0,0900 |
| 6 | 0,70 | 1 | 0,0900 |
| 7 | 0,70 | 0 | 0,4900 |
| 8 | 0,70 | 1 | 0,0900 |
| 9 | 0,90 | 1 | 0,0100 |
| 10 | 0,90 | 0 | **0,8100** |
| 11 | 0,90 | 0 | **0,8100** |
| 12 | 0,90 | 1 | 0,0100 |

**Étape 2 : score de Brier global**
```
Somme des (p − o)² = 3,315
Brier = 3,315 / 12 = 0,2762
```

**Étape 3 : comparaison à la baseline**
```
Brier = 0,2762  >  0,25 (baseline)
```
Le journal fait ici **légèrement moins bien** que "toujours dire 0,50". Surprenant ? L'analyse par tranches l'explique.

**Étape 4 : analyse de calibration par tranches**

| Tranche | n | p moyen annoncé | Réalisés (o=1) | Fréquence réelle | Diagnostic |
|---------|---|-----------------|----------------|------------------|------------|
| 50-60 % | 4 | 0,55 | 2 | 0,50 | quasi calibrée |
| 70 % | 4 | 0,70 | 3 | 0,75 | bien calibrée |
| 90 % | 4 | 0,90 | 2 | 0,50 | **surconfiance forte** |

**Lecture** : deux tranches sur trois sont saines — quand on dit 70 %, ça arrive ~75 % du temps ; quand on dit ~55 %, ça arrive ~50 % du temps. Le problème vient de la tranche **90 %** : les événements annoncés "quasi certains" ne se réalisent qu'une fois sur deux. Ce sont les deux ratés #10 et #11 (chacun 0,81) qui dominent la somme et font passer le Brier au-dessus de la baseline.

**Conclusion** : la calibration n'est pas globale, elle se diagnostique **par niveau de confiance**. Ici, le levier d'amélioration est clair — **cesser de surestimer les "90 %"** (les ramener vers 0,60-0,70 tant que la fréquence réelle ne suit pas). C'est exactement le travail de recalibration vu dans l'exercice Hard 1.

---

## Exercice 2 — Comparer deux prévisionnistes sur les mêmes événements

**Issues observées** : `1, 0, 1, 1, 0, 1, 0, 0, 1, 1`

**Alex**

| # | p | o | (p − o)² |
|---|------|---|----------|
| 1 | 0,80 | 1 | 0,0400 |
| 2 | 0,30 | 0 | 0,0900 |
| 3 | 0,70 | 1 | 0,0900 |
| 4 | 0,90 | 1 | 0,0100 |
| 5 | 0,20 | 0 | 0,0400 |
| 6 | 0,60 | 1 | 0,1600 |
| 7 | 0,40 | 0 | 0,1600 |
| 8 | 0,10 | 0 | 0,0100 |
| 9 | 0,75 | 1 | 0,0625 |
| 10 | 0,85 | 1 | 0,0225 |

```
Somme Alex = 0,685
Brier Alex = 0,685 / 10 = 0,0685
```

**Bo**

| # | p | o | (p − o)² |
|---|------|---|----------|
| 1 | 0,55 | 1 | 0,2025 |
| 2 | 0,45 | 0 | 0,2025 |
| 3 | 0,55 | 1 | 0,2025 |
| 4 | 0,60 | 1 | 0,1600 |
| 5 | 0,45 | 0 | 0,2025 |
| 6 | 0,55 | 1 | 0,2025 |
| 7 | 0,50 | 0 | 0,2500 |
| 8 | 0,45 | 0 | 0,2025 |
| 9 | 0,55 | 1 | 0,2025 |
| 10 | 0,60 | 1 | 0,1600 |

```
Somme Bo = 1,9875
Brier Bo = 1,9875 / 10 = 0,1987
```

**Appels directionnels** (p > 0,5 ↔ o = 1, p < 0,5 ↔ o = 0) :
- Alex : **10 / 10**
- Bo : **10 / 10**

**Décision** : Alex est **mieux noté** — `0,0685 < 0,1987` (plus bas = meilleur). Les deux sont sous la baseline 0,25, mais Alex de très loin.

**Interprétation (résolution vs calibration)** : les deux prévisionnistes ont **exactement les mêmes appels directionnels** (10/10). Pourtant leurs Brier diffèrent d'un facteur ~3. La raison : **la résolution**.
- Bo se colle à 0,50 (entre 0,45 et 0,60) : il a "raison" sur le sens, mais en restant timide il encaisse de gros (p − o)² même quand il vise juste (0,55 contre une issue = 1 coûte 0,2025).
- Alex **ose** des probabilités tranchées (0,90, 0,10, 0,85…) et elles se vérifient : le Brier le récompense.

Un bon Brier combine donc **deux qualités** : la **calibration** (les X % se réalisent bien X % du temps) **et** la **résolution** (s'écarter de 0,50 à bon escient). Bo est honnête mais peu informatif ; Alex est informatif **et** justifié. Note : si Alex avait été tranché **mais souvent à côté**, son Brier aurait explosé — la confiance ne paie que si elle est calibrée.

---

## Exercice 3 — Appliquer SIFT à une affirmation virale douteuse

Affirmation à vérifier : *"Étude 2023 du « Global Productivity Institute », publiée dans le Journal of Applied Cognitive Performance : travailler debout 2 h/jour augmente la productivité de 42 %."*

**S — Stop**
Avant tout partage : reconnaître le réflexe émotionnel ("c'est exactement ce que je voulais croire / c'est spectaculaire"). Ne rien repartager tant que ce n'est pas vérifié. Le chiffre rond et flatteur (+42 %) est précisément le genre de contenu conçu pour circuler vite.

**I — Investigate the source (lecture latérale)**
On ne reste pas sur la page qui fait la promesse ; on ouvre de nouveaux onglets pour enquêter sur la source :
```
"Global Productivity Institute"
"Global Productivity Institute" institut crédibilité financement
"Journal of Applied Cognitive Performance" indexé Scopus OR DOAJ
"Journal of Applied Cognitive Performance" facteur d'impact éditeur
```
Objectif : l'institut a-t-il une existence vérifiable indépendamment ? La revue est-elle indexée dans des bases reconnues (DOAJ, Scopus, PubMed) ?

**F — Find better coverage**
On cherche si des sources fiables ont relayé ou démenti le chiffre :
```
étude productivité travail debout 42 %
bureau debout productivité méta-analyse
standing desk productivity 42% study debunk
```
Si seuls des blogs et posts se renvoient la même phrase sans source primaire, c'est un cercle d'auto-citation, pas une couverture.

**T — Trace to original**
On remonte à l'étude réelle :
```
Google Scholar : "travailler debout" productivité 2023
Google Scholar : titre exact entre guillemets si on en trouve un
```
Et on teste tout DOI fourni sur `https://doi.org/<le-doi>`. Si l'étude est introuvable sur Scholar et qu'aucun DOI ne résout, la source primaire n'existe (probablement) pas.

**Signaux d'alerte (≥ 3)**
1. Chiffre rond, précis et spectaculaire (+42 %) — typique du contenu viral.
2. Institut au nom générique et grandiloquent, invérifiable.
3. Revue jamais entendue / non indexée dans les bases reconnues.
4. Aucun lien vers la source primaire, pas d'auteurs nommés.
5. Formulation "une étude montre" sans méthodologie, taille d'échantillon ni durée.

**Conclusion calibrée** : tant que la source primaire n'est pas retrouvée, on classe l'affirmation comme **non vérifiée** (probabilité de fiabilité faible) — ce n'est pas "faux avec certitude", c'est "à ne pas relayer et à fort soupçon de fabrication". L'honnêteté épistémique consiste à dire "je n'ai pas pu confirmer", pas à inventer un démenti symétrique.
