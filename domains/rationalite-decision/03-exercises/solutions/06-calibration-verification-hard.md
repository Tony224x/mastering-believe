# Solutions Hard — Module 06 : Calibration & Vérification de l'Information

*Les scores de Brier peuvent être vérifiés avec le script :*
`python domains/rationalite-decision/02-code/06-calibration-verification.py`

*Rappel : `B = (1/N) Σ (p − o)²`. 0 = parfait, 0,25 = baseline, 1 = pire ; plus bas = mieux.*

---

## Exercice 1 — Diagnostiquer puis recalibrer un journal de prévisions

**Étape 1 : (p − o)² et somme par tranche**

| Tranche (p) | Outcomes | (p − o)² par item | Somme tranche |
|-------------|----------|-------------------|---------------|
| 0,60 (n=5) | 1,1,0,0,1 | 0,16 · 0,16 · 0,36 · 0,36 · 0,16 | **1,2000** |
| 0,75 (n=4) | 1,1,1,0 | 0,0625 · 0,0625 · 0,0625 · 0,5625 | **0,7500** |
| 0,85 (n=4) | 1,0,0,1 | 0,0225 · 0,7225 · 0,7225 · 0,0225 | **1,4900** |
| 0,95 (n=3) | 1,0,0 | 0,0025 · 0,9025 · 0,9025 | **1,8075** |

```
Somme totale = 1,2000 + 0,7500 + 1,4900 + 1,8075 = 5,2475
N = 16
Brier global = 5,2475 / 16 = 0,3280
```
`0,3280 > 0,25` → le prévisionniste fait **moins bien que la baseline**. Le diagnostic dit pourquoi.

**Étape 2 : décomposition de la calibration**

| Tranche | p annoncé | Fréquence réelle | Écart (freq − p) | Verdict |
|---------|-----------|------------------|------------------|---------|
| 0,60 | 0,60 | 3/5 = 0,600 | 0,000 | calibrée |
| 0,75 | 0,75 | 3/4 = 0,750 | 0,000 | calibrée |
| 0,85 | 0,85 | 2/4 = 0,500 | **−0,350** | surconfiance |
| 0,95 | 0,95 | 1/3 ≈ 0,333 | **−0,617** | surconfiance forte |

Les tranches basses (0,60 ; 0,75) sont **parfaitement calibrées**. Toute la perte vient des **hautes confiances** : quand le prévisionniste dit 0,85 ou 0,95, les événements se réalisent bien moins souvent. Direction de l'erreur : **surconfiance** (p systématiquement au-dessus de la fréquence réelle).

**Étape 3 : règle de recalibration prescrite**

On ne touche pas aux tranches saines. On **rapproche les hautes confiances de leur fréquence observée** :
- Prédictions annoncées **0,85 → ramenées à 0,65** (la fréquence réelle est 0,50 ; on raccourcit sans surcorriger, l'échantillon étant petit).
- Prédictions annoncées **0,95 → ramenées à 0,70** (fréquence réelle ≈ 0,33 ; même prudence).
- Tranches **0,60 et 0,75 : inchangées**.

Principe : "shrink" (rétrécissement) des prédictions trop confiantes vers le centre, calibré sur les fréquences réelles plutôt que sur l'intuition.

**Étape 4 : effet chiffré sur le Brier**

On réapplique la règle aux **mêmes issues** :

| Tranche (recalibrée) | p | Outcomes | Somme (p − o)² |
|----------------------|------|----------|----------------|
| 0,60 (inchangée) | 0,60 | 1,1,0,0,1 | 1,2000 |
| 0,75 (inchangée) | 0,75 | 1,1,1,0 | 0,7500 |
| 0,65 (ex-0,85) | 0,65 | 1,0,0,1 | 0,1225 · 0,4225 · 0,4225 · 0,1225 = **1,0900** |
| 0,70 (ex-0,95) | 0,70 | 1,0,0 | 0,09 · 0,49 · 0,49 = **1,0700** |

```
Somme recalibrée = 1,2000 + 0,7500 + 1,0900 + 1,0700 = 4,1100
Brier recalibré  = 4,1100 / 16 = 0,2569
Δ Brier = 0,2569 − 0,3280 = −0,0711
```

**Lecture** : la recalibration fait baisser le Brier de **~0,071** (de 0,328 à 0,257) — il repasse au niveau de la baseline. La recalibration **ne change que les tranches surconfiantes** : rapprocher p de la fréquence réelle réduit mécaniquement la moyenne des (p − o)², sans rien dégrader ailleurs (les tranches 0,60 et 0,75 sont identiques avant/après). Levier d'entraînement concret : tenir le journal, repérer la surconfiance **par tranche**, et appliquer un shrink jusqu'à ce que les fréquences réelles rejoignent les probabilités annoncées.

---

## Exercice 2 — Protocole de vérification d'une réponse d'IA (citation + image)

Affirmation à vérifier : une plante d'intérieur réduirait les particules fines de **37 % en 24 h**, "selon Okonkwo & Lindqvist (2021), *Indoor phytoremediation efficiency in office environments*, International Journal of Atmospheric Bioremediation, DOI 10.1099/ijab.2021.88421", avec une photo "du bureau test".

### 1. Citation — lecture latérale
On quitte la réponse de l'IA et on enquête ailleurs :
```
"International Journal of Atmospheric Bioremediation" indexé Scopus OR DOAJ
"International Journal of Atmospheric Bioremediation" éditeur ISSN
Okonkwo Lindqvist phytoremediation indoor air auteurs
```
But : la revue existe-t-elle dans une base reconnue (DOAJ, Scopus, PubMed) ? Les auteurs ont-ils d'autres travaux traçables ?

### 2. Citation — Google Scholar (titre exact entre guillemets)
```
Google Scholar : "Indoor phytoremediation efficiency in office environments"
```
Les guillemets forcent la correspondance exacte du titre. **0 résultat ⇒ forte présomption de fabrication** : un vrai article publié est indexé. (On peut élargir avec les noms d'auteurs si le titre ne renvoie rien.)

### 3. Citation — DOI
On ouvre l'URL canonique :
```
https://doi.org/10.1099/ijab.2021.88421
```
Un DOI valide **résout** vers la page de l'article chez l'éditeur. S'il renvoie une erreur "DOI Not Found" ou ne mène nulle part, c'est un **signal majeur** : le DOI est inventé (les LLM fabriquent volontiers des DOI à l'allure plausible).

### 4. Image — reverse image search
On vérifie si la photo appartient vraiment à l'étude :
```
Google Images / Google Lens : importer l'image → "rechercher la source"
TinEye : importer l'image → trier par "oldest"
```
On regarde la **plus ancienne occurrence** et son contexte. Si l'image apparaît d'abord dans une banque d'images ou un article sans rapport, elle est **hors contexte** — repartagée pour donner une fausse caution visuelle.

### 5. Signaux d'hallucination (≥ 4)
1. Revue au nom générique et grandiloquent, non indexée → invérifiable.
2. **DOI qui ne résout pas** sur doi.org.
3. **Titre absent de Google Scholar** (recherche entre guillemets : 0 résultat).
4. Chiffre précis et flatteur (37 % en 24 h) sans méthodologie.
5. Aucune trace indépendante des auteurs.
6. Image non rattachable à l'étude (origine antérieure / hors sujet).
7. Assurance excessive du modèle ("Oui." catégorique) sur une question où la littérature réelle est nuancée.

### 6. Conclusion calibrée
Verdict : **probablement halluciné — à pondérer en conséquence.** On n'affirme pas "c'est faux à 100 %" (on n'a pas prouvé une négation), mais on **abaisse fortement** la probabilité accordée à la réponse : tant que la source primaire n'est pas retrouvée (Scholar + DOI), elle est traitée comme non fiable et non citable.

**Synthèse calibration + vérification** : la vérification n'est pas un interrupteur "vrai/faux", c'est un **ajustement bayésien de notre confiance**. Une citation invérifiable + un DOI mort + une image hors contexte font chuter la probabilité de fiabilité de la réponse de l'IA. C'est exactement le geste du module : convertir des signaux de vérification en une **croyance calibrée**, plutôt que d'accepter ou de rejeter l'affirmation en bloc.
