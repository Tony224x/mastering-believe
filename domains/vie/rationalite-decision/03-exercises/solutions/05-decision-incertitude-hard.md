# Solutions Hard — Module 05 : Décision sous Incertitude

*Montants en k€ (ex. 1) et en € (ex. 2). Probabilités a posteriori arrondies à trois décimales ; les espérances tombent sur des valeurs exactes (147/0,525 = 280 ; −32/0,475 ≈ −67,4).*

---

## Exercice 1 — Arbre complet : étude de marché puis lancement (valeur de l'information)

**Données :** demande forte P = 0,45 (lancement +500), demande faible P = 0,55 (lancement −200), ne pas lancer 0. Étude coûte 40, fiabilité P(positif|forte) = 0,80, P(positif|faible) = 0,30.

### 1. Branche sans étude

```
E[lancer direct] = 0,45 × 500 + 0,55 × (−200) = 225 − 110 = 115 k€
```

115 > 0 → **sans étude, on lance**. Meilleure valeur sans information = **115 k€**.

### 2. Branche avec étude — probabilités et espérances conditionnelles

Probabilités jointes :

```
P(forte, positif) = 0,45 × 0,80 = 0,360
P(faible, positif) = 0,55 × 0,30 = 0,165
P(forte, négatif) = 0,45 × 0,20 = 0,090
P(faible, négatif) = 0,55 × 0,70 = 0,385
```

Marginales :

```
P(positif) = 0,360 + 0,165 = 0,525
P(négatif) = 0,090 + 0,385 = 0,475
```

A posteriori (Bayes) :

```
P(forte | positif) = 0,360 / 0,525 ≈ 0,686
P(forte | négatif) = 0,090 / 0,475 ≈ 0,189
```

Espérances de lancement après chaque rapport (en passant par les jointes) :

```
E[lancer | positif] = (0,360 × 500 + 0,165 × (−200)) / 0,525
                    = (180 − 33) / 0,525 = 147 / 0,525 = 280 k€   → on lance

E[lancer | négatif] = (0,090 × 500 + 0,385 × (−200)) / 0,475
                    = (45 − 77) / 0,475 = −32 / 0,475 ≈ −67,4 k€  → on n'lance pas (0)
```

### 3. Remontée et décision sur l'étude

```
Valeur au nœud étude = 0,525 × max(280, 0) + 0,475 × max(−67,4, 0)
                     = 0,525 × 280 + 0,475 × 0 = 147 k€
Nette du coût        = 147 − 40 = 107 k€
```

Comparaison : 107 k€ (avec étude) **<** 115 k€ (sans étude). **Politique optimale : ne pas faire l'étude, lancer directement.**

### 4. Valeur de l'information (EVSI)

L'étude vaut **147 k€** brut contre **115 k€** sans étude :

```
EVSI = 147 − 115 = 32 k€
```

Le test fournit donc 32 k€ d'information utile (essentiellement en évitant les lancements après un rapport négatif). Mais il **coûte 40 k€** : comme 40 > 32, **on ne l'achète pas**. Une étude peut être informative sans être rentable.

### 5. Sensibilité

- **Sur l'a priori de demande forte.** Lancer directement vaut p × 500 + (1 − p) × (−200) = 700p − 200. Ce montant s'annule pour **p = 200/700 ≈ 0,286**. En dessous de ~28,6 % de chances de demande forte, le lancement direct devient non rentable (on choisirait « ne pas lancer »), et la valeur de l'information augmenterait.
- **Sur le coût de l'étude.** L'étude deviendrait optimale dès que son coût passerait **sous l'EVSI de 32 k€** (ou si sa fiabilité, donc l'EVSI, augmentait). Décision robuste tant que le coût reste au-dessus de ce seuil.

---

## Exercice 2 — Paradoxe d'Ellsberg : ambiguïté, axiome violé, et choix d'un fournisseur

**Urne :** 90 jetons = 30 rouges + 60 « noirs ou jaunes » (proportion inconnue). Pari gagnant paie 100 €.

### 1. Espérances connues (A et D)

```
P(rouge)         = 30/90 = 1/3        → E[A] = (1/3) × 100 ≈ 33,33 €
P(noir ou jaune) = 60/90 = 2/3        → E[D] = (2/3) × 100 ≈ 66,67 €
```

Ces deux quantités sont **exactes quelle que soit** la répartition noir/jaune : on connaît le nombre de rouges (30) et le nombre de « non-rouges » (60). A et D ne dépendent donc d'**aucune** hypothèse sur le mélange ambigu.

### 2. B et C en fonction de l'inconnu

Posons P(noir) = b (inconnu, entre 0 et 2/3). Alors P(jaune) = 2/3 − b.

```
E[B] = P(noir) × 100               = 100·b €
E[C] = P(rouge ou jaune) × 100     = (1/3 + (2/3 − b)) × 100 = (1 − b)·100 €
```

### 3. Démonstration de l'incohérence

Pour un agent qui attribue **une** probabilité unique b :

```
A ≻ B  ⟺  33,33 > 100·b      ⟺  b < 1/3
D ≻ C  ⟺  66,67 > (1 − b)·100 ⟺  1 − b < 2/3  ⟺  b > 1/3
```

Le profil de choix modal **A ≻ B et D ≻ C** exige donc **simultanément** b < 1/3 **et** b > 1/3 : **contradiction**. Aucune probabilité unique b ne rationalise les deux choix.

Vérification numérique (3 valeurs de b) :

| b (P noir) | E[B] | E[C] | A−B | D−C |
|------------|------|------|-----|-----|
| 0,067 | 6,67 € | 93,33 € | +26,67 | −26,67 |
| 0,333 | 33,33 € | 66,67 € | 0 | 0 |
| 0,400 | 40,00 € | 60,00 € | −6,67 | +6,67 |

On voit que A−B et D−C sont **toujours de signes opposés** (sauf à l'égalité b = 1/3) : on ne peut jamais préférer A à B **et** D à C en même temps avec une probabilité unique.

**Axiome violé :** le **principe de la chose-sûre** (l'analogue de l'axiome d'indépendance chez Savage). Les paris C et D ne diffèrent de A et B que par l'ajout du résultat « jaune » commun ; ajouter un même événement aux deux options ne devrait pas inverser la préférence — or il l'inverse. Le phénomène est l'**aversion à l'ambiguïté** : les gens préfèrent les probabilités **connues** (rouge = 1/3, noir+jaune = 2/3) aux probabilités **inconnues** (noir seul, rouge+jaune).

### 4. Transfert : fournisseur « sûr » F1 vs fournisseur « inconnu » F2

F1 a un taux de défaut **connu = 1/3** ; F2 a un taux de défaut **inconnu** (potentiellement meilleur ou pire).

- **Cas où préférer F1 est défendable.** Si F2 risque d'être sélectionné défavorablement — le vendeur en sait plus que vous, ou les « bonnes » offres ne restent pas inconnues longtemps (sélection adverse) — alors l'inconnu penche statistiquement vers le mauvais côté. De même si un défaut grave est **catastrophique ou irréversible** (rappel produit, sécurité) : on paie une prime pour borner le pire cas. L'aversion à l'ambiguïté est ici une heuristique prudente.
- **Cas où c'est un piège.** Si l'ambiguïté est **symétrique et bénigne** (F2 a autant de chances d'être meilleur que pire) et que l'espérance de F2 dépasse celle de F1, refuser F2 par simple inconfort fait **renoncer à de la valeur**. C'est exactement l'erreur d'Ellsberg transposée : confondre « je ne connais pas la probabilité » avec « la probabilité est mauvaise ».
- **Action qui réduit l'ambiguïté au lieu de la fuir.** Plutôt que d'écarter F2 par principe, **transformer l'ambiguïté en risque mesuré** : passer une **commande pilote** limitée, demander un **audit** ou des références, ou imposer une **période d'essai** avec clause de sortie. On apprend le vrai taux de défaut à faible coût, puis on décide sur des probabilités estimées — ce qui est précisément la posture du module (méthode > conclusion).
